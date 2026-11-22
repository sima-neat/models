#!/usr/bin/env python3
"""Build LightGlue's matcher from the converted weights with surgery.py, compile it into a Modalix MPK and measure
its numerical fidelity.

surgery.py exports a static ONNX: two images' 512 keypoint encodings and descriptors in, their soft assignment out.
Calibration and numerical fidelity use one seeded order of the yfcc100m-keypoints cal pairs: the INT8-activation
build, w8a8, calibrates on the first pairs, and fidelity compares the quantized graph's assignment with the ONNX's on
a later span. A W8A8 build runs the operations of its bf16_regions, if it declares any, in BF16.
A build whose matches overlap the ONNX's too little stops there, before the MLA compile.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import INPUT_SHAPES, KEYPOINTS, calibration_inputs, fidelity_inputs, matches
from model_registry.compile_script import component_parameters, fidelity_report, read_mpk_manifest
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import BF16_REGIONS, INPUTS, annotate_bf16_regions, assignment_matrix, export_matcher

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "lightglue_superpoint"
COMPONENT_ID = "model"
OUTPUT_NAME = "assignment"
# The least overlap of a build's matches with the ONNX's. The assignment's own cosine overstates the change: its entries
# sit near 0 or 1, and a keypoint whose two best partners are close can swap them under BF16 rounding. BF16 activations
# keep most matches and INT8 ones may lose more but must not collapse; a broken operator or a collapsed build shows up
# as a low overlap. The floors are per precision.
MATCH_OVERLAP_FLOORS = {"w16a16": 0.9, "w8a16": 0.9, "w8a8": 0.6}
# Softmaxes in the attention region: one per attention, in nine layers of self and cross attention on two images.
ATTENTION_SOFTMAXES = 36


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _bf16_activations(precision: str) -> bool:
    """w16a16 and w8a16 keep activations in BF16, which need no calibrated ranges; w8a8's INT8 activations do."""
    return precision in ("w16a16", "w8a16")


def _bf16_regions(profile_id: str, parameters: dict[str, Any]) -> list[str]:
    """The regions the build profile runs in BF16, checking that its ID marks it mixed."""
    regions = parameters.get("bf16_regions", [])
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 build: {regions}")
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    if unknown:
        raise ValueError(f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def _require_bf16_regions(quantized: Any, regions: list[str]) -> None:
    """Fail when the compiler ignored the BF16 annotations, which would silently build plain W8A8."""
    # The quantized model keeps its network private, and the compiler renames every node, so count by operation.
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.BFLOAT_16
    )
    print(f"BF16 operations: {dict(sorted(counts.items()))}", flush=True)
    softmaxes = ATTENTION_SOFTMAXES if "attention" in regions else 0
    if not counts or counts["SoftmaxOp"] != softmaxes:
        raise RuntimeError(f"expected BF16 operations including {softmaxes} softmaxes, found {dict(counts)}")


def _quantization_config(parameters: dict[str, Any]) -> QuantizationParams:
    """The compiler's activation and weight formats for a build."""
    precision = parameters["precision"]
    if precision == "w8a8":
        return QuantizationParams(
            # Histogram MSE picks each activation range from the calibration pairs.
            calibration_method=HistogramMSEMethod(int(parameters["calibration_bins"])),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each keypoint's activations get their own INT8 scale before an INT8-weight convolution.
            per_token_quant=True,
        )
    raise ValueError(f"this recipe supports only the declared Modalix BF16 and INT8 builds, not {precision!r}")


def quantize(source: Path, profile_id: str, dataset: Path | None, seed: int) -> Any:
    """Quantize the ONNX; the INT8-activation build calibrates on the first pairs of the cal pairs' seeded order."""
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    config = _quantization_config(parameters)
    regions = _bf16_regions(profile_id, parameters)
    graph = source
    if regions:
        graph = source.with_name("annotated.onnx")
        marks = annotate_bf16_regions(source, graph, regions)
        print(f"Marked {marks['bfloat16']} operations BF16 and {marks['int8']} INT8 for {regions}", flush=True)
    shapes = {name: tuple(shape) for name, shape in INPUTS.items()}
    importer = onnx_source(str(graph), shape_dict=shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    calibration_data = None
    if not _bf16_activations(parameters["precision"]):
        samples = calibration_inputs(dataset, seed, int(parameters["calibration_samples"]))
        print(f"Calibrating on {int(parameters['calibration_samples'])} yfcc100m-keypoints cal pairs", flush=True)
        data = {InputName(name): samples[name] for name in INPUT_SHAPES}
        calibration_data = convert_data_generator_to_iterable(DataGenerator(data))
    quantized = loaded.quantize(
        calibration_data=calibration_data,
        quantization_config=config,
        model_name=f"{MODEL_ID}_{profile_id}",
        log_level=logging.INFO,
    )
    if regions:
        _require_bf16_regions(quantized, regions)
    return quantized


def audit_mpk(archive: Path) -> dict[str, Any]:
    """Require one MLA plugin, no CPU plugins and the matcher's four float32 inputs."""
    names, manifest = read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = manifest.get("plugins", [])
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    expected = {name: math.prod(shape) * 4 for name, shape in INPUT_SHAPES.items()}
    if public_inputs != expected:
        raise RuntimeError(f"MPK input interface mismatch: {public_inputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins)}


def compile_model(quantized: Any, component_dir: Path, precision: str) -> Path:
    component_dir.mkdir(parents=True, exist_ok=False)
    quantized.compile(
        output_path=str(component_dir),
        batch_size=1,
        compress=True,
        log_level=logging.INFO,
        tessellate_parameters=None,
        l2_caching_mode=L2CachingMode.NONE,
        preserve=True,
        # The option applies to every layer norm, and INT8 ones reject it.
        layer_norm_use_fp32_intermediates=_bf16_activations(precision),
    )
    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    audit_mpk(archives[0])
    print(f"Compiled MPK: {archives[0]}", flush=True)
    return archives[0]


def match_overlap(assignments: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """The overlap of the matches LightGlue extracts from each pair's two assignments, over every pair:
    |A & B| / sqrt(|A| |B|) for A and B the two sets of all pairs' matches."""
    shared = expected_count = actual_count = 0
    for expected, actual in assignments:
        expected_matches = {tuple(row) for row in matches(expected).tolist()}
        actual_matches = {tuple(row) for row in matches(actual).tolist()}
        shared += len(expected_matches & actual_matches)
        expected_count, actual_count = expected_count + len(expected_matches), actual_count + len(actual_matches)
    return shared / math.sqrt(expected_count * actual_count) if expected_count and actual_count else 0.0


def require_match_overlap(precision: str, overlap: float) -> None:
    """Stop a build whose matches overlap the ONNX's no more than its precision's floor."""
    floor = MATCH_OVERLAP_FLOORS[precision]
    if not overlap > floor:
        raise RuntimeError(f"{precision}'s matches overlap the ONNX's by {overlap:.4f}, not above the {floor} floor")


def measure_fidelity(
    source: Path, quantized: Any, dataset: Path, fidelity: dict[str, Any], profile_id: str
) -> tuple[dict[str, Any], float]:
    """Compare the quantized graph with the ONNX source on the declared cal pairs: the fidelity report of the
    assignment, and the overlap of the matches LightGlue extracts from the two assignments."""
    selection = fidelity["selection"]
    samples = fidelity_inputs(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    assignments = []
    for inputs in samples:
        # The inputs are channels-last, as the compiler takes them; the ONNX takes them channels-first.
        (output,) = reference.run(
            [OUTPUT_NAME], {name: np.ascontiguousarray(value.transpose(0, 3, 1, 2)) for name, value in inputs.items()})
        expected = assignment_matrix(output)
        outputs = quantized.execute(inputs={InputName(name): value for name, value in inputs.items()}, use_jax=True)
        (actual,) = [np.asarray(output) for output in outputs]
        # The compiler returns the assignment channels-last, [1, 1, i, j].
        if actual.shape != (1, 1, KEYPOINTS, KEYPOINTS):
            raise RuntimeError(f"the quantized graph returns its assignment as {actual.shape}, expected "
                               f"{(1, 1, KEYPOINTS, KEYPOINTS)}")
        assignments.append((expected, actual.reshape(KEYPOINTS, KEYPOINTS)))
    report = fidelity_report(
        MODEL_ID,
        profile_id,
        fidelity,
        [{OUTPUT_NAME: assignment} for assignment in assignments],
        reference="ONNX source (ONNX Runtime)",
        candidate="Quantized AFE graph (JAX execute)",
    )
    return report, match_overlap(assignments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")

    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth" or not weights.is_file():
        parser.error(f"--model must be the converted weights.pth, not {weights}")
    precision = str(component_parameters(MODEL_DIR, args.build_profile, COMPONENT_ID)["precision"])
    if not _bf16_activations(precision) and args.dataset is None:
        parser.error(f"{args.build_profile} calibrates; pass --dataset")
    fidelity = _metadata()["numerical_fidelity"]
    if args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    output_dir = args.output_dir.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_matcher(weights, Path(temporary) / "model.onnx")
        # Calibration takes the first pairs of the order whose later span numerical fidelity measures.
        quantized = quantize(source, args.build_profile, dataset, int(fidelity["selection"]["seed"]))
        # Before the MLA compile, so a broken quantization stops the build early. Printed, not logged: the Model
        # Compiler's logging setup drops this script's INFO records.
        print(f"Measuring numerical fidelity on {fidelity['selection']['samples']} pairs", flush=True)
        report, overlap = measure_fidelity(source, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity,
                                           args.build_profile)
        for metric in report["metrics"]:
            print(f"Fidelity {metric['name']}: cosine {metric['cosine_similarity']:.4f}, "
                  f"max |error| {metric['max_absolute_error']:.4f}", flush=True)
        print(f"Fidelity matches: overlap {overlap:.4f} with the ONNX's", flush=True)
        require_match_overlap(precision, overlap)
        compile_model(quantized, output_dir / COMPONENT_ID, precision)
    (output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
