#!/usr/bin/env python3
"""Build LightStereo-S from the converted weights with surgery.py, compile it into one Modalix MPK, and measure its
disparity fidelity.

Calibration and numerical fidelity use one seeded order of the scene-flow cal pairs: calibration takes the first pairs,
and fidelity compares the quantized graph's disparity with the ONNX's on a later span, before the MLA compile, so a
quantized graph that returns non-finite disparities stops the build early. Fidelity is diagnostic: the log and
fidelity.json record the difference from the ONNX, and score both disparities against the pairs' true disparity.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
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
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from model_registry.compile_script import component_parameters, fidelity_report, read_mpk_manifest
from model_registry.model import load_model as load_metadata
from scene_flow import INPUT, end_point_error, read_pairs, selected_pairs
from sima_utils.data.data_generator import DataGenerator
from surgery import INTERFACE, OUTPUT, annotate_bf16_regions, export_network

# Importing afe leaves the root logger at WARNING, so the recipe logs through its own INFO logger.
LOG = logging.getLogger("lightstereo_s.compile")
LOG.setLevel(logging.INFO)
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "lightstereo_s"
COMPONENT_ID = "model"
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INTERFACE[INPUT]) * 4}
OUTPUT_BYTES = [math.prod(INTERFACE[OUTPUT]) * 4]


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _bf16_regions(profile_id: str) -> list[str]:
    """Return the regions the build profile runs BF16, checking that its ID marks it mixed."""
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    regions = parameters.get("bf16_regions", [])
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 build: {regions}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def _quantization(parameters: dict[str, Any]) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision.

    Every precision calibrates: each instance normalization ends in a square root, which the Model Compiler runs as
    an INT8 lookup table even when activations are BF16, so its input range comes from the calibration pairs.
    """
    precision = parameters["precision"]
    calibration = HistogramMSEMethod(int(parameters["calibration_bins"]))
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


def _require_bf16_readout(quantized: Any) -> None:
    """Fail when the Model Compiler ignored the read-out's BF16 marks, which would silently build plain W8A8."""
    # The quantized model keeps its network private, and the compiler renames every node, so count by operation.
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.BFLOAT_16
    )
    LOG.info("BF16 operations: %s", dict(sorted(counts.items())))
    if counts["SoftmaxOp"] != 2:
        raise RuntimeError(f"expected both read-out softmaxes in BF16, found {dict(counts)}")


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
    names, manifest = read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    outputs = [node.get("size") for node in plugins[-1].get("output_nodes", [])]
    if outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def quantize(source: Path, profile_id: str, dataset: Path, seed: int) -> Any:
    """Quantize one build profile on the first calibration pairs of the seeded order."""
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    quantization = _quantization(parameters)
    count = int(parameters["calibration_samples"])
    frames, _ = read_pairs(dataset, selected_pairs(dataset, seed, 0, count))
    importer = onnx_source(
        str(source), shape_dict={INPUT: tuple(INTERFACE[INPUT])}, layout="NCHW", flexible_batch_size=True
    )
    loaded = load_model(importer, target=gen2_target)
    LOG.info("Quantizing %s, calibrating on %d scene-flow cal pairs", parameters["precision"], count)
    try:
        return loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): frames})),
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
    except SystemExit as error:
        # AFE reports a failed quantization or compile by raising SystemExit, which would exit with status 0.
        raise RuntimeError(f"Model Compiler failed to quantize {MODEL_ID} {profile_id}") from error


def compile_model(quantized: Any, output_dir: Path, profile_id: str) -> Path:
    """Compile the quantized graph into the build's MPK and audit it."""
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        try:
            quantized.compile(
                output_path=str(build_dir / "mpk"),
                batch_size=1,
                compress=True,
                log_level=logging.INFO,
                tessellate_parameters=None,
                l2_caching_mode=L2CachingMode.NONE,
            )
        except SystemExit as error:
            raise RuntimeError(f"Model Compiler failed to compile {MODEL_ID} {profile_id}") from error
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        # Keep the archive before auditing it, so a rejected MPK stays available for diagnosis.
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    audit_mpk(archive)
    return archive


def measure_fidelity(
    source: Path, candidate: Any, dataset: Path, config: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    """Compare the quantized graph's disparity with the ONNX source on the declared Scene Flow pairs."""
    selection = config["selection"]
    selected = selected_pairs(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    frames, truths = read_pairs(dataset, selected)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    compared = []
    scores: dict[str, list[float]] = {"ONNX": [], "quantized": []}
    for index, (left, _, _) in enumerate(selected):
        frame = frames[index : index + 1]
        expected = reference.run([OUTPUT], {INPUT: frame.transpose(0, 3, 1, 2)})[0].astype(np.float64)
        outputs = candidate.execute(inputs={InputName(INPUT): frame}, use_jax=True)
        actual = np.asarray(outputs[0], dtype=np.float64).reshape(expected.shape)
        compared.append({OUTPUT: (expected, actual)})
        scores["ONNX"].append(end_point_error(expected, truths[index]))
        scores["quantized"].append(end_point_error(actual, truths[index]))
        LOG.info(
            "Fidelity %s: %.4f px from the ONNX; Scene Flow EPE %.3f px ONNX, %.3f px quantized",
            left,
            float(np.mean(np.abs(actual - expected))),
            scores["ONNX"][-1],
            scores["quantized"][-1],
        )
    onnx_epe, quantized_epe = float(np.mean(scores["ONNX"])), float(np.mean(scores["quantized"]))
    LOG.info(
        "Scene Flow EPE on the %d fidelity pairs: %.3f px ONNX, %.3f px quantized (%+.1f%%)",
        len(selected),
        onnx_epe,
        quantized_epe,
        100 * (quantized_epe / onnx_epe - 1),
    )
    return fidelity_report(
        MODEL_ID,
        profile_id,
        config,
        compared,
        reference="ONNX source (ONNX Runtime)",
        candidate="Quantized AFE graph (JAX execute)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, required=True, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth":
        parser.error("--model must be the verified weights.pth from the conversion workflow")
    if not weights.is_file():
        parser.error(f"model does not exist: {weights}")
    fidelity = _metadata()["numerical_fidelity"]
    output_dir = args.output_dir.expanduser().resolve()
    seed = int(fidelity["selection"]["seed"])
    regions = _bf16_regions(args.build_profile)
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_network(weights, Path(temporary) / "model.onnx")
        # Fidelity runs the unmarked export in ONNX Runtime, so the marks go on a copy.
        graph = source
        if regions:
            graph = Path(temporary) / "model-bf16-regions.onnx"
            LOG.info("Marked %d operations BF16 for %s", annotate_bf16_regions(source, graph, regions), regions)
        quantized = quantize(graph, args.build_profile, args.dataset.expanduser().resolve(), seed)
        if regions:
            _require_bf16_readout(quantized)
        LOG.info("Measuring numerical fidelity on %d pairs", fidelity["selection"]["samples"])
        report = measure_fidelity(
            source, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity, args.build_profile
        )
        LOG.info("Fidelity: %s", report["metrics"][0])
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The calibration frames and the float graph are released; collecting them lowers the compile's peak memory.
        gc.collect()
        # Inside the block: the MPK manifest records the checksum of the ONNX it compiled from.
        LOG.info("Compiled MPK: %s", compile_model(quantized, output_dir, args.build_profile))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
