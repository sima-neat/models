#!/usr/bin/env python3
"""Compile the converted SegFormer-B0 ONNX into one Modalix MPK and measure its numerical fidelity."""

from __future__ import annotations

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
    BiasCorrectionType,
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
from calibration import IMAGE_SIZE, calibration_samples, fidelity_names, preprocess_image, read_images
from model_registry.compile_script import component_parameters, fidelity_report, parse_arguments, read_mpk_manifest
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import BF16_REGIONS, annotate_bf16_regions

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "segformer_b0_ade_512"
COMPONENT_ID = "model"
INPUT = "pixel_values"
INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
CLASS_COUNT = 150
LOGITS_SHAPE = (1, CLASS_COUNT, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(LOGITS_SHAPE) * 4]
# In the mixed build, four patch-embedding LayerNorms normalize INT8 inputs; the other 20 normalize BF16 inputs.
# AFE lowers all 24 marked LayerNorms' scale and shift as BF16 convolutions.
# Unfused GELUs are identified by their Erf operation.
REGION_BF16_OPERATIONS = {
    "attention_core": {("SoftmaxOp",): 8},
    "norms": {("LayerNormOp",): 20, ("ConvAddActivationOp",): 24},
    "gelu": {("GeluOp", "ErfOp"): 8},
}


def _bf16_regions(profile_id: str, parameters: dict[str, Any]) -> list[str]:
    """Return the regions the build profile runs BF16, checking that its ID marks it mixed."""
    regions = list(parameters.get("bf16_regions", []))
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 build: {regions}")
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    if unknown:
        raise ValueError(f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def _check_precision(quantized: Any, marks: dict[str, str], regions: list[str]) -> None:
    """Verify source precision marks and the BF16 operation counts after AFE lowering."""
    bf16, mismatches = Counter(), []
    for node in quantized._net.iter_nodes_recursive():
        config = getattr(node.ir, "quant_config", None)
        if config is None or config.quantization_precision.value is None:
            continue
        precision = config.quantization_precision.value
        kind = type(node.ir.operation).__name__
        if precision == QuantizationPrecision.BFLOAT_16:
            bf16[kind] += 1
        # LayerNorm's mark applies to its affine convolution; its normalization is checked by count.
        if kind == "LayerNormOp":
            continue
        marked = {QuantizationPrecision(marks[name]) for name in node.source_node_names or () if name in marks}
        if marked and marked != {precision}:
            mismatches.append(
                f"{node.name} ({kind}): {getattr(precision, 'value', '-')}, "
                f"expected {sorted(item.value for item in marked)} from {node.source_node_names}"
            )
    if mismatches:
        raise RuntimeError("Operations did not quantize at their marked precision:\n" + "\n".join(mismatches))
    _require_bf16_regions(bf16, regions)


def _require_bf16_regions(bf16: Counter[str], regions: list[str]) -> None:
    """Fail when AFE ignored a BF16 mark, which would silently requantize the region to INT8."""
    logging.info("BF16 operations: %s", dict(sorted(bf16.items())))
    expected, found = {}, {}
    for region, counts in REGION_BF16_OPERATIONS.items():
        for kinds, count in counts.items():
            key = f"{region}:{'|'.join(kinds)}"
            expected[key] = count if region in regions else 0
            found[key] = sum(bf16[kind] for kind in kinds)
    if found != expected:
        raise RuntimeError(f"expected BF16 operations by region {expected}, found {found}")


def _quantization(precision: str, calibration_bins: int | None) -> QuantizationParams:
    """Only INT8 activations need calibration; BF16 uses the SDK's skip-calibration path."""
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=HistogramMSEMethod(calibration_bins),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


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
    outputs = [node.get("size") for node in (plugins[-1] if plugins else {}).get("output_nodes", [])]
    if outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> tuple[Path, Any]:
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    precision = str(parameters["precision"])
    regions = _bf16_regions(profile_id, parameters)
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
    calibration_data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        samples = calibration_samples(dataset, int(parameters["calibration_samples"]))
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph, marks = source, {}
        if regions:
            # The marks go on a copy, so the fidelity check still compares with the unmarked source.
            graph = build_dir / "graphs" / "model.onnx"
            graph.parent.mkdir()
            shutil.copyfile(source, graph)
            marks = annotate_bf16_regions(graph, regions)
            logging.info("Precision marks for %s: %s", regions, dict(Counter(marks.values())))
        importer = onnx_source(str(graph), shape_dict={INPUT: INPUT_SHAPE}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        logging.info("Quantizing %s", precision)
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        if regions:
            _check_precision(quantized, marks, regions)
        quantized.compile(
            output_path=str(build_dir),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
        )
        archives = sorted(build_dir.glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0])
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    return archive, quantized


def _logits(outputs: Any) -> np.ndarray:
    """The quantized graph's logits in the ONNX NCHW order; the SDK returns them channels-last."""
    logits = np.asarray(outputs[0], dtype=np.float64)
    if logits.shape == (1, *LOGITS_SHAPE[2:], CLASS_COUNT):
        logits = logits.transpose(0, 3, 1, 2)
    if logits.shape != LOGITS_SHAPE:
        raise RuntimeError(f"unexpected logits shape {logits.shape}")
    return logits


def measure_fidelity(source: Path, candidate: Any, dataset: Path, config: dict[str, Any], profile_id: str) -> dict:
    """Compare the quantized graph's logits and per-pixel classes with the ONNX source on val2017 images."""
    selection = config["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    compared = []
    agreeing = pixels = 0
    for name in names:
        sample = preprocess_image(images[name])
        expected = reference.run(None, {INPUT: sample.transpose(2, 0, 1)[None]})[0]
        actual = _logits(candidate.execute(inputs={InputName(INPUT): sample[None]}, use_jax=True))
        compared.append({"logits": (expected, actual)})
        agreeing += int(np.sum(actual.argmax(1) == expected.argmax(1)))
        pixels += expected.shape[2] * expected.shape[3]
    # The fidelity schema holds only the numerical metrics, so the class agreement goes to the log.
    logging.info("Per-pixel class agreement with the ONNX source: %.4f over %d images", agreeing / pixels, len(names))
    return fidelity_report(
        MODEL_ID,
        profile_id,
        config,
        compared,
        reference="ONNX source (ONNX Runtime)",
        candidate="Quantized AFE graph (JAX execute)",
    )


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    # force replaces any handler an import installed first, which would otherwise drop these lines from the job log.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)

    fidelity = load_metadata(MODEL_DIR)["numerical_fidelity"]
    archive, quantized = compile_model(
        arguments.model, arguments.output_dir, arguments.build_profile, arguments.dataset
    )
    logging.info("Compiled MPK: %s", archive)
    logging.info("Measuring numerical fidelity on %d images", fidelity["selection"]["samples"])
    report = measure_fidelity(arguments.model, quantized, arguments.fidelity_dataset, fidelity, arguments.build_profile)
    (arguments.output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logging.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
