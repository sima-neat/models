#!/usr/bin/env python3
"""Rewrite the converted DETR ResNet-101 ONNX for the MLA and compile it into one Modalix MPK."""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

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
from calibration import calibration_images
from model_registry.compile_script import component_parameters, parse_arguments, read_mpk_manifest
from sima_utils.data.data_generator import DataGenerator
from surgery import BF16_REGIONS, INPUT, INPUT_SHAPE, OUTPUT_SHAPES, OUTPUTS, annotate_bf16_regions, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "detr_resnet_101"
COMPONENT_ID = "model"
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPES[name]) * 4 for name in OUTPUTS]
# What AFE lowers the attention blocks' BF16 operations to: one softmax per block, and a slice-concat for each of the
# 50 Q, K and V head splits and 17 head merges. The input projection's flatten into tokens is one more, which stays
# INT8 with the backbone.
ATTENTION_BF16_OPERATIONS = {"SoftmaxOp": 17, "SliceConcatOp": 67}


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


def _operation_counts(quantized: Any) -> tuple[Counter[str], Counter[str]]:
    """Count the quantized operations by type: all of them, and those that run BF16."""
    # The quantized model keeps its network private, and AFE renames every node, so count by operation.
    operations, bf16 = Counter(), Counter()
    for node in quantized._net.iter_nodes_recursive():
        config = getattr(node.ir, "quant_config", None)
        if config is None:
            continue
        kind = type(node.ir.operation).__name__
        operations[kind] += 1
        if config.quantization_precision.value == QuantizationPrecision.BFLOAT_16:
            bf16[kind] += 1
    return operations, bf16


def _require_bf16_regions(bf16: Counter[str], regions: list[str]) -> None:
    """Fail when AFE ignored a BF16 annotation, which would silently requantize the attention to INT8."""
    logging.info("BF16 operations: %s", dict(sorted(bf16.items())))
    expected = {kind: count if "attention" in regions else 0 for kind, count in ATTENTION_BF16_OPERATIONS.items()}
    found = {kind: bf16[kind] for kind in ATTENTION_BF16_OPERATIONS}
    if not bf16 or found != expected:
        raise RuntimeError(f"expected BF16 attention operations {expected}, found {found}")


def _quantization(precision: str, calibration_bins: int | None) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. Only INT8 activations need calibration."""
    if precision == "w8a8":
        return QuantizationParams(
            # Histogram MSE picks each activation range from the calibration images.
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
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel != "pass_through" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
    regions = _bf16_regions(profile_id, parameters)
    calibration_data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        samples = calibration_images(dataset, int(parameters["calibration_samples"]))
        logging.info("Calibrating %s with %d sample(s)", precision, len(samples))
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = prepare_model(source, build_dir / "graphs")

        if regions:
            # ONNX Runtime cannot run the annotations, so they go in after prepare_model's numerical check.
            logging.info("Precision marks for %s: %s", regions, annotate_bf16_regions(graph, regions))
        importer = onnx_source(
            str(graph), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NCHW", flexible_batch_size=True
        )
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        operations, bf16 = _operation_counts(quantized)
        logging.info("Quantized operations: %s", dict(sorted(operations.items())))
        if regions:
            _require_bf16_regions(bf16, regions)
        quantized.compile(
            output_path=str(build_dir / "mpk"),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            # The option applies to every layer norm and INT8 ones reject it, so a build keeps FP32 intermediates
            # only when all its layer norms run BF16.
            layer_norm_use_fp32_intermediates=bf16["LayerNormOp"] == operations["LayerNormOp"] > 0,
        )
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0])
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    return archive


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    # force replaces any handler an import installed first, which would otherwise drop these lines from the job log.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    archive = compile_model(arguments.model, arguments.output_dir, arguments.build_profile, arguments.dataset)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
