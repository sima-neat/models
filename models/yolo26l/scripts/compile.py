#!/usr/bin/env python3
"""Cut the converted YOLO26l ONNX at its raw heads and compile it into one Modalix MPK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import IMAGE_SIZE, calibration_images
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import INPUT, INPUT_SHAPE, OUTPUT_SHAPES, OUTPUTS, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "yolo26l"
COMPONENT_ID = "model"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPES[name]) * 4 for name in OUTPUTS]


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str, calibration_bins: int | None) -> Any:
    int8_weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    if precision == "w8a8":
        config = (
            default_quantization.with_activation_quantization(
                quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            )
            .with_weight_quantization(int8_weights)
            .with_calibration(HistogramMSEMethod(calibration_bins))
            .with_bias_correction(BiasCorrectionType.REGULAR)
        )
    elif precision == "w16a16":
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(bfloat16_scheme())
        )
    elif precision == "w8a16":
        # BF16 activations need no calibrated ranges, and bias correction would measure the placeholder sample.
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(int8_weights)
            .with_calibration(SkipCalibration())
            .with_bias_correction(BiasCorrectionType.NONE)
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
    # with_activation_quantization turns per-token quantization on for BF16; this recipe keeps it off.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(False)


def _read_mpk_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        return names, json.load(source)


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
    names, manifest = _read_mpk_manifest(archive)
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
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}"
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
    parameters = _parameters(profile_id)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        samples = calibration_images(dataset, int(parameters["calibration_samples"]))
    else:
        # BF16 derives nothing from representative data, but quantize() still needs one sample.
        samples = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = prepare_model(source, build_dir / "graphs")
        importer = onnx_source(
            str(graph), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NCHW", flexible_batch_size=True
        )
        loaded = load_model(importer, target=gen2_target)
        logging.info("Quantizing %s with %d sample(s)", precision, len(samples))
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples})),
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        quantized.compile(
            output_path=str(build_dir / "mpk"),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
        )
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0])
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the conversion workflow")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
