#!/usr/bin/env python3
"""Collapse SESR's linear blocks and compile the network into one Modalix MPK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
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
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from model_registry.model import load_model as load_metadata
from sesr import INPUT, INPUT_SHAPE, NUM_BLOCKS, OUTPUT_SHAPE, SCALE, TILE
from sima_utils.data.data_generator import DataGenerator
from surgery import prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = f"sesr_m{NUM_BLOCKS}_{SCALE}x"
COMPONENT_ID = "model"
# DIV2K's calibration images are already bicubic-downscaled 2x, the same kind of input the model upscales.
CALIBRATION_IMAGES = "DIV2K_calibration_LR_bicubic"
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPE) * 4]


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def calibration_images(dataset: Path, count: int) -> list[tuple[str, bytes]]:
    """The first `count` calibration PNGs by name, from the extracted archive or the dataset-registry tarball."""
    if dataset.is_dir():
        files = [(path.name, path.read_bytes()) for path in sorted((dataset / CALIBRATION_IMAGES).glob("*.png"))]
    elif tarfile.is_tarfile(dataset):
        files = []
        with tarfile.open(dataset) as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if member.isfile() and path.parent.name == CALIBRATION_IMAGES and path.suffix == ".png":
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"could not read {member.name} from the dataset")
                    files.append((path.name, stream.read()))
        files.sort()
    else:
        raise ValueError(f"unsupported calibration dataset input: {dataset}")
    if len(files) < count:
        raise ValueError(f"expected at least {count} images under {CALIBRATION_IMAGES}/, found {len(files)}")
    return files[:count]


def calibration_tiles(dataset: Path, count: int) -> list[np.ndarray]:
    """One seeded 224x224 crop from each of the first `count` calibration images: RGB in [0, 1], HWC float32."""
    import cv2

    chooser = random.Random(0)
    tiles = []
    for name, content in calibration_images(dataset, count):
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < TILE:
            raise ValueError(f"cannot take a {TILE}-pixel crop from {name}")
        top = chooser.randrange(image.shape[0] - TILE + 1)
        left = chooser.randrange(image.shape[1] - TILE + 1)
        crop = cv2.cvtColor(image[top : top + TILE, left : left + TILE], cv2.COLOR_BGR2RGB)
        tiles.append(crop.astype(np.float32) / np.float32(255))
    return tiles


def _quantization(precision: str, calibration_bins: int | None) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. Only INT8 activations need calibration."""
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=HistogramMSEMethod(calibration_bins),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


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
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    outputs = [node.get("size") for node in (plugins[-1] if plugins else {}).get("output_nodes", [])]
    if outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
    calibration_data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        samples = calibration_tiles(dataset, int(parameters["calibration_samples"]))
        logging.info("Calibrating %s with %d tile(s)", precision, len(samples))
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = prepare_model(source, build_dir / "graphs")
        importer = onnx_source(str(graph), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=calibration_data,
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
