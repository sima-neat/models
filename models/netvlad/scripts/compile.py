#!/usr/bin/env python3
"""Compile the converted NetVLAD ONNX into one Modalix MPK and measure its descriptor fidelity."""

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
import onnxruntime as ort
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import IMAGE_HEIGHT, IMAGE_WIDTH, load_images, select_images
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

# Importing afe installs a root handler at WARNING, so the recipe logs through its own INFO logger.
LOG = logging.getLogger("netvlad.compile")
LOG.setLevel(logging.INFO)
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "netvlad"
COMPONENT_ID = "model"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
INPUT = "image"
INPUT_SHAPE = [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]
OUTPUT = "descriptor"
DESCRIPTOR_DIM = 4096
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [DESCRIPTOR_DIM * 4]


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _parameters(profile_id: str) -> dict[str, Any]:
    profiles = [profile for profile in _metadata()["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str, calibration_bins: int | None) -> Any:
    if precision == "w8a16":
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
        # AFE keeps a few operators in INT8 even with BF16 activations, so W8A16 still calibrates their ranges.
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(weights)
            .with_calibration(HistogramMSEMethod(calibration_bins))
            .with_bias_correction(BiasCorrectionType.NONE)
        )
    elif precision == "w16a16":
        config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(
            bfloat16_scheme()
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
    # with_activation_quantization turns per-token quantization on for BF16; only W8A16 keeps it, as for YOLO26.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(precision == "w8a16")


def calibration_names(dataset: Path, sample_count: int) -> tuple[str, ...]:
    """The first images of the fidelity seed's shuffle, which the fidelity comparison must start after."""
    selection = _metadata()["numerical_fidelity"]["selection"]
    if int(selection["offset"]) < sample_count:
        raise ValueError("the fidelity selection must begin after the calibration images")
    return select_images(dataset, int(selection["seed"]), 0, sample_count)


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
    # The last EV74 kernel converts the MLA's output back to float32 (cast_transform for BF16), so the audit
    # checks where the descriptor comes from and its size rather than one kernel name.
    terminal = plugins[-1] if plugins else {}
    processor = str(terminal.get("processor", "")).upper()
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if processor != "EV74" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {processor} returns output bytes {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> tuple[Path, Any]:
    parameters = _parameters(profile_id)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a16" else None
    quantization = _quantization(precision, calibration_bins)
    if precision == "w8a16":
        if dataset is None:
            raise ValueError("the W8A16 build requires --dataset from dataset-registry")
        samples = load_images(dataset, calibration_names(dataset, int(parameters["calibration_samples"])))
    else:
        # BF16 derives nothing from representative data, but quantize() still needs one sample.
        samples = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        importer = onnx_source(str(source), shape_dict={INPUT: tuple(INPUT_SHAPE)}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        LOG.info("Quantizing %s with %d sample(s)", precision, len(samples))
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples})),
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        try:
            quantized.compile(
                output_path=str(build_dir / "mpk"),
                batch_size=1,
                compress=True,
                log_level=logging.INFO,
                tessellate_parameters=None,
                l2_caching_mode=L2CachingMode.NONE,
                deployable=True,
            )
        except SystemExit as exc:
            # AFE reports a compile error by raising SystemExit without a code, which would exit with status 0.
            raise RuntimeError("the Model Compiler failed to compile the model; its log above names the layer") from exc
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    # Audit the kept copy, so a rejected MPK stays in the output for diagnosis while the build still fails.
    audit_mpk(archive)
    return archive, quantized


def measure_fidelity(source: Path, candidate: Any, dataset: Path, profile_id: str) -> dict[str, Any]:
    """Compare the quantized graph's descriptors with the ONNX source on the declared calibration-set images."""
    config = _metadata()["numerical_fidelity"]
    selection = config["selection"]
    names = select_images(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    totals = {"count": 0, "dot": 0.0, "expected": 0.0, "actual": 0.0, "error": 0.0, "absolute": 0.0, "max": 0.0}
    for index, (name, sample) in enumerate(zip(names, load_images(dataset, names), strict=True), start=1):
        LOG.info("Fidelity image %d/%d: %s", index, len(names), name)
        expected = reference.run([OUTPUT], {INPUT: sample.transpose(2, 0, 1)[None]})[0].astype(np.float64)
        outputs = candidate.execute(inputs={InputName(INPUT): sample[None]}, use_jax=True)
        actual = np.asarray(outputs[0], dtype=np.float64).reshape(expected.shape)
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise RuntimeError(f"descriptors for {name} contain non-finite values")
        error = actual - expected
        totals["count"] += expected.size
        totals["dot"] += float(np.dot(expected.ravel(), actual.ravel()))
        totals["expected"] += float(np.sum(expected**2))
        totals["actual"] += float(np.sum(actual**2))
        totals["error"] += float(np.sum(error**2))
        totals["absolute"] += float(np.sum(np.abs(error)))
        totals["max"] = max(totals["max"], float(np.max(np.abs(error))))
    count = totals["count"]
    expected_rms = max(np.sqrt(totals["expected"] / count), np.finfo(np.float64).eps)
    norm = np.sqrt(totals["expected"] * totals["actual"])
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "ONNX source (ONNX Runtime)",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [
            {
                "name": OUTPUT,
                "cosine_similarity": float(np.clip(totals["dot"] / norm if norm else 1.0, -1.0, 1.0)),
                "normalized_rmse": float(np.sqrt(totals["error"] / count) / expected_rms),
                "mean_absolute_error": float(totals["absolute"] / count),
                "max_absolute_error": totals["max"],
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the conversion workflow")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    fidelity = _metadata().get("numerical_fidelity")
    if fidelity is not None and args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    output_dir = args.output_dir.expanduser().resolve()
    archive, quantized = compile_model(source, output_dir, args.build_profile, dataset)
    LOG.info("Compiled MPK: %s", archive)
    if fidelity is not None:
        report = measure_fidelity(source, quantized, args.fidelity_dataset.expanduser().resolve(), args.build_profile)
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
