#!/usr/bin/env python3
"""Compile YuNet-s's raw heads into one Modalix MPK and measure its numerical fidelity."""

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
from calibration import CALIBRATION_SEED, calibration_images, fidelity_names, preprocess_image, read_images
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import INPUT, INPUT_SHAPE, OUTPUT_SHAPES, OUTPUTS, SOURCE_OUTPUT_SHAPES, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "yunet_s"
COMPONENT_ID = "model"
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPES[name]) * 4 for name in OUTPUTS]


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def check_fidelity_selection(selection: dict[str, Any]) -> None:
    """Fail unless the fidelity images come after every calibration image in the same seeded order."""
    calibrated = [
        int(item["parameters"]["calibration_samples"])
        for profile in load_metadata(MODEL_DIR)["build_profiles"]
        for item in profile["components"]
        if "calibration_samples" in item["parameters"]
    ]
    if int(selection["seed"]) != CALIBRATION_SEED:
        raise ValueError(
            f"numerical_fidelity draws with seed {selection['seed']}, but calibration uses {CALIBRATION_SEED}"
        )
    if int(selection["offset"]) < max(calibrated, default=0):
        raise ValueError(
            f"numerical_fidelity starts at image {selection['offset']}, inside the first {max(calibrated)} "
            "calibration images"
        )


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
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel != "pass_through" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> tuple[Path, Any]:
    parameters = _parameters(profile_id)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
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
    return archive, quantized


def _flatten(name: str, value: Any) -> np.ndarray:
    """A head map in upstream's [1, H*W, C] layout; the SDK returns maps channels-last, the cut ONNX channels-first."""
    array = np.asarray(value, dtype=np.float64)
    nchw = tuple(OUTPUT_SHAPES[name])
    if array.shape == nchw:
        array = array.transpose(0, 2, 3, 1)
    elif array.shape != (nchw[0], nchw[2], nchw[3], nchw[1]):
        raise RuntimeError(f"unexpected {name} shape {array.shape}")
    return array.reshape(SOURCE_OUTPUT_SHAPES[name])


def measure_fidelity(source: Path, candidate: Any, dataset: Path, config: dict[str, Any], profile_id: str) -> dict:
    """Compare the quantized graph's 12 head maps with the upstream ONNX outputs on WIDER FACE training images."""
    selection = config["selection"]
    names = fidelity_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    images = read_images(dataset, names)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    reference_names = [value.name for value in reference.get_outputs()]
    totals = {name: {"dot": 0.0, "expected": 0.0, "actual": 0.0, "error": 0.0, "absolute": 0.0, "max": 0.0, "count": 0}
              for name in config["outputs"]}
    for image_name in names:
        sample = preprocess_image(images[image_name])
        reference_outputs = reference.run(None, {INPUT: sample.transpose(2, 0, 1)[None]})
        expected = dict(zip(reference_names, reference_outputs, strict=True))
        outputs = candidate.execute(inputs={InputName(INPUT): sample[None]}, use_jax=True)
        actual = {name: _flatten(name, value) for name, value in zip(OUTPUTS, outputs, strict=True)}
        for name, total in totals.items():
            reference_value, compiled = expected[name].astype(np.float64), actual[name]
            if not (np.isfinite(reference_value).all() and np.isfinite(compiled).all()):
                raise RuntimeError(f"{name} for {image_name} contains non-finite values")
            error = compiled - reference_value
            total["dot"] += float(np.dot(reference_value.ravel(), compiled.ravel()))
            total["expected"] += float(np.sum(reference_value**2))
            total["actual"] += float(np.sum(compiled**2))
            total["error"] += float(np.sum(error**2))
            total["absolute"] += float(np.sum(np.abs(error)))
            total["max"] = max(total["max"], float(np.max(np.abs(error))))
            total["count"] += reference_value.size
    metrics = []
    for name, total in totals.items():
        count = total["count"]
        expected_rms = max(np.sqrt(total["expected"] / count), np.finfo(np.float64).eps)
        norm = np.sqrt(total["expected"] * total["actual"])
        metrics.append(
            {
                "name": name,
                "cosine_similarity": float(np.clip(total["dot"] / norm if norm else 1.0, -1.0, 1.0)),
                "normalized_rmse": float(np.sqrt(total["error"] / count) / expected_rms),
                "mean_absolute_error": float(total["absolute"] / count),
                "max_absolute_error": total["max"],
            }
        )
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "Upstream ONNX (ONNX Runtime)",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": metrics,
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
        parser.error("--model must be the pinned upstream ONNX")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    fidelity = load_metadata(MODEL_DIR).get("numerical_fidelity")
    if fidelity is not None and args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    if fidelity is not None:
        check_fidelity_selection(fidelity["selection"])
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    # force replaces any handler an import installed first, which would otherwise drop these lines from the job log.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    output_dir = args.output_dir.expanduser().resolve()
    archive, quantized = compile_model(source, output_dir, args.build_profile, dataset)
    logging.info("Compiled MPK: %s", archive)
    if fidelity is not None:
        report = measure_fidelity(
            source, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity, args.build_profile
        )
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The compiler reconfigures logging while it runs, so the summary is printed too.
        summary = {metric["name"]: round(metric["cosine_similarity"], 6) for metric in report["metrics"]}
        print(f"Fidelity cosine similarity: {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
