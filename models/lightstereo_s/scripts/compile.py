#!/usr/bin/env python3
"""Compile the converted LightStereo-S ONNX into one Modalix MPK and measure its disparity fidelity."""

from __future__ import annotations

import argparse
import gc
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
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from scene_flow import HEIGHT, INPUTS, WIDTH, fidelity_pairs, read_pairs
from sima_utils.data.data_generator import DataGenerator

# Importing afe leaves the root logger at WARNING, so the recipe logs through its own INFO logger.
LOG = logging.getLogger("lightstereo_s.compile")
LOG.setLevel(logging.INFO)
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "lightstereo_s"
COMPONENT_ID = "model"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
INPUT_SHAPE = [1, 3, HEIGHT, WIDTH]
OUTPUT = "disparity"
OUTPUT_SHAPE = [1, 1, HEIGHT, WIDTH]
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = dict.fromkeys(INPUTS, math.prod(INPUT_SHAPE) * 4)
OUTPUT_BYTES = [math.prod(OUTPUT_SHAPE) * 4]


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _parameters(profile_id: str) -> dict[str, Any]:
    profiles = [profile for profile in _metadata()["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str) -> Any:
    if precision != "w16a16":
        raise ValueError(f"unsupported precision: {precision}")
    config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(
        bfloat16_scheme()
    )
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
    outputs = [node.get("size") for node in plugins[-1].get("output_nodes", [])]
    if outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str) -> tuple[Path, Any]:
    """Quantize and compile one build profile, returning its MPK and the quantized graph."""
    precision = str(_parameters(profile_id)["precision"])
    quantization = _quantization(precision)
    # BF16 derives nothing from representative data, but quantize() still needs one sample.
    samples = {name: np.zeros((1, HEIGHT, WIDTH, 3), dtype=np.float32) for name in INPUTS}
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        importer = onnx_source(str(source), shape_dict={name: tuple(INPUT_SHAPE) for name in INPUTS}, layout="NCHW")
        loaded = load_model(importer, target=gen2_target)
        LOG.info("Quantizing %s", precision)
        data = DataGenerator({InputName(name): batch for name, batch in samples.items()})
        try:
            quantized = loaded.quantize(
                calibration_data=convert_data_generator_to_iterable(data),
                quantization_config=quantization,
                model_name=f"{MODEL_ID}_{profile_id}",
                log_level=logging.INFO,
            )
            # The calibration sample and the float graph are not needed to compile; freeing them lowers peak memory.
            del data, samples, loaded
            gc.collect()
            quantized.compile(
                output_path=str(build_dir / "mpk"),
                batch_size=1,
                compress=True,
                log_level=logging.INFO,
                tessellate_parameters=None,
                l2_caching_mode=L2CachingMode.NONE,
                deployable=True,
            )
        except SystemExit as error:
            # AFE reports a failed quantization or compile by raising SystemExit, which would exit with status 0.
            raise RuntimeError(f"Model Compiler failed to quantize or compile {MODEL_ID} {profile_id}") from error
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        # Keep the archive before auditing it, so a rejected MPK stays available for diagnosis.
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    audit_mpk(archive)
    return archive, quantized


def measure_fidelity(
    source: Path, candidate: Any, dataset: Path, config: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    """Compare the quantized graph's disparity with the ONNX source on the declared Scene Flow pairs."""
    selection = config["selection"]
    selected = fidelity_pairs(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    pairs = read_pairs(dataset, selected)
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    totals = {"count": 0, "dot": 0.0, "expected": 0.0, "actual": 0.0, "error": 0.0, "absolute": 0.0, "max": 0.0}
    for index, (left, _) in enumerate(selected):
        sample = {name: pairs[name][index : index + 1] for name in INPUTS}
        feeds = {name: batch.transpose(0, 3, 1, 2) for name, batch in sample.items()}
        expected = reference.run([OUTPUT], feeds)[0].astype(np.float64)
        outputs = candidate.execute(inputs={InputName(name): batch for name, batch in sample.items()}, use_jax=True)
        actual = np.asarray(outputs[0], dtype=np.float64).reshape(expected.shape)
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise RuntimeError(f"disparity for {left} contains non-finite values")
        error = actual - expected
        totals["count"] += expected.size
        totals["dot"] += float(np.dot(expected.ravel(), actual.ravel()))
        totals["expected"] += float(np.sum(expected**2))
        totals["actual"] += float(np.sum(actual**2))
        totals["error"] += float(np.sum(error**2))
        totals["absolute"] += float(np.sum(np.abs(error)))
        totals["max"] = max(totals["max"], float(np.max(np.abs(error))))
        LOG.info("Fidelity %s: end-point error %.4f px", left, float(np.mean(np.abs(error))))
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
        "samples": len(selected),
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
    parser.add_argument("--dataset", type=Path, help="calibration archive; no build profile of this model calibrates")
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
    if args.dataset is not None:
        parser.error("no build profile declares calibration_dataset, so the recipe takes no --dataset")
    output_dir = args.output_dir.expanduser().resolve()
    archive, candidate = compile_model(source, output_dir, args.build_profile)
    LOG.info("Compiled MPK: %s", archive)
    if fidelity is not None:
        LOG.info("Measuring numerical fidelity on %d pairs", fidelity["selection"]["samples"])
        report = measure_fidelity(
            source, candidate, args.fidelity_dataset.expanduser().resolve(), fidelity, args.build_profile
        )
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        LOG.info("Fidelity: %s", report["metrics"][0])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
