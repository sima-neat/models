#!/usr/bin/env python3
"""Compile LightGlue's matcher from the converted ONNX into a Modalix MPK and measure its numerical fidelity.

The converted ONNX is static: two images' 512 keypoints and descriptors in, their soft assignment out. Calibration and
numerical fidelity use one seeded order of the yfcc100m-keypoints cal pairs: calibration takes the first pairs, and
fidelity compares the quantized graph's assignment with the ONNX's on a later span.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
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
from calibration import INPUT_SHAPES, OUTPUT_SHAPE, calibration_inputs, fidelity_inputs
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

LOG = logging.getLogger("lightglue_superpoint.compile")
MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "lightglue_superpoint"
COMPONENT_ID = "model"
OUTPUT_NAME = "assignment"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]


def _metadata() -> dict[str, Any]:
    return load_metadata(MODEL_DIR)


def _parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = [item for item in metadata["build_profiles"] if item["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one build profile named {profile_id!r}")
    (binding,) = [item for item in profiles[0]["components"] if item["id"] == COMPONENT_ID]
    return binding["parameters"]


def _quantization_config(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError(f"this recipe supports only the declared Modalix BF16 and INT8 builds, not {precision!r}")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        # AFE quantizes the graph with calibrated ranges, even at BF16.
        .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        # with_activation_quantization turns per-token quantization on for BF16; these builds do not use it.
        .with_per_token_quant(False)
    )


def quantize(source: Path, profile_id: str, dataset: Path) -> Any:
    parameters = _parameters(_metadata(), profile_id)
    config = _quantization_config(parameters)
    importer = onnx_source(str(source), shape_dict=INPUT_SHAPES, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    samples = calibration_inputs(dataset, int(parameters["calibration_samples"]))
    LOG.info("Calibrating on %d yfcc100m-keypoints cal pairs", int(parameters["calibration_samples"]))
    data = {InputName(name): samples[name] for name in INPUT_SHAPES}
    return loaded.quantize(
        calibration_data=convert_data_generator_to_iterable(DataGenerator(data)),
        quantization_config=config,
        model_name=f"{MODEL_ID}_{profile_id}",
        log_level=logging.INFO,
    )


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


def audit_mpk(archive: Path) -> dict[str, Any]:
    """Require one MLA plugin, no CPU plugins, the pinned compiler and the matcher's four float32 inputs."""
    names, manifest = _read_mpk_manifest(archive)
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
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(f"MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}")
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
        deployable=True,
        retained_temporary_directory_name=None,
        use_power_limits=False,
        max_mla_power=None,
        # The option applies to every layer norm, and INT8 ones reject it.
        layer_norm_use_fp32_intermediates=precision == "w16a16",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    audit_mpk(archives[0])
    LOG.info("Compiled MPK: %s", archives[0])
    return archives[0]


@dataclass
class _OutputError:
    """Running sums of the assignment's error against the ONNX, over every fidelity pair."""

    name: str
    count: int = 0
    dot: float = 0.0
    expected_squared: float = 0.0
    actual_squared: float = 0.0
    error_squared: float = 0.0
    error_absolute: float = 0.0
    max_absolute_error: float = 0.0

    def add(self, expected: np.ndarray, actual: np.ndarray) -> None:
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{self.name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{self.name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        self.count += expected.size
        self.dot += float(np.dot(expected.ravel(), actual.ravel()))
        self.expected_squared += float(np.sum(np.square(expected)))
        self.actual_squared += float(np.sum(np.square(actual)))
        self.error_squared += float(np.sum(np.square(error)))
        self.error_absolute += float(np.sum(absolute))
        self.max_absolute_error = max(self.max_absolute_error, float(np.max(absolute)))

    def summary(self) -> dict[str, Any]:
        expected_rms = max(np.sqrt(self.expected_squared / self.count), np.finfo(np.float64).eps)
        norm = np.sqrt(self.expected_squared * self.actual_squared)
        cosine = self.dot / norm if norm else float(self.expected_squared == self.actual_squared)
        return {
            "name": self.name,
            "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
            "normalized_rmse": float(np.sqrt(self.error_squared / self.count) / expected_rms),
            "mean_absolute_error": float(self.error_absolute / self.count),
            "max_absolute_error": self.max_absolute_error,
        }


def measure_fidelity(source: Path, quantized: Any, dataset: Path, config: dict[str, Any], profile_id: str) -> dict:
    """Compare the quantized graph's assignment with the ONNX source's on the declared cal pairs."""
    selection = config["selection"]
    samples = fidelity_inputs(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    error = _OutputError(OUTPUT_NAME)
    for inputs in samples:
        (expected,) = reference.run([OUTPUT_NAME], inputs)
        outputs = quantized.execute(inputs={InputName(name): value for name, value in inputs.items()}, use_jax=True)
        (actual,) = [np.asarray(output) for output in outputs]
        if actual.shape != OUTPUT_SHAPE:
            raise RuntimeError(f"the quantized graph returns its assignment as {actual.shape}, expected {OUTPUT_SHAPE}")
        error.add(expected, actual)
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "ONNX source (ONNX Runtime)",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(samples),
        "metrics": [error.summary()],
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
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    if args.dataset is None:
        parser.error("every build profile calibrates; pass --dataset")
    fidelity = _metadata().get("numerical_fidelity")
    if fidelity is not None and args.fidelity_dataset is None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    output_dir = args.output_dir.expanduser().resolve()
    precision = str(_parameters(_metadata(), args.build_profile)["precision"])
    quantized = quantize(source, args.build_profile, args.dataset.expanduser().resolve())
    report = None
    if fidelity is not None:
        # Before the MLA compile, so a broken quantization stops the build early.
        LOG.info("Measuring numerical fidelity on %d pairs", fidelity["selection"]["samples"])
        report = measure_fidelity(source, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity,
                                  args.build_profile)
        LOG.info("Fidelity: %s", report["metrics"][0])
    compile_model(quantized, output_dir / COMPONENT_ID, precision)
    if report is not None:
        (output_dir / "fidelity.json").write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
