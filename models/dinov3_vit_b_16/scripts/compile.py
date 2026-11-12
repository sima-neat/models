#!/usr/bin/env python3
"""Build DINOv3 ViT-B/16 from the converted weights with surgery.py, compile it into a Modalix MPK and measure its
numerical fidelity.

surgery.py exports a static ONNX: a 224x224 image in, the class embedding and the patch embeddings out. Numerical
fidelity compares the quantized graph with reference.onnx, timm's own DINOv3, on the declared imagenet-2012 cal
images.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    InputName,
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, select_names
from model_registry.conversion import conversion_outputs
from model_registry.model import load_model as load_metadata
from surgery import INPUTS, OUTPUTS, export_encoder

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = MODEL_DIR.name
COMPONENT_ID = "model"
(INPUT_NAME,) = INPUTS


def _parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = [item for item in metadata["build_profiles"] if item["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one build profile named {profile_id!r}")
    (binding,) = [item for item in profiles[0]["components"] if item["id"] == COMPONENT_ID]
    return binding["parameters"]


def _quantization_config(precision: str) -> QuantizationParams:
    """The compiler's activation and weight formats for a build. BF16 activations need no calibrated ranges."""
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
            # Each token's activations get their own INT8 scale before an INT8-weight convolution.
            per_token_quant=True,
        )
    raise ValueError(f"this recipe supports only the declared Modalix BF16-activation builds, not {precision!r}")


def quantize(source: Path, profile_id: str, precision: str) -> Any:
    shapes = {name: tuple(shape) for name, shape in INPUTS.items()}
    importer = onnx_source(str(source), shape_dict=shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    try:
        return loaded.quantize(
            calibration_data=None,
            quantization_config=_quantization_config(precision),
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
    except SystemExit as error:
        # AFE reports a failed quantization or compile by raising SystemExit, which would exit with status 0.
        raise RuntimeError(f"Model Compiler failed to quantize {MODEL_ID} {profile_id}") from error


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
    """Require one MLA plugin, no CPU plugins and the image input."""
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
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    expected = {name: math.prod(shape) * 4 for name, shape in INPUTS.items()}
    if public_inputs != expected:
        raise RuntimeError(f"MPK input interface mismatch: {public_inputs}")


def compile_model(quantized: Any, component_dir: Path) -> Path:
    component_dir.mkdir(parents=True, exist_ok=False)
    try:
        quantized.compile(
            output_path=str(component_dir),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
            # Not the compiler's default (False); no build of this recipe was measured without it.
            layer_norm_use_fp32_intermediates=True,
        )
    except SystemExit as error:
        raise RuntimeError(f"Model Compiler failed to compile {MODEL_ID}") from error
    archives = sorted(component_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected exactly one MPK archive, found {len(archives)} in {component_dir}")
    audit_mpk(archives[0])
    print(f"Compiled MPK: {archives[0]}", flush=True)
    return archives[0]


@dataclass
class _OutputError:
    """Running sums of one output's error against reference.onnx, over every fidelity image."""

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


def measure_fidelity(
    reference: Path, quantized: Any, dataset: Path, fidelity: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    """Compare the quantized graph's embeddings with reference.onnx's on the declared cal images."""
    selection = fidelity["selection"]
    names = select_names(dataset, int(selection["seed"]), int(selection["offset"]), int(selection["samples"]))
    session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    errors = [_OutputError(name) for name in fidelity["outputs"]]
    for image in load_images(dataset, names):
        # The image is channels-last, as the compiler takes it; reference.onnx takes it channels-first.
        expected = dict(zip(OUTPUTS, session.run(list(OUTPUTS), {INPUT_NAME: image.transpose(2, 0, 1)[None]}),
                            strict=True))
        # The outputs' shapes differ, so each quantized output is matched to its name by its shape.
        executed = quantized.execute(inputs={InputName(INPUT_NAME): image[None]}, use_jax=True)
        outputs = {np.asarray(output).shape: np.asarray(output) for output in executed}
        for error in errors:
            shape = tuple(OUTPUTS[error.name])
            if shape not in outputs:
                raise RuntimeError(f"the quantized model returned no {error.name} of shape {shape}: {list(outputs)}")
            error.add(expected[error.name], outputs[shape])
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": fidelity["status"],
        "reference": "timm's DINOv3 as reference.onnx (ONNX Runtime)",
        "candidate": "Quantized AFE graph (JAX execute)",
        "dataset": {key: fidelity["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": len(names),
        "metrics": [error.summary() for error in errors],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--fidelity-dataset", type=Path, required=True, help="dataset-registry fidelity archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")

    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth" or not weights.is_file():
        parser.error(f"--model must be the converted weights.pth, not {weights}")
    reference = weights.with_name(conversion_outputs(MODEL_DIR)[1])
    metadata = load_metadata(MODEL_DIR)
    precision = str(_parameters(metadata, args.build_profile)["precision"])
    fidelity = metadata["numerical_fidelity"]
    output_dir = args.output_dir.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_encoder(weights, reference, Path(temporary) / "model.onnx")
        quantized = quantize(source, args.build_profile, precision)
        # Before the MLA compile, so a broken quantization shows early. Printed, not logged: the Model Compiler's
        # logging setup drops this script's INFO records.
        print(f"Measuring numerical fidelity on {fidelity['selection']['samples']} images", flush=True)
        report = measure_fidelity(reference, quantized, args.fidelity_dataset.expanduser().resolve(), fidelity,
                                  args.build_profile)
        for metric in report["metrics"]:
            print(f"Fidelity {metric['name']}: cosine {metric['cosine_similarity']:.5f}, "
                  f"normalized RMSE {metric['normalized_rmse']:.5f}", flush=True)
        compile_model(quantized, output_dir / COMPONENT_ID)
    (output_dir / "fidelity.json").write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
