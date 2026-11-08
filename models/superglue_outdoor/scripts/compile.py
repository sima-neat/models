#!/usr/bin/env python3
"""Build SuperGlue outdoor's matcher from the converted weights with surgery.py and compile it into one Modalix MPK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from afe.apis.defines import (
    QuantizationParams,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.load.importers.general_importer import onnx_source
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from surgery import export_matcher

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "superglue_outdoor"
COMPONENT_ID = "model"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
# The exported graph's [1, C, 1, N] inputs and [1, N+1, 1, N+1] couplings output.
INPUT_SHAPES = {
    "keypoints0": [1, 2, 1, KEYPOINTS],
    "scores0": [1, 1, 1, KEYPOINTS],
    "descriptors0": [1, DESCRIPTOR_SIZE, 1, KEYPOINTS],
    "keypoints1": [1, 2, 1, KEYPOINTS],
    "scores1": [1, 1, 1, KEYPOINTS],
    "descriptors1": [1, DESCRIPTOR_SIZE, 1, KEYPOINTS],
}
OUTPUT_SHAPES = {"couplings": [1, KEYPOINTS + 1, 1, KEYPOINTS + 1]}
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {name: math.prod(shape) * 4 for name, shape in INPUT_SHAPES.items()}
OUTPUT_BYTES = [math.prod(shape) * 4 for shape in OUTPUT_SHAPES.values()]
# The terminal EV74 plugin only converts the MLA result to the float32 interface: BF16 builds end in a cast.
OUTPUT_KERNELS = {"pass_through", "cast_transform"}
# Share of re-observed keypoints whose quantized best match must equal FP32's.
MIN_MATCH_AGREEMENT = 0.95


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. BF16 activations need no calibration."""
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
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    raise ValueError(f"unsupported precision: {precision}")


@dataclass(frozen=True)
class SyntheticPair:
    """Channels-last matcher inputs and the image-0 and image-1 indices of the keypoints image 1 re-observes."""

    inputs: dict[str, np.ndarray]
    shared: np.ndarray
    target: np.ndarray


def synthetic_pair() -> SyntheticPair:
    """One seeded pair in which half of image 1's keypoints re-observe image-0 keypoints.

    Keypoints are in SuperGlue's normalized frame and descriptors are unit vectors, as SuperPoint gives them.
    """
    rng = np.random.RandomState(0)
    keypoints = [rng.uniform(-0.7, 0.7, (KEYPOINTS, 2)) for _ in range(2)]
    descriptors = [rng.normal(0, 1, (KEYPOINTS, DESCRIPTOR_SIZE)) for _ in range(2)]
    shared = rng.permutation(KEYPOINTS)[: KEYPOINTS // 2]
    target = rng.permutation(KEYPOINTS)[: KEYPOINTS // 2]
    keypoints[1][target] = keypoints[0][shared] * 0.95 + 0.02 + rng.normal(0, 0.003, (len(shared), 2))
    descriptors[1][target] = descriptors[0][shared] + rng.normal(0, 0.3, (len(shared), DESCRIPTOR_SIZE))
    columns = {
        "keypoints": keypoints,
        "scores": [rng.uniform(0.005, 0.5, (KEYPOINTS, 1)) for _ in range(2)],
        "descriptors": [value / np.linalg.norm(value, axis=1, keepdims=True) for value in descriptors],
    }
    inputs = {
        f"{kind}{image}": values[image].astype(np.float32)[None, None]
        for kind, values in columns.items()
        for image in range(2)
    }
    return SyntheticPair(inputs, shared, target)


def check_quantized_matches(source: Path, quantized: Any) -> float:
    """Require the quantized matcher's couplings to be finite and to keep FP32's best match for re-observed keypoints.

    Runs the quantized model through the Model Compiler's own execution and the source ONNX through ONNX Runtime
    on the synthetic pair; returns the share of re-observed image-0 keypoints whose best image-1 keypoint
    agrees.
    """
    import onnxruntime as ort

    pair = synthetic_pair()
    inputs, shared = pair.inputs, pair.shared
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(
        None, {name: np.ascontiguousarray(value.transpose(0, 3, 1, 2)) for name, value in inputs.items()}
    )[0]
    # ONNX gives [1, j, 1, i]; the compiler's execution returns channels-last [1, 1, i, j].
    expected = np.asarray(reference).reshape(KEYPOINTS + 1, KEYPOINTS + 1).T[:KEYPOINTS, :KEYPOINTS]
    output = np.asarray(quantized.execute(inputs, fast_mode=False)[0], dtype=np.float32)
    actual = output.reshape(KEYPOINTS + 1, KEYPOINTS + 1)[:KEYPOINTS, :KEYPOINTS]
    if not np.isfinite(output).all():
        raise RuntimeError(f"quantized couplings hold {int((~np.isfinite(output)).sum())} non-finite values")
    agreement = float((actual[shared].argmax(1) == expected[shared].argmax(1)).mean())
    if agreement < MIN_MATCH_AGREEMENT:
        raise RuntimeError(f"quantized best matches agree with FP32 for only {agreement:.3f} of re-observed keypoints")
    return agreement


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
    """Require the whole matcher on one MLA segment, with no host code, and the declared interface."""
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
    if kernel not in OUTPUT_KERNELS or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: kernel {kernel!r}, sizes {outputs}")


def compile_model(source: Path, output_dir: Path, profile_id: str) -> Path:
    precision = str(_parameters(profile_id)["precision"])
    quantization = _quantization(precision)
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        importer = onnx_source(
            str(source),
            shape_dict={name: tuple(shape) for name, shape in INPUT_SHAPES.items()},
            layout="NCHW",
            flexible_batch_size=True,
        )
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=None,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        agreement = check_quantized_matches(source, quantized)
        # The Model Compiler configures logging before this script runs, so the script prints its own lines.
        print(f"Quantized best matches agree with FP32 for {agreement:.3f} of re-observed keypoints", flush=True)
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
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth":
        parser.error("--model must be the verified weights.pth from the conversion workflow")
    if not weights.is_file():
        parser.error(f"model does not exist: {weights}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_matcher(weights, Path(temporary) / "model.onnx")
        archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    print(f"Compiled MPK: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
