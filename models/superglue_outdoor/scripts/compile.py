#!/usr/bin/env python3
"""Compile the converted SuperGlue outdoor matcher into one Modalix MPK."""

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
import onnx
from afe.apis.defines import (
    BiasCorrectionType,
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
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "superglue_outdoor"
COMPONENT_ID = "model"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
# The converted graph's [1, C, 1, N] inputs and [1, N+1, 1, N+1] couplings output.
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


def _quantization(precision: str) -> Any:
    if precision == "w16a16":
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(bfloat16_scheme())
        )
    elif precision == "w8a16":
        # BF16 activations need no calibrated ranges, and bias correction would measure the placeholder sample.
        config = (
            default_quantization.with_activation_quantization(bfloat16_scheme())
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_calibration(SkipCalibration())
            .with_bias_correction(BiasCorrectionType.NONE)
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
    # with_activation_quantization turns per-token quantization on for BF16; only W8A16 keeps it.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(precision == "w8a16")


def representative_pair() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """One channels-last sample per input, in which half of image 1's keypoints re-observe image-0 keypoints.

    Keypoints are in SuperGlue's normalized frame and descriptors are unit vectors, as SuperPoint gives them.
    Returns the inputs and the image-0 and image-1 indices of the re-observed keypoints.
    BF16 derives nothing from representative data, but quantize() still needs one sample.
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
    return inputs, shared, target


def representative_inputs() -> dict[str, np.ndarray]:
    return representative_pair()[0]


def check_quantized_matches(source: Path, quantized: Any) -> float:
    """Require the quantized matcher's couplings to be finite and to keep FP32's best match for re-observed keypoints.

    Runs the quantized model through the Model Compiler's own execution and the source ONNX through ONNX Runtime
    on the representative pair; returns the share of re-observed image-0 keypoints whose best image-1 keypoint
    agrees.
    """
    import onnxruntime as ort

    inputs, shared, _ = representative_pair()
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


def validate_source(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    values = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in [*model.graph.input, *model.graph.output]
    }
    if values != {**INPUT_SHAPES, **OUTPUT_SHAPES}:
        raise ValueError(f"Unsupported SuperGlue graph: {values}")


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
    validate_source(source)
    # BF16 activations derive nothing from representative data, but quantize() still needs one sample.
    samples = representative_inputs()
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
        logging.info("Quantizing %s with %d sample(s)", precision, len(next(iter(samples.values()))))
        quantized = loaded.quantize(
            calibration_data=convert_data_generator_to_iterable(
                DataGenerator({InputName(name): value for name, value in samples.items()})
            ),
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        agreement = check_quantized_matches(source, quantized)
        logging.info("Quantized best matches agree with FP32 for %.3f of re-observed keypoints", agreement)
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
    parser.add_argument("--dataset", type=Path, help="unused: no build of this model takes a calibration dataset")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the conversion workflow")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
