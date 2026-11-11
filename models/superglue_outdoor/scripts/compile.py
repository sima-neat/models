#!/usr/bin/env python3
"""Build SuperGlue outdoor's matcher from the converted weights with surgery.py and compile it into one Modalix MPK.

A W8A8 build calibrates its INT8 activations on yfcc100m-keypoints cal pairs, and runs the operations of its
bf16_regions, if it declares any, in BF16.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from afe.apis.defines import (
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
from calibration import calibration_inputs
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import BF16_REGIONS, annotate_bf16_regions, export_matcher

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "superglue_outdoor"
COMPONENT_ID = "model"
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
# The terminal EV74 plugin only converts the MLA result to the float32 interface: BF16 couplings end in a cast and
# INT8 couplings in a dequantization.
OUTPUT_KERNELS = {"pass_through", "cast_transform", "dequantization_transform"}
# Share of re-observed keypoints whose quantized best match must equal FP32's. W8A8 with its BF16 regions keeps
# 0.926; plain W8A8 keeps 0.641 and does not pass.
MIN_MATCH_AGREEMENT = {"w16a16": 0.95, "w8a16": 0.95, "w8a8": 0.85}
# Softmaxes in the matcher: one per attention, in 18 layers on two images.
SOFTMAXES = 36


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _bf16_regions(profile_id: str, parameters: dict[str, Any]) -> list[str]:
    """The regions the build profile runs in BF16, checking that its ID marks it mixed."""
    regions = parameters.get("bf16_regions", [])
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 build: {regions}")
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    if unknown:
        raise ValueError(f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def _quantization(parameters: dict[str, Any]) -> QuantizationParams:
    """The compiler's activation and weight formats for a precision. BF16 activations need no calibration."""
    precision = parameters["precision"]
    if precision == "w8a8":
        return QuantizationParams(
            # Histogram MSE picks each activation range from the calibration pairs.
            calibration_method=HistogramMSEMethod(int(parameters["calibration_bins"])),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        )
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


def _require_bf16_regions(quantized: Any, regions: list[str]) -> None:
    """Fail when the compiler ignored the BF16 annotations, which would silently build plain W8A8."""
    # The quantized model keeps its network private, and the compiler renames every node, so count by operation.
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.BFLOAT_16
    )
    print(f"BF16 operations: {dict(sorted(counts.items()))}", flush=True)
    softmaxes = SOFTMAXES if "attention" in regions else 0
    if not counts or counts["SoftmaxOp"] != softmaxes:
        raise RuntimeError(f"expected BF16 operations including {softmaxes} softmaxes, found {dict(counts)}")


def check_quantized_matches(source: Path, quantized: Any, minimum: float = MIN_MATCH_AGREEMENT["w16a16"]) -> float:
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
    if agreement < minimum:
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
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel not in OUTPUT_KERNELS or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: kernel {kernel!r}, sizes {outputs}")


def _calibration_data(parameters: dict[str, Any], dataset: Path | None) -> Any:
    """The calibration pairs of a build with INT8 activations; BF16 activations take none."""
    if parameters["precision"] != "w8a8":
        return None
    samples = calibration_inputs(dataset, int(parameters["calibration_samples"]))
    print(f"Calibrating on {int(parameters['calibration_samples'])} yfcc100m-keypoints cal pairs", flush=True)
    return convert_data_generator_to_iterable(DataGenerator({InputName(name): samples[name] for name in INPUT_SHAPES}))


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)
    regions = _bf16_regions(profile_id, parameters)
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = source
        if regions:
            graph = build_dir / "annotated.onnx"
            marks = annotate_bf16_regions(source, graph, regions)
            print(f"Marked {marks['bfloat16']} operations BF16 and {marks['int8']} INT8 for {regions}", flush=True)
        importer = onnx_source(
            str(graph),
            shape_dict={name: tuple(shape) for name, shape in INPUT_SHAPES.items()},
            layout="NCHW",
            flexible_batch_size=True,
        )
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=_calibration_data(parameters, dataset),
            quantization_config=_quantization(parameters),
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
        if regions:
            _require_bf16_regions(quantized, regions)
        agreement = check_quantized_matches(source, quantized, MIN_MATCH_AGREEMENT[parameters["precision"]])
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
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    if _parameters(args.build_profile)["precision"] == "w8a8" and args.dataset is None:
        parser.error(f"{args.build_profile} calibrates; pass --dataset")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    weights = args.model.expanduser().resolve()
    if weights.suffix != ".pth":
        parser.error("--model must be the verified weights.pth from the conversion workflow")
    if not weights.is_file():
        parser.error(f"model does not exist: {weights}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-onnx-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        source = export_matcher(weights, Path(temporary) / "model.onnx")
        archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset)
    print(f"Compiled MPK: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
