#!/usr/bin/env python3
"""Compile the converted SuperPoint ONNX into one Modalix MPK and check the quantized keypoints against FP32."""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
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
from calibration import calibration_images
from keypoints import (
    CELL,
    DESCRIPTOR_SIZE,
    HEIGHT,
    KEYPOINT_CHANNELS,
    WIDTH,
    decode,
    sample_descriptors,
    synthetic_image,
)
from model_registry.compile_script import compiler_log_level, component_parameters, parse_arguments, read_mpk_manifest
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "superpoint"
COMPONENT_ID = "model"
INPUT = "image"
INPUT_SHAPE = (1, 1, HEIGHT, WIDTH)
OUTPUT_SHAPES = {
    "logits": (1, KEYPOINT_CHANNELS, HEIGHT // CELL, WIDTH // CELL),
    "descriptors": (1, DESCRIPTOR_SIZE, HEIGHT // CELL, WIDTH // CELL),
}
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(shape) * 4 for shape in OUTPUT_SHAPES.values()]
# INT8 builds pass the heads through; BF16 builds cast them back to the float32 interface.
OUTPUT_KERNELS = {"pass_through", "cast_transform"}
# The gate compares the quantized model's keypoints with FP32's on seeded synthetic scenes.
GATE_SEEDS = range(101, 105)
GATE_KEYPOINTS = 500
GATE_PIXELS = 2.0
MIN_KEYPOINT_RECALL = 0.8
MIN_DESCRIPTOR_SIMILARITY = 0.9


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
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


def _channels_first(outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Name the quantized model's channels-last outputs by their shapes and return them channels-first."""
    named = {}
    for output in outputs:
        array = np.asarray(output, dtype=np.float32).transpose(0, 3, 1, 2)
        names = [name for name, shape in OUTPUT_SHAPES.items() if array.shape == shape]
        if len(names) != 1:
            raise RuntimeError(f"unexpected quantized output shape: {np.shape(output)}")
        named[names[0]] = array
    if set(named) != set(OUTPUT_SHAPES):
        raise RuntimeError(f"quantized outputs are {sorted(named)}, expected {sorted(OUTPUT_SHAPES)}")
    return named


def keypoint_agreement(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray]) -> tuple[float, float]:
    """Share of FP32's best keypoints the quantized model also finds within GATE_PIXELS, and the median cosine
    similarity of the two models' descriptors at those keypoints."""
    for name, values in actual.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"quantized {name} contains non-finite values")
    reference, _, reference_descriptors = decode(*(expected[name][0] for name in OUTPUT_SHAPES), limit=GATE_KEYPOINTS)
    found, _, _ = decode(*(actual[name][0] for name in OUTPUT_SHAPES), limit=GATE_KEYPOINTS)
    if len(reference) == 0:
        raise RuntimeError("FP32 finds no keypoints in the gate image")
    if len(found) == 0:
        return 0.0, 0.0
    distance = np.linalg.norm(reference[:, None] - found[None], axis=2).min(axis=1)
    recall = float((distance <= GATE_PIXELS).mean())
    # The quantized descriptors are sampled at FP32's keypoints, so both describe the same points.
    sampled = sample_descriptors(reference, actual["descriptors"][0])
    similarity = float(np.median((sampled * reference_descriptors).sum(axis=1)))
    return recall, similarity


def check_quantized_keypoints(source: Path, quantized: Any) -> tuple[float, float]:
    """Require the quantized model to find FP32's keypoints and describe them alike on synthetic scenes."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    recalls, similarities = [], []
    for seed in GATE_SEEDS:
        image = synthetic_image(seed)
        expected = dict(zip(OUTPUT_SHAPES, session.run(None, {INPUT: image[None, None]})))
        actual = _channels_first(list(quantized.execute({INPUT: image[None, :, :, None]}, fast_mode=False)))
        recall, similarity = keypoint_agreement(expected, actual)
        recalls.append(recall)
        similarities.append(similarity)
    recall, similarity = float(np.mean(recalls)), float(np.mean(similarities))
    logging.info("Quantized keypoint recall %.3f, descriptor similarity %.3f", recall, similarity)
    if recall < MIN_KEYPOINT_RECALL or similarity < MIN_DESCRIPTOR_SIMILARITY:
        raise RuntimeError(
            f"quantized SuperPoint drifts from FP32: keypoint recall {recall:.3f} (minimum {MIN_KEYPOINT_RECALL}), "
            f"descriptor similarity {similarity:.3f} (minimum {MIN_DESCRIPTOR_SIMILARITY})"
        )
    return recall, similarity


def validate_source(path: Path) -> None:
    """Require the converted interface: one grayscale image in, the two raw heads out."""
    graph = onnx.load(str(path), load_external_data=False).graph
    shapes = {
        value.name: tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        for value in [*graph.input, *graph.output]
    }
    if shapes != {INPUT: INPUT_SHAPE, **OUTPUT_SHAPES}:
        raise ValueError(f"Unsupported SuperPoint graph: {shapes}")


def audit_mpk(archive: Path) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
    names, manifest = read_mpk_manifest(archive)
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


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    validate_source(source)
    parameters = component_parameters(MODEL_DIR, profile_id, COMPONENT_ID)
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
        importer = onnx_source(str(source), shape_dict={INPUT: INPUT_SHAPE}, layout="NCHW", flexible_batch_size=True)
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=compiler_log_level(),
        )
        check_quantized_keypoints(source, quantized)
        quantized.compile(
            output_path=str(build_dir / "mpk"),
            batch_size=1,
            compress=True,
            log_level=compiler_log_level(),
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
        )
        archives = sorted((build_dir / "mpk").glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise RuntimeError(f"expected one MPK, found {len(archives)}")
        audit_mpk(archives[0])
        archive = component_dir / archives[0].name
        shutil.copyfile(archives[0], archive)
    return archive


def main() -> int:
    arguments = parse_arguments(__doc__, MODEL_DIR)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    archive = compile_model(arguments.model, arguments.output_dir, arguments.build_profile, arguments.dataset)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
