#!/usr/bin/env python3
"""Partition, validate, and compile the verified RF-DETR Large ONNX."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
import yaml
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
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
from calibration import load_images, select_images, select_top_proposals, to_afe_layout
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    BOUNDARY_SHAPES,
    BOXES,
    FEATURES,
    INPUT,
    INPUT_SHAPE,
    OUTPUT_SHAPES,
    OUTPUTS,
    PROPOSALS,
    SCORES,
    graph_paths,
    prepare_model,
    validate_float_split,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "rf_detr_large"
INVOCATION_ID = "rf_detr"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
# Precisions with BF16 activations. Their layer norms keep FP32 intermediates, as every measured build did.
A16_PRECISIONS = ("w8a16", "w16a16")
COMPONENT_IDS = ("backbone", "decoder")
# Files the compilation phases pass to each other in the temporary directory.
REFERENCE_OUTPUTS = "reference_outputs.npz"
BACKBONE_OUTPUTS = "backbone_outputs.npz"
FIDELITY_REPORT = "fidelity.json"


def _float32_bytes(shape: list[int]) -> int:
    return math.prod(shape) * 4


# MPK manifests record each public tensor's size in bytes.
COMPONENT_INTERFACES = {
    "backbone": {
        "inputs": {INPUT: _float32_bytes(INPUT_SHAPE)},
        "outputs": [_float32_bytes(BOUNDARY_SHAPES[name]) for name in (FEATURES, SCORES, BOXES)],
    },
    "decoder": {
        "inputs": {name: _float32_bytes(BOUNDARY_SHAPES[name]) for name in (FEATURES, PROPOSALS)},
        "outputs": [_float32_bytes(OUTPUT_SHAPES[name]) for name in OUTPUTS],
    },
}


def _component_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    """Return each component's build parameters in the build profile."""
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}


def _component_name(profile_id: str, component_id: str) -> str:
    """Name a component's quantized model and its build directory."""
    return f"{MODEL_ID}_{profile_id}_{component_id}"


def _quantization_config(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    int8_weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = int8_weights
        calibration = HistogramMSEMethod(int(parameters["calibration_bins"]))
        # Bias correction measured +0.1 COCO mAP on the RF-DETR Small INT8 backbone.
        bias_correction = BiasCorrectionType.REGULAR
    elif precision in A16_PRECISIONS:
        # BF16 activations need no calibrated ranges.
        activation = bfloat16_scheme()
        weights = int8_weights if precision == "w8a16" else bfloat16_scheme()
        calibration = SkipCalibration()
        bias_correction = BiasCorrectionType.NONE
    else:
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(calibration)
        .with_bias_correction(bias_correction)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        # with_activation_quantization turns per-token quantization on for BF16; only declared components keep it.
        .with_per_token_quant(bool(parameters.get("per_token", False)))
    )


def _quantize_component(
    graph: Path,
    name: str,
    parameters: dict[str, Any],
    calibration_inputs: dict[str, np.ndarray] | None,
    input_shapes: dict[str, tuple[int, ...]],
) -> Any:
    importer = onnx_source(str(graph), shape_dict=input_shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    data = None
    if calibration_inputs is not None:
        data = convert_data_generator_to_iterable(
            DataGenerator({InputName(key): to_afe_layout(value) for key, value in calibration_inputs.items()})
        )
    return loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization_config(parameters),
        model_name=name,
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


def audit_mpk(archive: Path, component_id: str) -> dict[str, Any]:
    contract = COMPONENT_INTERFACES[component_id]
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, "
            f"host={len(host_plugins)}, shared_objects={shared_objects}"
        )
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"{component_id} MPK compiler mismatch: expected {COMPILER_VERSION}, "
            f"found {manifest.get('model_sdk_version')}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if public_inputs != contract["inputs"]:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {public_inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    output_sizes = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel != "pass_through" or output_sizes != contract["outputs"]:
        raise RuntimeError(f"{component_id} MPK output interface mismatch: {output_sizes}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins), "output_sizes": output_sizes}


def _find_mpk(build_dir: Path, component_id: str) -> Path:
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def _compile_component(
    quantized: Any,
    build_dir: Path,
    component_id: str,
    precision: str,
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=False)
    quantized.compile(
        output_path=str(build_dir),
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
        layer_norm_use_fp32_intermediates=precision in A16_PRECISIONS,
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archive = _find_mpk(build_dir, component_id)
    audit_mpk(archive, component_id)
    return archive


def _features_to_nchw(features: np.ndarray) -> np.ndarray:
    """Return backbone features in the NCHW layout of the ONNX boundary, whichever layout AFE returns."""
    batch, channels, height, width = BOUNDARY_SHAPES[FEATURES]
    if features.shape == (batch, height, width, channels):
        return features.transpose(0, 3, 1, 2)
    if features.shape != (batch, channels, height, width):
        raise RuntimeError(f"unexpected backbone feature shape: {features.shape}")
    return features


def _execute_backbone(backbone: Any, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the quantized backbone on one image and return its features, scores and boxes."""
    outputs = backbone.execute(inputs={InputName(INPUT): to_afe_layout(image[None])}, use_jax=True)
    features, scores, boxes = (np.asarray(output) for output in outputs)
    return _features_to_nchw(features), scores, boxes


def _decoder_calibration_inputs(backbone: Any, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the features and top proposals the quantized backbone produces for each calibration image."""
    features = []
    proposals = []
    for image in images:
        image_features, scores, boxes = _execute_backbone(backbone, image)
        features.append(image_features[0].astype(np.float32))
        proposals.append(select_top_proposals(scores, boxes)[0])
    return np.stack(features), np.stack(proposals)


@dataclass
class _OutputError:
    """Running sums of one output's error against the original ONNX, over every fidelity image."""

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


def _precision_label(parameters: dict[str, Any]) -> str:
    """Name a component's precision, such as W8A16 per-token."""
    return parameters["precision"].upper() + (" per-token" if parameters.get("per_token") else "")


def _pipeline_description(component_parameters: dict[str, dict[str, Any]]) -> str:
    """Name the compiled pipeline that fidelity compares with the original ONNX."""
    backbone = _precision_label(component_parameters["backbone"])
    decoder = _precision_label(component_parameters["decoder"])
    return f"AFE {backbone} backbone → CPU TopK/Gather → AFE {decoder} decoder (JAX execute)"


def _calibration_names(
    component_parameters: dict[str, dict[str, Any]],
    calibration_dataset: Path | None,
    fidelity_dataset: Path,
    selection: dict[str, Any],
) -> tuple[str, ...]:
    """Select the calibration images of the W8A8 components; components with BF16 activations need none."""
    counts = {
        int(parameters["calibration_samples"])
        for parameters in component_parameters.values()
        if parameters["precision"] == "w8a8"
    }
    if not counts:
        return ()
    if len(counts) > 1:
        raise ValueError("W8A8 components must calibrate on the same images")
    (count,) = counts
    if calibration_dataset is None:
        raise ValueError("W8A8 components require --dataset for calibration")
    if calibration_dataset != fidelity_dataset:
        raise ValueError("RF-DETR calibration and fidelity must use the same verified dataset")
    if int(selection["offset"]) < count:
        raise ValueError("fidelity selection must begin after the calibration images")
    return select_images(calibration_dataset, int(selection["seed"]), 0, count)


def _write_fidelity(
    output_dir: Path,
    profile_id: str,
    config: dict[str, Any],
    candidate: str,
    metrics: list[dict[str, Any]],
) -> None:
    selection = config["selection"]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "component_id": config["result_component"],
        "scope": "full_pipeline",
        "status": config["status"],
        "reference": "Original full ONNX (ONNX Runtime)",
        "candidate": candidate,
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": selection["samples"],
        "metrics": metrics,
    }
    (output_dir / FIDELITY_REPORT).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reference_outputs(source: Path, images: np.ndarray) -> dict[str, np.ndarray]:
    """Run the original ONNX on the fidelity images."""
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    outputs = {name: [] for name in OUTPUTS}
    for image in images:
        for name, value in zip(OUTPUTS, session.run(None, {INPUT: image[None]}), strict=True):
            outputs[name].append(value[0])
    return {name: np.stack(values) for name, values in outputs.items()}


def _measure_fidelity(
    decoder: Any,
    backbone_outputs: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare the compiled pipeline's outputs on each fidelity image with the original ONNX's."""
    errors = [_OutputError(name) for name in OUTPUTS]
    for index, (features, scores, boxes) in enumerate(
        zip(
            backbone_outputs["evaluation_features"],
            backbone_outputs["evaluation_scores"],
            backbone_outputs["evaluation_boxes"],
            strict=True,
        )
    ):
        inputs = {
            InputName(FEATURES): to_afe_layout(features[None]),
            InputName(PROPOSALS): select_top_proposals(scores[None], boxes[None]),
        }
        outputs = decoder.execute(inputs=inputs, use_jax=True)
        for error, output in zip(errors, outputs, strict=True):
            error.add(reference[error.name][index][None], np.asarray(output))
    return [error.summary() for error in errors]


def _prepare_phase(
    source: Path,
    fidelity_dataset: Path,
    evaluation_names: tuple[str, ...],
    temporary_dir: Path,
) -> None:
    """Split the ONNX into the two component graphs, check the split, and save the reference outputs."""
    backbone_graph, decoder_graph = prepare_model(source, temporary_dir / "graphs")
    images = load_images(fidelity_dataset, evaluation_names)
    validate_float_split(source, backbone_graph, decoder_graph, images[0])
    np.savez(temporary_dir / REFERENCE_OUTPUTS, **_reference_outputs(source, images))


def _backbone_phase(
    graph: Path,
    temporary_dir: Path,
    profile_id: str,
    component_parameters: dict[str, dict[str, Any]],
    calibration_dataset: Path | None,
    calibration_names: tuple[str, ...],
    fidelity_dataset: Path,
    evaluation_names: tuple[str, ...],
) -> None:
    """Compile the backbone and save its outputs for the decoder phase."""
    parameters = component_parameters["backbone"]
    calibration_images = load_images(calibration_dataset, calibration_names) if calibration_names else None
    name = _component_name(profile_id, "backbone")
    backbone = _quantize_component(
        graph,
        name,
        parameters,
        {INPUT: calibration_images} if parameters["precision"] == "w8a8" else None,
        {INPUT: tuple(INPUT_SHAPE)},
    )
    _compile_component(backbone, temporary_dir / name, "backbone", parameters["precision"])

    features, scores, boxes = zip(
        *(_execute_backbone(backbone, image) for image in load_images(fidelity_dataset, evaluation_names)),
        strict=True,
    )
    outputs = {
        "evaluation_features": np.concatenate(features),
        "evaluation_scores": np.concatenate(scores),
        "evaluation_boxes": np.concatenate(boxes),
    }
    if component_parameters["decoder"]["precision"] == "w8a8":
        # A W8A8 decoder calibrates on the proposals this quantized backbone selects.
        outputs["calibration_features"], outputs["calibration_proposals"] = _decoder_calibration_inputs(
            backbone, calibration_images
        )
    np.savez(temporary_dir / BACKBONE_OUTPUTS, **outputs)


def _decoder_phase(
    graph: Path,
    temporary_dir: Path,
    profile_id: str,
    parameters: dict[str, Any],
    fidelity_config: dict[str, Any],
    candidate: str,
) -> None:
    """Compile the decoder, then measure the whole pipeline's fidelity and write its report."""
    with np.load(temporary_dir / BACKBONE_OUTPUTS) as saved:
        backbone_outputs = dict(saved)
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = dict(saved)
    calibration_inputs = None
    if parameters["precision"] == "w8a8":
        calibration_inputs = {
            FEATURES: backbone_outputs["calibration_features"],
            PROPOSALS: backbone_outputs["calibration_proposals"],
        }
    name = _component_name(profile_id, "decoder")
    decoder = _quantize_component(
        graph,
        name,
        parameters,
        calibration_inputs,
        {tensor: tuple(BOUNDARY_SHAPES[tensor]) for tensor in (FEATURES, PROPOSALS)},
    )
    _compile_component(decoder, temporary_dir / name, "decoder", parameters["precision"])
    metrics = _measure_fidelity(decoder, backbone_outputs, reference)
    _write_fidelity(temporary_dir, profile_id, fidelity_config, candidate, metrics)


def _execute_phase(target: Callable[..., None], arguments: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    target(**arguments)


def _run_phase(name: str, target: Callable[..., None], **arguments: Any) -> None:
    logging.info("Starting %s", name)
    process = multiprocessing.get_context("spawn").Process(name=name, target=_execute_phase, args=(target, arguments))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.exitcode}")
    logging.info("Completed %s", name)


def _copy_component_archives(temporary_dir: Path, profile_id: str, output_dir: Path) -> dict[str, Path]:
    """Copy each component's MPK to <output_dir>/<component>/."""
    archives = {}
    for component_id in COMPONENT_IDS:
        built = _find_mpk(temporary_dir / _component_name(profile_id, component_id), component_id)
        component_dir = output_dir / component_id
        component_dir.mkdir()
        archives[component_id] = component_dir / built.name
        shutil.copyfile(built, archives[component_id])
    return archives


def _write_compile_outputs(
    output_dir: Path,
    profile_id: str,
    archives: dict[str, Path],
    graph_hashes: dict[str, str],
    fidelity_component: str,
) -> None:
    """Write compile-outputs.json, which records each MPK, derived graph, and the fidelity report."""
    fidelity_path = output_dir / fidelity_component / FIDELITY_REPORT
    manifest = {
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "compile_invocation_id": INVOCATION_ID,
        "outputs": [
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archives[component_id].name}",
                "mpk_sha256": sha256_file(archives[component_id])[0],
            }
            for component_id in COMPONENT_IDS
        ],
        "derived_graphs": [
            {"component_id": component_id, "sha256": graph_hashes[component_id]} for component_id in COMPONENT_IDS
        ],
        "correctness": {
            "component_id": fidelity_component,
            "path": f"{fidelity_component}/{FIDELITY_REPORT}",
            "sha256": sha256_file(fidelity_path)[0],
        },
    }
    (output_dir / "compile-outputs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compile_model(
    source: Path,
    profile_id: str,
    calibration_dataset: Path | None,
    fidelity_dataset: Path,
    output_dir: Path,
) -> dict[str, Path]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    component_parameters = _component_parameters(metadata, profile_id)
    fidelity_config = next(entry for entry in metadata["fidelity"] if profile_id in entry["build_profiles"])
    selection = fidelity_config["selection"]
    calibration_names = _calibration_names(component_parameters, calibration_dataset, fidelity_dataset, selection)
    evaluation_names = select_images(
        fidelity_dataset,
        int(selection["seed"]),
        int(selection["offset"]),
        int(selection["samples"]),
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    temporary_root = os.environ.get("RUNNER_TEMP")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=temporary_root) as temporary:
        temporary_dir = Path(temporary)
        backbone_graph, decoder_graph = graph_paths(temporary_dir / "graphs")
        _run_phase(
            "RF-DETR graph preparation",
            _prepare_phase,
            source=source,
            fidelity_dataset=fidelity_dataset,
            evaluation_names=evaluation_names,
            temporary_dir=temporary_dir,
        )
        graph_hashes = {
            "backbone": sha256_file(backbone_graph)[0],
            "decoder": sha256_file(decoder_graph)[0],
        }
        _run_phase(
            "RF-DETR backbone compilation",
            _backbone_phase,
            graph=backbone_graph,
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            component_parameters=component_parameters,
            calibration_dataset=calibration_dataset,
            calibration_names=calibration_names,
            fidelity_dataset=fidelity_dataset,
            evaluation_names=evaluation_names,
        )
        _run_phase(
            "RF-DETR decoder compilation and fidelity",
            _decoder_phase,
            graph=decoder_graph,
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            parameters=component_parameters["decoder"],
            fidelity_config=fidelity_config,
            candidate=_pipeline_description(component_parameters),
        )
        output_dir.mkdir(parents=True)
        archives = _copy_component_archives(temporary_dir, profile_id, output_dir)
        fidelity_component = fidelity_config["result_component"]
        shutil.copyfile(temporary_dir / FIDELITY_REPORT, output_dir / fidelity_component / FIDELITY_REPORT)
        _write_compile_outputs(output_dir, profile_id, archives, graph_hashes, fidelity_component)
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="calibration images; required when a component is W8A8")
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != INVOCATION_ID:
        parser.error(f"unsupported compile invocation: {args.compile_invocation}")
    source = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the manual conversion workflow")
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = compile_model(
        source,
        args.build_profile,
        dataset,
        fidelity_dataset,
        args.output_dir.expanduser().resolve(),
    )
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
