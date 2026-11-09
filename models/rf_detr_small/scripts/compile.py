#!/usr/bin/env python3
"""Partition, validate, and compile the verified RF-DETR Small ONNX."""

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
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, select_images, select_top_proposals, to_afe_layout
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    BF16_REGIONS,
    BOUNDARY_SHAPES,
    BOXES,
    FEATURES,
    GLOBAL_ATTENTION_BLOCKS,
    INPUT,
    INPUT_SHAPE,
    OUTPUT_SHAPES,
    OUTPUTS,
    PROPOSALS,
    SCORES,
    annotate_bf16_regions,
    graph_paths,
    prepare_model,
    validate_float_split,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "rf_detr_small"
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


def _bf16_regions(profile_id: str, component_parameters: dict[str, dict[str, Any]]) -> list[str]:
    """Return the backbone regions the build profile runs BF16, checking that its ID marks it mixed."""
    if "bf16_regions" in component_parameters["decoder"]:
        raise ValueError("only the backbone declares bf16_regions")
    parameters = component_parameters["backbone"]
    regions = parameters.get("bf16_regions", [])
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"bf16_regions must name distinct regions of a W8A8 backbone: {regions}")
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    if unknown:
        raise ValueError(f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    if bool(regions) != ("mixed" in profile_id.split("_")):
        raise ValueError(f"build profile {profile_id!r} must have the mixed token exactly when it has bf16_regions")
    return regions


def _calibrates(parameters: dict[str, Any]) -> bool:
    return "calibration_samples" in parameters


def _quantization_config(parameters: dict[str, Any]) -> QuantizationParams:
    """The compiler's activation and weight formats for a component's precision."""
    precision = parameters["precision"]
    # BF16 activations need no calibrated ranges, except where AFE keeps an operator in INT8: it has no
    # BF16 Sin or Cos and runs them as INT8 lookup tables. The decoder's query sine embedding is such a
    # Sin, so the decoder calibrates even at W16A16, while a W8A16 backbone skips calibration.
    calibration = (
        HistogramMSEMethod(int(parameters["calibration_bins"])) if _calibrates(parameters) else SkipCalibration()
    )
    if parameters.get("per_token") and precision != "w8a16":
        raise ValueError("only a W8A16 component can declare per_token")
    if precision == "w8a8":
        if not _calibrates(parameters):
            raise ValueError("a W8A8 component must declare calibration_bins and calibration_samples")
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Bias correction measured +0.1 COCO mAP on the RF-DETR Small INT8 backbone.
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=bool(parameters.get("per_token", False)),
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=calibration,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


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


def _require_bf16_regions(quantized: Any, regions: list[str]) -> None:
    """Fail when AFE ignored the BF16 annotations, which would silently build a W8A8 backbone."""
    # The quantized model keeps its network private, and AFE renames every node, so count by operation.
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.BFLOAT_16
    )
    logging.info("BF16 backbone operations: %s", dict(sorted(counts.items())))
    # Softmax stays one AFE node, unlike LayerNorm, whose scale and bias AFE splits off.
    softmaxes = GLOBAL_ATTENTION_BLOCKS if "attention_global" in regions else 0
    if not counts or counts["SoftmaxOp"] != softmaxes:
        raise RuntimeError(f"expected BF16 operations including {softmaxes} softmaxes, found {dict(counts)}")


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
        # The option applies to every layer norm and INT8 ones reject it, so BF16 regions keep BF16 intermediates.
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
class _BackboneOutputs:
    """What the backbone phase hands the decoder phase, saved to BACKBONE_OUTPUTS between their processes."""

    evaluation_features: np.ndarray
    evaluation_scores: np.ndarray
    evaluation_boxes: np.ndarray
    # Only when the decoder calibrates: the features and top proposals of each calibration image.
    calibration_features: np.ndarray | None = None
    calibration_proposals: np.ndarray | None = None

    def save(self, path: Path) -> None:
        np.savez(path, **{name: value for name, value in vars(self).items() if value is not None})

    @classmethod
    def load(cls, path: Path) -> _BackboneOutputs:
        with np.load(path) as saved:
            return cls(**saved)


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
    """Name a component's precision, such as W8A16 per-token or W8A8 mixed (BF16 residual)."""
    label = parameters["precision"].upper() + (" per-token" if parameters.get("per_token") else "")
    regions = parameters.get("bf16_regions")
    return label + (f" mixed (BF16 {', '.join(regions)})" if regions else "")


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
    """Select the calibration images of the components that calibrate."""
    counts = {
        int(parameters["calibration_samples"])
        for parameters in component_parameters.values()
        if _calibrates(parameters)
    }
    if not counts:
        return ()
    if len(counts) > 1:
        raise ValueError("calibrated components must use the same images")
    (count,) = counts
    if calibration_dataset is None:
        raise ValueError("calibrated components require --dataset")
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
    backbone_outputs: _BackboneOutputs,
    reference: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare the compiled pipeline's outputs on each fidelity image with the original ONNX's."""
    errors = [_OutputError(name) for name in OUTPUTS]
    for index, (features, scores, boxes) in enumerate(
        zip(
            backbone_outputs.evaluation_features,
            backbone_outputs.evaluation_scores,
            backbone_outputs.evaluation_boxes,
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
    bf16_regions: list[str],
    temporary_dir: Path,
) -> None:
    """Split the ONNX into the two component graphs, check the split, and save the reference outputs.

    Then mark the backbone's BF16 regions, which ONNX Runtime cannot run.
    """
    backbone_graph, decoder_graph = prepare_model(source, temporary_dir / "graphs")
    images = load_images(fidelity_dataset, evaluation_names)
    validate_float_split(source, backbone_graph, decoder_graph, images[0])
    np.savez(temporary_dir / REFERENCE_OUTPUTS, **_reference_outputs(source, images))

    if bf16_regions:
        marks = annotate_bf16_regions(backbone_graph, bf16_regions)
        logging.info(
            "Marked %d backbone operations BF16 and %d INT8 for %s", marks["bfloat16"], marks["int8"], bf16_regions
        )


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
        {INPUT: calibration_images} if _calibrates(parameters) else None,
        {INPUT: tuple(INPUT_SHAPE)},
    )

    if parameters.get("bf16_regions"):
        _require_bf16_regions(backbone, parameters["bf16_regions"])
    _compile_component(backbone, temporary_dir / name, "backbone", parameters["precision"])

    features, scores, boxes = zip(
        *(_execute_backbone(backbone, image) for image in load_images(fidelity_dataset, evaluation_names)),
        strict=True,
    )
    outputs = _BackboneOutputs(np.concatenate(features), np.concatenate(scores), np.concatenate(boxes))
    if _calibrates(component_parameters["decoder"]):
        # The decoder calibrates on the proposals this quantized backbone selects.
        outputs.calibration_features, outputs.calibration_proposals = _decoder_calibration_inputs(
            backbone, calibration_images
        )
    outputs.save(temporary_dir / BACKBONE_OUTPUTS)


def _decoder_phase(
    graph: Path,
    temporary_dir: Path,
    profile_id: str,
    parameters: dict[str, Any],
    fidelity_config: dict[str, Any],
    candidate: str,
) -> None:
    """Compile the decoder, then measure the whole pipeline's fidelity and write its report."""
    backbone_outputs = _BackboneOutputs.load(temporary_dir / BACKBONE_OUTPUTS)
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = dict(saved)

    calibration_inputs = None
    if _calibrates(parameters):
        calibration_inputs = {
            FEATURES: backbone_outputs.calibration_features,
            PROPOSALS: backbone_outputs.calibration_proposals,
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
    try:
        target(**arguments)
    except SystemExit as error:
        # AFE reports a failed quantization or compilation by raising SystemExit(), which exits with code 0.
        raise RuntimeError(f"{target.__name__} stopped in AFE; the AFE log above names the cause") from error


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
    output_dir: Path, profile_id: str, archives: dict[str, Path], graph_hashes: dict[str, str]
) -> None:
    """Write compile-outputs.json, which records each MPK and derived graph."""
    manifest = {
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "compile_invocation_id": profile_id,
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
    metadata = load_metadata(MODEL_DIR)
    component_parameters = _component_parameters(metadata, profile_id)
    bf16_regions = _bf16_regions(profile_id, component_parameters)
    fidelity_config = metadata["numerical_fidelity"]
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
            bf16_regions=bf16_regions,
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
        shutil.copyfile(temporary_dir / FIDELITY_REPORT, output_dir / FIDELITY_REPORT)
        _write_compile_outputs(output_dir, profile_id, archives, graph_hashes)
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="calibration images; required when a component calibrates")
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
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
