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
    bfloat16_scheme,
    default_quantization,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import (
    execute_backbone,
    load_images,
    select_images,
    stable_topk_gather,
    to_afe_layout,
    transformer_inputs,
)
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
PROFILE_ID = "modalix_int8_bf16"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]


def _float32_bytes(shape: list[int]) -> int:
    return math.prod(shape) * 4


# MPK manifests record each public tensor's size in bytes.
COMPONENT_INTERFACES = {
    "backbone": {
        "inputs": {INPUT: _float32_bytes(INPUT_SHAPE)},
        "outputs": [_float32_bytes(BOUNDARY_SHAPES[name]) for name in (FEATURES, SCORES, BOXES)],
    },
    "transformer": {
        "inputs": {name: _float32_bytes(BOUNDARY_SHAPES[name]) for name in (FEATURES, PROPOSALS)},
        "outputs": [_float32_bytes(OUTPUT_SHAPES[name]) for name in OUTPUTS],
    },
}


def _profile() -> dict[str, Any]:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    return next(profile for profile in model["build_profiles"] if profile["id"] == PROFILE_ID)


def _quantization(precision: str, calibration_bins: int) -> Any:
    if precision == "int8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "bf16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(HistogramMSEMethod(calibration_bins))
        # INT8 bias correction measured +0.1 COCO mAP on the RF-DETR Small backbone; BF16 needs none.
        .with_bias_correction(BiasCorrectionType.REGULAR if precision == "int8" else BiasCorrectionType.NONE)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        .with_prefer_int8_udf(True)
        .with_per_token_quant(False)
    )


def _quantize_component(
    graph: Path,
    temporary_dir: Path,
    name: str,
    precision: str,
    calibration_bins: int,
    inputs: dict[str, np.ndarray],
    input_shapes: dict[str, tuple[int, ...]],
    output_names: list[str],
) -> tuple[Any, Path]:
    build_dir = temporary_dir / name
    build_dir.mkdir(parents=True, exist_ok=False)
    importer = onnx_source(str(graph), shape_dict=input_shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    data = convert_data_generator_to_iterable(
        DataGenerator({InputName(key): to_afe_layout(value) for key, value in inputs.items()})
    )
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization(precision, calibration_bins),
        model_name=name,
        log_level=logging.INFO,
    )
    return quantized, build_dir


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


def _compile_component(
    quantized: Any,
    build_dir: Path,
    component_id: str,
    precision: str,
) -> Path:
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
        layer_norm_use_fp32_intermediates=precision == "bf16",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    audit_mpk(archives[0], component_id)
    return archives[0]


def _record_metrics(
    totals: dict[str, dict[str, float]],
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    names: list[str],
) -> None:
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        expected = np.asarray(expected, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{name} shape mismatch: {expected.shape} != {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{name} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        total = totals.setdefault(
            name,
            {
                "count": 0.0,
                "dot": 0.0,
                "expected_squared": 0.0,
                "actual_squared": 0.0,
                "error_squared": 0.0,
                "error_absolute": 0.0,
                "max_absolute_error": 0.0,
            },
        )
        total["count"] += expected.size
        total["dot"] += float(np.dot(expected.ravel(), actual.ravel()))
        total["expected_squared"] += float(np.sum(np.square(expected)))
        total["actual_squared"] += float(np.sum(np.square(actual)))
        total["error_squared"] += float(np.sum(np.square(error)))
        total["error_absolute"] += float(np.sum(absolute))
        total["max_absolute_error"] = max(total["max_absolute_error"], float(np.max(absolute)))


def _finish_metrics(totals: dict[str, dict[str, float]], names: list[str]) -> list[dict[str, Any]]:
    results = []
    for name in names:
        total = totals[name]
        count = total["count"]
        expected_rms = max(np.sqrt(total["expected_squared"] / count), np.finfo(np.float64).eps)
        norm = np.sqrt(total["expected_squared"] * total["actual_squared"])
        cosine = total["dot"] / norm if norm else float(total["expected_squared"] == total["actual_squared"])
        results.append(
            {
                "name": name,
                "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
                "normalized_rmse": float(np.sqrt(total["error_squared"] / count) / expected_rms),
                "mean_absolute_error": float(total["error_absolute"] / count),
                "max_absolute_error": total["max_absolute_error"],
            }
        )
    return results


def _write_fidelity(
    output_dir: Path,
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> Path:
    selection = config["selection"]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": PROFILE_ID,
        "component_id": config["result_component"],
        "scope": "full_pipeline",
        "status": config["status"],
        "reference": "Original full ONNX (ONNX Runtime)",
        "candidate": "AFE INT8 backbone → CPU TopK/Gather → AFE BF16 transformer (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": selection["samples"],
        "metrics": metrics,
    }
    path = output_dir / "fidelity.json"
    path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _prepare_graphs_and_reference(
    source: Path,
    dataset: Path,
    evaluation_names: tuple[str, ...],
    temporary_dir: Path,
) -> None:
    _, backbone_graph, transformer_graph = prepare_model(source, temporary_dir / "graphs")
    validation_image = load_images(dataset, evaluation_names[:1])[0]
    validate_float_split(
        source,
        backbone_graph,
        transformer_graph,
        validation_image,
    )

    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    outputs = {name: [] for name in OUTPUTS}
    for image in load_images(dataset, evaluation_names):
        values = session.run(None, {INPUT: image[None]})
        for name, value in zip(OUTPUTS, values, strict=True):
            outputs[name].append(value[0])
    np.savez(temporary_dir / "reference_outputs.npz", **{name: np.stack(outputs[name]) for name in OUTPUTS})


def _compile_backbone(
    graph: Path,
    dataset: Path,
    calibration_names: tuple[str, ...],
    evaluation_names: tuple[str, ...],
    temporary_dir: Path,
    precision: str,
    calibration_bins: int,
) -> None:
    calibration_images = load_images(dataset, calibration_names)
    backbone, build_dir = _quantize_component(
        graph,
        temporary_dir,
        f"{MODEL_ID}_backbone_{PROFILE_ID}",
        precision,
        calibration_bins,
        {INPUT: calibration_images},
        {INPUT: tuple(INPUT_SHAPE)},
        [FEATURES, SCORES, BOXES],
    )
    _compile_component(backbone, build_dir, "backbone", precision)

    calibration_features, calibration_proposals = transformer_inputs(backbone, calibration_images, InputName)
    evaluation = [execute_backbone(backbone, image, InputName) for image in load_images(dataset, evaluation_names)]
    np.savez(
        temporary_dir / "backbone_outputs.npz",
        calibration_features=calibration_features,
        calibration_proposals=calibration_proposals,
        evaluation_features=np.stack([values[0][0] for values in evaluation]),
        evaluation_scores=np.stack([values[1][0] for values in evaluation]),
        evaluation_boxes=np.stack([values[2][0] for values in evaluation]),
    )


def _compile_transformer(
    graph: Path,
    temporary_dir: Path,
    precision: str,
    calibration_bins: int,
    fidelity_config: dict[str, Any],
) -> None:
    with np.load(temporary_dir / "backbone_outputs.npz") as boundaries:
        calibration_features = boundaries["calibration_features"]
        calibration_proposals = boundaries["calibration_proposals"]
        evaluation_features = boundaries["evaluation_features"]
        evaluation_scores = boundaries["evaluation_scores"]
        evaluation_boxes = boundaries["evaluation_boxes"]

        transformer, build_dir = _quantize_component(
            graph,
            temporary_dir,
            f"{MODEL_ID}_transformer_{PROFILE_ID}",
            precision,
            calibration_bins,
            {FEATURES: calibration_features, PROPOSALS: calibration_proposals},
            {name: tuple(BOUNDARY_SHAPES[name]) for name in (FEATURES, PROPOSALS)},
            OUTPUTS,
        )
        _compile_component(transformer, build_dir, "transformer", precision)

        totals: dict[str, dict[str, float]] = {}
        with np.load(temporary_dir / "reference_outputs.npz") as reference:
            for index in range(len(evaluation_features)):
                proposals = stable_topk_gather(
                    evaluation_scores[index][None],
                    evaluation_boxes[index][None],
                )
                inputs = {
                    InputName(FEATURES): to_afe_layout(evaluation_features[index][None]),
                    InputName(PROPOSALS): proposals,
                }
                actual = [np.asarray(output) for output in transformer.execute(inputs=inputs, use_jax=True)]
                expected = [reference[name][index][None] for name in OUTPUTS]
                _record_metrics(totals, expected, actual, OUTPUTS)
    metrics = _finish_metrics(totals, OUTPUTS)
    _write_fidelity(temporary_dir, fidelity_config, metrics)


def _execute_phase(target: Callable[..., None], args: tuple[Any, ...]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    target(*args)


def _run_phase(name: str, target: Callable[..., None], *args: Any) -> None:
    logging.info("Starting %s", name)
    process = multiprocessing.get_context("spawn").Process(name=name, target=_execute_phase, args=(target, args))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.exitcode}")
    logging.info("Completed %s", name)


def _component_archive(temporary_dir: Path, component_id: str) -> Path:
    build_dir = temporary_dir / f"{MODEL_ID}_{component_id}_{PROFILE_ID}"
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def compile_model(
    source: Path,
    calibration_dataset: Path,
    correctness_dataset: Path,
    output_dir: Path,
) -> dict[str, Path]:
    profile = _profile()
    bindings = {binding["id"]: binding["parameters"] for binding in profile["components"]}
    calibration_samples = int(bindings["backbone"]["calibration_samples"])
    transformer_samples = int(bindings["transformer"]["calibration_samples"])
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    fidelity_config = next(entry for entry in metadata["validation"] if entry["id"] == "full_pipeline_fidelity")
    selection = fidelity_config["selection"]
    if transformer_samples != calibration_samples:
        raise ValueError("backbone and transformer must use the same calibration sample count")
    if int(selection["offset"]) != calibration_samples:
        raise ValueError("fidelity selection must begin after the calibration partition")
    if calibration_dataset.resolve() != correctness_dataset.resolve():
        raise ValueError("RF-DETR calibration and fidelity must use the same verified dataset")
    calibration_names, evaluation_names = select_images(
        calibration_dataset,
        calibration_samples,
        int(selection["samples"]),
        int(selection["seed"]),
    )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    temporary_root = os.environ.get("RUNNER_TEMP")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=temporary_root) as temporary:
        temporary_dir = Path(temporary)
        _, backbone_graph, transformer_graph = graph_paths(temporary_dir / "graphs")
        _run_phase(
            "RF-DETR graph preparation",
            _prepare_graphs_and_reference,
            source,
            correctness_dataset,
            evaluation_names,
            temporary_dir,
        )
        graph_hashes = {
            "backbone": sha256_file(backbone_graph)[0],
            "transformer": sha256_file(transformer_graph)[0],
        }
        _run_phase(
            "RF-DETR backbone compilation",
            _compile_backbone,
            backbone_graph,
            calibration_dataset,
            calibration_names,
            evaluation_names,
            temporary_dir,
            str(bindings["backbone"]["precision"]),
            int(bindings["backbone"]["calibration_bins"]),
        )
        _run_phase(
            "RF-DETR transformer compilation and fidelity",
            _compile_transformer,
            transformer_graph,
            temporary_dir,
            str(bindings["transformer"]["precision"]),
            int(bindings["transformer"]["calibration_bins"]),
            fidelity_config,
        )
        backbone_archive_source = _component_archive(temporary_dir, "backbone")
        transformer_archive_source = _component_archive(temporary_dir, "transformer")

        output_dir.mkdir(parents=True)
        archives = {}
        for component_id, source in (
            ("backbone", backbone_archive_source),
            ("transformer", transformer_archive_source),
        ):
            component_dir = output_dir / component_id
            component_dir.mkdir()
            archive = component_dir / source.name
            shutil.copyfile(source, archive)
            archives[component_id] = archive

        fidelity_component = fidelity_config["result_component"]
        fidelity_path = output_dir / fidelity_component / "fidelity.json"
        shutil.copyfile(temporary_dir / "fidelity.json", fidelity_path)
        manifest = {
            "schema_version": 2,
            "model_id": MODEL_ID,
            "build_profile_id": PROFILE_ID,
            "compile_invocation_id": INVOCATION_ID,
            "outputs": [
                {
                    "component_id": component_id,
                    "mpk_path": f"{component_id}/{archives[component_id].name}",
                    "mpk_sha256": sha256_file(archives[component_id])[0],
                }
                for component_id in ("backbone", "transformer")
            ],
            "derived_graphs": [
                {"component_id": component_id, "sha256": graph_hashes[component_id]}
                for component_id in ("backbone", "transformer")
            ],
            "correctness": {
                "component_id": fidelity_component,
                "path": f"{fidelity_component}/fidelity.json",
                "sha256": sha256_file(fidelity_path)[0],
            },
        }
        (output_dir / "compile-outputs.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--correctness-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.build_profile != PROFILE_ID:
        parser.error(f"unsupported build profile: {args.build_profile}")
    if args.compile_invocation != INVOCATION_ID:
        parser.error(f"unsupported compile invocation: {args.compile_invocation}")
    source = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    correctness_dataset = args.correctness_dataset.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the manual conversion workflow")
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not correctness_dataset.exists():
        parser.error(f"correctness dataset does not exist: {correctness_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = compile_model(
        source,
        dataset,
        correctness_dataset,
        args.output_dir.expanduser().resolve(),
    )
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
