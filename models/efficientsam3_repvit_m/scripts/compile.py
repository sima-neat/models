#!/usr/bin/env python3
"""Export EfficientSAM3 from its converted weights, split it, and compile its text encoder and image model into BF16
MPKs.

surgery.py checks its reconstruction against the upstream grid-72 PyTorch reference, then reduces the token grid
to 48 and checks the ONNX export against PyTorch at that grid. The BF16 text encoder skips calibration. The image
model calibrates its two INT8 GroupNorm rsqrt UDFs on real images and quantized text features, as the board runs
them. Numerical fidelity compares the whole pipeline with the exported ONNX. Calibration takes the first images
of the coco-2017 cal archive's seeded order and fidelity a later span; each image gets one COCO category prompt,
spread over the 80 names.
"""

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
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import load_images, one_hot, packed, prompts, select_images, tokens
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    IMAGE_INPUTS,
    IMAGE_OUTPUTS,
    INTERFACE,
    MASK_SIZE,
    QUERIES,
    RUNTIME_FILES,
    TEXT_INPUT,
    TEXT_OUTPUT,
    export_model,
    graph_paths,
    prepare_model,
    validate_float_split,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "efficientsam3_repvit_m"
COMPONENT_IDS = ("text_encoder", "image_model")
# Files the compilation phases pass to each other in the temporary directory.
REFERENCE_OUTPUTS = "reference_outputs.npz"
TEXT_OUTPUTS = "text_outputs.npz"
FIDELITY_REPORT = "fidelity.json"
# The compiled pipeline that fidelity compares with the original ONNX; _quantization builds only W16A16 components.
CANDIDATE = "AFE W16A16 text encoder → CPU text packing → AFE W16A16 image model (JAX execute)"


def _float32_bytes(name: str) -> int:
    return math.prod(INTERFACE[name]) * 4


# MPK manifests record each public tensor's size in bytes; the MPKs keep the ONNX float32 interface.
COMPONENT_INTERFACES = {
    "text_encoder": {"inputs": {TEXT_INPUT: _float32_bytes(TEXT_INPUT)}, "outputs": [_float32_bytes(TEXT_OUTPUT)]},
    "image_model": {
        "inputs": {name: _float32_bytes(name) for name in IMAGE_INPUTS},
        "outputs": [_float32_bytes(name) for name in IMAGE_OUTPUTS],
    },
}


def _component_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    """Return each component's build parameters in the build profile."""
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}


def _component_name(profile_id: str, component_id: str) -> str:
    """Name a component's quantized model, its MPK and its build directory."""
    return f"{MODEL_ID}_{profile_id}_{component_id}"


def _quantization(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision != "w16a16":
        raise ValueError(f"unsupported precision: {precision}")
    bins = parameters.get("calibration_bins")
    return QuantizationParams(
        activation_quantization_scheme=bfloat16_scheme(),
        weight_quantization_scheme=bfloat16_scheme(),
        requantization_mode=RequantizationMode.sima,
        # Only the image model needs ranges: its mask-head GroupNorm has two INT8 rsqrt UDFs.
        calibration_method=SkipCalibration() if bins is None else HistogramMSEMethod(int(bins)),
        biascorr_type=BiasCorrectionType.NONE,
        channel_equalization=False,
        smooth_quant=False,
        prefer_int8_udf=True,
        per_token_quant=False,
    )


def _quantize_component(
    graph: Path, name: str, parameters: dict[str, Any], inputs: dict[str, np.ndarray] | None = None
) -> Any:
    shapes = {key: tuple(INTERFACE[key]) for key in (inputs if inputs is not None else [TEXT_INPUT])}
    importer = onnx_source(str(graph), shape_dict=shapes, layout="NCHW", flexible_batch_size=False)
    loaded = load_model(importer, target=gen2_target)
    data = None if inputs is None else convert_data_generator_to_iterable(
        DataGenerator({InputName(key): value for key, value in inputs.items()})
    )
    return loaded.quantize(
        calibration_data=data,
        quantization_config=_quantization(parameters),
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
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if public_inputs != contract["inputs"]:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {public_inputs}")
    # A single-output MPK ends in the float32 cast; a multi-output one gathers its casts in a pass-through.
    output_sizes = [node.get("size") for node in plugins[-1].get("output_nodes", [])] if plugins else []
    if output_sizes != contract["outputs"]:
        raise RuntimeError(f"{component_id} MPK output interface mismatch: {output_sizes}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins), "output_sizes": output_sizes}


def _find_mpk(build_dir: Path, component_id: str) -> Path:
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def _compile_component(quantized: Any, build_dir: Path, component_id: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=False)
    quantized.compile(
        output_path=str(build_dir),
        log_level=logging.INFO,
        preserve=True,
        layer_norm_use_fp32_intermediates=True,
    )
    archive = _find_mpk(build_dir, component_id)
    audit_mpk(archive, component_id)
    return archive


def _image_outputs(outputs: list[Any]) -> list[np.ndarray]:
    """Return the image model's detections and NCHW mask logits from an AFE execution."""
    detections, masks = (np.asarray(output, dtype=np.float32) for output in outputs)
    if masks.shape == (1, MASK_SIZE, MASK_SIZE, QUERIES):
        masks = masks.transpose(0, 3, 1, 2)
    shapes = [list(detections.shape), list(masks.shape)]
    if shapes != [INTERFACE[name] for name in IMAGE_OUTPUTS]:
        raise RuntimeError(f"unexpected image model outputs: {shapes}")
    return [detections, masks]


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


def _image_selections(
    component_parameters: dict[str, dict[str, Any]],
    calibration_dataset: Path,
    fidelity_dataset: Path,
    selection: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select the calibration images, the first of the seeded order, and the later span fidelity measures."""
    count = int(component_parameters["image_model"]["calibration_samples"])
    if calibration_dataset != fidelity_dataset:
        raise ValueError("EfficientSAM3 calibration and fidelity must use the same verified dataset")
    seed, offset = int(selection["seed"]), int(selection["offset"])
    if offset < count:
        raise ValueError("fidelity selection must begin after the calibration images")
    return (
        select_images(calibration_dataset, seed, 0, count),
        select_images(fidelity_dataset, seed, offset, int(selection["samples"])),
    )


def _write_fidelity(output_dir: Path, profile_id: str, config: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    selection = config["selection"]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "FP32 PyTorch export (ONNX Runtime)",
        "candidate": CANDIDATE,
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": selection,
        "samples": selection["samples"],
        "metrics": metrics,
    }
    (output_dir / FIDELITY_REPORT).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _measure_fidelity(
    model: Any, images: np.ndarray, texts: np.ndarray, reference: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Compare the compiled pipeline's outputs on each fidelity image with the original ONNX's."""
    errors = [_OutputError(name) for name in IMAGE_OUTPUTS]
    for index, (image, text) in enumerate(zip(images, texts, strict=True)):
        inputs = {InputName("pixel_values"): image[None], InputName("text"): text[None]}
        outputs = _image_outputs(list(model.execute(inputs=inputs, use_jax=True)))
        for error, output in zip(errors, outputs, strict=True):
            error.add(reference[error.name][index][None], output)
    return [error.summary() for error in errors]


def _prepare_phase(source: Path, fidelity_dataset: Path, fidelity_names: tuple[str, ...], temporary_dir: Path) -> None:
    """Export the weights to ONNX, split it into the two component graphs, check the split, and save the reference
    outputs."""
    exported = export_model(source, temporary_dir / "graphs" / f"{MODEL_ID}.onnx")
    text_graph, image_graph = prepare_model(exported, temporary_dir / "graphs")
    images = load_images(fidelity_dataset, fidelity_names)
    report, reference = validate_float_split(exported, text_graph, image_graph, images, prompts(len(fidelity_names)))
    logging.info("Float split matches the exported graph: %s", json.dumps(report))
    np.savez(temporary_dir / REFERENCE_OUTPUTS, **reference)
    # The components compile from their own graphs; the 1.6 GB export is not needed again.
    exported.unlink()


def _text_encoder_phase(
    graph: Path,
    temporary_dir: Path,
    profile_id: str,
    parameters: dict[str, Any],
    calibration_prompts: list[str],
    fidelity_prompts: list[str],
) -> None:
    """Compile the BF16 text encoder without calibration, then prepare image calibration and fidelity features."""
    name = _component_name(profile_id, "text_encoder")
    encoder = _quantize_component(graph, name, parameters)
    _compile_component(encoder, temporary_dir / name, "text_encoder")

    # The image model calibrates and is checked on the quantized encoder's features, as the board runs.
    def packed_text(prompt: str) -> np.ndarray:
        (features,) = encoder.execute(inputs={InputName(TEXT_INPUT): one_hot(tokens(prompt))}, use_jax=True)
        return packed(np.asarray(features), tokens(prompt))

    np.savez(
        temporary_dir / TEXT_OUTPUTS,
        calibration=np.concatenate([packed_text(prompt) for prompt in calibration_prompts]),
        fidelity=np.concatenate([packed_text(prompt) for prompt in fidelity_prompts]),
    )


def _image_model_phase(
    graph: Path,
    temporary_dir: Path,
    profile_id: str,
    parameters: dict[str, Any],
    dataset: Path,
    calibration_names: tuple[str, ...],
    fidelity_names: tuple[str, ...],
    fidelity_config: dict[str, Any],
) -> None:
    """Compile the image model, then measure the whole pipeline's fidelity and write its report."""
    with np.load(temporary_dir / TEXT_OUTPUTS) as text:
        calibration_text, fidelity_text = text["calibration"], text["fidelity"]
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = dict(saved)
    name = _component_name(profile_id, "image_model")
    inputs = {"pixel_values": load_images(dataset, calibration_names), "text": calibration_text}
    model = _quantize_component(graph, name, parameters, inputs)
    del inputs
    _compile_component(model, temporary_dir / name, "image_model")

    metrics = _measure_fidelity(model, load_images(dataset, fidelity_names), fidelity_text, reference)
    _write_fidelity(temporary_dir, profile_id, fidelity_config, metrics)


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
        "runtime_files": [
            {"path": f"tokenizer/{name}", "sha256": sha256_file(output_dir / "tokenizer" / name)[0]}
            for name in RUNTIME_FILES
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
    source: Path, profile_id: str, calibration_dataset: Path, fidelity_dataset: Path, output_dir: Path
) -> dict[str, Path]:
    metadata = load_metadata(MODEL_DIR)
    component_parameters = _component_parameters(metadata, profile_id)
    fidelity_config = metadata["numerical_fidelity"]
    calibration_names, fidelity_names = _image_selections(
        component_parameters, calibration_dataset, fidelity_dataset, fidelity_config["selection"]
    )

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        text_graph, image_graph = graph_paths(temporary_dir / "graphs")
        _run_phase(
            "EfficientSAM3 graph preparation",
            _prepare_phase,
            source=source,
            fidelity_dataset=fidelity_dataset,
            fidelity_names=fidelity_names,
            temporary_dir=temporary_dir,
        )
        graph_hashes = {"text_encoder": sha256_file(text_graph)[0], "image_model": sha256_file(image_graph)[0]}

        _run_phase(
            "EfficientSAM3 text encoder compilation",
            _text_encoder_phase,
            graph=text_graph,
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            parameters=component_parameters["text_encoder"],
            calibration_prompts=prompts(len(calibration_names)),
            fidelity_prompts=prompts(len(fidelity_names)),
        )

        _run_phase(
            "EfficientSAM3 image model compilation and fidelity",
            _image_model_phase,
            graph=image_graph,
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            parameters=component_parameters["image_model"],
            dataset=calibration_dataset,
            calibration_names=calibration_names,
            fidelity_names=fidelity_names,
            fidelity_config=fidelity_config,
        )

        output_dir.mkdir(parents=True)
        archives = _copy_component_archives(temporary_dir, profile_id, output_dir)
        shutil.copytree(temporary_dir / "graphs" / "tokenizer", output_dir / "tokenizer")
        shutil.copyfile(temporary_dir / FIDELITY_REPORT, output_dir / FIDELITY_REPORT)
        _write_compile_outputs(output_dir, profile_id, archives, graph_hashes)
        return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="calibration images")
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve()
    if source.suffix not in {".pt", ".pth"}:
        parser.error("--model must be the weights bundle from the conversion workflow")
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    output_dir = args.output_dir.expanduser().resolve()
    outputs = compile_model(source, args.build_profile, dataset, fidelity_dataset, output_dir)
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
