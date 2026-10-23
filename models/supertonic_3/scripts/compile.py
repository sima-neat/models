#!/usr/bin/env python3
"""Prepare, quantize, verify, and compile Supertonic 3's vector estimator and vocoder."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_model as load_metadata
from model_registry.model import mpk_filename
from runtime_inputs import MAX_SEQUENCE_LENGTH, build_time_table, pack_rope_input
from surgery import prepare_model, sha256
from vector_contract import COMPILED_INPUT_SHAPES, COMPILED_OUTPUT_SHAPES
from vocoder_surgery import EXPANDED_LENGTH, LATENT_CHANNELS, LATENT_LENGTH, PCM_PER_FRAME

if TYPE_CHECKING:
    from afe.apis.defines import InputName

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "supertonic_3"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
COMPONENT_IDS = ("vector_estimator", "vocoder")
# The source is Supertone's ONNX folder; the vector estimator is its main file and the vocoder sits beside it.
VOCODER_SOURCE = "vocoder.onnx"
DENOISING_STEPS = 8
# Graph surgery fixes every sequence at its longest length and builds the step table for DENOISING_STEPS.
SURGERY_PARAMETERS: dict[str, dict[str, Any]] = {
    "vector_estimator": {
        "component": "vector_estimator",
        "text_length": MAX_SEQUENCE_LENGTH,
        "latent_length": LATENT_LENGTH,
        "steps": DENOISING_STEPS,
    },
    "vocoder": {"component": "vocoder", "latent_length": LATENT_LENGTH},
}
# Activations are BF16, so quantization needs no measured ranges: one seeded synthetic sample calibrates.
CALIBRATION_SEED = 1101
RUNTIME_DATA = "supertonic_runtime_data.npz"
COMPILE_REPORT = "compile-report.json"

COMPONENTS: dict[str, dict[str, Any]] = {
    "vector_estimator": {
        "inputs": dict(COMPILED_INPUT_SHAPES),
        "outputs": dict(COMPILED_OUTPUT_SHAPES),
    },
    "vocoder": {
        "inputs": {"latent": (1, LATENT_CHANNELS, 1, LATENT_LENGTH)},
        "outputs": {"wav_frames": (1, PCM_PER_FRAME, 1, EXPANDED_LENGTH)},
    },
}


def _component_parameters(metadata: dict[str, Any], profile_id: str) -> dict[str, dict[str, Any]]:
    """Return each component's build parameters in the build profile."""
    profiles = [profile for profile in metadata["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}


def _batch_size(parameters: dict[str, Any]) -> int:
    """Input sets one call takes, stacked on a leading axis."""
    batch_size = int(parameters.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, found {batch_size}")
    return batch_size


def _to_afe_layout(value: np.ndarray) -> np.ndarray:
    if value.ndim != 4:
        raise ValueError(f"expected rank-4 NCHW input, found {value.shape}")
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1), dtype=np.float32)


def _calibration_inputs(component_id: str, runtime_data: Path | None) -> dict[str, np.ndarray]:
    """One seeded synthetic input set with the masks, step table and rotary tables the engine builds."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    shapes = COMPONENTS[component_id]["inputs"]
    if component_id == "vocoder":
        return {"latent": rng.standard_normal(shapes["latent"], dtype=np.float32)}

    if runtime_data is None:
        raise ValueError("vector-estimator preparation did not produce runtime data")
    latent_mask = np.zeros(shapes["latent_mask"], dtype=np.float32)
    latent_mask[..., : MAX_SEQUENCE_LENGTH - 32] = 1.0
    text_mask = np.zeros(shapes["text_mask"], dtype=np.float32)
    text_mask[:, : MAX_SEQUENCE_LENGTH - 64, ...] = 1.0
    with np.load(runtime_data) as archive:
        style_key = np.asarray(archive["conditional_style_key"], dtype=np.float32)
        rope_bank = np.asarray(archive["rope_sin_cos_by_length"], dtype=np.float32)
    inputs = {
        "noisy_latent": rng.standard_normal(shapes["noisy_latent"], dtype=np.float32),
        "text_emb": rng.standard_normal(shapes["text_emb"], dtype=np.float32),
        "style_ttl": rng.standard_normal(shapes["style_ttl"], dtype=np.float32),
        "style_key": np.ascontiguousarray(style_key.transpose(0, 2, 1)[:, :, None, :]),
        "latent_mask": latent_mask,
        "text_mask": text_mask,
        "time_sinusoidal": build_time_table(DENOISING_STEPS)[0],
        "rope_tables": pack_rope_input(rope_bank, latent_mask, text_mask),
    }
    for name, shape in shapes.items():
        if inputs[name].shape != shape:
            raise RuntimeError(f"{name} shape {inputs[name].shape} != {shape}")
    return inputs


def _quantization(parameters: dict[str, Any]) -> Any:
    from afe.apis.defines import (
        BiasCorrectionType,
        CalibrationMethod,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        quantization_scheme,
    )

    precision = parameters["precision"]
    if precision == "w8a16":
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        weights = bfloat16_scheme()
    else:
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(bfloat16_scheme())
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("mse"))
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        .with_prefer_int8_udf(True)
        .with_per_token_quant(False)
    )


def _load_and_quantize(
    graph: Path,
    component_id: str,
    model_name: str,
    parameters: dict[str, Any],
    physical_inputs: dict[str, np.ndarray],
) -> tuple[Any, Any, dict[InputName, np.ndarray]]:
    from afe.apis.defines import InputName, gen2_target
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.load.importers.general_importer import onnx_source
    from sima_utils.data.data_generator import DataGenerator

    contract = COMPONENTS[component_id]
    importer = onnx_source(str(graph), shape_dict=contract["inputs"], layout="NCHW")
    loaded = load_model(importer, target=gen2_target)
    afe_inputs = {InputName(name): _to_afe_layout(physical_inputs[name]) for name in contract["inputs"]}
    calibration = convert_data_generator_to_iterable(DataGenerator(afe_inputs))
    quantized = loaded.quantize(
        calibration_data=calibration,
        quantization_config=_quantization(parameters),
        model_name=model_name,
        log_level=logging.INFO,
    )
    return loaded, quantized, afe_inputs


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if expected.shape != actual.shape:
        raise RuntimeError(f"verification shape mismatch: {expected.shape} != {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise RuntimeError("verification produced a non-finite tensor")
    delta = actual - expected
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    norm_product = max(
        float(np.linalg.norm(expected) * np.linalg.norm(actual)),
        np.finfo(np.float64).tiny,
    )
    return {
        "shape": list(actual.shape),
        "cosine_similarity": float(np.vdot(expected.ravel(), actual.ravel()) / norm_product),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def _to_nchw(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return a quantized output in the graph's NCHW layout, whichever layout AFE returns."""
    value = np.asarray(value)
    if value.shape == shape:
        return value
    batch, channels, height, width = shape
    if value.shape == (batch, height, width, channels):
        return value.transpose(0, 3, 1, 2)
    raise RuntimeError(f"unexpected quantized output shape {value.shape} for {shape}")


def _verify_quantized(
    graph: Path,
    component_id: str,
    quantized: Any,
    physical_inputs: dict[str, np.ndarray],
    afe_inputs: dict[InputName, np.ndarray],
) -> list[dict[str, Any]]:
    """Compare the quantized graph with the float graph on the calibration sample.

    ONNX Runtime runs the float graph: the compiler's own float execution of this graph crashes with
    Model Compiler 8681769b526b.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    reference = session.run(None, {name: physical_inputs[name] for name in COMPONENTS[component_id]["inputs"]})
    candidate = list(quantized.execute(inputs=afe_inputs, use_jax=True))
    if len(reference) != len(candidate):
        raise RuntimeError(f"output count mismatch: {len(reference)} != {len(candidate)}")
    shapes = COMPONENTS[component_id]["outputs"].values()
    return [
        _metrics(expected, _to_nchw(actual, shape))
        for expected, actual, shape in zip(reference, candidate, shapes, strict=True)
    ]


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


def audit_mpk(archive: Path, component_id: str, batch_size: int = 1) -> dict[str, Any]:
    names, manifest = _read_mpk_manifest(archive)
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla = [item for item in plugins if str(item.get("processor", "")).upper() == "MLA"]
    host = [item for item in plugins if str(item.get("processor", "")).upper() in {"A65", "APU"}]
    elfs = [name for name in names if name.endswith(".elf")]
    shared = [name for name in names if name.endswith(".so") or ".so." in name]
    if len(mla) != 1 or len(elfs) != 1 or host or shared:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla)}, ELF={len(elfs)}, host={len(host)}, shared_objects={shared}"
        )
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(f"compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}")
    expected_inputs = {
        name: int(np.prod(shape, dtype=np.int64)) * 4 * batch_size
        for name, shape in COMPONENTS[component_id]["inputs"].items()
    }
    actual_inputs = {item.get("name"): item.get("size") for item in manifest.get("input_nodes", [])}
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"public input contract mismatch: {actual_inputs} != {expected_inputs}")
    terminal = plugins[-1] if plugins else {}
    expected_outputs = [
        int(np.prod(shape, dtype=np.int64)) * 4 * batch_size for shape in COMPONENTS[component_id]["outputs"].values()
    ]
    actual_outputs = [item.get("size") for item in terminal.get("output_nodes", [])]
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"public output contract mismatch: {actual_outputs} != {expected_outputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla), "output_sizes": actual_outputs}


def _compile(quantized: Any, build_dir: Path, component_id: str, batch_size: int) -> Path:
    quantized.compile(
        output_path=str(build_dir),
        batch_size=batch_size,
        compress=True,
        preserve=True,
        log_level=logging.INFO,
        layer_norm_use_fp32_intermediates=True,
    )
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one MPK archive, found {len(archives)}")
    audit_mpk(archives[0], component_id, batch_size)
    return archives[0]


def _component_phase(
    source: Path, temporary_dir: Path, profile_id: str, component_id: str, parameters: dict[str, Any]
) -> None:
    """Prepare, quantize, verify and compile one component into <temporary_dir>/<component_id>/."""
    work_dir = temporary_dir / component_id
    prepared = prepare_model(source, work_dir / "surgery", component_id, SURGERY_PARAMETERS[component_id])
    graph = prepared["model"]
    model_name = mpk_filename(load_metadata(MODEL_DIR), profile_id, component_id).removesuffix("_mpk.tar.gz")
    inputs = _calibration_inputs(component_id, prepared.get("runtime_data"))
    _, quantized, afe_inputs = _load_and_quantize(graph, component_id, model_name, parameters, inputs)
    verification = _verify_quantized(graph, component_id, quantized, inputs, afe_inputs)
    build_dir = work_dir / "compiled"
    build_dir.mkdir()
    batch_size = _batch_size(parameters)
    archive = _compile(quantized, build_dir, component_id, batch_size)
    report = {
        "derived_graph_sha256": sha256(graph),
        "mpk": archive.name,
        "batch_size": batch_size,
        "surgery": prepared["report"],
        "quantized_verification": verification,
        "mpk_audit": audit_mpk(archive, component_id, batch_size),
    }
    if prepared.get("runtime_data") is not None:
        shutil.copyfile(prepared["runtime_data"], work_dir / RUNTIME_DATA)
    (work_dir / COMPILE_REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _execute_phase(target: Callable[..., None], arguments: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    try:
        target(**arguments)
    except SystemExit as error:
        # AFE reports a failed quantization or compilation by raising SystemExit(), which exits with code 0.
        raise RuntimeError(f"{target.__name__} stopped in AFE; the AFE log above names the cause") from error


def _run_phase(name: str, target: Callable[..., None], **arguments: Any) -> None:
    """Run one phase in its own process, so each component's compiler memory is released before the next."""
    logging.info("Starting %s", name)
    process = multiprocessing.get_context("spawn").Process(name=name, target=_execute_phase, args=(target, arguments))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{name} failed with exit code {process.exitcode}")
    logging.info("Completed %s", name)


def _write_outputs(
    temporary_dir: Path, source: Path, profile_id: str, output_dir: Path
) -> dict[str, Path]:
    """Copy each MPK to <output_dir>/<component>/ and write compile-outputs.json and the compile report."""
    archives: dict[str, Path] = {}
    reports: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True)
    for component_id in COMPONENT_IDS:
        work_dir = temporary_dir / component_id
        reports[component_id] = json.loads((work_dir / COMPILE_REPORT).read_text(encoding="utf-8"))
        component_dir = output_dir / component_id
        component_dir.mkdir()
        archives[component_id] = component_dir / reports[component_id]["mpk"]
        shutil.copyfile(work_dir / "compiled" / reports[component_id]["mpk"], archives[component_id])
    # The engine reads these tables at run time; they come from the same surgery as the compiled graph.
    shutil.copyfile(temporary_dir / "vector_estimator" / RUNTIME_DATA, output_dir / RUNTIME_DATA)
    outputs = {
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "compile_invocation_id": profile_id,
        "outputs": [
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archives[component_id].name}",
                "mpk_sha256": sha256(archives[component_id]),
            }
            for component_id in COMPONENT_IDS
        ],
        "derived_graphs": [
            {"component_id": component_id, "sha256": reports[component_id]["derived_graph_sha256"]}
            for component_id in COMPONENT_IDS
        ],
    }
    (output_dir / "compile-outputs.json").write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n", "utf-8")
    report = {
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "compiler_version": COMPILER_VERSION,
        "source_sha256": {path.name: sha256(path) for path in (source, source.with_name(VOCODER_SOURCE))},
        "runtime_data_sha256": sha256(output_dir / RUNTIME_DATA),
        "components": reports,
    }
    (output_dir / COMPILE_REPORT).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return archives


def compile_model(source: Path, profile_id: str, output_dir: Path) -> dict[str, Path]:
    from afe.apis.release_v1 import get_model_sdk_version

    compiler_version = str(get_model_sdk_version())
    if compiler_version != COMPILER_VERSION:
        raise RuntimeError(f"Supertonic requires Model Compiler {COMPILER_VERSION}; found {compiler_version}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    parameters = _component_parameters(load_metadata(MODEL_DIR), profile_id)
    sources = {"vector_estimator": source, "vocoder": source.with_name(VOCODER_SOURCE)}
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        for component_id in COMPONENT_IDS:
            _run_phase(
                f"Supertonic {component_id} compilation",
                _component_phase,
                source=sources[component_id],
                temporary_dir=temporary_dir,
                profile_id=profile_id,
                component_id=component_id,
                parameters=parameters[component_id],
            )
        return _write_outputs(temporary_dir, source, profile_id, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the verified onnx/vector_estimator.onnx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    for path in (source, source.with_name(VOCODER_SOURCE)):
        if not path.is_file():
            parser.error(f"source does not exist: {path}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    archives = compile_model(source, args.build_profile, args.output_dir.expanduser().resolve())
    for component_id, archive in archives.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
