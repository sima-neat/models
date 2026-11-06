#!/usr/bin/env python3
"""Split the verified CLIP ViT-B/16 ONNX into its image and text encoders, compile each, and measure fidelity.

The text encoder calibrates and is compared on the tokenized prompts the conversion embeds as constant outputs:
the first TEXT_CALIBRATION prompts calibrate it, and fidelity compares each held-out prompt's end-of-text row,
the row an application reads.
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
import onnx
import onnxruntime as ort
from afe.apis.defines import (
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
from calibration import load_images, select_names
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "clip_vit_b_16"
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
COMPONENT_IDS = ("image_encoder", "text_encoder")
# Each component's graph inputs and outputs in the converted ONNX, with their shapes.
COMPONENT_GRAPHS = {
    "image_encoder": {"inputs": {"pixel_values": (1, 3, 224, 224)}, "outputs": {"image_embeds": (1, 512), "logits": (1, 1000)}},
    "text_encoder": {"inputs": {"token_embeds": (1, 77, 512)}, "outputs": {"text_embeds": (1, 77, 512)}},
}
PROMPTS, PROMPT_ENDS = "calibration_token_embeds", "calibration_ends"
TEXT_CALIBRATION = 32
# Files the compilation phases pass to each other in the temporary directory.
PROMPTS_FILE = "prompts.npz"
REFERENCE_OUTPUTS = "reference_outputs.npz"
FIDELITY_REPORT = "fidelity.json"


def _float32_bytes(shape: tuple[int, ...]) -> int:
    return math.prod(shape) * 4


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
    if precision == "w8a8":
        activation = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w8a16":
        activation = bfloat16_scheme()
        weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    elif precision == "w16a16":
        activation = weights = bfloat16_scheme()
    else:
        raise ValueError(f"unsupported precision: {precision}")
    return (
        default_quantization.with_activation_quantization(activation)
        .with_weight_quantization(weights)
        .with_requantization_mode(RequantizationMode.sima)
        # AFE refuses to quantize these encoders without calibrated ranges, even at BF16.
        .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
        .with_channel_equalization(False)
        .with_smooth_quant(False)
        # with_activation_quantization turns per-token quantization on for BF16. w8a16 keeps it, scaling each token's
        # activations on its own before an INT8-weight operation; w16a16 and w8a8 do not use it.
        .with_per_token_quant(precision == "w8a16")
    )


def _afe_layout(array: np.ndarray) -> np.ndarray:
    """AFE takes images in NHWC order and other tensors as they are."""
    return np.ascontiguousarray(array.transpose(0, 2, 3, 1) if array.ndim == 4 else array)


def _quantize_component(graph: Path, name: str, component_id: str, parameters: dict[str, Any],
                        calibration: np.ndarray) -> Any:
    shapes = COMPONENT_GRAPHS[component_id]["inputs"]
    importer = onnx_source(str(graph), shape_dict=shapes, layout="NCHW", flexible_batch_size=True)
    loaded = load_model(importer, target=gen2_target)
    (input_name,) = shapes
    data = convert_data_generator_to_iterable(DataGenerator({InputName(input_name): _afe_layout(calibration)}))
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
    """Require one MLA plugin, no CPU plugins, the pinned compiler and the component's input."""
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = manifest.get("plugins", [])
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, "
            f"host={len(host_plugins)}, shared_objects={shared_objects}"
        )
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"{component_id} MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}"
        )
    public_inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    expected = {name: _float32_bytes(shape) for name, shape in COMPONENT_GRAPHS[component_id]["inputs"].items()}
    if public_inputs != expected:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {public_inputs}")
    return {"elf": elfs[0], "mla_plugins": len(mla_plugins)}


def _find_mpk(build_dir: Path, component_id: str) -> Path:
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    return archives[0]


def _compile_component(quantized: Any, build_dir: Path, component_id: str, precision: str) -> Path:
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
        # The option applies to every layer norm, and ones with INT8 activations (w8a8) reject it.
        layer_norm_use_fp32_intermediates=precision != "w8a8",
        rms_norm_use_fp32_intermediates=False,
        enable_filter_sharing=False,
        layout_search_effort_level=0,
    )
    archive = _find_mpk(build_dir, component_id)
    audit_mpk(archive, component_id)
    return archive


@dataclass
class _OutputError:
    """Running sums of one output's error against the original ONNX, over every fidelity sample."""

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


def _graph_paths(graphs_dir: Path) -> dict[str, Path]:
    return {component_id: graphs_dir / f"{component_id}.onnx" for component_id in COMPONENT_IDS}


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _full_inputs(**inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Inputs for the full ONNX: the given ones, and zeros for the other encoder's input."""
    shapes = {name: shape for graph in COMPONENT_GRAPHS.values() for name, shape in graph["inputs"].items()}
    return {name: inputs.get(name, np.zeros(shape, dtype=np.float32)) for name, shape in shapes.items()}


def _prepare_phase(source: Path, fidelity_dataset: Path, fidelity_names: list[str], temporary_dir: Path) -> None:
    """Split the ONNX into the two encoders, check the split, and save the prompts and the reference outputs."""
    full = onnx.load(source)
    extractor = onnx.utils.Extractor(full)
    paths = _graph_paths(temporary_dir / "graphs")
    paths["image_encoder"].parent.mkdir(parents=True)
    for component_id, path in paths.items():
        graph = COMPONENT_GRAPHS[component_id]
        onnx.save(extractor.extract_model(list(graph["inputs"]), list(graph["outputs"])), path)
    del extractor, full

    reference = _session(source)
    prompts, ends = reference.run([PROMPTS, PROMPT_ENDS], _full_inputs())
    if prompts.shape[0] <= TEXT_CALIBRATION or prompts.shape[1:] != COMPONENT_GRAPHS["text_encoder"]["inputs"]["token_embeds"][1:]:
        raise RuntimeError(f"unexpected embedded prompts: {prompts.shape}")
    np.savez(temporary_dir / PROMPTS_FILE, prompts=prompts, ends=ends)

    images = load_images(fidelity_dataset, fidelity_names).transpose(0, 3, 1, 2)
    outputs: dict[str, list[np.ndarray]] = {"image_embeds": [], "logits": [], "text_embeds": []}
    for image in images:
        image_embeds, logits = reference.run(["image_embeds", "logits"], _full_inputs(pixel_values=image[None]))
        outputs["image_embeds"].append(image_embeds)
        outputs["logits"].append(logits)
    for prompt, end in zip(prompts[TEXT_CALIBRATION:], ends[TEXT_CALIBRATION:], strict=True):
        (text_embeds,) = reference.run(["text_embeds"], _full_inputs(token_embeds=prompt[None]))
        outputs["text_embeds"].append(text_embeds[:, end])
    np.savez(temporary_dir / REFERENCE_OUTPUTS, **{name: np.concatenate(values) for name, values in outputs.items()})

    # Each encoder alone must reproduce the full ONNX.
    image_encoder, text_encoder = _session(paths["image_encoder"]), _session(paths["text_encoder"])
    expected_image = reference.run(["image_embeds", "logits"], _full_inputs(pixel_values=images[:1]))
    expected_text = reference.run(["text_embeds"], _full_inputs(token_embeds=prompts[:1]))
    for expected, actual in (
        (expected_image, image_encoder.run(["image_embeds", "logits"], {"pixel_values": images[:1]})),
        (expected_text, text_encoder.run(["text_embeds"], {"token_embeds": prompts[:1]})),
    ):
        for want, got in zip(expected, actual, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def _execute(model: Any, input_name: str, value: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """Run a quantized encoder on one sample; its outputs, keyed by their distinct shapes."""
    outputs = model.execute(inputs={InputName(input_name): _afe_layout(value)}, use_jax=True)
    return {np.asarray(output).shape: np.asarray(output) for output in outputs}


def _output(outputs: dict[tuple[int, ...], np.ndarray], component_id: str, name: str) -> np.ndarray:
    shape = COMPONENT_GRAPHS[component_id]["outputs"][name]
    if shape not in outputs:
        raise RuntimeError(f"the quantized {component_id} returned no {name} of shape {shape}: {list(outputs)}")
    return outputs[shape]


def _image_phase(graph: Path, temporary_dir: Path, profile_id: str, parameters: dict[str, Any],
                 calibration_dataset: Path, calibration_names: list[str], fidelity_dataset: Path,
                 fidelity_names: list[str]) -> None:
    """Compile the image encoder and compare its outputs on the fidelity images."""
    calibration = load_images(calibration_dataset, calibration_names).transpose(0, 3, 1, 2)
    name = _component_name(profile_id, "image_encoder")
    encoder = _quantize_component(graph, name, "image_encoder", parameters, calibration)
    _compile_component(encoder, temporary_dir / name, "image_encoder", parameters["precision"])
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = {key: saved[key] for key in ("image_embeds", "logits")}
    errors = [_OutputError(key) for key in reference]
    for index, image in enumerate(load_images(fidelity_dataset, fidelity_names).transpose(0, 3, 1, 2)):
        outputs = _execute(encoder, "pixel_values", image[None])
        for error in errors:
            error.add(reference[error.name][index : index + 1], _output(outputs, "image_encoder", error.name))
    (temporary_dir / "image_encoder_fidelity.json").write_text(json.dumps([error.summary() for error in errors]))


def _text_phase(graph: Path, temporary_dir: Path, profile_id: str, parameters: dict[str, Any]) -> None:
    """Compile the text encoder and compare the held-out prompts' end-of-text rows."""
    with np.load(temporary_dir / PROMPTS_FILE) as saved:
        prompts, ends = saved["prompts"], saved["ends"]
    name = _component_name(profile_id, "text_encoder")
    encoder = _quantize_component(graph, name, "text_encoder", parameters, prompts[:TEXT_CALIBRATION])
    _compile_component(encoder, temporary_dir / name, "text_encoder", parameters["precision"])
    with np.load(temporary_dir / REFERENCE_OUTPUTS) as saved:
        reference = saved["text_embeds"]
    error = _OutputError("text_embeds")
    for index, (prompt, end) in enumerate(zip(prompts[TEXT_CALIBRATION:], ends[TEXT_CALIBRATION:], strict=True)):
        text_embeds = _output(_execute(encoder, "token_embeds", prompt[None]), "text_encoder", "text_embeds")
        error.add(reference[index : index + 1], text_embeds[:, end])
    (temporary_dir / "text_encoder_fidelity.json").write_text(json.dumps([error.summary()]))


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


def _precision_label(component_parameters: dict[str, dict[str, Any]]) -> str:
    precisions = {parameters["precision"].upper() for parameters in component_parameters.values()}
    if len(precisions) != 1:
        raise ValueError("both encoders of a build profile use one precision")
    return precisions.pop()


def _write_fidelity(temporary_dir: Path, profile_id: str, config: dict[str, Any], precision: str) -> None:
    metrics = [
        *json.loads((temporary_dir / "image_encoder_fidelity.json").read_text()),
        *json.loads((temporary_dir / "text_encoder_fidelity.json").read_text()),
    ]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile_id,
        "status": config["status"],
        "reference": "Original full ONNX (ONNX Runtime)",
        "candidate": f"AFE {precision} image encoder and text encoder (JAX execute)",
        "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": config["selection"],
        "samples": config["selection"]["samples"],
        "metrics": metrics,
    }
    (temporary_dir / FIDELITY_REPORT).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def _write_compile_outputs(output_dir: Path, profile_id: str, archives: dict[str, Path],
                           graph_hashes: dict[str, str]) -> None:
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
    (output_dir / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")


def compile_model(source: Path, profile_id: str, calibration_dataset: Path, fidelity_dataset: Path,
                  output_dir: Path) -> dict[str, Path]:
    metadata = load_metadata(MODEL_DIR)
    component_parameters = _component_parameters(metadata, profile_id)
    precision = _precision_label(component_parameters)
    fidelity_config = metadata["numerical_fidelity"]
    selection = fidelity_config["selection"]
    seed, offset, samples = (int(selection[key]) for key in ("seed", "offset", "samples"))
    image_parameters = component_parameters["image_encoder"]
    count = int(image_parameters["calibration_samples"])
    if offset < count:
        raise ValueError("the fidelity selection must begin after the calibration images")
    calibration_names = select_names(calibration_dataset, seed, 0, count)
    fidelity_names = select_names(fidelity_dataset, seed, offset, samples)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        graphs = _graph_paths(temporary_dir / "graphs")
        _run_phase(
            "CLIP graph preparation",
            _prepare_phase,
            source=source,
            fidelity_dataset=fidelity_dataset,
            fidelity_names=fidelity_names,
            temporary_dir=temporary_dir,
        )
        graph_hashes = {component_id: sha256_file(path)[0] for component_id, path in graphs.items()}
        _run_phase(
            "CLIP image encoder compilation and fidelity",
            _image_phase,
            graph=graphs["image_encoder"],
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            parameters=image_parameters,
            calibration_dataset=calibration_dataset,
            calibration_names=calibration_names,
            fidelity_dataset=fidelity_dataset,
            fidelity_names=fidelity_names,
        )
        _run_phase(
            "CLIP text encoder compilation and fidelity",
            _text_phase,
            graph=graphs["text_encoder"],
            temporary_dir=temporary_dir,
            profile_id=profile_id,
            parameters=component_parameters["text_encoder"],
        )
        _write_fidelity(temporary_dir, profile_id, fidelity_config, precision)
        output_dir.mkdir(parents=True)
        archives = _copy_component_archives(temporary_dir, profile_id, output_dir)
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
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the manual conversion workflow")
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs = compile_model(source, args.build_profile, dataset, fidelity_dataset, args.output_dir.expanduser().resolve())
    for component_id, archive in outputs.items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
