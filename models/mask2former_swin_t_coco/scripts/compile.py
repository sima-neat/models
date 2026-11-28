#!/usr/bin/env python3
"""Split the verified Mask2Former ONNX into its components, quantize, and compile each to an MPK.

The converted ONNX holds the five components (pipeline.py) side by side. Each is extracted, quantized to its build
precision and compiled in its own process, and its MPK is audited: one MLA segment, no host code. A component that
declares calibration_samples calibrates on seeded cal2017 images, each component fed the float32 outputs of the
components before it; the others quantize from one synthetic sample.
compile-outputs.json records every MPK.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import multiprocessing
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    RequantizationMode,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from pipeline import COMPONENTS, INPUT_SHAPES, OUTPUT_SHAPES, preprocess, route, tensor_name
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "mask2former_swin_t_coco"
SAMPLE_SEED = 0
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261101
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")


def _component_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}
    if list(parameters) != list(COMPONENTS):
        raise ValueError(f"build profile {profile_id} must bind {', '.join(COMPONENTS)} in order")
    return parameters


def _quantization(parameters: dict[str, Any]) -> Any:
    precision = str(parameters["precision"])
    if precision != "w16a16":
        raise ValueError(f"unsupported precision: {precision}")
    config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(
        bfloat16_scheme()
    )
    # Some elementwise functions run as INT8 lookup tables even at BF16; a component that declares bins calibrates
    # their ranges on real images.
    if parameters.get("calibration_bins"):
        config = config.with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
    # with_activation_quantization turns per-token quantization on for BF16, which this build does not use.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(False)


def validate_source(source: Path) -> None:
    model = onnx.load(str(source), load_external_data=False)
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()}
    expected_outputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unsupported Mask2Former graph: inputs {inputs}, outputs {outputs}")


def split(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for component in COMPONENTS:
        graphs[component] = directory / f"{component}.onnx"
        onnx.utils.extract_model(
            str(source),
            str(graphs[component]),
            [tensor_name(component, name) for name in INPUT_SHAPES[component]],
            [tensor_name(component, name) for name in OUTPUT_SHAPES[component]],
        )
    return graphs


def channels_last(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1)) if value.ndim == 4 else value


def onnx_session(graph: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])


def calibration_images(dataset: Path, count: int) -> list[np.ndarray]:
    """A seeded sample of cal2017 images from the dataset-registry archive or directory, decoded to RGB."""
    from PIL import Image

    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    else:
        with tarfile.open(dataset, "r:*") as archive:
            names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    names = sorted(
        name
        for name in names
        if name.lower().endswith(IMAGE_SUFFIXES) and PurePosixPath(name).parent.name == CALIBRATION_SPLIT
    )
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    if dataset.is_dir():
        content = {name: (dataset / name).read_bytes() for name in selected}
    else:
        content, wanted = {}, set(selected)
        with tarfile.open(dataset, "r|*") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name in wanted and member.isfile() and (stream := archive.extractfile(member)) is not None:
                    content[name] = stream.read()
    images = []
    for name in selected:
        with Image.open(io.BytesIO(content[name])) as source:
            images.append(np.asarray(source.convert("RGB")))
    return images


def float_chain(graphs: dict[str, Path], images: list[np.ndarray]) -> dict[str, list[dict[str, np.ndarray]]]:
    """Each component's NCHW inputs for the images: the preprocessed image, then the float32 ONNX outputs of the
    components before it, routed by name."""
    feeds: dict[str, list[dict[str, np.ndarray]]] = {component: [] for component in COMPONENTS}
    sessions = {component: onnx_session(graphs[component]) for component in COMPONENTS[:-1]}
    for image in images:
        available = {"image": preprocess(image).transpose(2, 0, 1)[None]}
        for component in COMPONENTS:
            current = dict(zip(INPUT_SHAPES[component], route(component, available)))
            feeds[component].append(current)
            if component in sessions:
                names = list(OUTPUT_SHAPES[component])
                outputs = sessions[component].run(
                    [tensor_name(component, name) for name in names],
                    {tensor_name(component, key): value for key, value in current.items()},
                )
                available.update(zip(names, outputs))
    return feeds


def calibration_data(component: str, samples: list[dict[str, np.ndarray]]) -> Any:
    """The quantizer's channels-last calibration batches, one input set at a time."""
    stacked = {
        InputName(tensor_name(component, name)): np.concatenate([channels_last(sample[name]) for sample in samples])
        for name in INPUT_SHAPES[component]
    }
    return convert_data_generator_to_iterable(DataGenerator(stacked))


def _read_mpk_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        manifests = [member for member in members if member.name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {[member.name for member in manifests]}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0].name}")
        return json.load(source)


def audit_mpk(archive: Path, component: str) -> None:
    """Require the component on one MLA segment, with no host code."""
    manifest = _read_mpk_manifest(archive)
    processors = [str(plugin.get("processor", "")).upper() for plugin in manifest.get("plugins", [])]
    if processors.count("MLA") != 1 or {"A65", "APU"} & set(processors):
        raise RuntimeError(f"invalid {component} MPK: plugins run on {processors}")


def _build(
    graph: Path, build_dir: Path, name: str, component: str, parameters: dict[str, Any], samples: list[dict]
) -> Path:
    """Quantize and compile one component to its MPK."""
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    importer = onnx_source(
        str(graph),
        shape_dict={tensor_name(component, key): shape for key, shape in INPUT_SHAPES[component].items()},
        layout="NCHW",
    )
    loaded = load_model(importer, target=gen2_target)
    logging.info("Quantizing %s as %s with %d sample(s)", component, parameters["precision"], len(samples))
    try:
        quantized = loaded.quantize(
            calibration_data=calibration_data(component, samples),
            quantization_config=_quantization(parameters),
            model_name=name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        raise RuntimeError(f"quantizing {component} failed; the AFE log above names the cause") from error
    build_dir.mkdir(parents=True)
    try:
        quantized.compile(output_path=str(build_dir), batch_size=1, compress=True, log_level=logging.INFO)
    except SystemExit as error:
        raise RuntimeError(f"compiling {component} failed; the AFE log above names the cause") from error
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component} MPK, found {len(archives)}")
    audit_mpk(archives[0], component)
    return archives[0]


def _in_process(*arguments: Any) -> Path:
    # Each component in its own process: the compiler's memory goes back to the system between them.
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(_build, arguments)


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> dict[str, Path]:
    parameters = _component_parameters(profile_id)
    validate_source(source)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build = Path(temporary)
        graphs = split(source, build / "graphs")

        calibrated = [component for component in COMPONENTS if "calibration_samples" in parameters[component]]
        if calibrated and dataset is None:
            raise ValueError("a calibrated component requires --dataset from dataset-registry")
        samples: dict[str, list[dict[str, np.ndarray]]] = {}
        if calibrated:
            count = max(int(parameters[component]["calibration_samples"]) for component in calibrated)
            feeds = float_chain(graphs, calibration_images(dataset, count))
            for component in calibrated:
                samples[component] = feeds[component][: int(parameters[component]["calibration_samples"])]
        # An uncalibrated component derives nothing from representative data, but quantize() still needs one sample.
        rng = np.random.RandomState(SAMPLE_SEED)
        for component in COMPONENTS:
            synthetic = {
                name: rng.standard_normal(shape).astype(np.float32) for name, shape in INPUT_SHAPES[component].items()
            }
            samples.setdefault(component, [synthetic])

        output_dir.mkdir(parents=True)
        archives = {}
        for component in COMPONENTS:
            name = f"{MODEL_ID}_{profile_id}_{component}"
            built = _in_process(
                graphs[component], build / name, name, component, parameters[component], samples[component]
            )
            (output_dir / component).mkdir()
            archives[component] = output_dir / component / built.name
            shutil.copyfile(built, archives[component])
            logging.info("Compiled %s: %s", component, archives[component])
        manifest = {
            "model_id": MODEL_ID,
            "build_profile_id": profile_id,
            "compile_invocation_id": profile_id,
            "outputs": [
                {
                    "component_id": component,
                    "mpk_path": f"{component}/{archives[component].name}",
                    "mpk_sha256": sha256_file(archives[component])[0],
                }
                for component in COMPONENTS
            ],
            "derived_graphs": [
                {"component_id": component, "sha256": sha256_file(graphs[component])[0]} for component in COMPONENTS
            ],
        }
        (output_dir / "compile-outputs.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return archives


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
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx" or not source.is_file():
        parser.error(f"--model must be the verified ONNX from the conversion workflow: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
