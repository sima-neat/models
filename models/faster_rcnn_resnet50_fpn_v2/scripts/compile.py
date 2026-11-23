#!/usr/bin/env python3
"""Split Faster R-CNN ResNet-50 FPN v2 into its backbone and box head and compile each into a Modalix MPK.

The phases run in separate processes, so each one's Model Compiler memory is freed before the next starts:
graph preparation, the backbone, then the box head. The box head takes as many proposals as its build profile's
proposals parameter sets. An INT8 box head calibrates on those regions, as the FP32 backbone and calibration.py's
proposals and RoIAlign give them for each calibration image.
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
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
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
from calibration import INPUT_SIZE, calibration_images, grid_cols, proposals, roi_align, tile_regions, unpack_backbone
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    BACKBONE_SHAPE,
    INPUT,
    ROIS,
    box_head_shapes,
    graph_paths,
    max_split_difference,
    prepare_model,
    rois_shape,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "faster_rcnn_resnet50_fpn_v2"
COMPONENT_IDS = ("backbone", "box_head")
# The split must reproduce the source's networks; float32 reordering in the box head stays far below this.
SPLIT_TOLERANCE = 1e-4
BACKBONE_INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
INPUT_NAMES = {"backbone": INPUT, "box_head": ROIS}
# The plugin that hands an MPK's outputs to the application: a pass-through gathers several, and a single output
# ends in its own dequantization (INT8) or cast (BF16).
TERMINAL_KERNELS = {"pass_through", "dequantization_transform", "cast_transform"}


def input_shape(component_id: str, regions: int) -> tuple[int, ...]:
    return BACKBONE_INPUT_SHAPE if component_id == "backbone" else rois_shape(regions)


def component_interface(component_id: str, regions: int) -> dict[str, Any]:
    """A component's MPK interface: manifests record each public tensor's size in bytes, and the MPKs keep the ONNX
    float32 interfaces."""
    if component_id == "backbone":
        return {"inputs": {INPUT: math.prod(BACKBONE_INPUT_SHAPE) * 4}, "outputs": [math.prod(BACKBONE_SHAPE) * 4]}
    return {
        "inputs": {ROIS: math.prod(rois_shape(regions)) * 4},
        "outputs": [math.prod(shape) * 4 for shape in box_head_shapes(regions).values()],
    }


def box_head_regions(parameters: dict[str, dict[str, Any]]) -> int:
    """The proposals a build profile's box head takes; grid_cols rejects a count the grid cannot hold."""
    regions = parameters["box_head"].get("proposals")
    if not isinstance(regions, int) or isinstance(regions, bool):
        raise ValueError("the box_head component must set an integer proposals parameter")
    grid_cols(regions)
    return regions


def _component_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}


def _quantization(parameters: dict[str, Any]) -> Any:
    precision = parameters["precision"]
    if precision == "w8a8":
        config = (
            default_quantization.with_activation_quantization(
                quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            )
            .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
            .with_calibration(HistogramMSEMethod(int(parameters["calibration_bins"])))
            .with_bias_correction(BiasCorrectionType.REGULAR)
        )
    elif precision == "w16a16":
        config = default_quantization.with_activation_quantization(bfloat16_scheme()).with_weight_quantization(
            bfloat16_scheme()
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
    # with_activation_quantization turns per-token quantization on for BF16; neither build uses it.
    return config.with_requantization_mode(RequantizationMode.sima).with_per_token_quant(False)


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


def audit_mpk(archive: Path, component_id: str, regions: int) -> None:
    """Require the component on one MLA segment, with no host code, and its declared interface."""
    contract = component_interface(component_id, regions)
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    host_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() in {"A65", "APU"}]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects or host_plugins:
        raise RuntimeError(
            f"invalid {component_id} MPK: MLA={len(mla_plugins)}, ELF={len(elfs)}, host={len(host_plugins)}, "
            f"shared_objects={shared_objects}"
        )
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != contract["inputs"]:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel not in TERMINAL_KERNELS or outputs != contract["outputs"]:
        raise RuntimeError(f"{component_id} MPK output interface mismatch: {outputs}")


def _compile_component(
    graph: Path,
    component_id: str,
    parameters: dict[str, Any],
    regions: int,
    samples: np.ndarray,
    name: str,
    build_dir: Path,
) -> None:
    """Quantize one component graph on NHWC samples, compile it and audit its MPK."""
    importer = onnx_source(
        str(graph),
        shape_dict={INPUT_NAMES[component_id]: input_shape(component_id, regions)},
        layout="NCHW",
        flexible_batch_size=True,
    )
    loaded = load_model(importer, target=gen2_target)
    logging.info("Quantizing the %s %s with %d sample(s)", parameters["precision"], component_id, len(samples))
    quantized = loaded.quantize(
        calibration_data=convert_data_generator_to_iterable(
            DataGenerator({InputName(INPUT_NAMES[component_id]): samples})
        ),
        quantization_config=_quantization(parameters),
        model_name=name,
        log_level=logging.INFO,
    )
    quantized.compile(
        output_path=str(build_dir),
        batch_size=1,
        compress=True,
        log_level=logging.INFO,
        tessellate_parameters=None,
        l2_caching_mode=L2CachingMode.NONE,
        deployable=True,
    )
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component_id} MPK, found {len(archives)}")
    audit_mpk(archives[0], component_id, regions)


def _calibrates(parameters: dict[str, Any]) -> bool:
    return parameters["precision"] == "w8a8"


def _bf16_sample(component_id: str, regions: int) -> np.ndarray:
    """BF16 derives nothing from representative data, but quantize() still needs one NHWC sample."""
    _, channels, height, width = input_shape(component_id, regions)
    return np.zeros((1, height, width, channels), np.float32)


def _prepare_phase(source: Path, graphs: Path, regions: int) -> None:
    backbone, box_head = prepare_model(source, graphs, regions)
    differences = max_split_difference(source, backbone, box_head, regions)
    logging.info("Split against the source: %s", differences)
    if max(differences.values()) > SPLIT_TOLERANCE:
        raise RuntimeError(f"the component graphs differ from the source: {differences}")


def _backbone_phase(graph: Path, parameters: dict[str, Any], dataset: Path | None, name: str, build_dir: Path) -> None:
    if _calibrates(parameters):
        if dataset is None:
            raise ValueError("an INT8 backbone requires --dataset from dataset-registry")
        count = int(parameters["calibration_samples"])
        samples = np.stack([image for image, _ in calibration_images(dataset, count)])
    else:
        samples = _bf16_sample("backbone", 0)
    _compile_component(graph, "backbone", parameters, 0, samples, name, build_dir)


def _box_head_samples(backbone: Path, dataset: Path, selected: int, count: int, regions: int) -> np.ndarray:
    """The first count of the backbone's selected calibration images' regions from the FP32 backbone, as many as
    the box head takes, each tiled as the box head's NHWC input."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(backbone), providers=["CPUExecutionProvider"])
    _, channels, height, width = rois_shape(regions)
    samples = np.empty((count, height, width, channels), np.float32)
    for index, (image, size) in enumerate(islice(calibration_images(dataset, selected), count)):
        (output,) = session.run(None, {INPUT: np.ascontiguousarray(image.transpose(2, 0, 1))[None]})
        features, rpn = unpack_backbone(output)
        boxes = proposals(rpn, size, regions)
        samples[index] = tile_regions(roi_align(features, boxes), regions)
    return samples


def _box_head_phase(
    graph: Path,
    backbone: Path,
    parameters: dict[str, Any],
    regions: int,
    selected: int,
    dataset: Path | None,
    name: str,
    build_dir: Path,
) -> None:
    if _calibrates(parameters):
        if dataset is None:
            raise ValueError("an INT8 box head requires --dataset from dataset-registry")
        count = int(parameters["calibration_samples"])
        if count > selected:
            raise ValueError("the box head calibrates on a subset of the backbone's calibration images")
        samples = _box_head_samples(backbone, dataset, selected, count, regions)
    else:
        samples = _bf16_sample("box_head", regions)
    _compile_component(graph, "box_head", parameters, regions, samples, name, build_dir)


def _execute_phase(target: Callable[..., None], arguments: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    try:
        target(**arguments)
    except SystemExit as error:
        # AFE reports a failed quantization or compilation by raising SystemExit(), which exits with code 0.
        raise RuntimeError(f"{target.__name__} stopped in AFE; the AFE log above names the cause") from error


def _run_phase(label: str, target: Callable[..., None], **arguments: Any) -> None:
    logging.info("Starting %s", label)
    process = multiprocessing.get_context("spawn").Process(name=label, target=_execute_phase, args=(target, arguments))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.exitcode}")
    logging.info("Completed %s", label)


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> dict[str, Path]:
    parameters = _component_parameters(profile_id)
    if set(parameters) != set(COMPONENT_IDS):
        raise ValueError(f"build profile {profile_id!r} must configure {COMPONENT_IDS}")
    regions = box_head_regions(parameters)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        backbone_graph, box_head_graph = graph_paths(temporary_dir / "graphs")
        builds = {component_id: temporary_dir / component_id for component_id in COMPONENT_IDS}
        _run_phase("graph preparation", _prepare_phase, source=source, graphs=temporary_dir / "graphs", regions=regions)
        _run_phase(
            "backbone compilation",
            _backbone_phase,
            graph=backbone_graph,
            parameters=parameters["backbone"],
            dataset=dataset,
            name=f"{MODEL_ID}_{profile_id}_backbone",
            build_dir=builds["backbone"],
        )
        _run_phase(
            "box head compilation",
            _box_head_phase,
            graph=box_head_graph,
            backbone=backbone_graph,
            parameters=parameters["box_head"],
            regions=regions,
            selected=int(parameters["backbone"].get("calibration_samples", 0)),
            dataset=dataset,
            name=f"{MODEL_ID}_{profile_id}_box_head",
            build_dir=builds["box_head"],
        )
        graph_hashes = {"backbone": sha256_file(backbone_graph)[0], "box_head": sha256_file(box_head_graph)[0]}
        archives = {}
        for component_id, build_dir in builds.items():
            (built,) = sorted(build_dir.glob("*_mpk.tar.gz"))
            (output_dir / component_id).mkdir(parents=True)
            archives[component_id] = output_dir / component_id / built.name
            shutil.copyfile(built, archives[component_id])
    _write_compile_outputs(output_dir, profile_id, archives, graph_hashes)
    return archives


def _write_compile_outputs(
    output_dir: Path, profile_id: str, archives: dict[str, Path], graph_hashes: dict[str, str]
) -> None:
    """Write compile-outputs.json, which records each component's MPK and derived graph."""
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    for component_id, archive in compile_model(
        source, args.output_dir.expanduser().resolve(), args.build_profile, dataset
    ).items():
        logging.info("Compiled %s MPK: %s", component_id, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
