#!/usr/bin/env python3
"""Cut SSD-MobileNet V1 at its raw box and class heads and compile it into one Modalix MPK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    RequantizationMode,
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
from model_registry.dependencies import resolve_model_sdk_version
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, prepare_model

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "ssd_mobilenet_v1"
COMPONENT_ID = "model"
MODEL_SDK_VERSION = resolve_model_sdk_version(MODEL_DIR.parents[1])
# MPK manifests record each public tensor's size in bytes; the MPK keeps the ONNX float32 interface.
INPUT_BYTES = {INPUT_NAME: math.prod(INPUT_SHAPE) * 4}
OUTPUT_BYTES = [math.prod(shape) * 4 for _, shape in OUTPUTS]


def _parameters(profile_id: str) -> dict[str, Any]:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    return next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)


def _quantization(precision: str, calibration_bins: int | None) -> QuantizationParams:
    """Only INT8 activation ranges depend on representative calibration images."""
    if precision == "w8a8":
        config = QuantizationParams(
            node_names={""},  # Preserve the shipped default's literal node exclusions.
            calibration_method=HistogramMSEMethod(calibration_bins),
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            biascorr_type=BiasCorrectionType.REGULAR,
        )
    elif precision == "w16a16":
        config = QuantizationParams(
            node_names={""},  # Preserve the shipped default's literal node exclusions.
            calibration_method=SkipCalibration(),
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    else:
        raise ValueError(f"unsupported precision: {precision}")
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


def _audit_head_contract(plugins: list[dict[str, Any]], precision: str) -> None:
    """Check this recipe's named float heads and their transform lineage, not Core's carrier parser.

    The public PassThrough names are positional. Its inputs retain the surgery names; their producers
    declare logical NHWC shapes and convert MLA values (possibly channel-sliced) to public float32.
    """
    if precision not in {"w8a8", "w16a16"}:
        raise RuntimeError(f"unsupported SSD precision: {precision}")
    terminal = plugins[-1]
    inputs, outputs = terminal.get("input_nodes", []), terminal.get("output_nodes", [])
    if len(inputs) != len(OUTPUTS) or len(outputs) != len(OUTPUTS):
        raise RuntimeError("MPK must expose all twelve SSD heads")
    if len({node.get("name") for node in outputs}) != len(OUTPUTS):
        raise RuntimeError("MPK public output names must be unique")
    producers: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for plugin in plugins[:-1]:
        for node in plugin.get("output_nodes", []):
            name = node.get("name")
            if not isinstance(name, str) or name in producers:
                raise RuntimeError(f"ambiguous MPK producer: {name}")
            producers[name] = (plugin, node)

    def upstream(node: dict[str, Any], consumer: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        producer = producers.get(node.get("name"))
        if producer is None or producer[0]["sequence"] >= consumer["sequence"]:
            raise RuntimeError(f"missing or late MPK producer: {node.get('name')}")
        if node.get("size") != producer[1].get("size"):
            raise RuntimeError(f"MPK producer/consumer byte-size mismatch: {node.get('name')}")
        return producer

    origins = set()
    for (name, shape), input_node, output_node in zip(OUTPUTS, inputs, outputs, strict=True):
        expected_shape = [shape[0], shape[2], shape[3], shape[1]]
        expected_bytes = math.prod(shape) * 4
        if str(input_node.get("name", "")).rsplit("/", 1)[-1] != name:
            raise RuntimeError(f"MPK output identity/order mismatch: expected {name}")
        if input_node.get("size") != expected_bytes or output_node.get("size") != expected_bytes:
            raise RuntimeError(f"MPK public head byte-size mismatch: {name}")
        converter, _ = upstream(input_node, terminal)
        config = converter.get("config_params", {})
        params = config.get("params", {})
        converter_inputs = converter.get("input_nodes", [])
        scalar_bytes = 1 if precision == "w8a8" else 2
        if precision == "w8a8":
            valid_type = config.get("kernel") == "dequantization_transform" and params.get("input_data_type") == "int8"
        else:
            valid_type = (
                config.get("kernel") == "cast_transform"
                and params.get("in_dtype") == "bfloat16"
                and params.get("out_dtype") == "float32"
            )
        if (
            not valid_type
            or params.get("input_shapes") != [expected_shape]
            or params.get("output_shapes") != [expected_shape]
        ):
            raise RuntimeError(f"MPK head shape/type mismatch: {name}")
        if len(converter_inputs) != 1 or converter_inputs[0].get("size") != math.prod(shape) * scalar_bytes:
            raise RuntimeError(f"MPK head converter input mismatch: {name}")
        origin, origin_node = upstream(converter_inputs[0], converter)
        if origin.get("config_params", {}).get("kernel") == "slice_transform":
            slice_params = origin["config_params"].get("params", {})
            source_nodes = origin.get("input_nodes", [])
            if slice_params.get("output_shapes") != [expected_shape] or len(source_nodes) != 1:
                raise RuntimeError(f"MPK head slice mismatch: {name}")
            origin, origin_node = upstream(source_nodes[0], origin)
        if str(origin.get("processor", "")).upper() != "MLA" or origin_node["name"] in origins:
            raise RuntimeError(f"MPK head must have its own MLA origin: {name}")
        origins.add(origin_node["name"])


def audit_mpk(archive: Path, precision: str) -> None:
    """Require the whole network on one MLA segment, with no host code, and the declared interface."""
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
    if str(manifest.get("model_sdk_version")) != MODEL_SDK_VERSION:
        raise RuntimeError(f"MPK SDK mismatch: expected {MODEL_SDK_VERSION}, found {manifest.get('model_sdk_version')}")
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    if inputs != INPUT_BYTES:
        raise RuntimeError(f"MPK input interface mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel != "pass_through" or outputs != OUTPUT_BYTES:
        raise RuntimeError(f"MPK output interface mismatch: {outputs}")
    _audit_head_contract(plugins, precision)


def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)
    precision = str(parameters["precision"])
    calibration_bins = int(parameters["calibration_bins"]) if precision == "w8a8" else None
    quantization = _quantization(precision, calibration_bins)
    calibration_data = None
    if precision == "w8a8":
        if dataset is None:
            raise ValueError("the INT8 build requires --dataset from dataset-registry")
        samples = calibration_images(dataset, int(parameters["calibration_samples"]))
        logging.info("Calibrating %s with %d sample(s)", precision, len(samples))
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT_NAME): samples}))
    component_dir = output_dir / COMPONENT_ID
    component_dir.mkdir(parents=True, exist_ok=False)

    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build_dir = Path(temporary)
        graph = prepare_model(source, build_dir / "graphs" / f"{MODEL_ID}_heads.onnx")
        importer = onnx_source(
            str(graph), shape_dict={INPUT_NAME: tuple(INPUT_SHAPE)}, layout="NCHW", flexible_batch_size=True
        )
        loaded = load_model(importer, target=gen2_target)
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=f"{MODEL_ID}_{profile_id}",
            log_level=logging.INFO,
        )
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
        audit_mpk(archives[0], precision)
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
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile, dataset)
    logging.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
