#!/usr/bin/env python3
"""Split the verified PointNet++ SSG ONNX into its eight components, check them, and compile each to an MPK.

The converted ONNX holds the eight stages side by side. Each component is extracted, quantized, gated
numerically against FP32 on a synthetic block run through the whole pipeline, compiled, and audited: one MLA
segment, no host code, the declared interface. compile-outputs.json records every MPK and derived graph.
"""

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

import numpy as np
import onnx
from afe.apis.defines import (
    BiasCorrectionType,
    InputName,
    RequantizationMode,
    SkipCalibration,
    bfloat16_scheme,
    default_quantization,
    gen2_target,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from pointcloud import (
    COMPONENTS,
    INPUT_SHAPES,
    OUTPUT_SHAPES,
    input_name,
    output_name,
    segment,
    synthetic_block,
)
from sima_utils.data.data_generator import DataGenerator

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "pointnet2_ssg_s3dis"
# The terminal EV74 plugin only converts the MLA result to the float32 interface: BF16 builds end in a cast.
OUTPUT_KERNELS = {"pass_through", "cast_transform"}
# Share of a synthetic block's points whose compiled-pipeline label must equal FP32's.
MIN_LABEL_AGREEMENT = 0.95


def _component_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}
    if list(parameters) != list(COMPONENTS):
        raise ValueError(f"build profile {profile_id} must bind {', '.join(COMPONENTS)} in order")
    return parameters


def _component_name(profile_id: str, component_id: str) -> str:
    return f"{MODEL_ID}_{profile_id}_{component_id}"


def _quantization(precision: str) -> Any:
    if precision != "w16a16":
        raise ValueError(f"unsupported precision: {precision}")
    # BF16 activations need no calibrated ranges, and bias correction would only measure the one sample.
    return (
        default_quantization.with_activation_quantization(bfloat16_scheme())
        .with_weight_quantization(bfloat16_scheme())
        .with_calibration(SkipCalibration())
        .with_bias_correction(BiasCorrectionType.NONE)
        .with_requantization_mode(RequantizationMode.sima)
        # with_activation_quantization turns per-token quantization on for BF16; this recipe keeps it off.
        .with_per_token_quant(False)
    )


def validate_source(source: Path) -> None:
    model = onnx.load(str(source), load_external_data=False)
    inputs = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in model.graph.input}
    outputs = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in model.graph.output}
    expected_inputs = {input_name(name): INPUT_SHAPES[name] for name in COMPONENTS}
    expected_outputs = {output_name(name): OUTPUT_SHAPES[name] for name in COMPONENTS}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unsupported PointNet++ graph: inputs {inputs}, outputs {outputs}")


def split(source: Path, directory: Path) -> dict[str, Path]:
    """Extract each component's graph from the side-by-side ONNX."""
    directory.mkdir(parents=True)
    graphs = {}
    for name in COMPONENTS:
        graphs[name] = directory / f"{name}.onnx"
        onnx.utils.extract_model(str(source), str(graphs[name]), [input_name(name)], [output_name(name)])
    return graphs


def _channels_last(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1))


def reference_inputs(graphs: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each component's FP32 input on the synthetic block, and the block's FP32 logits."""
    import onnxruntime as ort

    sessions = {
        name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for name, path in graphs.items()
    }
    inputs = {}

    def run(component, value):
        inputs[component] = value
        return sessions[component].run(None, {input_name(component): value})[0]

    return inputs, segment(synthetic_block(), run)


def label_agreement(quantized: dict[str, Any], reference: np.ndarray) -> float:
    """Share of the synthetic block's points whose label from the quantized components equals FP32's."""

    def run(component, value):
        (output,) = quantized[component].execute({input_name(component): _channels_last(value)}, fast_mode=False)
        return np.asarray(output, np.float32)

    logits = segment(synthetic_block(), run)
    if not np.isfinite(logits).all():
        raise RuntimeError("the quantized pipeline produced non-finite logits")
    return float((logits.argmax(1) == reference.argmax(1)).mean())


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


def audit_mpk(archive: Path, component_id: str) -> None:
    """Require the component on one MLA segment, with no host code, and its declared interface."""
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
    if inputs != {input_name(component_id): math.prod(INPUT_SHAPES[component_id]) * 4}:
        raise RuntimeError(f"{component_id} MPK input interface mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = [node.get("size") for node in terminal.get("output_nodes", [])]
    if kernel not in OUTPUT_KERNELS or outputs != [math.prod(OUTPUT_SHAPES[component_id]) * 4]:
        raise RuntimeError(f"{component_id} MPK output interface mismatch: kernel {kernel!r}, sizes {outputs}")


def _quantize(graph: Path, name: str, component: str, precision: str, sample: np.ndarray) -> Any:
    importer = onnx_source(
        str(graph),
        shape_dict={input_name(component): INPUT_SHAPES[component]},
        layout="NCHW",
        flexible_batch_size=True,
    )
    loaded = load_model(importer, target=gen2_target)
    try:
        return loaded.quantize(
            # BF16 derives nothing from data, but quantize() still needs one sample.
            calibration_data=convert_data_generator_to_iterable(
                DataGenerator({InputName(input_name(component)): _channels_last(sample)})
            ),
            quantization_config=_quantization(precision),
            model_name=name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        # AFE reports a failed quantization by raising SystemExit().
        raise RuntimeError(f"quantizing {component} failed; the AFE log above names the cause") from error


def _compile(quantized: Any, build_dir: Path, component: str) -> Path:
    build_dir.mkdir(parents=True)
    try:
        quantized.compile(
            output_path=str(build_dir),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
        )
    except SystemExit as error:
        raise RuntimeError(f"compiling {component} failed; the AFE log above names the cause") from error
    archives = sorted(build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {component} MPK, found {len(archives)}")
    audit_mpk(archives[0], component)
    return archives[0]


def compile_model(source: Path, output_dir: Path, profile_id: str) -> dict[str, Path]:
    parameters = _component_parameters(profile_id)
    validate_source(source)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        build = Path(temporary)
        graphs = split(source, build / "graphs")
        samples, reference = reference_inputs(graphs)
        quantized = {}
        for component in COMPONENTS:
            name = _component_name(profile_id, component)
            logging.info("Quantizing %s as %s", component, parameters[component]["precision"])
            quantized[component] = _quantize(
                graphs[component], name, component, str(parameters[component]["precision"]), samples[component]
            )
        agreement = label_agreement(quantized, reference)
        logging.info("Quantized pipeline labels agree with FP32 for %.4f of the synthetic block", agreement)
        if agreement < MIN_LABEL_AGREEMENT:
            raise RuntimeError(f"quantized labels agree with FP32 for only {agreement:.4f} of the synthetic block")
        output_dir.mkdir(parents=True)
        archives = {}
        for component in COMPONENTS:
            name = _component_name(profile_id, component)
            built = _compile(quantized[component], build / name, component)
            (output_dir / component).mkdir()
            archives[component] = output_dir / component / built.name
            shutil.copyfile(built, archives[component])
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
        (output_dir / "compile-outputs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, help="unused: no build of this model takes a calibration dataset")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    source = args.model.expanduser().resolve()
    if source.suffix != ".onnx":
        parser.error("--model must be the verified ONNX from the conversion workflow")
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    # The Model Compiler configures the root logger on import; force this format and level over it.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    archives = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    for component, archive in archives.items():
        logging.info("Compiled %s: %s", component, archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
