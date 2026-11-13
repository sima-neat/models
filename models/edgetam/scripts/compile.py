#!/usr/bin/env python3
"""Build EdgeTAM's four per-frame graphs with surgery.py, quantize each, and compile each into a Modalix MPK.

compile.py receives the converted image_encoder.onnx as --model, with weights.pth beside it. Every component
calibrates on the inputs that tracking DAVIS cal objects with the FP32 graphs gives it (calibration.py), and runs the
operations of its bf16_regions, if it declares any, in BF16. A W16A16 image encoder runs its GELUs as INT8 lookup
tables. Each component quantizes and compiles in its own process,
so the compiler's memory returns to the system between them. Each quantized component's agreement with FP32 on a
tracked clip it did not calibrate on is printed. compile-outputs.json records every MPK and derived graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from afe.apis.defines import (
    HistogramMSEMethod,
    InputName,
    QuantizationParams,
    bfloat16_scheme,
    gen2_target,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.apis.model import L2CachingMode
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import CalibrationSet, calibration_sets
from model_registry.conversion import conversion_outputs
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    BF16_REGIONS,
    COMPONENTS,
    INT8_GELUS,
    INTERFACES,
    annotate_bf16_regions,
    annotate_int8_gelus,
    export_components,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "edgetam"
# Cal sequences tracked for calibration, the first frame of each prompted. The last one only checks the quantized
# components, on CHECK_FRAMES of its frames, and is not calibrated on.
CALIBRATION_CLIPS = 12
CHECK_FRAMES = 8
# The softmaxes the memory attention's attention region runs in BF16: two self and two cross attentions.
BF16_SOFTMAXES = 4
# The terminal EV74 plugin only converts the MLA result to the float32 interface.
OUTPUT_KERNELS = {"pass_through", "cast_transform", "dequantization_transform"}


def component_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    profiles = [profile for profile in load_metadata(MODEL_DIR)["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = {binding["id"]: binding["parameters"] for binding in profiles[0]["components"]}
    if list(parameters) != COMPONENTS:
        raise ValueError(f"build profile {profile_id} must bind {', '.join(COMPONENTS)} in order")
    return parameters


def bf16_regions(component: str, parameters: dict[str, Any]) -> list[str]:
    """The regions a component runs in BF16; only a W8A8 component has them."""
    regions = parameters.get("bf16_regions", [])
    if regions and (parameters["precision"] != "w8a8" or len(set(regions)) != len(regions)):
        raise ValueError(f"{component} bf16_regions must name distinct regions of a W8A8 component: {regions}")
    unknown = sorted(set(regions) - set(BF16_REGIONS[component]))
    if unknown:
        raise ValueError(f"unknown {component} BF16 regions {unknown}; choose from {list(BF16_REGIONS[component])}")
    return regions


def int8_gelus(component: str, parameters: dict[str, Any]) -> bool:
    """Whether the component's GELUs run as INT8 lookup tables: those surgery.py lists, in a W16A16 component."""
    return parameters["precision"] == "w16a16" and component in INT8_GELUS


def check_profile_id(profile_id: str, parameters: dict[str, dict[str, Any]]) -> None:
    """The profile ID is each component's precision token, with mixed after one that has BF16 regions, in component
    order and with a repeated token written once (naming-conventions.yaml)."""
    tokens = []
    for component in COMPONENTS:
        token = parameters[component]["precision"] + (
            "_mixed" if bf16_regions(component, parameters[component]) else ""
        )
        if not tokens or tokens[-1] != token:
            tokens.append(token)
    if profile_id != "_".join(tokens):
        raise ValueError(f"build profile {profile_id!r} must be named {'_'.join(tokens)!r} after its precisions")


def quantization(parameters: dict[str, Any]) -> QuantizationParams:
    """The compiler's calibration and its activation and weight formats for a precision.

    Every component calibrates: with BF16 activations its Sin and Cos still run as INT8 lookup tables.
    """
    precision = parameters["precision"]
    calibration_method = HistogramMSEMethod(int(parameters["calibration_bins"]))
    if precision == "w8a16":
        return QuantizationParams(
            calibration_method=calibration_method,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
            # Each token's activations get their own INT8 scale before an INT8-weight operation.
            per_token_quant=True,
        )
    if precision == "w8a8":
        return QuantizationParams(
            calibration_method=calibration_method,
            activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
            weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        )
    if precision == "w16a16":
        return QuantizationParams(
            calibration_method=calibration_method,
            activation_quantization_scheme=bfloat16_scheme(),
            weight_quantization_scheme=bfloat16_scheme(),
        )
    raise ValueError(f"unsupported precision: {precision}")


def channels_last(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(0, 2, 3, 1)) if value.ndim == 4 else value


def channels_first(value: np.ndarray, shape: list[int]) -> np.ndarray:
    value = np.asarray(value, np.float32)
    if len(shape) == 4:
        value = value.reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
    return value.reshape(shape)


def calibration_data(component: str, samples: list[dict[str, np.ndarray]]) -> Any:
    """The quantizer's channels-last calibration inputs, the samples stacked along the batch axis."""
    stacked = {
        InputName(name): np.concatenate([channels_last(sample[name]) for sample in samples])
        for name in INTERFACES[component]["inputs"]
    }
    return convert_data_generator_to_iterable(DataGenerator(stacked))


def require_bf16_regions(quantized: Any, component: str, regions: list[str]) -> None:
    """Fail unless every softmax of the regions runs in BF16; ignored annotations would silently build plain W8A8."""
    counts = Counter(
        type(node.ir.operation).__name__
        for node in quantized._net.iter_nodes_recursive()
        if getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.BFLOAT_16
    )
    print(f"{component} BF16 operations: {dict(sorted(counts.items()))}", flush=True)
    if counts["SoftmaxOp"] != BF16_SOFTMAXES:
        raise RuntimeError(
            f"{component} declares BF16 regions {regions}, but {counts['SoftmaxOp']} of {BF16_SOFTMAXES} softmaxes"
            " run in BF16"
        )


def require_int8_gelus(quantized: Any, component: str) -> None:
    """Fail unless every listed GELU runs in INT8; ignored annotations would silently build the BF16 erf."""
    gelus = sum(
        type(node.ir.operation).__name__ == "GeluOp"
        and getattr(node.ir, "quant_config", None) is not None
        and node.ir.quant_config.quantization_precision.value == QuantizationPrecision.INT_8
        for node in quantized._net.iter_nodes_recursive()
    )
    if gelus != INT8_GELUS[component][1]:
        raise RuntimeError(f"{gelus} of {INT8_GELUS[component][1]} {component} GELUs run in INT8")


def cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
    return float(expected @ actual / max(np.linalg.norm(expected) * np.linalg.norm(actual), 1e-12))


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


def audit_mpk(archive: Path, component: str) -> None:
    """Require the whole component on one MLA segment, with no host code, and its declared float32 interface."""
    names, manifest = _read_mpk_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    processors = [str(plugin.get("processor", "")).upper() for plugin in plugins]
    if len(elfs) != 1 or processors.count("MLA") != 1 or {"A65", "APU"} & set(processors):
        raise RuntimeError(f"invalid {component} MPK: {len(elfs)} ELF(s), plugins run on {processors}")
    interface = INTERFACES[component]
    inputs = {node.get("name"): node.get("size") for node in manifest.get("input_nodes", [])}
    expected = {name: int(np.prod(shape)) * 4 for name, shape in interface["inputs"].items()}
    if inputs != expected:
        raise RuntimeError(f"{component} MPK input interface {inputs}, expected {expected}")
    terminal = plugins[-1] if plugins else {}
    kernel = str(terminal.get("config_params", {}).get("kernel", ""))
    outputs = sorted(node.get("size") for node in terminal.get("output_nodes", []))
    if kernel not in OUTPUT_KERNELS or outputs != sorted(
        int(np.prod(shape)) * 4 for shape in interface["outputs"].values()
    ):
        raise RuntimeError(f"{component} MPK output interface mismatch: kernel {kernel!r}, sizes {outputs}")


@dataclass(frozen=True)
class ComponentBuild:
    """What the process that quantizes and compiles one component needs."""

    component: str
    graph: Path
    calibration: Path
    build_dir: Path
    model_name: str
    parameters: dict[str, Any]


def quantize_component(build: ComponentBuild, samples: list[dict[str, np.ndarray]]) -> Any:
    """Import the component's graph, with its BF16 regions or INT8 GELUs marked if it has any, and quantize it on
    the samples."""
    component, parameters = build.component, build.parameters
    regions = bf16_regions(component, parameters)
    gelus = int8_gelus(component, parameters)
    source = build.graph
    annotated = build.build_dir.with_name(f"{component}-annotated.onnx")
    if regions:
        source = annotated
        marks = annotate_bf16_regions(build.graph, source, component, regions)
        print(f"Marked {marks['bfloat16']} {component} operations BF16 and {marks['int8']} INT8 for {regions}")
    if gelus:
        source = annotated
        marks = annotate_int8_gelus(build.graph, source, component)
        print(f"Marked {marks['int8']} {component} GELUs INT8 and the {marks['bfloat16']} operations after them BF16")
    importer = onnx_source(
        str(source),
        shape_dict={key: tuple(shape) for key, shape in INTERFACES[component]["inputs"].items()},
        layout="NCHW",
        flexible_batch_size=False,
    )
    print(f"Quantizing {component} as {parameters['precision']} with {len(samples)} samples", flush=True)
    try:
        quantized = load_model(importer, target=gen2_target).quantize(
            calibration_data=calibration_data(component, samples),
            quantization_config=quantization(parameters),
            model_name=build.model_name,
            log_level=logging.INFO,
        )
    except SystemExit as error:
        raise RuntimeError(f"quantizing {component} failed; the AFE log above names the cause") from error
    if regions:
        require_bf16_regions(quantized, component, regions)
    if gelus:
        require_int8_gelus(quantized, component)
    return quantized


def check_against_fp32(quantized: Any, build: ComponentBuild, checks: list[dict[str, np.ndarray]]) -> None:
    """Print each quantized output's cosine with the FP32 graph's on the check clip; fail when one is not finite.

    The cosine is a diagnostic of one component on FP32 inputs. accuracy.py measures what the build profile's
    components do together.
    """
    import onnxruntime as ort

    outputs = INTERFACES[build.component]["outputs"]
    session = ort.InferenceSession(str(build.graph), providers=["CPUExecutionProvider"])
    expected, actual = {key: [] for key in outputs}, {key: [] for key in outputs}
    for feeds in checks:
        # JAX execution: the TensorFlow kernels of fast_mode have no CPU AvgPool3D, which the image encoder needs.
        result = quantized.execute(
            inputs={InputName(key): channels_last(value) for key, value in feeds.items()}, use_jax=True
        )
        for key, reference, value in zip(outputs, session.run(list(outputs), feeds), result):
            expected[key].append(reference)
            actual[key].append(channels_first(value, outputs[key]))
    similarity = {key: cosine(np.stack(expected[key]), np.stack(actual[key])) for key in outputs}
    print(f"{build.component} quantized/FP32 cosine on the check clip: {similarity}", flush=True)
    if not all(np.isfinite(value) for value in similarity.values()):
        raise RuntimeError(f"{build.component} quantized outputs are not finite on the check clip: {similarity}")


def compile_component(quantized: Any, build: ComponentBuild) -> Path:
    """Compile the quantized component into build_dir; return its audited MPK."""
    build.build_dir.mkdir(parents=True)
    try:
        quantized.compile(
            output_path=str(build.build_dir),
            batch_size=1,
            compress=True,
            log_level=logging.INFO,
            tessellate_parameters=None,
            l2_caching_mode=L2CachingMode.NONE,
            deployable=True,
        )
    except SystemExit as error:
        raise RuntimeError(f"compiling {build.component} failed; the AFE log above names the cause") from error
    archives = sorted(build.build_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one {build.component} MPK, found {len(archives)}")
    audit_mpk(archives[0], build.component)
    return archives[0]


def build_component(build: ComponentBuild) -> Path:
    """Quantize, check and compile one component; return its MPK."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    calibration = CalibrationSet.load(build.calibration)
    quantized = quantize_component(build, calibration.samples)
    check_against_fp32(quantized, build, calibration.checks)
    return compile_component(quantized, build)


def build_in_process(build: ComponentBuild) -> Path:
    """Build one component in its own process, so the compiler's memory returns to the system after it."""
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(build_component, (build,))


def write_compile_outputs(
    output_dir: Path, profile_id: str, archives: dict[str, Path], graphs: dict[str, Path]
) -> None:
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


def compile_model(
    image_encoder: Path, weights: Path, output_dir: Path, profile_id: str, dataset: Path
) -> dict[str, Path]:
    parameters = component_parameters(profile_id)
    check_profile_id(profile_id, parameters)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    sample_counts = {component: int(parameters[component]["calibration_samples"]) for component in COMPONENTS}
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary_dir = Path(temporary)
        graphs = export_components(image_encoder, weights, temporary_dir / "graphs")
        # Each set waits on disk, so this process holds no calibration data while a component builds.
        calibration_paths = {}
        for component, calibration in calibration_sets(
            graphs, dataset, CALIBRATION_CLIPS, sample_counts, CHECK_FRAMES
        ).items():
            calibration_paths[component] = temporary_dir / f"{component}-calibration.pkl"
            calibration.save(calibration_paths[component])
        archives = {}
        for component in COMPONENTS:
            built = build_in_process(
                ComponentBuild(
                    component=component,
                    graph=graphs[component],
                    calibration=calibration_paths[component],
                    build_dir=temporary_dir / "mpk" / component,
                    model_name=f"{MODEL_ID}_{profile_id}_{component}",
                    parameters=parameters[component],
                )
            )
            archives[component] = output_dir / component / built.name
            archives[component].parent.mkdir(parents=True)
            shutil.copyfile(built, archives[component])
        write_compile_outputs(output_dir, profile_id, archives, graphs)
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the converted image_encoder.onnx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="the davis-2017-cal Dataset Registry archive")
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    image_encoder = args.model.expanduser().resolve()
    _, weights_name = conversion_outputs(MODEL_DIR)
    weights = image_encoder.with_name(weights_name)
    if image_encoder.suffix != ".onnx" or not image_encoder.is_file() or not weights.is_file():
        parser.error(f"--model must be the converted image encoder ONNX, with {weights_name} beside it")
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    archives = compile_model(
        image_encoder, weights, args.output_dir.expanduser().resolve(), args.build_profile, dataset
    )
    for component, archive in archives.items():
        print(f"Compiled {component} MPK: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
