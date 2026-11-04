#!/usr/bin/env python3
"""Compile Small/default precision profiles using the registry's existing flow.

Every decoder calibrates on its own profile's quantized backbone. The candidate
W16A16 profile uses the pinned compiler defaults. Mixed profiles consume a
frozen Neural Compressor precision map, never run a search during compilation.
The reference-logit approximation lives in surgery.py and is checked against the
original floating model before quantization. Full-model fidelity is diagnostic
because quantization can reorder selected queries; COCO evaluation is the task gate.
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnxruntime as ort
from afe.apis.defines import (
    BiasCorrectionType,
    HistogramMSEMethod,
    InputName,
    bfloat16_scheme,
    default_quantization,
    quantization_scheme,
)
from afe.apis.loaded_net import load_model
from afe.core.configs import QuantizationPrecision
from afe.core.utils import convert_data_generator_to_iterable
from afe.load.importers.general_importer import onnx_source
from calibration import decoder_inputs, load_images, select_images, to_afe_layout
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import sha256_file
from model_registry.model import load_model as load_metadata
from sima_utils.data.data_generator import DataGenerator
from surgery import (
    OUTPUTS,
    annotate_backbone,
    prepare_model,
    reference_precision_map,
    region_precision_map,
    validate_bf16_regions,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "dfine_s"
PROFILES = {
    "w16a16_w16a16": "w16a16",
    "w8a16_w16a16": "w8a16",
    "w8a8_w16a16": "w8a8",
    "w8a8_mixed_w16a16": "w8a8",
}
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
COMPONENTS = ("backbone", "decoder")
BACKBONE_OUTPUTS = ("attention_memory", "query_content", "proposal_scores", "reference_logits")
DECODER_INPUTS = ("attention_memory", "query_content", "reference_logits")
SHAPES = {
    "backbone": {"input": (1, 3, 640, 640)},
    "decoder": dict(zip(DECODER_INPUTS, ((1, 8400, 256), (1, 300, 256), (1, 300, 4)), strict=True)),
}
OUTPUT_SHAPES = {
    "backbone": ((1, 8400, 256), (1, 8400, 256), (1, 8400), (1, 8400, 4)),
    "decoder": tuple(OUTPUTS.values()),
}


def component_parameters(metadata, profile_id):
    if metadata["id"] != MODEL_ID:
        raise ValueError("Expected D-FINE Small metadata")
    profiles = [p for p in metadata["build_profiles"] if p["id"] == profile_id]
    if profile_id not in PROFILES or len(profiles) != 1:
        raise ValueError(f"Unknown or undeclared precision profile: {profile_id}")
    bindings = profiles[0]["components"]
    if [b["id"] for b in bindings] != list(COMPONENTS):
        raise ValueError("Expected ordered backbone and decoder components")
    parameters = {b["id"]: b["parameters"] for b in bindings}
    for name, value in parameters.items():
        mixed = name == "backbone" and profile_id == "w8a8_mixed_w16a16"
        expected = {"precision", "calibration_bins", "calibration_samples"}
        if mixed:
            expected.add("bf16_regions")
        if (
            set(value) != expected
            or value["precision"] != (PROFILES[profile_id] if name == "backbone" else "w16a16")
            or any(type(value[k]) is not int or value[k] < 1 for k in ("calibration_bins", "calibration_samples"))
        ):
            raise ValueError(f"Invalid {name} precision or calibration parameters for {profile_id}")
        if mixed:
            validate_bf16_regions(value["bf16_regions"])
    if parameters["backbone"]["calibration_samples"] != parameters["decoder"]["calibration_samples"]:
        raise ValueError("Both components must calibrate on the same images")
    return parameters


def quantization_config(parameters):
    precision = parameters["precision"]
    if precision not in ("w16a16", "w8a16", "w8a8"):
        raise ValueError(f"Unsupported precision: {precision}")
    int8_weights = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
    config = (
        default_quantization.with_activation_quantization(
            quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            if precision == "w8a8"
            else bfloat16_scheme()
        )
        .with_weight_quantization(bfloat16_scheme() if precision == "w16a16" else int8_weights)
        .with_calibration(HistogramMSEMethod(parameters["calibration_bins"]))
    )
    if precision == "w8a8":
        # Initial candidate inherited from Nano; Small qualification is pending.
        config = config.with_bias_correction(BiasCorrectionType.REGULAR).with_per_token_quant(False)
    return config


def quantize_component(graph, name, parameters, arrays):
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Expected finite float32 calibration tensors")
    loaded = load_model(onnx_source(str(graph), layout="NCHW"))
    data = convert_data_generator_to_iterable(
        DataGenerator({InputName(key): to_afe_layout(value) for key, value in arrays.items()})
    )
    quantized = loaded.quantize(
        calibration_data=data,
        quantization_config=quantization_config(parameters),
        model_name=name,
        log_level=logging.INFO,
    )
    if "bf16_regions" in parameters:
        require_mixed_precision(quantized)
    return quantized


def require_mixed_precision(quantized):
    """Check applied arithmetic, not only annotations, as RF-DETR does.

    Ignore placeholders and adapters: INT8 ingress does not prove an INT8
    convolution survived AFE's precision propagation and fusion.
    """
    counts = Counter(
        node.ir.quant_config.quantization_precision.value
        for node in quantized._net.iter_nodes_recursive()
        if type(getattr(node.ir, "operation", None)).__name__ == "ConvAddActivationOp"
        and getattr(node.ir, "quant_config", None) is not None
    )
    logging.info("Mixed backbone convolution precisions: %s", dict(counts))
    if not counts[QuantizationPrecision.BFLOAT_16] or not counts[QuantizationPrecision.INT_8]:
        raise RuntimeError(f"Expected BF16 and INT8 convolutions in the mixed backbone, found {dict(counts)}")


def audit_mpk(archive, component):
    """Check hardware placement and the public ABI, not incidental adapter topology."""
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"Unsafe MPK member: {member.name}")
        names = [m.name for m in members]
        manifests = [n for n in names if n.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise ValueError("Expected one MPK manifest")
        manifest = json.load(stream.extractfile(manifests[0]))
    plugins = sorted(manifest["plugins"], key=lambda n: n["sequence"])
    mla = [n for n in plugins if n.get("processor") == "MLA"]
    if (
        len(mla) != 1
        or len([n for n in names if n.endswith(".elf")]) != 1
        or any(n.endswith(".so") or ".so." in n for n in names)
        or any(n.get("processor") not in ("MLA", "EV74") for n in plugins)
    ):
        raise ValueError("Expected one MLA partition and EV74 adapters, without host model fallback")
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise ValueError("MPK Model Compiler version differs from the repository pin")
    inputs = {n["name"]: n["size"] for n in manifest["input_nodes"]}
    expected = {name: math.prod(shape) * 4 for name, shape in SHAPES[component].items()}
    terminal = plugins[-1]
    if (
        inputs != expected
        or terminal.get("config_params", {}).get("kernel") != "pass_through"
        or [n["size"] for n in terminal["output_nodes"]] != [math.prod(shape) * 4 for shape in OUTPUT_SHAPES[component]]
    ):
        raise ValueError(f"Unexpected {component} public input/output contract")
    if component == "backbone":
        producers = {value["name"]: node for node in plugins for value in node["output_nodes"]}
        for value in terminal["input_nodes"]:
            producer = producers[value["name"]]["config_params"]
            params = producer.get("params", {})
            if (
                producer.get("kernel") != "cast_transform"
                or params.get("in_dtype") != "bfloat16"
                or params.get("out_dtype") != "float32"
            ):
                raise ValueError("Expected uniform BF16 backbone outputs with FP32 publication")
    return manifest


def compile_component(model, root, component):
    output = root / component
    output.mkdir()
    model.compile(output_path=str(output), batch_size=1, log_level=logging.INFO)
    archives = list(output.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise ValueError(f"Expected one {component} model package")
    audit_mpk(archives[0], component)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def execute_backbone(model, image):
    tensors = tuple(
        np.asarray(v) for v in model.execute(inputs={InputName("input"): to_afe_layout(image[None])}, use_jax=True)
    )
    if len(tensors) != 4 or any(
        v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all()
        for v, shape in zip(tensors, OUTPUT_SHAPES["backbone"], strict=True)
    ):
        raise ValueError("Unexpected quantized backbone outputs")
    return tensors


def decoder_feed(outputs):
    return {
        name: np.ascontiguousarray(value) for name, value in zip(DECODER_INPUTS, decoder_inputs(outputs), strict=True)
    }


def prepare_phase(source, dataset, evaluation_names, work, parameters):
    backbone, decoder = prepare_model(source, work / "graphs")
    reference, before, after = session(source), session(backbone), session(decoder)
    values = {name: [] for name in OUTPUTS}
    for image in load_images(dataset, evaluation_names):
        expected = reference.run(None, {"input": image[None]})
        heads = before.run(None, {"input": image[None]})
        actual = after.run(None, decoder_feed(heads))
        for name, original, modified in zip(OUTPUTS, expected, actual, strict=True):
            np.testing.assert_allclose(
                modified,
                original,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"{name}: floating split/reference-logit approximation",
            )
            values[name].append(original[0])
    np.savez(work / "reference.npz", **{k: np.stack(v) for k, v in values.items()})
    if parameters["precision"] == "w8a8":
        # ONNX Runtime validates the unannotated floating split before AFE-only annotations.
        precision_map = (
            region_precision_map(backbone, parameters["bf16_regions"])
            if "bf16_regions" in parameters
            else reference_precision_map(backbone)
        )
        # Keep a uniform BF16 MLA output boundary without promoting the whole body.
        # Merge only promotions: reference masking and other mixed regions stay BF16.
        precision_map["nodes"].update(
            {
                name: precision
                for name, precision in region_precision_map(backbone, ["outputs"])["nodes"].items()
                if precision == "bfloat16"
            }
        )
        (work / "precision-map.json").write_text(json.dumps(precision_map, indent=2) + "\n")
        annotate_backbone(backbone, precision_map)


def backbone_phase(profile, parameters, dataset, calibration_names, evaluation_names, work):
    images = load_images(dataset, calibration_names)
    model = quantize_component(
        work / "graphs/backbone.onnx", f"{MODEL_ID}_{profile}_backbone", parameters, {"input": images}
    )
    calibration = {name: [] for name in DECODER_INPUTS}
    for image in images:
        for key, value in decoder_feed(execute_backbone(model, image)).items():
            # AFE may recycle output storage between executions.
            calibration[key].append(value[0].copy())
    np.savez(work / "decoder-calibration.npz", **{k: np.stack(v) for k, v in calibration.items()})
    evaluation = {name: [] for name in BACKBONE_OUTPUTS}
    for image in load_images(dataset, evaluation_names):
        for key, value in zip(BACKBONE_OUTPUTS, execute_backbone(model, image), strict=True):
            evaluation[key].append(value[0].copy())
    np.savez(work / "backbone-evaluation.npz", **{k: np.stack(v) for k, v in evaluation.items()})
    compile_component(model, work, "backbone")


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


def decoder_phase(profile, parameters, metadata, work):
    with np.load(work / "decoder-calibration.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    model = quantize_component(work / "graphs/decoder.onnx", f"{MODEL_ID}_{profile}_decoder", parameters, arrays)
    compile_component(model, work, "decoder")
    errors = [_OutputError(name) for name in OUTPUTS]
    with (
        np.load(work / "backbone-evaluation.npz", allow_pickle=False) as upstream,
        np.load(work / "reference.npz", allow_pickle=False) as reference,
    ):
        count = len(upstream[BACKBONE_OUTPUTS[0]])
        for i in range(count):
            outputs = [upstream[name][i][None] for name in BACKBONE_OUTPUTS]
            actual = model.execute(inputs={InputName(k): v for k, v in decoder_feed(outputs).items()}, use_jax=True)
            for error, value in zip(errors, actual, strict=True):
                error.add(reference[error.name][i][None], np.asarray(value))
    config = metadata["numerical_fidelity"]
    report = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "build_profile_id": profile,
        "status": config["status"],
        "reference": "Original full ONNX (ONNX Runtime)",
        "candidate": f"AFE profile {profile}: backbone -> CPU Top300/Gather -> bounded-reference decoder (JAX execute)",
        "dataset": {k: config["dataset"][k] for k in ("id", "variant", "sha256")},
        "selection": config["selection"],
        "samples": count,
        "metrics": [e.summary() for e in errors],
    }
    (work / "fidelity.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")


def execute_phase(target, arguments):
    # AFE can use SystemExit(None) on failure; it must not produce a successful worker.
    try:
        target(**arguments)
    except BaseException as error:
        raise RuntimeError(f"{target.__name__} failed; see its compiler log") from error


def run_phase(target, **arguments):
    """Release compiler/JAX memory between phases, as the RF-DETR recipe does."""
    process = multiprocessing.get_context("spawn").Process(
        name=target.__name__, target=execute_phase, args=(target, arguments)
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{target.__name__} failed with exit code {process.exitcode}; see its compiler log")


def compile_model(source, profile, dataset, fidelity_dataset, output_dir):
    metadata = load_metadata(MODEL_DIR)
    parameters = component_parameters(metadata, profile)
    selection = metadata["numerical_fidelity"]["selection"]
    count = parameters["backbone"]["calibration_samples"]
    if dataset != fidelity_dataset or selection["offset"] < count:
        raise ValueError("Use the same verified dataset with disjoint calibration/fidelity images")
    calibration_names = select_images(dataset, selection["seed"], 0, count)
    evaluation_names = select_images(dataset, selection["seed"], selection["offset"], selection["samples"])
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        work = Path(temporary)
        run_phase(
            prepare_phase,
            source=source,
            dataset=dataset,
            evaluation_names=evaluation_names,
            work=work,
            parameters=parameters["backbone"],
        )
        run_phase(
            backbone_phase,
            profile=profile,
            parameters=parameters["backbone"],
            dataset=dataset,
            calibration_names=calibration_names,
            evaluation_names=evaluation_names,
            work=work,
        )
        run_phase(decoder_phase, profile=profile, parameters=parameters["decoder"], metadata=metadata, work=work)
        output_dir.mkdir(parents=True)
        outputs = []
        for name in COMPONENTS:
            (archive,) = (work / name).glob("*_mpk.tar.gz")
            destination = output_dir / name
            destination.mkdir()
            target = destination / archive.name
            shutil.copyfile(archive, target)
            outputs.append(
                {
                    "component_id": name,
                    "mpk_path": str(target.relative_to(output_dir)),
                    "mpk_sha256": sha256_file(target)[0],
                }
            )
        shutil.copyfile(work / "fidelity.json", output_dir / "fidelity.json")
        document = {
            "model_id": MODEL_ID,
            "build_profile_id": profile,
            "compile_invocation_id": profile,
            "outputs": outputs,
            "derived_graphs": [
                {"component_id": name, "sha256": sha256_file(work / "graphs" / f"{name}.onnx")[0]}
                for name in COMPONENTS
            ],
        }
        (output_dir / "compile-outputs.json").write_text(json.dumps(document, indent=2) + "\n")
    return {entry["component_id"]: output_dir / entry["mpk_path"] for entry in outputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fidelity-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")
    for path in (args.model, args.dataset, args.fidelity_dataset):
        if not path.exists():
            parser.error(f"input does not exist: {path}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    compile_model(
        args.model.resolve(),
        args.build_profile,
        args.dataset.resolve(),
        args.fidelity_dataset.resolve(),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
