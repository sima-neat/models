#!/usr/bin/env python3
"""Quantize and compile one pinned SSD-MobileNet V1 Modalix variant."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from calibration import (
    ARCHIVE_SHA256,
    CALIBRATION_ARRAY_SHA256,
    CALIBRATION_IMAGE_IDS,
    VALIDATION_ARRAY_SHA256,
    sha256_file,
    validate_dataset,
)
from model_registry.dependencies import resolve_model_compiler
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, prepare_model

LOG = logging.getLogger("ssd_mobilenet_v1.compile")
MODEL_DIR = Path(__file__).resolve().parent
COMPILER_VERSION = resolve_model_compiler(MODEL_DIR.parents[1])["version"]
PUBLIC_INPUT_SIZE = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
PUBLIC_OUTPUT_SIZES = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _, shape in OUTPUTS]


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one SSD-MobileNet V1 variant named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == "model")
    variant = {**profile, "parameters": binding["parameters"]}
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("precision") not in {"int8", "bf16"}:
        raise ValueError("SSD-MobileNet V1 supports only declared Modalix INT8 and BF16 variants")
    expected_quantization = (
        ("asymmetric_per_tensor_8bit", "symmetric_per_channel_8bit")
        if parameters["precision"] == "int8"
        else ("bfloat16", "bfloat16")
    )
    declared_quantization = (parameters.get("activation_quantization"), parameters.get("weight_quantization"))
    if declared_quantization != expected_quantization:
        raise ValueError(f"SSD quantization declaration mismatch: {declared_quantization}")
    if parameters.get("calibration_method") != "mse" or parameters.get("requantization") != "sima":
        raise ValueError("SSD variants require MSE calibration and SiMa requantization")
    if parameters.get("calibration_samples") != len(CALIBRATION_IMAGE_IDS):
        raise ValueError(f"SSD variants require exactly {len(CALIBRATION_IMAGE_IDS)} calibration samples")
    return variant


def calibration_source() -> SourceSpec:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    calibration = next(item for item in metadata["calibrations"] if item["id"] == "default")
    if calibration.get("status") != "ready":
        raise ValueError("SSD-MobileNet V1 requires pinned representative calibration data")
    return SourceSpec.from_mapping(calibration["source"])


def resolve_calibration_path() -> Path:
    override = os.environ.get("CALIBRATION_PATH", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"calibration snapshot does not exist: {path}")
        return path
    path, _cache_hit = Fetcher().fetch(calibration_source())
    return path


def _safe_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
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
        manifest = json.load(source)
    return names, manifest


def _kernel(plugin: dict[str, Any]) -> str:
    return str(plugin.get("config_params", {}).get("kernel", ""))


def _params(plugin: dict[str, Any]) -> dict[str, Any]:
    value = plugin.get("config_params", {}).get("params", {})
    return value if isinstance(value, dict) else {}


def audit_quantized_model(path: Path, precision: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"quantized SSD serialization has no layer topology: {path}") from exc
    convolutions = [layer for layer in layers if str(layer.get("class_name", "")).startswith("MLA_0/ConvAddActivation")]
    if len(convolutions) != 47:
        raise RuntimeError(f"expected 47 quantized SSD MLA convolutions, found {len(convolutions)}")
    expected_scalar_type = 6 if precision == "int8" else 13
    scalar_types = {layer.get("config", {}).get("conv_attrs.input_type", {}).get("value") for layer in convolutions}
    statuses = {layer.get("config", {}).get("_status") for layer in convolutions}
    if scalar_types != {expected_scalar_type} or statuses != {"SIMA_QUANTIZED"}:
        raise RuntimeError(
            f"{precision.upper()} SSD compute precision audit failed: types={scalar_types}, statuses={statuses}"
        )
    if precision == "int8":
        per_channel = {layer.get("config", {}).get("per_channel") for layer in convolutions}
        has_weight_quantization = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
        if per_channel != {True} or has_weight_quantization != {True}:
            raise RuntimeError("INT8 SSD weights are not represented as per-channel quantized convolution weights")
    else:
        has_bfloat_weights = {"weights_attrs.data" in layer.get("config", {}) for layer in convolutions}
        has_integer_weights = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
        if has_bfloat_weights != {True} or has_integer_weights != {False}:
            raise RuntimeError("BF16 SSD convolution weights do not match the serialized BF16 topology")
    return {
        "compute_convolution_count": len(convolutions),
        "compute_scalar_type": expected_scalar_type,
        "internal_precision": precision,
        "quantized_model_json_sha256": sha256_file(path),
        "weight_quantization": "per_channel_int8" if precision == "int8" else "bfloat16",
    }


def quantization_quality_report(
    reference_samples: list[list[np.ndarray]], candidate_samples: list[list[np.ndarray]]
) -> dict[str, Any]:
    if len(reference_samples) != len(candidate_samples) or not reference_samples:
        raise RuntimeError(
            "quantization validation requires the same non-zero number of floating and quantized samples"
        )
    accumulators = [
        {"dot": 0.0, "reference_energy": 0.0, "candidate_energy": 0.0, "error_energy": 0.0, "mae": 0.0}
        for _ in OUTPUTS
    ]
    element_counts = [0] * len(OUTPUTS)
    maximum_errors = [0.0] * len(OUTPUTS)
    for sample_index, (reference_outputs, candidate_outputs) in enumerate(
        zip(reference_samples, candidate_samples, strict=True)
    ):
        if len(reference_outputs) != len(OUTPUTS) or len(candidate_outputs) != len(OUTPUTS):
            raise RuntimeError(
                f"validation sample {sample_index} returned {len(reference_outputs)} floating and "
                f"{len(candidate_outputs)} quantized outputs; expected {len(OUTPUTS)}"
            )
        for output_index, ((name, expected_shape), reference, candidate) in enumerate(
            zip(OUTPUTS, reference_outputs, candidate_outputs, strict=True)
        ):
            reference_array = np.asarray(reference, dtype=np.float64)
            candidate_array = np.asarray(candidate, dtype=np.float64)
            if reference_array.shape != candidate_array.shape or reference_array.size != int(np.prod(expected_shape)):
                raise RuntimeError(
                    f"validation output {name} shape mismatch: floating={reference_array.shape}, "
                    f"quantized={candidate_array.shape}, expected_elements={int(np.prod(expected_shape))}"
                )
            if not np.isfinite(reference_array).all() or not np.isfinite(candidate_array).all():
                raise RuntimeError(f"validation output {name} contains non-finite values")
            difference = candidate_array - reference_array
            state = accumulators[output_index]
            state["dot"] += float(np.vdot(reference_array, candidate_array))
            state["reference_energy"] += float(np.vdot(reference_array, reference_array))
            state["candidate_energy"] += float(np.vdot(candidate_array, candidate_array))
            state["error_energy"] += float(np.vdot(difference, difference))
            state["mae"] += float(np.abs(difference).sum())
            element_counts[output_index] += reference_array.size
            maximum_errors[output_index] = max(maximum_errors[output_index], float(np.abs(difference).max()))

    outputs = []
    for output_index, ((name, _), state) in enumerate(zip(OUTPUTS, accumulators, strict=True)):
        reference_energy = state["reference_energy"]
        candidate_energy = state["candidate_energy"]
        if reference_energy <= 0.0 or candidate_energy <= 0.0:
            raise RuntimeError(f"validation output {name} has zero signal energy")
        outputs.append(
            {
                "cosine_similarity": state["dot"] / np.sqrt(reference_energy * candidate_energy),
                "maximum_absolute_error": maximum_errors[output_index],
                "mean_absolute_error": state["mae"] / element_counts[output_index],
                "name": name,
                "normalized_root_mean_square_error": np.sqrt(state["error_energy"] / reference_energy),
            }
        )
    return {
        "maximum_normalized_root_mean_square_error": max(
            output["normalized_root_mean_square_error"] for output in outputs
        ),
        "minimum_cosine_similarity": min(output["cosine_similarity"] for output in outputs),
        "outputs": outputs,
        "reference": "onnx_fp32_raw_heads",
        "sample_count": len(reference_samples),
    }


def validate_quantization(loaded: Any, quantized: Any, validation: np.ndarray, InputName: Any) -> dict[str, Any]:
    reference_samples: list[list[np.ndarray]] = []
    candidate_samples: list[list[np.ndarray]] = []
    for index in range(validation.shape[0]):
        inputs = {InputName(INPUT_NAME): validation[index : index + 1]}
        reference_samples.append([np.asarray(output) for output in loaded.execute(inputs=inputs)])
        candidate_samples.append([np.asarray(output) for output in quantized.execute(inputs=inputs, use_jax=True)])
    report = quantization_quality_report(reference_samples, candidate_samples)
    if report["sample_count"] != validation.shape[0]:
        raise RuntimeError("quantization validation did not evaluate every held-out sample")
    return report


def enforce_quantization_quality(report: dict[str, Any], precision: str) -> dict[str, Any]:
    specification = yaml.safe_load((MODEL_DIR / "tests" / "correctness.yaml").read_text(encoding="utf-8"))
    try:
        thresholds = specification["compiler_raw_head_thresholds"][precision]
        minimum_cosine = float(thresholds["minimum_cosine_similarity"])
        maximum_nrmse = float(thresholds["maximum_normalized_root_mean_square_error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"missing quantization quality thresholds for {precision}") from exc
    observed_cosine = float(report["minimum_cosine_similarity"])
    observed_nrmse = float(report["maximum_normalized_root_mean_square_error"])
    if observed_cosine < minimum_cosine or observed_nrmse > maximum_nrmse:
        raise RuntimeError(
            f"{precision.upper()} quantization quality failed: cosine={observed_cosine:.8f} "
            f"(minimum {minimum_cosine:.8f}), normalized RMSE={observed_nrmse:.8f} "
            f"(maximum {maximum_nrmse:.8f})"
        )
    report["gate"] = {
        "maximum_normalized_root_mean_square_error": maximum_nrmse,
        "minimum_cosine_similarity": minimum_cosine,
        "passed": True,
    }
    return report


def audit_mpk(archive: Path, variant_id: str) -> dict[str, Any]:
    variant = variant_config(variant_id)
    precision = str(variant["parameters"]["precision"])
    names, manifest = _safe_manifest(archive)
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    a65_plugins = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "A65"]
    if len(elfs) != 1 or len(mla_plugins) != 1 or shared_objects:
        raise RuntimeError(f"invalid SSD MPK packaging: MLA={len(mla_plugins)} ELF={len(elfs)} SO={shared_objects}")
    if str(manifest.get("model_sdk_version")) != COMPILER_VERSION:
        raise RuntimeError(
            f"SSD MPK compiler mismatch: expected {COMPILER_VERSION}, found {manifest.get('model_sdk_version')}"
        )

    input_nodes = manifest.get("input_nodes", [])
    if (
        len(input_nodes) != 1
        or input_nodes[0].get("name") != INPUT_NAME
        or input_nodes[0].get("size") != PUBLIC_INPUT_SIZE
    ):
        raise RuntimeError(f"SSD MPK public input ABI mismatch: {input_nodes}")
    terminal = plugins[-1] if plugins else {}
    output_sizes = [node.get("size") for node in terminal.get("output_nodes", [])]
    if _kernel(terminal) != "pass_through" or output_sizes != PUBLIC_OUTPUT_SIZES:
        raise RuntimeError(f"SSD MPK public output ABI mismatch: {output_sizes}")

    mla_outputs = mla_plugins[0].get("output_nodes", [])
    output_kernel = "dequantization_transform" if precision == "int8" else "cast_transform"
    conversions = [plugin for plugin in plugins if _kernel(plugin) == output_kernel]
    if precision == "bf16":
        conversions = [plugin for plugin in conversions if _params(plugin).get("out_dtype") == "float32"]
    dtype_key = "input_data_type" if precision == "int8" else "in_dtype"
    internal_types = [str(_params(plugin).get(dtype_key, "")).lower() for plugin in conversions]
    if len(mla_outputs) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} SSD MLA outputs, found {len(mla_outputs)}")
    aliases = {"int8"} if precision == "int8" else {"bf16", "bfloat16"}
    if len(internal_types) != len(OUTPUTS) or any(value not in aliases for value in internal_types):
        raise RuntimeError(f"{precision.upper()} SSD output transport audit failed: {internal_types}")

    if precision == "int8":
        quantize = [plugin for plugin in plugins if _kernel(plugin) == "quantization_transform"]
        dequantize = [plugin for plugin in plugins if _kernel(plugin) == "dequantization_transform"]
        quantized_types = [str(_params(plugin).get("output_data_type", "")).lower() for plugin in quantize]
        dequantized_types = [str(_params(plugin).get("input_data_type", "")).lower() for plugin in dequantize]
        if quantized_types != ["int8"] or dequantized_types != ["int8"] * len(OUTPUTS):
            raise RuntimeError(
                f"INT8 SSD boundary transforms are inconsistent: quantize={quantized_types}, "
                f"dequantize={dequantized_types}"
            )

    external_tessellation_stages = [
        plugin.get("name")
        for plugin in plugins
        if _kernel(plugin) in {"tessellation_transform", "detessellation_transform", "quantize_tessellate_transform"}
    ]
    if external_tessellation_stages or a65_plugins:
        raise RuntimeError(
            f"SSD MLA layout audit failed: external_stages={external_tessellation_stages}, A65={len(a65_plugins)}"
        )

    return {
        "archive": archive.name,
        "calibration_archive_sha256": ARCHIVE_SHA256,
        "calibration_tensor_sha256": CALIBRATION_ARRAY_SHA256,
        "compiler_version": COMPILER_VERSION,
        "elf": elfs[0],
        "output_transport_tensor_count": len(internal_types),
        "output_transport_types": internal_types,
        "mla_plugin_count": len(mla_plugins),
        "public_input": {"dtype": "float32", "name": INPUT_NAME, "shape": list(INPUT_SHAPE)},
        "public_output_dtype": "float32",
        "public_output_sizes": output_sizes,
        "shared_object_count": len(shared_objects),
        "tessellation": "mla",
        "tessellation_stages": external_tessellation_stages,
        "validation_tensor_sha256": VALIDATION_ARRAY_SHA256,
    }


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    precision = str(variant["parameters"]["precision"])

    # These imports intentionally stay inside the Vulcan-only path. Registry,
    # surgery, provenance, and MPK-audit tests do not require an SDK install.
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    calibration_path = resolve_calibration_path()
    calibration, validation = validate_dataset(calibration_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ssd-mobilenet-v1-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        prepared = prepare_model(source, Path(temporary) / "ssd_mobilenet_v1_heads.onnx")
        importer = ImporterParams(
            format=ModelFormat.onnx,
            file_paths=[str(prepared)],
            input_names=[INPUT_NAME],
            input_shapes=[INPUT_SHAPE],
            input_types=[ScalarType.float32],
            layout="NCHW",
            output_names=[name for name, _ in OUTPUTS],
        )
        loaded = load_model(importer, target=gen2_target)
        calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT_NAME): calibration}))
        if precision == "int8":
            activation_scheme = quantization_scheme(asymmetric=True, per_channel=False, bits=8)
            weight_scheme = quantization_scheme(asymmetric=False, per_channel=True, bits=8)
        else:
            activation_scheme = weight_scheme = bfloat16_scheme()
        quantization = (
            default_quantization.with_activation_quantization(activation_scheme)
            .with_weight_quantization(weight_scheme)
            .with_requantization_mode(RequantizationMode.sima)
            .with_calibration(CalibrationMethod.from_str("mse"))
        )
        model_name = f"ssd_mobilenet_v1_{variant_id}"
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            model_name=model_name,
            log_level=logging.INFO,
        )
        quantized.save(model_name=model_name, output_directory=str(output_dir))
        quantized_report = audit_quantized_model(output_dir / f"{model_name}.sima.json", precision)
        quantized_report["quantization_validation"] = enforce_quantization_quality(
            validate_quantization(loaded, quantized, validation, InputName), precision
        )
        quantized.compile(
            output_path=str(output_dir),
            batch_size=1,
            log_level=logging.INFO,
        )

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one SSD MPK, found {len(archives)} in {output_dir}")
    report = audit_mpk(archives[0], variant_id)
    report.update(quantized_report)
    (output_dir / "ssd-mobilenet-v1-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--component", required=True)
    args = parser.parse_args()
    if args.component != "model":
        parser.error(f"unsupported component: {args.component}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_model(source, args.output_dir.expanduser().resolve(), args.build_profile)
    LOG.info("Compiled MPK: %s", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
