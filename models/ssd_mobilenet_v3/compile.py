#!/usr/bin/env python3
"""Quantize and compile one pinned SSD-MobileNet V3 Modalix variant."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec
from surgery import INPUT_NAME, INPUT_SHAPE, OUTPUTS, prepare_int8_model, prepare_model

LOG = logging.getLogger("ssd_mobilenet_v3.compile")
MODEL_DIR = Path(__file__).resolve().parent
COMPILER_VERSION = "2.1.3"
PUBLIC_INPUT_SIZE = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
PUBLIC_OUTPUT_SIZES = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _, shape in OUTPUTS]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_normalize(samples: np.ndarray) -> np.ndarray:
    """Convert the pinned legacy [-1, 1] HWC tensors to TorchVision input."""

    if samples.ndim != 4 or samples.shape[-1] != 3 or samples.dtype != np.float32:
        raise ValueError(f"expected float32 NHWC RGB samples, found {samples.shape} {samples.dtype}")
    rgb_unit = (samples + np.float32(1.0)) / np.float32(2.0)
    return (rgb_unit - IMAGENET_MEAN) / IMAGENET_STD


def build_reference_evaluator(path: Path, precision: str) -> Any:
    """Build an ONNX reference executor, adapting QDQ dispatch in SDK 2.1.3."""

    from onnx.reference import ReferenceEvaluator

    if precision not in {"int8", "bf16"}:
        raise ValueError(f"unsupported SSD reference precision: {precision}")
    if precision != "int8":
        return ReferenceEvaluator(str(path))

    # The ONNX package in ModelSDK 2.1.3 exposes Q/DQ implementations for
    # opset 19, but its dispatcher has no entries for the identical scalar and
    # per-axis behavior used by this opset-17 QAT graph. Register the shipped
    # implementations under their canonical operator names and ignore only
    # newer optional schema attributes that cannot occur in the audited graph.
    from onnx.reference.ops.op_dequantize_linear import DequantizeLinear_19
    from onnx.reference.ops.op_quantize_linear import QuantizeLinear_19

    class DequantizeLinear(DequantizeLinear_19):
        op_domain = ""

        def _run(
            self,
            x: Any,
            x_scale: Any,
            x_zero_point: Any = None,
            axis: Any = None,
            block_size: Any = None,
            output_dtype: Any = None,
        ) -> Any:
            if block_size not in {None, 0} or output_dtype not in {None, 0}:
                raise ValueError("opset-17 DequantizeLinear cannot use blocked quantization or output_dtype")
            return super()._run(x, x_scale, x_zero_point, axis)

    class QuantizeLinear(QuantizeLinear_19):
        op_domain = ""

        def _run(
            self,
            x: Any,
            y_scale: Any,
            zero_point: Any = None,
            axis: Any = None,
            saturate: Any = None,
            block_size: Any = None,
            output_dtype: Any = None,
            precision: Any = None,
        ) -> Any:
            if block_size not in {None, 0} or output_dtype not in {None, 0} or precision not in {None, 0}:
                raise ValueError("opset-17 QuantizeLinear cannot use blocked quantization or output_dtype")
            return super()._run(x, y_scale, zero_point, axis, saturate)

    return ReferenceEvaluator(str(path), new_ops=[DequantizeLinear, QuantizeLinear])


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one SSD-MobileNet V3 variant named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == "model")
    variant = {**profile, "parameters": binding["parameters"]}
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("precision") not in {"int8", "bf16"}:
        raise ValueError("SSD-MobileNet V3 supports only declared Modalix INT8 and BF16 variants")
    if not isinstance(parameters.get("tessellation_mla"), bool):
        raise ValueError("tessellation_mla must be a boolean")
    expected_quantization = (
        ("asymmetric_per_tensor_8bit", "symmetric_per_channel_8bit")
        if parameters["precision"] == "int8"
        else ("bfloat16", "bfloat16")
    )
    declared_quantization = (parameters.get("activation_quantization"), parameters.get("weight_quantization"))
    if declared_quantization != expected_quantization:
        raise ValueError(f"SSD quantization declaration mismatch: {declared_quantization}")
    expected_calibration = "min_max" if parameters["precision"] == "int8" else "mse"
    if parameters.get("calibration_method") != expected_calibration or parameters.get("requantization") != "sima":
        raise ValueError(
            f"SSD {parameters['precision'].upper()} variants require {expected_calibration} calibration "
            "and SiMa requantization"
        )
    if parameters.get("quantization_method") != ("qat" if parameters["precision"] == "int8" else "ptq"):
        raise ValueError(f"SSD {parameters['precision'].upper()} quantization method declaration mismatch")
    if parameters.get("smooth_quant", False) or parameters.get("bias_correction", "none") != "none":
        raise ValueError("SSD QAT/BF16 sources must not receive PTQ SmoothQuant or bias correction")
    if parameters.get("calibration_samples") != len(CALIBRATION_IMAGE_IDS):
        raise ValueError(f"SSD variants require exactly {len(CALIBRATION_IMAGE_IDS)} calibration samples")
    return variant


def calibration_source() -> SourceSpec:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    calibration = next(item for item in metadata["calibrations"] if item["id"] == "default")
    if calibration.get("status") != "ready":
        raise ValueError("SSD-MobileNet V3 requires pinned representative calibration data")
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


def _tessellation_parameters(
    model: Any,
    *,
    TensorDRAMLayout: Any,
    TensorTessellateParameters: Any,
    node_is_tuple: Any,
) -> dict[str, Any]:
    if "MLA_0" not in model._net.nodes:
        raise RuntimeError("quantized SSD graph does not contain MLA_0")
    mla = model._net.nodes["MLA_0"]
    input_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC,
    )
    output_layout = TensorTessellateParameters(
        tile_shape=(0, 0, 0, 0),
        enable_mla=True,
        dram_layout=TensorDRAMLayout.HWC16,
    )
    parameters = {name: input_layout for name in mla.input_names}
    output_node = mla.ir.nodes[mla.ir.output_node_name]
    output_names = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    parameters.update({f"{name}_output": output_layout for name in output_names})
    if len(output_names) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} MLA outputs, found {len(output_names)}")
    return parameters


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
    if len(convolutions) != 98:
        raise RuntimeError(f"expected 98 quantized SSD MLA convolutions, found {len(convolutions)}")
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


def _normalize_validation_output(value: np.ndarray, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    batch, channels, height, width = expected_shape
    if array.shape == expected_shape:
        return array
    if array.shape == (batch, height, width, channels):
        return array.transpose(0, 3, 1, 2)
    if array.shape == (batch, width, channels, height):
        # Model.execute exposes these raw heads as NWCH on this compiler.
        return array.transpose(0, 2, 3, 1)
    raise RuntimeError(f"validation output {name} has unsupported shape {array.shape}; expected {expected_shape}")


def quantization_quality_report(
    reference_samples: list[list[np.ndarray]], candidate_samples: list[list[np.ndarray]]
) -> dict[str, Any]:
    if len(reference_samples) != len(candidate_samples) or not reference_samples:
        raise RuntimeError(
            "quantization validation requires the same non-zero number of floating and quantized samples"
        )
    accumulators = [
        {"dot": 0.0, "reference_energy": 0.0, "candidate_energy": 0.0, "error_energy": 0.0, "mae": 0.0} for _ in OUTPUTS
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
            reference_array = _normalize_validation_output(reference, name, expected_shape)
            candidate_array = _normalize_validation_output(candidate, name, expected_shape)
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
        "reference": "onnx_precision_source_raw_heads",
        "sample_count": len(reference_samples),
    }


def validate_quantization(
    reference: Any,
    quantized: Any,
    validation: np.ndarray,
    InputName: Any,
) -> dict[str, Any]:
    reference_samples: list[list[np.ndarray]] = []
    candidate_samples: list[list[np.ndarray]] = []
    for index in range(validation.shape[0]):
        sample = validation[index : index + 1]
        inputs = {InputName(INPUT_NAME): sample}
        LOG.info("Validation sample %d/%d: executing FP32 reference", index + 1, validation.shape[0])
        reference_samples.append(
            [np.asarray(output) for output in reference.run(None, {INPUT_NAME: sample.transpose(0, 3, 1, 2)})]
        )
        LOG.info("Validation sample %d/%d: executing quantized JAX graph", index + 1, validation.shape[0])
        candidate_samples.append([np.asarray(output) for output in quantized.execute(inputs=inputs, use_jax=True)])
        LOG.info("Validation sample %d/%d: complete", index + 1, validation.shape[0])
    report = quantization_quality_report(reference_samples, candidate_samples)
    report["executor"] = "onnx_reference_and_jax_quantized"
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
    tessellation_mla = bool(variant["parameters"]["tessellation_mla"])
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

    unpack = [plugin for plugin in plugins if _kernel(plugin) == "unpack_transform"]
    if len(unpack) != 1:
        raise RuntimeError(f"expected one SSD output unpack transform, found {len(unpack)}")
    internal_types = [str(value).lower() for value in _params(unpack[0]).get("tensor_types", [])]
    aliases = {"int8"} if precision == "int8" or not tessellation_mla else {"bf16", "bfloat16"}
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

    external_tessellation_stages = sorted(
        PurePosixPath(name).name
        for name in names
        if PurePosixPath(name).name in {"0_quanttess.json", "0_tessellate.json"}
    )
    if bool(external_tessellation_stages) == tessellation_mla or a65_plugins:
        placement = "in MLA" if tessellation_mla else "on EV74 outside MLA"
        raise RuntimeError(
            f"SSD tessellation placement audit failed for {placement}: "
            f"external_stages={external_tessellation_stages}, A65={len(a65_plugins)}"
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
        "tessellation": "mla" if tessellation_mla else "ev74",
        "tessellation_stages": external_tessellation_stages,
        "validation_tensor_sha256": VALIDATION_ARRAY_SHA256,
    }


def compile_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    precision = str(variant["parameters"]["precision"])
    tessellation_mla = bool(variant["parameters"]["tessellation_mla"])
    calibration_method = str(variant["parameters"]["calibration_method"])

    # These imports intentionally stay inside the Vulcan-only path. Registry,
    # surgery, provenance, and MPK-audit tests do not require an SDK install.
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        TensorDRAMLayout,
        TensorTessellateParameters,
        bfloat16_scheme,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.node import node_is_tuple
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    calibration_path = resolve_calibration_path()
    calibration, _validation = validate_dataset(calibration_path)
    calibration = imagenet_normalize(calibration)
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="ssd-mobilenet-v3-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        prepare = prepare_int8_model if precision == "int8" else prepare_model
        prepared = prepare(source, Path(temporary) / "ssd_mobilenet_v3_heads.onnx")
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
            .with_calibration(CalibrationMethod.from_str(calibration_method))
        )
        model_name = f"ssd_mobilenet_v3_{variant_id}"
        quantized = loaded.quantize(
            calibration_data=calibration_data,
            quantization_config=quantization,
            any_shape_on_mla=False,
            automatic_layout_conversion=False,
            model_name=model_name,
            log_level=logging.INFO,
        )
        quantized.save(model_name=model_name, output_directory=str(output_dir))
        quantized_report = audit_quantized_model(output_dir / f"{model_name}.sima.json", precision)
        tessellation = None
        if tessellation_mla:
            tessellation = _tessellation_parameters(
                quantized,
                TensorDRAMLayout=TensorDRAMLayout,
                TensorTessellateParameters=TensorTessellateParameters,
                node_is_tuple=node_is_tuple,
            )
        quantized.compile(
            output_path=str(output_dir),
            batch_size=1,
            log_level=logging.INFO,
            tessellate_parameters=tessellation,
        )

    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one SSD MPK, found {len(archives)} in {output_dir}")
    report = audit_mpk(archives[0], variant_id)
    report.update(quantized_report)
    (output_dir / "ssd-mobilenet-v3-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archives[0]


def validate_saved_model(source: Path, output_dir: Path, variant_id: str) -> Path:
    """Reload compiler outputs in a fresh process and run held-out validation."""

    precision = str(variant_config(variant_id)["parameters"]["precision"])

    from afe.apis.defines import InputName
    from afe.apis.model import Model

    LOG.info("Fresh validation process: resolving calibration asset")
    calibration_path = resolve_calibration_path()
    _calibration, validation = validate_dataset(calibration_path)
    validation = imagenet_normalize(validation)
    model_name = f"ssd_mobilenet_v3_{variant_id}"
    with tempfile.TemporaryDirectory(
        prefix="ssd-mobilenet-v3-validation-",
        dir=os.environ.get("RUNNER_TEMP"),
    ) as temporary:
        LOG.info("Fresh validation process: preparing %s precision source", precision.upper())
        prepare = prepare_int8_model if precision == "int8" else prepare_model
        prepared = prepare(source, Path(temporary) / "ssd_mobilenet_v3_heads.onnx")
        LOG.info("Fresh validation process: creating independent ONNX precision-source reference")
        reference = build_reference_evaluator(prepared, precision)
        LOG.info("Fresh validation process: loading serialized quantized graph")
        quantized = Model.load(model_name, str(output_dir))
        LOG.info("Fresh validation process: executing %d held-out samples", validation.shape[0])
        # Model Compiler 2.1.3's TVM floating executor segfaults for this V3
        # graph. Use the prepared ONNX as the independent floating reference;
        # the JAX executor is confined to the serialized quantized graph.
        validation_report = enforce_quantization_quality(
            validate_quantization(
                reference,
                quantized,
                validation,
                InputName,
            ),
            precision,
        )
        LOG.info(
            "Fresh validation process: metrics cosine=%.10f normalized_rmse=%.10f",
            validation_report["minimum_cosine_similarity"],
            validation_report["maximum_normalized_root_mean_square_error"],
        )

    audit_path = output_dir / "ssd-mobilenet-v3-audit.json"
    if not audit_path.is_file():
        raise RuntimeError(f"missing pre-validation SSD audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["quantization_validation"] = validation_report
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one validated SSD MPK, found {len(archives)} in {output_dir}")
    LOG.info("Fresh validation process: audit saved")
    return archives[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--validation-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.component != "model":
        parser.error(f"unsupported component: {args.component}")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    output_dir = args.output_dir.expanduser().resolve()
    if args.validation_only:
        archive = validate_saved_model(source, output_dir, args.build_profile)
        LOG.info("Compiled MPK: %s", archive)
        # The dedicated worker has persisted every validation result. Avoid
        # native Model Compiler teardown, which can crash after V3 execution.
        logging.shutdown()
        os._exit(0)
    else:
        compile_model(source, output_dir, args.build_profile)
        os.execv(
            sys.executable,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                str(source),
                "--output-dir",
                str(output_dir),
                "--build-profile",
                args.build_profile,
                "--component",
                args.component,
                "--validation-only",
            ],
        )
    raise AssertionError("compile process replacement unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
