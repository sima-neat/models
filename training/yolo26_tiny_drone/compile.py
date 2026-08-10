#!/usr/bin/env python3
"""Compile a qualified YOLO26-P2 QAT export into a Modalix batch-1 MPK."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import random
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from contract import INPUT_NAME, DeploymentContract
from dataset import FrameRecord, dataset_content_sha256, load_manifest, require_unchanged_dataset
from onnx_audit import audit_onnx, sha256_file
from PIL import Image

LOG = logging.getLogger("yolo26_tiny_drone.compile")
ARCHITECTURE = "yolo26n-p2-attention-one-class"
MODEL_NAME = "yolo26n_p2_tiny_drone_int8_qat_b1"
MINIMUM_CALIBRATION_IMAGES = 128
EXPECTED_ATTENTION_MATMULS = 8
EXPECTED_ATTENTION_SOFTMAXES = 4
SERIALIZED_INT8_SCALAR_TYPE = 1
EXPECTED_MODEL_SDK_VERSION = "2.1.3"


def snapshot_export(export_dir: Path, destination: Path) -> tuple[Path, Path]:
    """Copy one self-consistent export into a private compilation snapshot."""
    export_root = export_dir.resolve()
    report_source = (export_root / "export-report.json").resolve()
    if report_source.parent != export_root or not report_source.is_file():
        raise ValueError(f"missing or invalid export report: {report_source}")
    report_snapshot = destination / "export-report.json"
    shutil.copyfile(report_source, report_snapshot)
    try:
        report = json.loads(report_snapshot.read_text(encoding="utf-8"))
        declaration = report["onnx"]
        filename = declaration["file"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("export report has no readable ONNX declaration") from exc
    if not isinstance(filename, str):
        raise ValueError("export report ONNX filename must be a string")
    source = (export_root / filename).resolve()
    if source.parent != export_root or not source.is_file():
        raise ValueError("export report references an invalid ONNX source")
    source_snapshot = (destination / filename).resolve()
    if source_snapshot.parent != destination.resolve():
        raise ValueError("export report ONNX filename must not escape its directory")
    shutil.copyfile(source, source_snapshot)
    return report_snapshot, source_snapshot


def validate_export(
    export_dir: Path, manifest: Path, dataset_sha256: str, *, allow_smoke: bool
) -> tuple[Path, dict[str, Any], DeploymentContract]:
    report_path = export_dir / "export-report.json"
    if not report_path.is_file():
        raise ValueError(f"missing export report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("architecture") != ARCHITECTURE:
        raise ValueError("export report is not the YOLO26-P2 tiny-drone QAT contract")
    try:
        contract = DeploymentContract.from_dict(report.get("deployment"))
    except (TypeError, ValueError) as exc:
        raise ValueError("export report has an invalid deployment contract") from exc
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("compile manifest does not match the QAT/export lineage")
    if report.get("dataset_sha256") != dataset_sha256:
        raise ValueError("compile dataset contents do not match the QAT/export lineage")
    if type(report.get("qualification_eligible")) is not bool:
        raise ValueError("export report has no explicit qualification eligibility")
    if not report["qualification_eligible"] and not allow_smoke:
        raise ValueError("ineligible diagnostic export requires --allow-smoke")
    if int(report.get("observer_images", 0)) < MINIMUM_CALIBRATION_IMAGES and not allow_smoke:
        raise ValueError("export has fewer than 128 QAT observer images")

    declaration = report.get("onnx")
    if not isinstance(declaration, dict) or not isinstance(declaration.get("file"), str):
        raise ValueError("export report has no ONNX declaration")
    source = (export_dir / declaration["file"]).resolve()
    if source.parent != export_dir.resolve() or not source.is_file():
        raise ValueError("export report references an invalid ONNX source")
    audit = audit_onnx(source, contract)
    declared_audit = declaration.get("audit")
    if not isinstance(declared_audit, dict) or audit.get("sha256") != declared_audit.get("sha256"):
        raise ValueError("ONNX source digest does not match the export report")

    accuracy = report.get("accuracy")
    if not isinstance(accuracy, dict):
        raise ValueError("export report has no accuracy evidence")
    if not allow_smoke:
        required = {
            "ap50_drop": 0.02,
            "qat_ap50": 0.30,
            "qat_tiny_recall50": 0.50,
            "tiny_ground_truth_count": 100.0,
            "tiny_recall50_drop": 0.05,
        }
        try:
            values = {key: float(accuracy[key]) for key in required}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("export report contains incomplete accuracy evidence") from exc
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("export report contains incomplete accuracy evidence")
        if values["ap50_drop"] > required["ap50_drop"]:
            raise ValueError("export exceeds the 2-point AP50-drop gate")
        if values["tiny_recall50_drop"] > required["tiny_recall50_drop"]:
            raise ValueError("export exceeds the 5-point tiny-recall-drop gate")
        if values["qat_ap50"] < required["qat_ap50"]:
            raise ValueError("export is below the 0.30 QAT AP50 gate")
        if values["qat_tiny_recall50"] < required["qat_tiny_recall50"]:
            raise ValueError("export is below the 0.50 QAT tiny-recall gate")
        if values["tiny_ground_truth_count"] < required["tiny_ground_truth_count"]:
            raise ValueError("export has fewer than 100 held-out tiny objects")
    return source, report, contract


def select_calibration_records(
    records: list[FrameRecord], count: int, *, seed: int, allow_smoke: bool
) -> list[FrameRecord]:
    minimum = 1 if allow_smoke else MINIMUM_CALIBRATION_IMAGES
    if count < minimum:
        raise ValueError(f"calibration image count must be at least {minimum}")
    train = [record for record in records if record.split == "train"]
    unique = {record.image: record for record in train}
    if len(unique) < count:
        raise ValueError(f"requested {count} calibration images, but only {len(unique)} distinct train images exist")
    # A seeded sample avoids taking a contiguous run from one video while
    # preserving the source dataset's natural sequence distribution.
    population = sorted(unique.values(), key=lambda record: str(record.image))
    selected = random.Random(seed).sample(population, count)
    return sorted(selected, key=lambda record: str(record.image))


def preprocess_image(path: Path, contract: DeploymentContract) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(contract.model_size / width, contract.model_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (contract.model_size - resized_width) // 2
    top = (contract.model_size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (contract.model_size, contract.model_size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def build_calibration(records: list[FrameRecord], contract: DeploymentContract) -> tuple[np.ndarray, dict[str, Any]]:
    samples = np.stack([preprocess_image(record.image, contract) for record in records])
    if (
        samples.shape
        != (
            len(records),
            contract.model_size,
            contract.model_size,
            3,
        )
        or samples.dtype != np.float32
    ):
        raise RuntimeError(f"invalid calibration tensor: shape={samples.shape}, dtype={samples.dtype}")
    digest = hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()
    evidence = {
        "count": len(records),
        "images": [
            {
                "path": str(record.image),
                "sequence": record.sequence,
                "sha256": sha256_file(record.image),
            }
            for record in records
        ],
        "preprocessing": (f"rgb_letterbox_{contract.model_size}_fill_114_divide_255_nhwc_float32"),
        "tensor_sha256": digest,
    }
    return samples, evidence


def audit_quantized_model(path: Path, expected_convolutions: int) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"serialized model has no layer topology: {path}") from exc
    convolutions = [layer for layer in layers if str(layer.get("class_name", "")).startswith("MLA_0/ConvAddActivation")]
    if len(convolutions) != expected_convolutions:
        raise RuntimeError(f"expected {expected_convolutions} MLA convolutions, found {len(convolutions)}")

    def serialized_scalar_types(value: Any) -> tuple[int, ...]:
        values: list[int] = []
        if isinstance(value, dict):
            if value.get("__decode_key__") == "ScalarType" and isinstance(value.get("value"), int):
                values.append(value["value"])
            else:
                for child in value.values():
                    values.extend(serialized_scalar_types(child))
        elif isinstance(value, list):
            for child in value:
                values.extend(serialized_scalar_types(child))
        return tuple(values)

    def io_types(layer: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = layer.get("config", {})
        return (
            serialized_scalar_types(config.get("_input_types")),
            serialized_scalar_types(config.get("_output_types")),
        )

    convolution_io = {io_types(layer) for layer in convolutions}
    statuses = {layer.get("config", {}).get("_status") for layer in convolutions}
    per_channel = {layer.get("config", {}).get("per_channel") for layer in convolutions}
    integer_weights = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
    if (
        convolution_io != {((SERIALIZED_INT8_SCALAR_TYPE,), (SERIALIZED_INT8_SCALAR_TYPE,))}
        or statuses != {"SIMA_QUANTIZED"}
        or per_channel != {True}
        or integer_weights != {True}
    ):
        raise RuntimeError(
            "compiled compute graph is not fully per-channel INT8: "
            f"io_types={convolution_io}, statuses={statuses}, per_channel={per_channel}, weights={integer_weights}"
        )
    attention_kinds = {
        "BatchMatmul": EXPECTED_ATTENTION_MATMULS,
        "Softmax": EXPECTED_ATTENTION_SOFTMAXES,
    }
    attention_report = {}
    for kind, expected_count in attention_kinds.items():
        attention_layers = [layer for layer in layers if str(layer.get("class_name", "")).startswith(f"MLA_0/{kind}")]
        if len(attention_layers) != expected_count:
            raise RuntimeError(f"expected {expected_count} MLA {kind} layers, found {len(attention_layers)}")
        expected_inputs = (SERIALIZED_INT8_SCALAR_TYPE,) * (2 if kind == "BatchMatmul" else 1)
        expected_io = (expected_inputs, (SERIALIZED_INT8_SCALAR_TYPE,))
        attention_types = {io_types(layer) for layer in attention_layers}
        attention_statuses = {layer.get("config", {}).get("_status") for layer in attention_layers}
        if attention_types != {expected_io} or attention_statuses != {"SIMA_QUANTIZED"}:
            raise RuntimeError(
                f"compiled {kind} layers are not fully INT8: types={attention_types}, statuses={attention_statuses}"
            )
        attention_report[kind] = {
            "count": len(attention_layers),
            "input_scalar_type": "int8",
            "output_scalar_type": "int8",
            "status": "SIMA_QUANTIZED",
        }
    return {
        "compute_convolution_count": len(convolutions),
        "compute_scalar_type": "int8",
        "internal_precision": "int8",
        "attention": attention_report,
        "quantized_model_json_sha256": sha256_file(path),
        "weight_quantization": "per_channel_int8",
    }


def _safe_mpk(archive: Path) -> tuple[list[str], dict[str, Any]]:
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
            raise RuntimeError("cannot read MPK manifest")
        manifest = json.load(source)
    return names, manifest


def _kernel(plugin: dict[str, Any]) -> str:
    return str(plugin.get("config_params", {}).get("kernel", ""))


def build_mla_tessellation_parameters(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build direct MLA layout parameters from the quantized graph itself."""
    from afe.apis.defines import TensorDRAMLayout, TensorTessellateParameters
    from afe.ir.node import node_is_tuple

    try:
        mla = model._net.nodes["MLA_0"]
        input_keys = list(mla.input_names)
        output_node = mla.ir.nodes[mla.ir.output_node_name]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError("quantized model has no readable MLA_0 layout contract") from exc
    logical_outputs = output_node.input_node_names if node_is_tuple(output_node) else [output_node.name]
    output_keys = [f"{name}_output" for name in logical_outputs]
    if not input_keys or not output_keys or len(set(input_keys + output_keys)) != len(input_keys) + len(output_keys):
        raise RuntimeError(f"invalid MLA layout keys: inputs={input_keys}, outputs={output_keys}")

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
    parameters = {key: dataclasses.replace(input_layout) for key in input_keys}
    parameters.update({key: dataclasses.replace(output_layout) for key in output_keys})
    evidence = {
        "input_keys": input_keys,
        "input_dram_layout": "HWC",
        "output_keys": output_keys,
        "output_dram_layout": "HWC16",
        "tessellation": "mla",
    }
    return parameters, evidence


def audit_mpk(archive: Path, contract: DeploymentContract) -> dict[str, Any]:
    names, manifest = _safe_mpk(archive)
    plugins = sorted(manifest.get("plugins", []), key=lambda item: item.get("sequence", 0))
    mla = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "MLA"]
    a65 = [plugin for plugin in plugins if str(plugin.get("processor", "")).upper() == "A65"]
    elfs = [name for name in names if name.endswith(".elf")]
    shared_objects = [name for name in names if name.endswith(".so") or ".so." in name]
    if len(mla) != 1 or len(elfs) != 1 or a65 or shared_objects:
        raise RuntimeError(
            f"invalid MPK placement: MLA={len(mla)}, ELF={len(elfs)}, A65={len(a65)}, SO={shared_objects}"
        )
    inputs = manifest.get("input_nodes", [])
    public_input_size = int(np.prod(contract.input_shape)) * np.dtype(np.float32).itemsize
    public_output_sizes = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _name, shape in contract.outputs]
    if len(inputs) != 1 or inputs[0].get("name") != INPUT_NAME or inputs[0].get("size") != public_input_size:
        raise RuntimeError(f"MPK public input ABI mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    terminal_inputs = terminal.get("input_nodes", [])
    output_nodes = terminal.get("output_nodes", [])
    role_input_abi = [(str(node.get("name", "")).rsplit("/", 1)[-1], node.get("size")) for node in terminal_inputs]
    output_sizes = [node.get("size") for node in output_nodes]
    expected_output_abi = list(zip(contract.output_names(), public_output_sizes, strict=True))
    if (
        _kernel(terminal) != "pass_through"
        or role_input_abi != expected_output_abi
        or output_sizes != public_output_sizes
    ):
        raise RuntimeError(
            "MPK public output ABI mismatch: "
            f"expected {expected_output_abi}, got roles={role_input_abi}, output_sizes={output_sizes}"
        )
    unpack = [plugin for plugin in plugins if _kernel(plugin) == "unpack_transform"]
    tensor_types = (
        unpack[0].get("config_params", {}).get("params", {}).get("tensor_types", []) if len(unpack) == 1 else []
    )
    quantize = [plugin for plugin in plugins if _kernel(plugin) == "quantization_transform"]
    dequantize = [plugin for plugin in plugins if _kernel(plugin) == "dequantization_transform"]
    external_tessellate = [plugin for plugin in plugins if _kernel(plugin) == "tessellation_transform"]
    external_detessellate = [plugin for plugin in plugins if _kernel(plugin) == "detessellation_transform"]
    quantized_types = [plugin.get("config_params", {}).get("params", {}).get("output_data_type") for plugin in quantize]
    dequantized_types = [
        plugin.get("config_params", {}).get("params", {}).get("input_data_type") for plugin in dequantize
    ]
    if (
        tensor_types != ["int8"] * len(contract.outputs)
        or quantized_types != ["int8"]
        or dequantized_types != ["int8"] * len(contract.outputs)
    ):
        raise RuntimeError(
            "MPK INT8 transport mismatch: "
            f"unpack={tensor_types}, quantize={quantized_types}, dequantize={dequantized_types}"
        )
    if external_tessellate or external_detessellate:
        raise RuntimeError(
            "MPK does not implement tensor layout on MLA: "
            f"external_tessellation={[plugin.get('name') for plugin in external_tessellate]}, "
            f"external_detessellation={[plugin.get('name') for plugin in external_detessellate]}"
        )
    quantized_outputs = [node.get("name") for node in quantize[0].get("output_nodes", [])]
    mla_inputs = [node.get("name") for node in mla[0].get("input_nodes", [])]
    if len(quantized_outputs) != 1 or mla_inputs != quantized_outputs:
        raise RuntimeError(
            f"MPK quantizer does not feed MLA directly: quantized_outputs={quantized_outputs}, mla_inputs={mla_inputs}"
        )
    if str(manifest.get("model_sdk_version")) != EXPECTED_MODEL_SDK_VERSION:
        raise RuntimeError(
            "MPK compiler version mismatch: "
            f"expected {EXPECTED_MODEL_SDK_VERSION}, found {manifest.get('model_sdk_version')}"
        )
    return {
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "compiler_version": str(manifest.get("model_sdk_version")),
        "elf": elfs[0],
        "mla_plugin_count": len(mla),
        "a65_plugin_count": len(a65),
        "ev74_plugin_count": sum(str(plugin.get("processor", "")).upper() == "EV74" for plugin in plugins),
        "public_input": {
            "dtype": "float32",
            "name": INPUT_NAME,
            "shape": list(contract.input_shape),
        },
        "public_output_dtype": "float32",
        "public_output_names": [name for name, _size in role_input_abi],
        "public_output_sizes": output_sizes,
        "public_output_abi": role_input_abi,
        "output_transport_types": tensor_types,
        "quant_to_mla_direct": True,
        "external_tessellation_plugin_count": len(external_tessellate),
        "external_detessellation_plugin_count": len(external_detessellate),
        "shared_object_count": len(shared_objects),
        "tessellation": "mla",
    }


def compile_model(
    source: Path,
    calibration: np.ndarray,
    output_dir: Path,
    model_name: str,
    contract: DeploymentContract,
    expected_convolutions: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    # Keep SDK imports out of source-validation and unit-test paths.
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    importer = ImporterParams(
        format=ModelFormat.onnx,
        file_paths=[str(source)],
        input_names=[INPUT_NAME],
        input_shapes=[contract.input_shape],
        input_types=[ScalarType.float32],
        layout="NCHW",
        output_names=contract.output_names(),
    )
    loaded = load_model(importer, target=gen2_target)
    calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT_NAME): calibration}))
    config = (
        default_quantization.with_activation_quantization(
            quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        )
        .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("min_max"))
    )
    quantized = loaded.quantize(
        calibration_data=calibration_data,
        quantization_config=config,
        # Attention uses static rank-4 tensors and ModelSDK's documented
        # two-input Einsum representation for batch matrix multiplication.
        any_shape_on_mla=True,
        automatic_layout_conversion=False,
        model_name=model_name,
        log_level=logging.INFO,
    )
    if quantized is None:
        raise RuntimeError("ModelSDK quantization returned no model")
    quantized.save(model_name=model_name, output_directory=str(output_dir))
    quantized_report = audit_quantized_model(
        output_dir / f"{model_name}.sima.json",
        expected_convolutions,
    )
    tessellation_parameters, tessellation_report = build_mla_tessellation_parameters(quantized)
    quantized.compile(
        output_path=str(output_dir),
        batch_size=1,
        log_level=logging.INFO,
        tessellate_parameters=tessellation_parameters,
    )
    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one compiled MPK, found {archives}")
    return archives[0], quantized_report, tessellation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=MINIMUM_CALIBRATION_IMAGES)
    parser.add_argument("--calibration-seed", type=int, default=26)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    export_dir = args.export_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    if not manifest.is_file():
        parser.error(f"manifest does not exist: {manifest}")
    if re.fullmatch(r"[a-z0-9_]{1,80}", args.model_name) is None:
        parser.error("model-name must contain only lowercase letters, digits, and underscores")
    records = load_manifest(manifest)
    dataset_digest = dataset_content_sha256(manifest, records)
    with tempfile.TemporaryDirectory(prefix="yolo26-export-snapshot-") as temporary:
        snapshot_dir = Path(temporary)
        report_snapshot, source_snapshot = snapshot_export(export_dir, snapshot_dir)
        source, export_report, contract = validate_export(
            snapshot_dir,
            manifest,
            dataset_digest,
            allow_smoke=args.allow_smoke,
        )
        report_digest = sha256_file(report_snapshot)
        source_digest = sha256_file(source_snapshot)
        report_snapshot.chmod(0o400)
        source_snapshot.chmod(0o400)
        selected = select_calibration_records(
            records,
            args.calibration_images,
            seed=args.calibration_seed,
            allow_smoke=args.allow_smoke,
        )
        calibration, calibration_report = build_calibration(selected, contract)
        require_unchanged_dataset(
            manifest,
            records,
            manifest_sha256=export_report["manifest_sha256"],
            dataset_sha256=dataset_digest,
            context="calibration",
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        archive, quantized_report, tessellation_report = compile_model(
            source,
            calibration,
            output_dir,
            args.model_name,
            contract,
            int(export_report["onnx"]["audit"]["convolution_count"]),
        )
        validate_export(
            snapshot_dir,
            manifest,
            dataset_digest,
            allow_smoke=args.allow_smoke,
        )
        if sha256_file(report_snapshot) != report_digest or sha256_file(source_snapshot) != source_digest:
            raise RuntimeError("private export snapshot changed during compilation")
        report = {
            "schema_version": 1,
            "architecture": ARCHITECTURE,
            "deployment": contract.to_dict(),
            "source_qualification_eligible": export_report["qualification_eligible"],
            "qualification_eligible": export_report["qualification_eligible"] and not args.allow_smoke,
            "source": {
                "export_report_sha256": report_digest,
                "onnx_sha256": source_digest,
            },
            "calibration": calibration_report,
            "quantized_model": quantized_report,
            "mla_layout": tessellation_report,
            "mpk": audit_mpk(archive, contract),
        }
    (output_dir / "compile-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
