#!/usr/bin/env python3
"""Compile a qualified YOLO26-P2 QAT export into a Modalix batch-1 MPK."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from contract import INPUT_NAME, INPUT_SHAPE, MODEL_SIZE, OUTPUTS, output_names
from dataset import FrameRecord, load_manifest
from onnx_audit import audit_onnx, sha256_file
from PIL import Image

LOG = logging.getLogger("yolo26_tiny_drone.compile")
ARCHITECTURE = "yolo26n-p2-int8-conv-one-class"
MODEL_NAME = "yolo26n_p2_tiny_drone_int8_qat_b1"
MINIMUM_CALIBRATION_IMAGES = 128
EXPECTED_CONVOLUTIONS = 131
PUBLIC_INPUT_SIZE = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
PUBLIC_OUTPUT_SIZES = [int(np.prod(shape)) * np.dtype(np.float32).itemsize for _name, shape in OUTPUTS]


def validate_export(export_dir: Path, manifest: Path, *, allow_smoke: bool) -> tuple[Path, dict[str, Any]]:
    report_path = export_dir / "export-report.json"
    if not report_path.is_file():
        raise ValueError(f"missing export report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("architecture") != ARCHITECTURE:
        raise ValueError("export report is not the YOLO26-P2 tiny-drone QAT contract")
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("compile manifest does not match the QAT/export lineage")
    if int(report.get("observer_images", 0)) < MINIMUM_CALIBRATION_IMAGES and not allow_smoke:
        raise ValueError("export has fewer than 128 QAT observer images")

    declaration = report.get("onnx")
    if not isinstance(declaration, dict) or not isinstance(declaration.get("file"), str):
        raise ValueError("export report has no ONNX declaration")
    source = (export_dir / declaration["file"]).resolve()
    if source.parent != export_dir.resolve() or not source.is_file():
        raise ValueError("export report references an invalid ONNX source")
    audit = audit_onnx(source)
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
    return source, report


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


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (MODEL_SIZE - resized_width) // 2
    top = (MODEL_SIZE - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (MODEL_SIZE, MODEL_SIZE), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def build_calibration(records: list[FrameRecord]) -> tuple[np.ndarray, dict[str, Any]]:
    samples = np.stack([preprocess_image(record.image) for record in records])
    if samples.shape != (len(records), MODEL_SIZE, MODEL_SIZE, 3) or samples.dtype != np.float32:
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
        "preprocessing": "rgb_letterbox_640_fill_114_divide_255_nhwc_float32",
        "tensor_sha256": digest,
    }
    return samples, evidence


def audit_quantized_model(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    try:
        layers = document["modelTopology"]["model_config"]["config"]["layers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"serialized model has no layer topology: {path}") from exc
    convolutions = [layer for layer in layers if str(layer.get("class_name", "")).startswith("MLA_0/ConvAddActivation")]
    if len(convolutions) != EXPECTED_CONVOLUTIONS:
        raise RuntimeError(f"expected {EXPECTED_CONVOLUTIONS} MLA convolutions, found {len(convolutions)}")
    scalar_types = {layer.get("config", {}).get("conv_attrs.input_type", {}).get("value") for layer in convolutions}
    statuses = {layer.get("config", {}).get("_status") for layer in convolutions}
    per_channel = {layer.get("config", {}).get("per_channel") for layer in convolutions}
    integer_weights = {"weight_quant_data" in layer.get("config", {}) for layer in convolutions}
    if scalar_types != {6} or statuses != {"SIMA_QUANTIZED"} or per_channel != {True} or integer_weights != {True}:
        raise RuntimeError(
            "compiled compute graph is not fully per-channel INT8: "
            f"types={scalar_types}, statuses={statuses}, per_channel={per_channel}, weights={integer_weights}"
        )
    return {
        "compute_convolution_count": len(convolutions),
        "compute_scalar_type": 6,
        "internal_precision": "int8",
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


def audit_mpk(archive: Path) -> dict[str, Any]:
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
    if len(inputs) != 1 or inputs[0].get("name") != INPUT_NAME or inputs[0].get("size") != PUBLIC_INPUT_SIZE:
        raise RuntimeError(f"MPK public input ABI mismatch: {inputs}")
    terminal = plugins[-1] if plugins else {}
    output_sizes = [node.get("size") for node in terminal.get("output_nodes", [])]
    if _kernel(terminal) != "pass_through" or output_sizes != PUBLIC_OUTPUT_SIZES:
        raise RuntimeError(f"MPK public output ABI mismatch: {output_sizes}")
    unpack = [plugin for plugin in plugins if _kernel(plugin) == "unpack_transform"]
    tensor_types = (
        unpack[0].get("config_params", {}).get("params", {}).get("tensor_types", []) if len(unpack) == 1 else []
    )
    quantize = [plugin for plugin in plugins if _kernel(plugin) == "quantization_transform"]
    dequantize = [plugin for plugin in plugins if _kernel(plugin) == "dequantization_transform"]
    quantized_types = [plugin.get("config_params", {}).get("params", {}).get("output_data_type") for plugin in quantize]
    dequantized_types = [
        plugin.get("config_params", {}).get("params", {}).get("input_data_type") for plugin in dequantize
    ]
    if (
        tensor_types != ["int8"] * len(OUTPUTS)
        or quantized_types != ["int8"]
        or dequantized_types != ["int8"] * len(OUTPUTS)
    ):
        raise RuntimeError(
            "MPK INT8 transport mismatch: "
            f"unpack={tensor_types}, quantize={quantized_types}, dequantize={dequantized_types}"
        )
    return {
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "compiler_version": str(manifest.get("model_sdk_version")),
        "elf": elfs[0],
        "mla_plugin_count": len(mla),
        "a65_plugin_count": len(a65),
        "ev74_plugin_count": sum(str(plugin.get("processor", "")).upper() == "EV74" for plugin in plugins),
        "public_input": {"dtype": "float32", "name": INPUT_NAME, "shape": list(INPUT_SHAPE)},
        "public_output_dtype": "float32",
        "public_output_sizes": output_sizes,
        "output_transport_types": tensor_types,
        "shared_object_count": len(shared_objects),
        "tessellation": "ev74",
    }


def compile_model(
    source: Path, calibration: np.ndarray, output_dir: Path, model_name: str
) -> tuple[Path, dict[str, Any]]:
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
        input_shapes=[INPUT_SHAPE],
        input_types=[ScalarType.float32],
        layout="NCHW",
        output_names=output_names(),
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
        any_shape_on_mla=False,
        automatic_layout_conversion=False,
        model_name=model_name,
        log_level=logging.INFO,
    )
    if quantized is None:
        raise RuntimeError("ModelSDK quantization returned no model")
    quantized.save(model_name=model_name, output_directory=str(output_dir))
    quantized_report = audit_quantized_model(output_dir / f"{model_name}.sima.json")
    quantized.compile(
        output_path=str(output_dir),
        batch_size=1,
        log_level=logging.INFO,
        tessellate_parameters=None,
    )
    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one compiled MPK, found {archives}")
    return archives[0], quantized_report


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
    source, export_report = validate_export(export_dir, manifest, allow_smoke=args.allow_smoke)
    records = load_manifest(manifest)
    selected = select_calibration_records(
        records,
        args.calibration_images,
        seed=args.calibration_seed,
        allow_smoke=args.allow_smoke,
    )
    calibration, calibration_report = build_calibration(selected)
    output_dir.mkdir(parents=True, exist_ok=False)
    archive, quantized_report = compile_model(source, calibration, output_dir, args.model_name)
    report = {
        "schema_version": 1,
        "architecture": ARCHITECTURE,
        "qualification_eligible": not args.allow_smoke,
        "source": {
            "export_report_sha256": sha256_file(export_dir / "export-report.json"),
            "onnx_sha256": export_report["onnx"]["audit"]["sha256"],
        },
        "calibration": calibration_report,
        "quantized_model": quantized_report,
        "mpk": audit_mpk(archive),
    }
    (output_dir / "compile-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
