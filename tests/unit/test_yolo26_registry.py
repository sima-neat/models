from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from model_registry.model import load_and_validate_model

from models.yolo26_tiny_drone.output_labels import restore_public_output_labels
from models.yolo26_tiny_drone.quantized_snapshot import (
    EXPECTED_CALIBRATION_TENSOR_SHA256,
    EXPECTED_ONNX_SHA256,
    MODEL_FILES,
    MODEL_NAME,
    build_archive,
    materialize_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "yolo26_tiny_drone"


def test_registry_entry_pins_qualified_qat_and_calibration_sources() -> None:
    model = load_and_validate_model(MODEL_DIR)
    profile = model["build_profiles"][0]
    binding = profile["components"][0]

    assert model["components"][0]["source"] == {
        "type": "file",
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/508126877",
        "sha256": "2627762fc29dbd377a3f471a3ab0407569dd39c13cd324d46ca426629ec80f66",
        "size_bytes": 2798186,
        "media_type": "application/onnx",
    }
    assert model["calibrations"][0]["source"]["sha256"] == (
        "8418b35d20ced1a9135eb8862e5554544f261023963ec6290e587057d2b11b2b"
    )
    assert profile["id"] == "modalix_int8_tessellation_mla"
    assert binding["parameters"]["calibration_samples"] == 128
    assert binding["parameters"]["calibration_tensor_sha256"] == (
        "d8830d19b33fe336c5c2cec4822c3b278be4169d125ca249be476771be5b544e"
    )
    assert binding["parameters"]["quantized_source"] == {
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/508162355",
        "sha256": "9d8444f989563e71f32ab2569f4f49e02607036b92c376f0c64b79e3ac4530cc",
        "size_bytes": 9644979,
        "media_type": "application/gzip",
    }


def test_registry_recipe_reuses_strict_training_compiler_contract() -> None:
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_strings = {
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for value in node.args
        if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith("--")
    }

    assert {"--model", "--output-dir", "--build-profile"} <= option_strings
    assert "audit_onnx(source, contract)" in source
    assert "validate_calibration(" in source
    assert "materialize=False" in source
    assert "materialize_snapshot(" in source
    assert "Model.load(MODEL_NAME" in source
    assert "restore_public_output_labels(quantized, contract.output_names())" in source
    assert '"mpk": audit_mpk(archive, contract)' in source
    assert "tessellate_parameters" not in source


def test_modelsdk_output_labels_are_restored_and_verified() -> None:
    class Net:
        def __init__(self) -> None:
            self._output_labels = None

        @property
        def output_labels(self):
            return self._output_labels

    class Model:
        def __init__(self) -> None:
            self._net = Net()

    model = Model()
    report = restore_public_output_labels(model, ["bbox_0", "class_logit_0"])

    assert model._net.output_labels == ["bbox_0", "class_logit_0"]
    assert report == {
        "names": ["bbox_0", "class_logit_0"],
        "restored_after_load": True,
    }


def test_modelsdk_output_labels_refuse_to_overwrite_an_abi_change() -> None:
    class Net:
        _output_labels = ["different"]
        output_labels = _output_labels

    class Model:
        _net = Net()

    with pytest.raises(RuntimeError, match="public output labels changed"):
        restore_public_output_labels(Model(), ["bbox_0"])


def test_registry_accuracy_gates_match_qualified_export() -> None:
    accuracy = yaml.safe_load((MODEL_DIR / "tests" / "correctness.yaml").read_text(encoding="utf-8"))
    metrics = {item["name"]: item for item in accuracy["metrics"]}

    assert accuracy["status"] == "qualified_reference"
    assert accuracy["validation_tiny_ground_truth_count"] == 197828
    assert metrics["qat_ap50"]["measured"] > metrics["qat_ap50"]["minimum"]
    assert metrics["qat_tiny_recall50"]["measured"] > metrics["qat_tiny_recall50"]["minimum"]
    assert metrics["ap50_drop"]["measured"] < metrics["ap50_drop"]["maximum"]
    assert metrics["tiny_recall50_drop"]["measured"] < metrics["tiny_recall50_drop"]["maximum"]


def test_calibration_archive_builder_is_deterministic_and_fail_closed() -> None:
    source = (MODEL_DIR / "calibration.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert "EXPECTED_IMAGES = 128" in source
    assert 'gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)' in source
    assert "member.issym()" in source and "member.islnk()" in source
    assert "calibration tensor digest mismatch" in source


def test_quantized_snapshot_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for index, name in enumerate(MODEL_FILES):
        (source_dir / name).write_bytes(f"qualified-{index}\n".encode())
    quantized_json = source_dir / f"{MODEL_NAME}.sima.json"
    quantized_digest = hashlib.sha256(quantized_json.read_bytes()).hexdigest()
    payload_digests = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in MODEL_FILES
    }
    report = {
        "schema_version": 1,
        "architecture": "yolo26n-p2-attention-one-class",
        "qualification_eligible": True,
        "source_qualification_eligible": True,
        "source": {"onnx_sha256": EXPECTED_ONNX_SHA256},
        "calibration": {"count": 128, "tensor_sha256": EXPECTED_CALIBRATION_TENSOR_SHA256},
        "quantized_model": {
            "compute_convolution_count": 129,
            "compute_scalar_type": "int8",
            "weight_quantization": "per_channel_int8",
            "quantized_model_json_sha256": quantized_digest,
            "serialized_model_sha256": payload_digests,
            "attention": {
                "BatchMatmul": {"count": 8, "input_scalar_type": "int8", "output_scalar_type": "int8"},
                "Softmax": {"count": 4, "input_scalar_type": "int8", "output_scalar_type": "int8"},
            },
        },
        "mpk": {"compiler_version": "2.1.3"},
    }
    report_path = tmp_path / "compile-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    build_archive(report_path, source_dir, first)
    build_archive(report_path, source_dir, second)

    assert first.read_bytes() == second.read_bytes()
    output = tmp_path / "materialized"
    manifest = materialize_snapshot(first, output)
    assert manifest["model_name"] == MODEL_NAME
    assert {path.name for path in output.iterdir()} == set(MODEL_FILES)

    for index, name in enumerate(MODEL_FILES):
        path = source_dir / name
        original = path.read_bytes()
        path.write_bytes(b"mutated payload")
        with pytest.raises(
            ValueError,
            match=rf"serialized model payload does not match its qualification report: {re.escape(name)}",
        ):
            build_archive(report_path, source_dir, tmp_path / f"mutated-{index}.tar.gz")
        path.write_bytes(original)
