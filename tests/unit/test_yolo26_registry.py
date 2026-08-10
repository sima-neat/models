from __future__ import annotations

import ast
from pathlib import Path

import yaml
from model_registry.model import load_and_validate_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "yolo26_tiny_drone"


def test_registry_entry_pins_qualified_qat_and_calibration_sources() -> None:
    model = load_and_validate_model(MODEL_DIR)
    variant = model["compile_variants"][0]

    assert model["source"] == {
        "uri": "https://api.github.com/repos/sima-neat/models/releases/assets/508126877",
        "sha256": "2627762fc29dbd377a3f471a3ab0407569dd39c13cd324d46ca426629ec80f66",
        "size_bytes": 2798186,
        "media_type": "application/onnx",
    }
    assert model["calibration"]["source"]["sha256"] == (
        "8418b35d20ced1a9135eb8862e5554544f261023963ec6290e587057d2b11b2b"
    )
    assert variant["id"] == "modalix_int8_tessellation_mla"
    assert variant["parameters"]["calibration_samples"] == 128
    assert variant["parameters"]["calibration_tensor_sha256"] == (
        "d8830d19b33fe336c5c2cec4822c3b278be4169d125ca249be476771be5b544e"
    )


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

    assert {"--model", "--output-dir", "--variant"} <= option_strings
    assert "audit_onnx(source, contract)" in source
    assert "validate_calibration(" in source
    assert "compile_model(" in source
    assert '"mpk": audit_mpk(archive, contract)' in source
    assert 'parameters.get("tessellation_owner") != "mla"' in source


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
