from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
from model_registry.accuracy import (
    AccuracyError,
    _classification_scores,
    convert_sample_layout,
    index_split,
    preprocess_image,
    select_sample_indices,
)
from model_registry.accuracy_config import select_accuracy_config, validate_accuracy_model_contract
from model_registry.cli import main
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "resnet_50"


def test_resnet_accuracy_config_is_executable_and_schema_valid() -> None:
    model = load_and_validate_model(MODEL_DIR)
    path, config = select_accuracy_config(MODEL_DIR, model)

    assert path == MODEL_DIR / "tests" / "correctness.yaml"
    assert config["engine"] == "model_registry"
    assert config["status"] == "provisional"
    assert model["tests"]["correctness"][0]["build_profiles"] == ["modalix_int8"]
    assert config["profiles"]["regular"]["samples"] == 500
    assert config["profiles"]["qualification"]["samples"] == "all"


def test_static_accuracy_contract_rejects_unsupported_model_operation() -> None:
    model = load_and_validate_model(MODEL_DIR)
    path, config = select_accuracy_config(MODEL_DIR, model)
    model["components"][0]["preprocessing"]["pipeline"].append(
        {"op": "run_arbitrary_python", "parameters": {}}
    )

    with pytest.raises(ValidationError, match="unsupported preprocessing operation"):
        validate_accuracy_model_contract(model, config, path, model["tests"]["correctness"][0])


def test_evaluator_rejects_profile_outside_accuracy_scope(tmp_path: Path) -> None:
    from model_registry.accuracy import evaluate_accuracy

    with pytest.raises(AccuracyError, match="does not apply to build profile 'modalix_bf16'"):
        evaluate_accuracy(MODEL_DIR, "modalix_bf16", "model", tmp_path / "candidate.sima")


def test_declarative_preprocessing_matches_resnet_contract() -> None:
    model = load_and_validate_model(MODEL_DIR)
    image = Image.new("RGB", (320, 240), color=(255, 0, 128))
    stream = io.BytesIO()
    image.save(stream, format="WEBP", lossless=True)

    result = preprocess_image(stream.getvalue(), model["components"][0]["preprocessing"]["pipeline"])

    assert result.shape == (3, 224, 224)
    assert result.dtype == np.float32
    expected = (np.asarray([1.0, 0.0, 128 / 255], dtype=np.float32) - [0.485, 0.456, 0.406]) / [
        0.229,
        0.224,
        0.225,
    ]
    assert result[:, 112, 112] == pytest.approx(expected, abs=1e-5)


def test_layout_conversion_preserves_explicit_tensor_contract() -> None:
    sample = np.zeros((3, 224, 224), dtype=np.float32)

    nchw = convert_sample_layout(sample, "NCHW", "NCHW")
    nhwc = convert_sample_layout(sample, "NCHW", "NHWC")

    assert nchw.shape == (1, 3, 224, 224)
    assert nhwc.shape == (1, 224, 224, 3)


def test_classification_postprocessing_enforces_metric_top_k() -> None:
    pipeline = (
        {"op": "softmax", "parameters": {"axis": 1}},
        {"op": "top_k", "parameters": {"k": 5}},
    )
    scores = _classification_scores(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), pipeline, 5)

    assert scores.sum() == pytest.approx(1.0)
    with pytest.raises(AccuracyError, match="smaller than requested"):
        _classification_scores(scores, ({"op": "top_k", "parameters": {"k": 1}},), 5)


def test_smoke_selection_is_deterministic_and_spans_validation_split() -> None:
    first = select_sample_indices(5000, 500)
    second = select_sample_indices(5000, 500)

    assert first == second
    assert len(first) == len(set(first)) == 500
    assert first[0] == 0
    assert first[-1] == 4999


def test_sampling_rejects_more_than_available() -> None:
    with pytest.raises(AccuracyError, match="1..5000"):
        select_sample_indices(5000, 5001)


def test_dataset_resolution_falls_back_to_current_python_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from model_registry import accuracy

    dataset = tmp_path / "dataset.tgz"
    dataset.write_bytes(b"registered dataset")
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"path": str(dataset)}), stderr="")

    monkeypatch.setattr(accuracy.shutil, "which", lambda _name: None)
    monkeypatch.setattr(accuracy.sys, "executable", "/missing/compiler/bin/python")
    monkeypatch.setattr(accuracy.subprocess, "run", run)
    config = {
        "id": "fixture",
        "variant": "mini",
        "sha256": accuracy.sha256_file(dataset),
        "size_bytes": dataset.stat().st_size,
    }

    assert accuracy.resolve_dataset(config, None) == dataset
    assert commands[0][:3] == ["/missing/compiler/bin/python", "-m", "dataset_registry"]


def test_dataset_index_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tgz"
    payload = b"not-an-image"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("../escape.webp")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    config = {
        "loader": {"image_suffixes": [".webp"]},
        "splits": {"validation": {"prefix": "val", "samples": 1}},
    }

    with pytest.raises(AccuracyError, match="unsafe dataset archive member"):
        index_split(archive_path, config, "validation")


def test_evaluate_cli_writes_generic_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    report = {
        "model": {"id": "resnet_50", "build_profile_id": "modalix_int8", "component_id": "model"},
        "dataset": {"evaluated_samples": 500, "available_samples": 5000},
        "metrics": {"candidate": {"top1_accuracy": 0.76}},
    }
    monkeypatch.setattr("model_registry.cli.evaluate_accuracy", lambda *args, **kwargs: report)
    output = tmp_path / "accuracy.json"

    result = main(
        [
            "evaluate",
            str(MODEL_DIR),
            "--build-profile",
            "modalix_int8",
            "--component",
            "model",
            "--candidate-model",
            str(tmp_path / "candidate.sima"),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "samples=500/5000" in capsys.readouterr().out
