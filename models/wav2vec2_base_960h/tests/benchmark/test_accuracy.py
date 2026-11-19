"""Check the complete FLEURS corpus and the measured accuracy baseline."""

from pathlib import Path

from model_registry.model_tests import measured_shortfalls


def test_accuracy_covers_the_full_fleurs_test_set(accuracy):
    result = accuracy["datasets"]["fleurs-en-us-test"]
    assert result["split"] == "test"
    assert result["available_samples"] == result["evaluated_samples"] == 647


def test_accuracy_meets_baseline(accuracy, model_dir: Path, build_profile: str):
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_accuracy_gate_rejects_regression(model_dir: Path):
    report = {"datasets": {"fleurs-en-us-test": {"metrics": {"wer": 0.5, "cer": 0.3}}}}
    assert measured_shortfalls(model_dir, "accuracy", "w8a16", report)
