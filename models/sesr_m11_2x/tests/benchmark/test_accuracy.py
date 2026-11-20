from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_accuracy_scores_every_validation_image(accuracy: dict[str, Any]) -> None:
    (result,) = accuracy["datasets"].values()
    assert result["evaluated_samples"] == result["available_samples"] == 100
