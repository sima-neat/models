from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/fresunet_mpk.tar.gz"]) == Path("/data/fresunet_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/fresunet_mpk.tar.gz"])


def test_accuracy_scores_the_test_split_of_oscd(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert (script.DATASET, script.SPLIT) == ("oscd-default", "test")


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_accuracy_scores_every_test_city(accuracy: dict[str, Any]) -> None:
    (result,) = accuracy["datasets"].values()
    assert result["evaluated_samples"] == result["available_samples"] == 10
