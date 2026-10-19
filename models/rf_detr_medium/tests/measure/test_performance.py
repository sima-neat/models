from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert [item["component_id"] for item in performance["components"]] == ["backbone", "decoder"]
    for item in performance["components"]:
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_performance_records_each_components_layer_timings(performance: dict[str, Any]) -> None:
    for item in performance["components"]:
        layers = item["layer_stats"]
        assert any(layer["total_execution_time_us"] > 0 for layer in layers), item["component_id"]
