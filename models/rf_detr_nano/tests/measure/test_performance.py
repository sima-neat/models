from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", performance) == []


def test_pipeline_meets_its_expected_values(pipeline: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", pipeline) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    plan = performance["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_performance_records_compiled_layer_timings(performance: dict[str, Any]) -> None:
    layers = performance["benchmark_report"]["layer_stats"]
    assert layers, "Core returned no MLA layer timings"
    assert any(layer["total_execution_time_us"] > 0 for layer in layers)
