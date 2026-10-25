from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["backbone", "decoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id
