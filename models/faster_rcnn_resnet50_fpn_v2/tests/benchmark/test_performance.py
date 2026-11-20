from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from model_registry.model_tests import measured_shortfalls


def test_the_letterboxed_frame_fills_the_input_width(load_script: Callable[[str], ModuleType]) -> None:
    window = load_script("performance").letterbox_window()

    assert (window.left, window.right, window.scale) == (0, 1344, 1.05)
    assert (window.top, window.bottom) == pytest.approx((294, 1050))


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


@pytest.mark.parametrize("component", ["backbone", "box_head"])
def test_performance_records_each_components_plan_and_mla_stats(performance: dict[str, Any], component: str) -> None:
    measured = performance["components"][component]
    plan = measured["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]
    layer_stats, stage_stats = measured["mla_layer_stats"], measured["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"
