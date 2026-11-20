from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_benchmark_picks_a_seeded_test_city(load_script: Callable[[str], ModuleType], tmp_path: Path) -> None:
    script = load_script("performance")
    root = tmp_path / "OSCD"
    root.mkdir()
    (root / "test.txt").write_text("brasilia,montpellier,norcia,rio\n")
    picked: list[str] = []
    script.city_input = lambda _root, city: picked.append(city) or city

    workload, sample = script.dataset_sample(tmp_path)

    expected = random.Random(0).choice(["brasilia", "montpellier", "norcia", "rio"])
    assert picked == [expected] and sample == expected
    assert workload == f"oscd-default test city {expected}"


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"
