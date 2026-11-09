from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    (frame,) = component["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"


def test_performance_reads_the_directory_the_runner_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The runner looks through the archive's one top-level directory, so images/ sits directly under --dataset."""
    arguments = ["--mpk", "model=m.tar.gz", "--dataset", f"pitts30k-val={tmp_path}", "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["performance.py", *arguments])
    # performance.py parses its arguments and runs as it loads.
    with pytest.raises(SystemExit, match=f"{tmp_path} has no images under images/"):
        load_script("performance")
