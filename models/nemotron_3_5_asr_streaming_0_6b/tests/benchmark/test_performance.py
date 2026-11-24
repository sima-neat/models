from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_transcribes_faster_than_real_time(performance: dict[str, Any]) -> None:
    metrics = performance["metrics"]
    assert 0 < metrics["real_time_factor"] < 1
    assert metrics["time_to_first_text_ms"] > 0
