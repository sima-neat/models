from __future__ import annotations

from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", performance) == []
