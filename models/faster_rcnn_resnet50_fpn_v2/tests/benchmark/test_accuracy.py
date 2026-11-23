from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from model_registry.model_tests import measured_shortfalls


def test_results_map_boxes_back_to_the_original_image(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    found = (np.array([[100, 50, 300, 250]], np.float32), np.array([0.9], np.float32), np.array([18]))

    rows = accuracy.result_rows(7, 640, 480, (800, 1066), found)

    np.testing.assert_allclose(rows[0, 1:5], [100 * 640 / 1066, 50 * 0.6, 200 * 640 / 1066, 200 * 0.6], rtol=1e-6)
    assert rows[0, 0] == 7 and rows[0, 6] == 18


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
