from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def scorer(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_each_query_keeps_its_best_class_other_than_no_object(scorer: ModuleType) -> None:
    logits = np.zeros((1, scorer.QUERY_COUNT, scorer.CLASSES), dtype=np.float32)
    logits[0, 0, 17] = 4.0
    # "No object" wins this query, so the detection takes the best real class and a low score.
    logits[0, 1, -1], logits[0, 1, 3] = 9.0, 2.0
    boxes = np.full((1, scorer.QUERY_COUNT, 4), 0.5, dtype=np.float32)
    rows = scorer.decode(logits, boxes, image_id=7, width=640, height=480)
    assert rows.shape == (scorer.QUERY_COUNT, 7)
    assert rows[0, 6] == 17 and rows[1, 6] == 3
    probabilities = np.exp(logits[0, 1]) / np.exp(logits[0, 1]).sum()
    assert rows[1, 5] == pytest.approx(probabilities[3])
    assert rows[0, 5] > rows[1, 5]


def test_boxes_scale_from_fractions_of_the_input_to_image_pixels(scorer: ModuleType) -> None:
    logits = np.zeros((1, scorer.QUERY_COUNT, scorer.CLASSES), dtype=np.float32)
    boxes = np.zeros((1, scorer.QUERY_COUNT, 4), dtype=np.float32)
    boxes[0, 0] = [0.25, 0.5, 0.5, 0.2]
    (x, y, w, h) = scorer.decode(logits, boxes, image_id=1, width=800, height=600)[0, 1:5]
    assert (x, y, w, h) == pytest.approx((0.0, 240.0, 400.0, 120.0))


@pytest.mark.parametrize("shape", [(1, 100, 91), (1, 300, 92)])
def test_decode_rejects_other_output_shapes(scorer: ModuleType, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        scorer.decode(np.zeros(shape, np.float32), np.zeros((1, shape[1], 4), np.float32), 1, 10, 10)
