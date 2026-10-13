from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_image_boxes_invert_the_letterbox_and_clip(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(500, 375)
    original = np.array([[10.0, 20.0, 110.0, 220.0], [-5.0, -1.0, 600.0, 400.0]])

    mapped = accuracy.image_boxes(original * gain + [left, top, left, top], 500, 375)

    np.testing.assert_allclose(mapped, [[10, 20, 110, 220], [0, 0, 500, 375]])


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:]:
        logits[:] = -20.0
    return outputs


def test_decode_selects_pairs_and_maps_boxes_like_ultralytics(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = [1, 2, 3, 4]
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    np.testing.assert_allclose(rows[:, :5], [[42, *expected_box], [42, *expected_box]])
    np.testing.assert_allclose(rows[:, 5], [1 / (1 + np.exp(-3.0)), 0.5])
    np.testing.assert_array_equal(rows[:, 6], [3, 8])


def test_decode_keeps_at_most_the_top_300_pairs(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, 0] = 2.0

    rows = accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert len(rows) == accuracy.TOP_K == 300


def _full_selection_rows(accuracy: ModuleType, outputs: list[np.ndarray], categories: np.ndarray) -> np.ndarray:
    """Decode with a sigmoid over every logit and two stable sorts, the straightforward form of the selection."""
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:]]))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > 0.001]
    anchor, label = anchors[order // 80], order % 80
    centres = accuracy.ANCHORS[anchor]
    corners = np.hstack([centres - distances[anchor, :2], centres + distances[anchor, 2:]])
    x1, y1, x2, y2 = accuracy.image_boxes(corners * accuracy.ANCHOR_STRIDES[anchor], 640, 427).T
    return np.stack([np.full(order.shape, 9), x1, y1, x2 - x1, y2 - y1, pairs[order], categories[label]], axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_decode_matches_the_full_sigmoid_selection(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape[:-1]) < 0.03
        logits[hot] += rng.uniform(3, 30, (hot.sum(), 80))
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 9, 640, 427, categories)

    np.testing.assert_array_equal(rows, _full_selection_rows(accuracy, outputs, categories))


def test_top_cuts_a_stable_descending_sort(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top(values, count), np.argsort(-values, kind="stable")[:count])


def test_logit_floor_cannot_reach_the_score_threshold(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.sigmoid(np.float32(accuracy.LOGIT_FLOOR)) < accuracy.SCORE_THRESHOLD


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (4, np.nan), (4, np.inf)])
def test_decode_rejects_non_finite_outputs(output: int, value: float, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 3] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_decode_rejects_another_output_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "accuracy", accuracy) == []
