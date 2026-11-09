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


def _one_hot_edges(distances: list[int]) -> np.ndarray:
    """Box edge distributions whose expected bins are exactly the given distances."""
    logits = np.full((4, 16), -50.0, np.float32)
    logits[range(4), distances] = 50.0
    return logits.ravel()


def test_edge_distances_are_the_expected_bin(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    uniform = np.zeros(64, np.float32)

    np.testing.assert_allclose(accuracy.edge_distances(_one_hot_edges([1, 2, 3, 4])), [[1, 2, 3, 4]])
    np.testing.assert_allclose(accuracy.edge_distances(uniform), [[7.5] * 4])


def test_decode_maps_one_anchor_like_ultralytics(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = _one_hot_edges([1, 2, 3, 4])
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    np.testing.assert_allclose(rows[:, :5], [[42, *expected_box], [42, *expected_box]], atol=1e-6)
    np.testing.assert_allclose(rows[:, 5], [1 / (1 + np.exp(-3.0)), 0.5])
    np.testing.assert_array_equal(rows[:, 6], [3, 8])


def test_nms_suppresses_overlaps_within_a_class_only(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Two neighbouring stride-32 anchors predict nearly the same box for class 0; one also predicts class 1.
    for column, logit in ((5, 2.0), (6, 1.0)):
        outputs[2][0, 5, column] = _one_hot_edges([4, 4, 4, 4])
        outputs[5][0, 5, column, 0] = logit
    outputs[5][0, 5, 6, 1] = 0.5

    rows = accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert [(round(score, 4), label) for score, label in rows[:, 5:]] == [(0.8808, 1), (0.6225, 2)]


def test_decode_keeps_at_most_300_detections(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Every stride-8 anchor predicts a separate small box for one class.
    outputs[0][:] = np.tile(_one_hot_edges([0, 0, 1, 1]), (80, 80, 1))
    outputs[3][0, :, :, 0] = 2.0

    rows = accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert len(rows) == accuracy.MAX_DETECTIONS_PER_IMAGE == 300


def _reference_rows(accuracy: ModuleType, outputs: list[np.ndarray], categories: np.ndarray) -> np.ndarray:
    """Decode every anchor, score every pair with a sigmoid, and run a plain per-class greedy NMS."""
    distributions = np.concatenate([output.reshape(-1, 64) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:]]))
    distances = accuracy.edge_distances(distributions)
    corners = np.hstack([accuracy.ANCHORS - distances[:, :2], accuracy.ANCHORS + distances[:, 2:]])
    corners *= accuracy.ANCHOR_STRIDES
    anchor, label = np.nonzero(scores > accuracy.SCORE_THRESHOLD)
    pair_scores = scores[anchor, label]
    order = np.argsort(-pair_scores, kind="stable")[: accuracy.MAX_CANDIDATES]
    anchor, label, pair_scores = anchor[order], label[order], pair_scores[order]
    kept: list[int] = []
    for index in range(len(pair_scores)):
        same_class = [other for other in kept if label[other] == label[index]]
        if all(accuracy.iou(corners[anchor[index]], corners[anchor[[other]]])[0] <= 0.7 for other in same_class):
            kept.append(index)
            # Candidates are score-sorted; later ones cannot change the first 300 survivors.
            if len(kept) == accuracy.MAX_DETECTIONS_PER_IMAGE:
                break
    kept = sorted(kept, key=lambda index: -pair_scores[index])[:300]
    x1, y1, x2, y2 = accuracy.image_boxes(corners[anchor[kept]], 640, 427).T
    return np.stack(
        [np.full(len(kept), 9), x1, y1, x2 - x1, y2 - y1, pair_scores[kept], categories[label[kept]]], axis=1
    )


@pytest.mark.parametrize("seed", range(3))
def test_decode_matches_the_full_reference_selection(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(seed)
    outputs = [rng.normal(0, 2, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:]:
        logits = rng.normal(-12, 2, shape)
        hot = rng.random(shape[:-1]) < 0.01
        logits[hot] += rng.uniform(3, 15, (hot.sum(), 80))
        # INT8 steps make ties.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 9, 640, 427, categories)

    np.testing.assert_allclose(rows, _reference_rows(accuracy, outputs, categories), rtol=1e-9, atol=1e-9)


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


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
