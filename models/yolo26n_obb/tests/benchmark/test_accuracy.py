from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

# The polygon-overlap tests need the real shapely, which only the DevKit's accuracy environment installs.
NEEDS_SHAPELY = pytest.mark.skipif(importlib.util.find_spec("shapely") is None, reason="needs shapely")


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:6]:
        logits[:] = -20.0
    return outputs


@pytest.mark.parametrize("angle", [0.0, 0.4, -1.2, math.pi / 2])
def test_decode_places_the_anchor_at_its_side_distances(angle: float, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Stride 16, row 15, column 20: the anchor centre is (20.5, 15.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 15, 20] = [1, 2, 3, 4]
    outputs[4][0, 15, 20, 9] = 3.0
    outputs[7][0, 15, 20, 0] = angle

    ((x, y, w, h, theta, score, label),) = accuracy.decode_tile(outputs)

    assert (theta, label) == (pytest.approx(angle), 9)
    np.testing.assert_allclose(score, 1 / (1 + np.exp(-3.0)), rtol=1e-6)
    np.testing.assert_allclose([w, h], [4 * 16, 6 * 16], rtol=1e-6)
    # In the box's own frame the anchor lies 1, 2, 3 and 4 cells from its left, top, right and bottom sides.
    cos, sin = math.cos(angle), math.sin(angle)
    dx, dy = 20.5 * 16 - x, 15.5 * 16 - y
    along, across = dx * cos + dy * sin, -dx * sin + dy * cos
    np.testing.assert_allclose(
        [w / 2 + along, h / 2 + across, w / 2 - along, h / 2 - across], [16, 32, 48, 64], atol=1e-4
    )


def test_decode_keeps_at_most_the_top_300_detections(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, :2] = 2.0

    assert len(accuracy.decode_tile(outputs)) == accuracy.TOP_K == 300


def test_decode_returns_nothing_when_no_score_clears_the_threshold(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    assert accuracy.decode_tile(outputs).shape == (0, 7)
    # A logit just under the threshold, yet above LOGIT_FLOOR, is ranked but never kept.
    just_below = math.log(accuracy.SCORE_THRESHOLD / (1 - accuracy.SCORE_THRESHOLD)) - 0.001
    assert just_below > accuracy.LOGIT_FLOOR
    outputs[5][0, :2, :2, 3] = just_below

    assert accuracy.decode_tile(outputs).shape == (0, 7)


def _full_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics' get_topk_index with a sigmoid over every logit and stable sorts, then the score threshold."""
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 15) for output in outputs[3:6]]))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > accuracy.SCORE_THRESHOLD]
    return anchors[order // 15], order % 15, pairs[order]


@pytest.mark.parametrize("seed", range(4))
def test_decode_matches_the_full_sigmoid_selection(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape) < 0.01
        logits[hot] += rng.uniform(3, 30, hot.sum())
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 1, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[6:]]
    anchors, labels, scores = _full_selection(accuracy, outputs)
    angles = np.concatenate([output.reshape(-1) for output in outputs[6:]])

    detections = accuracy.decode_tile(outputs)

    np.testing.assert_array_equal(detections[:, 5], scores)
    np.testing.assert_array_equal(detections[:, 6], labels)
    np.testing.assert_array_equal(detections[:, 4], angles[anchors])


def test_top_indices_cut_a_stable_descending_sort(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top_indices(values, count), np.argsort(-values, kind="stable")[:count])


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (4, np.nan), (4, np.inf), (7, np.nan)])
def test_decode_rejects_non_finite_outputs(output: int, value: float, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 0] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode_tile(outputs)


def test_decode_rejects_another_output_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode_tile(outputs)


def test_image_detections_undo_the_letterbox_and_add_the_window_offset(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    # An 800x600 tile is scaled by 1.28 and padded by 128 rows above; a 824x1024 edge tile is padded by 100 columns.
    tile = np.array([[512.0, 400.0, 128.0, 64.0, 0.3, 0.9, 2.0]])

    scaled = accuracy.image_detections(tile, (0, 0, 1024, 1024), 800, 600)
    padded = accuracy.image_detections(tile, (466, 1113, 1490, 2137), 824, 1024)

    np.testing.assert_allclose(scaled, [[400, 212.5, 100, 50, 0.3, 0.9, 2]])
    np.testing.assert_allclose(padded, [[878, 1513, 128, 64, 0.3, 0.9, 2]])


def test_probiou_matches_the_bhattacharyya_closed_form(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    box = np.array([[50.0, 40.0, 12.0, 6.0, 0.0]])
    # Equal covariances leave only the mean term: D = dx^2 / (8 * w^2 / 12) = 1/6 for dx = 4.
    shifted = box + [4.0, 0, 0, 0, 0]

    np.testing.assert_allclose(accuracy.probiou(box, shifted), [[1 - math.sqrt(1 - math.exp(-1 / 6))]], atol=1e-6)
    assert accuracy.probiou(box, box)[0, 0] > 0.999
    assert accuracy.probiou(box, box + [500.0, 0, 0, 0, 0])[0, 0] < 1e-3


def test_probiou_is_symmetric_and_turns_with_the_boxes(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(1)
    first = np.column_stack([rng.uniform(0, 50, (6, 2)), rng.uniform(2, 20, (6, 2)), rng.uniform(-1.5, 1.5, 6)])
    second = np.column_stack([rng.uniform(0, 50, (4, 2)), rng.uniform(2, 20, (4, 2)), rng.uniform(-1.5, 1.5, 4)])

    def turned(boxes: np.ndarray, angle: float) -> np.ndarray:
        cos, sin = math.cos(angle), math.sin(angle)
        x, y = boxes[:, 0], boxes[:, 1]
        return np.column_stack([x * cos - y * sin, x * sin + y * cos, boxes[:, 2:4], boxes[:, 4] + angle])

    overlaps = accuracy.probiou(first, second)
    np.testing.assert_allclose(overlaps, accuracy.probiou(second, first).T, atol=1e-9)
    np.testing.assert_allclose(overlaps, accuracy.probiou(turned(first, 0.7), turned(second, 0.7)), atol=1e-6)


def test_fast_nms_lets_a_dropped_box_still_drop_others(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    # Box 2 overlaps box 1, which overlaps box 0, but box 0 does not reach box 2. Greedy NMS would keep 1, 0 and 3.
    boxes = np.array([[8.0, 0, 10, 10, 0], [0, 0, 10, 10, 0], [4, 0, 10, 10, 0], [40, 0, 10, 10, 0]])
    overlaps = accuracy.probiou(boxes, boxes)
    assert overlaps[1, 0] < accuracy.MERGE_IOU <= min(overlaps[1, 2], overlaps[2, 0])

    assert accuracy.fast_nms(boxes, np.array([0.7, 0.9, 0.8, 0.1])).tolist() == [1, 3]


def test_fast_nms_blocks_match_one_full_comparison(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(2)
    boxes = np.column_stack([rng.uniform(0, 200, (500, 2)), rng.uniform(5, 40, (500, 2)), rng.uniform(-1.5, 1.5, 500)])
    scores = rng.random(500)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.triu(accuracy.probiou(boxes[order], boxes[order]) >= accuracy.MERGE_IOU, k=1).any(axis=0)

    monkeypatch.setattr(accuracy, "MERGE_BLOCK", 64)

    np.testing.assert_array_equal(accuracy.fast_nms(boxes, scores), order[~suppressed])


def test_merge_suppresses_overlaps_within_a_class_only(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    detections = np.array(
        [
            [0.0, 0, 10, 10, 0, 0.9, 1],
            [1.0, 0, 10, 10, 0, 0.8, 1],
            [1.0, 0, 10, 10, 0, 0.7, 4],
        ]
    )

    np.testing.assert_array_equal(accuracy.merge(detections), detections[[0, 2]])
    assert accuracy.merge(detections[:0]).shape == (0, 7)


def test_corners_follow_ultralytics_point_order(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    boxes = np.array([[10.0, 20.0, 4.0, 2.0, 0.0], [0.0, 0.0, 4.0, 2.0, math.pi / 2]])

    np.testing.assert_allclose(
        accuracy.corners(boxes),
        [[[12, 21], [12, 19], [8, 19], [8, 21]], [[-1, 2], [1, 2], [1, -2], [-1, -2]]],
        atol=1e-12,
    )


def test_ground_truth_reads_classes_and_difficult_flags(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    content = b"imagesource:GoogleEarth\ngsd:0.1\n0 0 4 0 4 2 0 2 ship 1\n1 1 3 1 3 3 1 3 swimming-pool 0\n"

    polygons, labels, difficult = accuracy.ground_truth(content)

    np.testing.assert_array_equal(polygons[0], [[0, 0], [4, 0], [4, 2], [0, 2]])
    assert labels.tolist() == [1, 14] and difficult.tolist() == [True, False]


@NEEDS_SHAPELY
def test_best_overlaps_pick_the_first_of_equal_truths(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    square = np.array([[0.0, 0], [2, 0], [2, 2], [0, 2]])
    truths = np.stack([square + [1, 0], square - [1, 0], square + [10, 0]])
    detections = np.stack([square, square + [10, 0], square + [50, 0]])

    best, index = accuracy.best_overlaps(detections, truths)

    np.testing.assert_allclose(best, [1 / 3, 1, 0])
    assert index.tolist() == [0, 2, -1]


def test_class_ap_counts_dota_task1_matches(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    # In score order: a hit on truth 0, a duplicate of it, a difficult truth 2, no overlap, then truth 1.
    best = np.array([0.8, 0.7, 0.9, 0.0, 0.6])
    truth = np.array([0, 0, 2, -1, 1])
    difficult = np.array([False, False, True])

    def ap(threshold: float) -> float:
        return accuracy.class_ap(best, truth, difficult, 2, threshold)

    # Recall reaches 0.5 at precision 1 and 1.0 at precision 0.5; the difficult match counts neither way.
    assert ap(0.5) == pytest.approx((6 * 1 + 5 * 0.5) / 11)
    assert ap(0.65) == pytest.approx(6 / 11)
    assert ap(0.85) == 0
    # An overlap must exceed the threshold.
    assert ap(0.6) == pytest.approx(6 / 11)


@NEEDS_SHAPELY
def test_score_averages_ap_over_labelled_classes_and_thresholds(
    tmp_path: Path, load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    (tmp_path / "P0001.txt").write_text("gsd:0.1\n100 100 130 100 130 110 100 110 plane 0\n")
    (tmp_path / "P0002.txt").write_text("200 200 230 200 230 210 200 210 ship 0\n")
    detections = {
        # The plane is found exactly; the ship box is shifted 6 pixels, IoU 240 / 360.
        "P0001.jpg": np.array([[115.0, 105, 30, 10, 0, 0.9, 0]]),
        "P0002.jpg": np.array([[221.0, 205, 30, 10, 0, 0.8, 1], [221.0, 205, 30, 10, 0, 0.7, 2]]),
    }

    metrics, class_ap50 = accuracy.score(detections, tmp_path)

    # The ship clears 0.50 to 0.65, four of the ten thresholds; the unlabelled third class is not averaged.
    assert class_ap50 == {"plane": 1.0, "ship": 1.0}
    assert metrics == {"dota_obb_map50": 1.0, "dota_obb_map": pytest.approx((1 + 0.4) / 2)}


def test_score_finds_each_truth_across_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")

    def axis_aligned_overlaps(detections: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Every box here has angle 0, so axis-aligned IoU equals polygon IoU and CI needs no shapely.
        if not len(detections) or not len(truths):
            return np.zeros(len(detections)), np.full(len(detections), -1)
        found_low, found_high = detections.min(1)[:, None], detections.max(1)[:, None]
        truth_low, truth_high = truths.min(1), truths.max(1)
        overlap = (np.minimum(found_high, truth_high) - np.maximum(found_low, truth_low)).clip(0).prod(2)
        iou = overlap / ((found_high - found_low).prod(2) + (truth_high - truth_low).prod(1) - overlap)
        best = iou.max(1)
        return best, np.where(best > 0, iou.argmax(1), -1)

    monkeypatch.setattr(accuracy, "best_overlaps", axis_aligned_overlaps)
    (tmp_path / "P0001.txt").write_text("100 100 130 100 130 110 100 110 plane 0\n")
    (tmp_path / "P0002.txt").write_text(
        "200 200 230 200 230 210 200 210 plane 1\n0 0 30 0 30 10 0 10 ship 0\n300 300 330 300 330 310 300 310 plane 0\n"
    )
    detections = {
        # A ship where P0001 labels none ranks first, so it is the ship's only miss.
        "P0001.jpg": np.array([[115.0, 105, 30, 10, 0, 0.9, 0], [500.0, 500, 30, 10, 0, 0.95, 1]]),
        # P0002's truths follow P0001's, its difficult plane first: a truth taken from the wrong image or line
        # turns the ignored match into a duplicate and the found ship into a difficult one.
        "P0002.jpg": np.array(
            [[215.0, 205, 30, 10, 0, 0.8, 0], [315.0, 305, 30, 10, 0, 0.7, 0], [15.0, 5, 30, 10, 0, 0.6, 1]]
        ),
    }

    metrics, class_ap50 = accuracy.score(detections, tmp_path)

    assert class_ap50 == {"plane": 1.0, "ship": 0.5}
    assert metrics == {"dota_obb_map50": 0.75, "dota_obb_map": 0.75}


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
