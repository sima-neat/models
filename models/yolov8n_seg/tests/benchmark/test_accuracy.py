from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "frPyObjects", lambda rles, height, width: rles)
    return module


def test_image_boxes_invert_the_letterbox_and_clip(accuracy_script: ModuleType) -> None:
    gain, _, (left, top, _, _) = accuracy_script.letterbox_geometry(500, 375)
    original = np.array([[10.0, 20.0, 110.0, 220.0], [-5.0, -1.0, 600.0, 400.0]])

    mapped = accuracy_script.image_boxes(original * gain + [left, top, left, top], 500, 375)

    np.testing.assert_allclose(mapped, [[10, 20, 110, 220], [0, 0, 500, 375]])


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:6]:
        logits[:] = -20.0
    return outputs


def _one_hot_edges(distances: list[int]) -> np.ndarray:
    """Box edge distributions whose expected bins are exactly the given distances."""
    logits = np.full((4, 16), -50.0, np.float32)
    logits[range(4), distances] = 50.0
    return logits.ravel()


def _rle_image(rle: dict) -> np.ndarray:
    """Expand an uncompressed column-major COCO RLE, background first."""
    height, width = rle["size"]
    values = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).astype(bool)
    return values.reshape(width, height).T


def test_edge_distances_are_the_expected_bin(accuracy_script: ModuleType) -> None:
    uniform = np.zeros(64, np.float32)

    np.testing.assert_allclose(accuracy_script.edge_distances(_one_hot_edges([1, 2, 3, 4])), [[1, 2, 3, 4]])
    np.testing.assert_allclose(accuracy_script.edge_distances(uniform), [[7.5] * 4], rtol=1e-6)


def test_decode_maps_one_anchor_like_ultralytics(accuracy_script: ModuleType) -> None:
    outputs = _board_outputs(accuracy_script)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = _one_hot_edges([1, 2, 3, 4])
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    results = accuracy_script.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    assert [item["image_id"] for item in results] == [42, 42]
    np.testing.assert_allclose([item["bbox"] for item in results], [expected_box, expected_box], atol=1e-5)
    np.testing.assert_allclose([item["score"] for item in results], [1 / (1 + np.exp(-3.0)), 0.5], rtol=1e-6)
    assert [item["category_id"] for item in results] == [3, 8]


def test_nms_suppresses_overlaps_within_a_class_only(accuracy_script: ModuleType) -> None:
    outputs = _board_outputs(accuracy_script)
    # Two neighbouring stride-32 anchors predict nearly the same box for class 0; one also predicts class 1.
    for column, logit in ((5, 2.0), (6, 1.0)):
        outputs[2][0, 5, column] = _one_hot_edges([4, 4, 4, 4])
        outputs[5][0, 5, column, 0] = logit
    outputs[5][0, 5, 6, 1] = 0.5

    results = accuracy_script.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert [(round(item["score"], 4), item["category_id"]) for item in results] == [(0.8808, 1), (0.6225, 2)]


def test_nms_offsets_classes_in_float32_like_ultralytics(accuracy_script: ModuleType) -> None:
    # These boxes overlap at IoU 0.700035. Ultralytics' float32 class offset rounds class 79's coordinates to 1/16
    # pixel, which keeps both; for class 0 the overlap stays above the threshold and NMS keeps one.
    corners = np.array([[100, 100, 200, 200], [100, 100, 200, 242.85]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)

    for label, kept in ((79, 2), (0, 1)):
        assert len(accuracy_script.nms(accuracy_script.class_aware(corners, np.full(2, label)), scores)) == kept


def test_decode_keeps_at_most_300_detections(accuracy_script: ModuleType) -> None:
    outputs = _board_outputs(accuracy_script)
    # Every stride-8 anchor predicts a separate small box for one class.
    outputs[0][:] = np.tile(_one_hot_edges([0, 0, 1, 1]), (80, 80, 1))
    outputs[3][0, :, :, 0] = 2.0

    boxes, scores, labels, coefficients = accuracy_script.select(outputs)

    assert len(boxes) == len(scores) == len(labels) == len(coefficients) == 300
    assert accuracy_script.MAX_DETECTIONS_PER_IMAGE == 300


def _reference_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Decode every anchor, score every pair with a sigmoid, and run a plain per-class greedy NMS."""
    distributions = np.concatenate([output.reshape(-1, 64) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:6]]))
    coefficients = np.concatenate([output.reshape(-1, 32) for output in outputs[6:9]])
    distances = accuracy.edge_distances(distributions)
    corners = np.hstack([accuracy.ANCHORS - distances[:, :2], accuracy.ANCHORS + distances[:, 2:]])
    corners *= accuracy.ANCHOR_STRIDES
    overlap_corners = corners.astype(np.float64)
    anchor, label = np.nonzero(scores > accuracy.SCORE_THRESHOLD)
    pair_scores = scores[anchor, label]
    order = np.argsort(-pair_scores, kind="stable")[: accuracy.MAX_CANDIDATES]
    anchor, label, pair_scores = anchor[order], label[order], pair_scores[order]
    kept: list[int] = []
    for index in range(len(pair_scores)):
        same_class = [other for other in kept if label[other] == label[index]]
        if all(
            accuracy.iou(overlap_corners[anchor[index]], overlap_corners[anchor[[other]]])[0] <= 0.7
            for other in same_class
        ):
            kept.append(index)
            # Scores are descending; lower-ranked candidates cannot change the first 300 greedy-NMS results.
            if len(kept) == accuracy.MAX_DETECTIONS_PER_IMAGE:
                break
    kept = sorted(kept, key=lambda index: -pair_scores[index])[:300]
    return corners[anchor[kept]], pair_scores[kept], label[kept], coefficients[anchor[kept]]


@pytest.mark.parametrize("seed", range(3))
def test_select_matches_the_full_reference_selection(seed: int, accuracy_script: ModuleType) -> None:
    rng = np.random.default_rng(seed)
    outputs = [rng.normal(0, 2, shape).astype(np.float32) for shape in accuracy_script.OUTPUT_SHAPES[:3]]
    for shape in accuracy_script.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-12, 2, shape)
        hot = rng.random(shape[:-1]) < 0.01
        logits[hot] += rng.uniform(3, 15, (hot.sum(), 80))
        # INT8 steps make ties.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 1, shape).astype(np.float32) for shape in accuracy_script.OUTPUT_SHAPES[6:]]

    for actual, expected in zip(accuracy_script.select(outputs), _reference_selection(accuracy_script, outputs)):
        np.testing.assert_array_equal(actual, expected)


def test_logit_floor_cannot_reach_the_score_threshold(accuracy_script: ModuleType) -> None:
    assert accuracy_script.sigmoid(np.float32(accuracy_script.LOGIT_FLOOR)) < accuracy_script.SCORE_THRESHOLD


@pytest.mark.parametrize("output", [0, 4, 7, 9])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_decode_rejects_non_finite_outputs(accuracy_script: ModuleType, output: int, value: float) -> None:
    outputs = _board_outputs(accuracy_script)
    outputs[output][0, 1, 2, 3] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy_script.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_decode_rejects_another_output_layout(accuracy_script: ModuleType) -> None:
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy_script)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy_script.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_resize_sources_follow_torch_bilinear_without_corner_alignment(accuracy_script: ModuleType) -> None:
    lower, upper, weight = accuracy_script.resize_sources(160, 640)

    # Target pixel i samples source position (i + 0.5) / 4 - 0.5, clamped to the first and last pixel.
    position = np.clip((np.arange(640) + 0.5) / 4 - 0.5, 0, 159)
    np.testing.assert_allclose((1 - weight) * lower + weight * upper, position)
    assert (upper - lower <= 1).all() and (upper[weight == 0] == lower[weight == 0]).all()
    lower, upper, weight = accuracy_script.resize_sources(426, 333)
    position = np.clip((np.arange(333) + 0.5) * 426 / 333 - 0.5, 0, 425)
    np.testing.assert_allclose((1 - weight) * lower + weight * upper, position, atol=1e-4)


def test_crop_bounds_round_and_wrap_like_slices_below_fifty_masks_and_use_exact_bounds_above(
    accuracy_script: ModuleType,
) -> None:
    boxes = np.array([[10.4, 10.6, 100.5, 639.7], [-0.6, -0.4, 700.0, 50.2]])

    # Rounded slice ends: -0.6 rounds to -1, which a slice reads as 639.
    np.testing.assert_array_equal(accuracy_script.crop_bounds(boxes), [[10, 11, 100, 640], [639, 0, 640, 50]])
    many = np.repeat(boxes, 25, axis=0)
    np.testing.assert_array_equal(accuracy_script.crop_bounds(many)[[0, -1]], [[11, 11, 101, 640], [0, 0, 640, 51]])


@pytest.mark.parametrize(("rows", "columns"), [((2, 6), (3, 8)), ((3, 7), (2, 6)), ((0, 7), (0, 9)), ((3, 7), (6, 9))])
def test_run_lengths_encode_column_major_with_background_first(
    accuracy_script: ModuleType, rows: tuple[int, int], columns: tuple[int, int]
) -> None:
    rng = np.random.default_rng(0)
    image = np.zeros((7, 9), dtype=bool)
    region = (slice(*rows), slice(*columns))
    image[region] = rng.random((rows[1] - rows[0], columns[1] - columns[0])) > 0.4
    # Runs that start the region and reach its end: the bottom row, before or at the image's last pixel.
    image[rows[0], columns[0]] = image[rows[1] - 1, columns[1] - 1] = True

    counts = accuracy_script.run_lengths(image[region], rows[0], columns[0], 7, 9)

    assert sum(counts) == 63 and all(counts[1:])
    np.testing.assert_array_equal(_rle_image({"size": [7, 9], "counts": counts}), image)


def test_a_positive_mask_over_the_whole_frame_sets_every_image_pixel(accuracy_script: ModuleType) -> None:
    proto = np.ones((160, 160, 32), np.float32)

    (rle,) = accuracy_script.instance_masks(proto, np.array([[0.0, 0.0, 640.0, 640.0]]), np.ones((1, 32)), 640, 427)

    assert rle["counts"] == [0, 640 * 427]


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
