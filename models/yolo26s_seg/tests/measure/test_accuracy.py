from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


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


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "frPyObjects", lambda rles, height, width: rles)
    return module


def _rle_image(rle: dict) -> np.ndarray:
    """Expand an uncompressed column-major COCO RLE, background first."""
    height, width = rle["size"]
    values = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).astype(bool)
    return values.reshape(width, height).T


def test_decode_selects_pairs_and_maps_boxes_like_ultralytics(accuracy_script: ModuleType) -> None:
    outputs = _board_outputs(accuracy_script)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = [1, 2, 3, 4]
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    results = accuracy_script.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    assert [item["image_id"] for item in results] == [42, 42]
    np.testing.assert_allclose([item["bbox"] for item in results], [expected_box, expected_box])
    np.testing.assert_allclose([item["score"] for item in results], [1 / (1 + np.exp(-3.0)), 0.5])
    assert [item["category_id"] for item in results] == [3, 8]


def test_decode_keeps_at_most_the_top_300_pairs(accuracy_script: ModuleType) -> None:
    outputs = _board_outputs(accuracy_script)
    outputs[3][0, :5, :, 0] = 2.0

    results = accuracy_script.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert len(results) == accuracy_script.TOP_K == 300


def _full_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Select with a sigmoid over every logit and two stable sorts, the straightforward form of the selection."""
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:6]]))
    coefficients = np.concatenate([output.reshape(-1, 32) for output in outputs[6:9]])
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > 0.001]
    anchor, label = anchors[order // 80], order % 80
    centres = accuracy.ANCHORS[anchor]
    corners = np.hstack([centres - distances[anchor, :2], centres + distances[anchor, 2:]])
    return corners * accuracy.ANCHOR_STRIDES[anchor], pairs[order], label, coefficients[anchor]


@pytest.mark.parametrize("seed", range(4))
def test_select_matches_the_full_sigmoid_selection(accuracy_script: ModuleType, seed: int) -> None:
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy_script.OUTPUT_SHAPES[:3]]
    for shape in accuracy_script.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape[:-1]) < 0.03
        logits[hot] += rng.uniform(3, 30, (hot.sum(), 80))
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 1, shape).astype(np.float32) for shape in accuracy_script.OUTPUT_SHAPES[6:]]

    for actual, expected in zip(accuracy_script.select(outputs), _full_selection(accuracy_script, outputs)):
        np.testing.assert_array_equal(actual, expected)


def test_top_cuts_a_stable_descending_sort(accuracy_script: ModuleType) -> None:
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy_script.top(values, count), np.argsort(-values, kind="stable")[:count])


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


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "accuracy", accuracy) == []
