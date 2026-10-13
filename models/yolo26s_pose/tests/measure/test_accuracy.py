from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_image_boxes_invert_the_letterbox_and_clip(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(500, 375)
    original = np.array([[10.0, 20.0, 110.0, 220.0], [-5.0, -1.0, 600.0, 400.0]])

    mapped = accuracy.image_boxes(original * gain + [left, top, left, top], 500, 375)

    np.testing.assert_allclose(mapped, [[10, 20, 110, 220], [0, 0, 500, 375]])


def test_image_points_invert_the_letterbox_and_clip(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(500, 375)
    original = np.array([[10.0, 20.0], [-5.0, 400.0], [510.0, -3.0]])

    mapped = accuracy.image_points(original * gain + [left, top], 500, 375)

    np.testing.assert_allclose(mapped, [[10, 20], [0, 375], [500, 0]])


def test_scored_images_are_those_with_labelled_keypoints(
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    ground_truth = MagicMock()
    ground_truth.dataset = {
        "annotations": [
            {"image_id": 5, "num_keypoints": 3},
            {"image_id": 5, "num_keypoints": 0},
            {"image_id": 2, "num_keypoints": 17},
            {"image_id": 9, "num_keypoints": 0},
        ]
    }
    monkeypatch.setattr(accuracy, "SCORED_IMAGES", 2)

    assert accuracy.scored_image_ids(ground_truth) == [2, 5]
    monkeypatch.setattr(accuracy, "SCORED_IMAGES", 2346)
    with pytest.raises(SystemExit, match="expected 2346 images"):
        accuracy.scored_image_ids(ground_truth)


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:6]:
        logits[:] = -20.0
    return outputs


def test_decode_maps_boxes_and_keypoints_like_ultralytics(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    # Stride 16, row 15, column 20: the anchor centre is (20.5, 15.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 15, 20] = [1, 2, 3, 4]
    outputs[4][0, 15, 20, 0] = 3.0
    # Keypoint 0 is one cell right of and two cells above the centre; keypoint 16 sits on it, visibility logit 2.
    outputs[7][0, 15, 20, 0:3] = [1.0, -2.0, 0.0]
    outputs[7][0, 15, 20, 48:51] = [0.0, 0.0, 2.0]
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(1280, 854)

    (person,) = accuracy.decode(outputs, 42, 1280, 854, 1)

    def original(x: float, y: float) -> list[float]:
        return [(x - left) / gain, (y - top) / gain]

    x1, y1 = original((20.5 - 1) * 16, (15.5 - 2) * 16)
    x2, y2 = original((20.5 + 3) * 16, (15.5 + 4) * 16)
    assert person["image_id"] == 42 and person["category_id"] == 1
    np.testing.assert_allclose(person["bbox"], [x1, y1, x2 - x1, y2 - y1])
    np.testing.assert_allclose(person["score"], 1 / (1 + np.exp(-3.0)))
    keypoints = np.reshape(person["keypoints"], (17, 3))
    np.testing.assert_allclose(keypoints[0], [*original(21.5 * 16, 13.5 * 16), 0.5])
    np.testing.assert_allclose(keypoints[16], [*original(20.5 * 16, 15.5 * 16), 1 / (1 + np.exp(-2.0))])
    np.testing.assert_allclose(keypoints[1, :2], original(20.5 * 16, 15.5 * 16))


def test_decode_keeps_at_most_the_top_300_people(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, 0] = 2.0

    people = accuracy.decode(outputs, 1, 640, 640, 1)

    assert len(people) == accuracy.TOP_K == 300


def test_decode_returns_no_people_when_none_clears_the_threshold(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    assert accuracy.decode(outputs, 1, 640, 640, 1) == []
    # sigmoid(-6.95) is just below SCORE_THRESHOLD.
    outputs[5][0, :2, :2, 0] = -6.95

    assert accuracy.decode(outputs, 1, 640, 640, 1) == []


def _full_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Select with a sigmoid over every logit and a stable sort, the straightforward form of the selection."""
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1) for output in outputs[3:6]]))
    order = np.argsort(-scores, kind="stable")[:300]
    order = order[scores[order] > 0.001]
    return order, scores[order]


@pytest.mark.parametrize("seed", range(4))
def test_decode_matches_the_full_sigmoid_selection(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape) < 0.03
        logits[hot] += rng.uniform(3, 30, hot.sum())
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 2, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[6:]]
    anchors, scores = _full_selection(accuracy, outputs)
    centres = accuracy.ANCHORS[anchors]
    strides = accuracy.ANCHOR_STRIDES[anchors]
    raw = np.concatenate([output.reshape(-1, 17, 3) for output in outputs[6:]])[anchors]

    people = accuracy.decode(outputs, 9, 640, 427, 1)

    np.testing.assert_array_equal([person["score"] for person in people], scores)
    keypoints = np.array([person["keypoints"] for person in people]).reshape(-1, 17, 3)
    expected = accuracy.image_points(((raw[..., :2] + centres[:, None]) * strides[:, None]).reshape(-1, 2), 640, 427)
    np.testing.assert_allclose(keypoints[..., :2].reshape(-1, 2), expected)
    np.testing.assert_allclose(keypoints[..., 2], accuracy.sigmoid(raw[..., 2]))


def test_top_cuts_a_stable_descending_sort(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top(values, count), np.argsort(-values, kind="stable")[:count])


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (4, np.nan), (4, np.inf), (7, np.nan)])
def test_decode_rejects_non_finite_outputs(output: int, value: float, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 0] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 640, 1)


def test_decode_rejects_another_output_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 640, 1)


# Lowest accepted value of each required metric per build profile: the build's latest result in the
# develop catalog at b64f221, less 0.01, rounded down.
MINIMUM_ACCURACY = {
    "modalix_bf16": {"coco_keypoint_map": 0.617},
    "modalix_int8": {"coco_keypoint_map": 0.5819},
}


def test_accuracy_meets_its_minimum(accuracy: dict[str, Any], build_profile: str) -> None:
    assert build_profile in MINIMUM_ACCURACY, f"no accuracy minimum for {build_profile}"
    for name, minimum in MINIMUM_ACCURACY[build_profile].items():
        value = accuracy["metrics"][name]["value"]
        assert value >= minimum, f"{name} {value:.4f} is below its minimum {minimum}"
