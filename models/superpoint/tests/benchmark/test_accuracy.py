from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

SHIFT = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])


def _features(accuracy: ModuleType, count: int = 300, seed: int = 0) -> tuple[tuple, tuple]:
    """Image-0 keypoints on a coarse grid and their exact images under SHIFT, with matching unit descriptors."""
    rng = np.random.default_rng(seed)
    cells = rng.permutation(30 * 20)[:count]
    keypoints0 = np.stack([cells % 30 * 20 + 30, cells // 30 * 20 + 30], axis=1).astype(np.float32)
    keypoints1 = accuracy.project(keypoints0, SHIFT).astype(np.float32)
    descriptors = rng.normal(size=(count, 256)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    scores = np.ones(count, np.float32)
    return (keypoints0, scores, descriptors), (keypoints1, scores, descriptors.copy())


def test_repeatability_counts_keypoints_both_images_see(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    (keypoints0, _, _), (keypoints1, _, _) = _features(accuracy)

    assert accuracy.repeatability(keypoints0, keypoints1, SHIFT) == 1.0
    # Moving half of image 1's keypoints 4 pixels away halves the repeated share.
    moved = keypoints1.copy()
    moved[::2] += [4, 0]
    assert accuracy.repeatability(keypoints0, moved, SHIFT) == pytest.approx(0.5, abs=0.01)
    # A keypoint the other image cannot see does not count either way.
    outside = np.vstack([keypoints0, [[635, 5]]])
    assert accuracy.repeatability(outside, keypoints1, SHIFT) == 1.0


def test_mutual_matches_keep_only_mutual_nearest_neighbours(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    descriptors0 = np.eye(3, dtype=np.float32)
    # Image-1 descriptor 0 and 2 both prefer image-0 descriptor 0; only descriptor 0 is mutual.
    descriptors1 = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], np.float32)

    rows, columns = accuracy.mutual_matches(descriptors0, descriptors1)

    assert rows.tolist() == [0, 1] and columns.tolist() == [0, 1]
    assert [len(part) for part in accuracy.mutual_matches(descriptors0[:0], descriptors1)] == [0, 0]


@pytest.mark.parametrize("wrong", [0, 60])
def test_score_pair_follows_the_homography(wrong: int, load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    features0, (keypoints1, scores1, descriptors1) = _features(accuracy)
    # Swap the first `wrong` image-1 descriptors pairwise: their matches land on the wrong keypoints.
    descriptors1 = descriptors1.copy()
    descriptors1[:wrong] = descriptors1[:wrong].reshape(-1, 2, 256)[:, ::-1].reshape(-1, 256)

    scores = accuracy.score_pair(features0, (keypoints1, scores1, descriptors1), SHIFT)

    assert scores["keypoint_repeatability"] == 1.0
    assert scores["matching_accuracy"] == pytest.approx(1 - wrong / 300)
    assert scores["homography_accuracy"] == 1.0
    assert scores["matches"] == 300


def test_homography_must_put_the_corners_within_the_tolerance(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    (keypoints0, _, _), (keypoints1, _, _) = _features(accuracy)

    assert accuracy.homography_correct(keypoints0, keypoints1, SHIFT)
    assert not accuracy.homography_correct(keypoints0, keypoints1 + [5, 0], SHIFT)
    assert not accuracy.homography_correct(keypoints0[:3], keypoints1[:3], SHIFT)


def test_image_pair_homography_maps_the_model_inputs(load_script: Callable[[str], ModuleType]) -> None:
    cv2 = pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    image = np.zeros((300, 500, 3), np.uint8)
    cv2.circle(image, (200, 120), 6, (255, 255, 255), -1)
    _, encoded = cv2.imencode(".png", image)

    image0, image1, matrix = accuracy.image_pair(encoded.tobytes(), 139)

    def centre(values: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(values > 0.5 * values.max())
        return np.array([[xs.mean(), ys.mean()]])

    np.testing.assert_allclose(accuracy.project(centre(image0), matrix), centre(image1), atol=1.5)


@pytest.mark.parametrize("shapes", [[(1, 60, 80, 65), (1, 60, 80, 256)], [(60, 80, 256), (60, 80, 65)]])
def test_heads_are_named_by_their_channels(shapes: list, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [np.full(shape, shape[-1], np.float32) for shape in shapes]

    logits, descriptors = accuracy.heads(outputs)

    assert logits.shape == (65, 60, 80) and (logits == 65).all()
    assert descriptors.shape == (256, 60, 80)


@pytest.mark.parametrize("case", ["layout", "nan"])
def test_heads_must_be_finite_and_channels_last(case: str, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    outputs = [np.zeros((1, 60, 80, 65), np.float32), np.zeros((1, 60, 80, 256), np.float32)]
    if case == "layout":
        outputs[0] = outputs[0].transpose(0, 3, 1, 2)
    else:
        outputs[1][0, 1, 2, 3] = np.nan

    with pytest.raises(ValueError):
        accuracy.heads(outputs)


def test_evaluate_averages_the_pairs(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    features0, features1 = _features(accuracy)
    empty = (np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros((0, 256), np.float32))
    outputs = iter([features0, features1, features0, empty])
    pairs = [(None, None, SHIFT), (None, None, SHIFT)]

    metrics, totals = accuracy.evaluate(iter(pairs), 2, lambda image: next(outputs))

    assert metrics == {"keypoint_repeatability": 0.5, "matching_accuracy": 0.5, "homography_accuracy": 0.5}
    assert totals == {"keypoints": 225.0, "matches": 150.0}


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
