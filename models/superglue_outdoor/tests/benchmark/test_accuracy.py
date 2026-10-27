from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

K0 = np.array([[900.0, 0.0, 800.0], [0.0, 900.0, 600.0], [0.0, 0.0, 1.0]])
K1 = np.array([[850.0, 0.0, 780.0], [0.0, 850.0, 590.0], [0.0, 0.0, 1.0]])


def _rotation(degrees: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.radians(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _scene(accuracy: ModuleType, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Two images of one 3D scene: image 1 sees image 0's first half of points; the rest are unrelated."""
    rng = np.random.default_rng(seed)
    size = accuracy.KEYPOINTS
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rotation((2.0, 8.0, -1.0)), [1.0, 0.05, 0.1]
    points = np.c_[rng.uniform(-4, 4, (size, 2)), rng.uniform(8, 16, size)]

    def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
        x = X @ K.T
        return x[:, :2] / x[:, 2:]

    keypoints0 = project(K0, points)
    keypoints1 = project(K1, points @ T[:3, :3].T + T[:3, 3])
    order = rng.permutation(size // 2)
    seen1 = np.empty_like(keypoints1)
    seen1[order] = keypoints1[: size // 2]
    seen1[size // 2:] = rng.uniform([0, 0], [1600, 1200], (size - size // 2, 2))
    descriptors = rng.normal(size=(2, size, accuracy.DESCRIPTOR_SIZE))
    descriptors /= np.linalg.norm(descriptors, axis=-1, keepdims=True)
    data = {
        "image_sizes": np.array([[1600, 1200], [1600, 1200]], np.int32),
        "keypoints": np.stack([keypoints0, seen1]).astype(np.float32),
        "scores": rng.uniform(0, 1, (2, size)).astype(np.float32),
        "descriptors": descriptors.astype(np.float32),
        "pair_images": np.array([[0, 1]] * accuracy.PAIRS, np.int32),
        "intrinsics": np.array([[K0, K1]] * accuracy.PAIRS),
        "poses": np.array([T] * accuracy.PAIRS),
    }
    truth = np.full(size, -1)
    truth[: size // 2] = order
    return data, truth


def _couplings(accuracy: ModuleType, truth: np.ndarray) -> np.ndarray:
    """Couplings that pair each image-0 keypoint with truth[i] and leave the others to the dustbin."""
    couplings = np.zeros((accuracy.KEYPOINTS + 1, accuracy.KEYPOINTS + 1), np.float32)
    rows = np.flatnonzero(truth >= 0)
    couplings[rows, truth[rows]] = 12.0
    return couplings


def test_pose_auc_integrates_the_recall_curve(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.pose_auc([0.0] * 10) == [1.0, 1.0, 1.0]
    assert accuracy.pose_auc([np.inf] * 10) == [0.0, 0.0, 0.0]
    # Half the pairs exact, half failed: half the area at every threshold.
    np.testing.assert_allclose(accuracy.pose_auc([0.0, np.inf] * 5), [0.5, 0.5, 0.5])
    # One pair with a 5-degree error: recall ramps from 0 to 1 over the first 5 degrees, then stays at 1. As in
    # SuperGlue's pose_auc, an error equal to the threshold does not count yet at that threshold.
    np.testing.assert_allclose(accuracy.pose_auc([5.0]), [0.0, 0.75, 0.875])


def test_true_correspondences_have_no_epipolar_error(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    data, truth = _scene(accuracy)
    rows = np.flatnonzero(truth >= 0)
    points0 = data["keypoints"][0][rows].astype(np.float64)
    points1 = data["keypoints"][1][truth[rows]].astype(np.float64)

    errors = accuracy.epipolar_errors(points0, points1, data["poses"][0], K0, K1)
    wrong = accuracy.epipolar_errors(points0, np.roll(points1, 1, axis=0), data["poses"][0], K0, K1)

    assert (errors < 1e-8).all()
    assert (wrong > accuracy.EPIPOLAR_THRESHOLD).mean() > 0.9


def test_pose_error_takes_the_larger_angle(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]

    assert accuracy.pose_error(T, np.eye(3), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    # The sign of the translation is ambiguous in the essential matrix: -t counts as exact.
    assert accuracy.pose_error(T, np.eye(3), np.array([-1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert accuracy.pose_error(T, _rotation((0.0, 0.0, 7.0)), np.array([1.0, np.tan(np.radians(3)), 0.0])) \
        == pytest.approx(7.0)


def test_score_pair_recovers_the_true_pose(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    data, truth = _scene(accuracy)

    scores = accuracy.score_pair(_couplings(accuracy, truth), data, 0)

    assert scores["matches"] == accuracy.KEYPOINTS // 2
    assert scores["precision"] == 1.0
    assert scores["matching_score"] == 0.5
    assert scores["pose_error"] < 0.5


def test_evaluate_reports_auc_precision_and_matching_score(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    data, truth = _scene(accuracy)
    outputs = iter([_couplings(accuracy, truth), np.zeros((accuracy.KEYPOINTS + 1,) * 2, np.float32)])

    metrics, totals = accuracy.evaluate(data, [0, 1], lambda inputs: next(outputs))

    assert set(metrics) == set(accuracy.METRICS)
    # One exact pose and one failed pair.
    assert 0.45 < metrics["pose_auc_5"] <= 0.5 and metrics["pose_auc_20"] <= 0.5
    assert metrics["match_precision"] == 0.5 and metrics["matching_score"] == 0.25
    assert totals == {"matches": accuracy.KEYPOINTS / 4}


def test_matcher_inputs_normalize_each_image_by_its_own_size(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    data, _ = _scene(accuracy)
    data["image_sizes"][1] = [1200, 1600]

    inputs = accuracy.matcher_inputs(data, 0)

    assert [value.shape for value in inputs] == list(accuracy.INPUTS.values())
    np.testing.assert_allclose(inputs[0][0, 0], accuracy.normalize_keypoints(data["keypoints"][0], 1600, 1200))
    np.testing.assert_allclose(inputs[3][0, 0], accuracy.normalize_keypoints(data["keypoints"][1], 1200, 1600))
    np.testing.assert_allclose(inputs[5][0, 0], data["descriptors"][1], rtol=1e-6)


@pytest.mark.parametrize("shape", [(1, 1, 513, 513), (1, 513, 513)])
def test_couplings_take_either_channels_last_layout(shape: tuple, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    output = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    matrix = accuracy.couplings_matrix(output)

    assert matrix.shape == (513, 513) and matrix[1, 0] == 513


@pytest.mark.parametrize("case", ["layout", "nan"])
def test_couplings_must_be_finite_and_channels_last(case: str, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    output = np.zeros((1, 513, 1, 513) if case == "layout" else (1, 1, 513, 513), np.float32)
    if case == "nan":
        output[0, 0, 3, 4] = np.nan

    with pytest.raises(ValueError):
        accuracy.couplings_matrix(output)


# A pinhole camera for SuperPoint's 640 x 480 input; the dataset's images are twice that size.
K_INPUT = np.array([[400.0, 0.0, 319.5], [0.0, 400.0, 239.5], [0.0, 0.0, 1.0]])


def _write_val(accuracy: ModuleType, root: Path, pose: np.ndarray, rotations: tuple[int, int] = (0, 0)) -> Path:
    """A yfcc100m val directory: val/a/0.jpg and val/a/1.jpg, and annotations/pairs_val.txt holding 4000 copies
    of their pair, with intrinsics of the 1280 x 960 originals."""
    import cv2

    directory = root / "yfcc100m-val"
    (directory / "val" / "a").mkdir(parents=True)
    (directory / "annotations").mkdir()
    for name in ("0.jpg", "1.jpg"):
        assert cv2.imwrite(str(directory / "val" / "a" / name), np.full((960, 1280, 3), 128, np.uint8))
    original = np.diag([2.0, 2.0, 1.0]) @ K_INPUT
    values = [*map(str, rotations), *(f"{v:.6f}" for v in [*original.ravel(), *original.ravel(), *pose.ravel()])]
    line = " ".join(["a/0.jpg", "a/1.jpg", *values])
    (directory / accuracy.PAIRS_FILE).write_text((line + "\n") * accuracy.PAIRS, encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"id": "yfcc100m", "variant": "val"}), encoding="utf-8")
    return directory


@pytest.mark.parametrize("case", ["valid", "short", "rotation", "dataset"])
def test_load_dataset_joins_superpoint_keypoints_and_ground_truth(
    case: str, tmp_path: Path, load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    accuracy = load_script("accuracy")
    data, _ = _scene(accuracy)
    directory = _write_val(accuracy, tmp_path, data["poses"][0])
    calls = []

    def keypoints(session, image):
        # load_dataset detects the images in name order: a/0.jpg, then a/1.jpg.
        side = len(calls)
        calls.append(side)
        count = 100 if case == "short" and side == 1 else accuracy.KEYPOINTS
        return data["keypoints"][side][:count], data["scores"][side][:count], data["descriptors"][side][:count]

    monkeypatch.setattr(accuracy, "superpoint_keypoints", keypoints)
    if case == "rotation":
        text = (directory / accuracy.PAIRS_FILE).read_text(encoding="utf-8").split("\n")
        text[5] = text[5].replace(".jpg 0 0 ", ".jpg 0 1 ")
        (directory / accuracy.PAIRS_FILE).write_text("\n".join(text), encoding="utf-8")
    elif case == "dataset":
        (directory / "manifest.json").write_text(json.dumps({"id": "yfcc100m", "variant": "cal"}), encoding="utf-8")

    if case in {"rotation", "dataset"}:
        with pytest.raises(SystemExit):
            accuracy.load_dataset(directory, session=None)
        return
    loaded = accuracy.load_dataset(directory, session=None)
    assert loaded["pair_images"].shape == (accuracy.PAIRS, 2) and (loaded["pair_images"] == [0, 1]).all()
    assert (loaded["image_sizes"] == [640, 480]).all() and (loaded["original_sizes"] == [1280, 960]).all()
    # The originals' intrinsics, scaled by half onto SuperPoint's input, give the input camera again.
    np.testing.assert_allclose(loaded["intrinsics"][0], [K_INPUT, K_INPUT], atol=1e-5)
    np.testing.assert_allclose(loaded["poses"][123], data["poses"][0], atol=1e-5)
    if case == "short":
        # An image with fewer than 512 keypoints is padded, and every pair holding it is left out.
        assert loaded["counts"].tolist() == [accuracy.KEYPOINTS, 100] and not loaded["usable"].any()
    else:
        np.testing.assert_array_equal(loaded["keypoints"], data["keypoints"])
        assert loaded["usable"].all()


@pytest.mark.parametrize("rotation", [0, 1])
def test_superpoint_input_is_640_by_480_grayscale_after_the_turn(
    rotation: int, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    import cv2

    accuracy = load_script("accuracy")
    pixels = np.zeros((600, 800, 3), np.uint8)
    pixels[:, :400] = 255  # white left half
    assert cv2.imwrite(str(tmp_path / "image.png"), pixels)

    image, original = accuracy.superpoint_image(tmp_path / "image.png", rotation)

    assert original == (800, 600) and image.shape == (480, 640) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    if rotation == 0:
        assert image[:, :300].mean() > 0.99 and image[:, 340:].mean() < 0.01
    else:
        # Resized to 480 x 640 first, then turned a quarter counter-clockwise: the white half ends up at the bottom.
        assert image[260:].mean() > 0.99 and image[:220].mean() < 0.01


def test_superpoint_decode_keeps_the_best_keypoints_first(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    logits = np.full((1, 65, 60, 80), -10.0, np.float32)
    logits[0, 64] = 10.0  # every cell is "no keypoint"
    for (cy, cx, channel, strength) in ((10, 20, 9, 12.0), (30, 40, 18, 14.0)):
        logits[0, channel, cy, cx] = strength  # one strong keypoint in each of two cells
    descriptors = np.random.default_rng(0).normal(size=(1, 256, 60, 80)).astype(np.float32)

    class Session:
        def run(self, outputs, inputs):
            assert inputs["image"].shape == (1, 1, 480, 640)
            return [logits, descriptors]

    keypoints, scores, sampled = accuracy.superpoint_keypoints(Session(), np.zeros((480, 640), np.float32))

    # Channel c of cell (cy, cx) is pixel (cx * 8 + c % 8, cy * 8 + c // 8); the stronger one comes first.
    assert keypoints[:2].tolist() == [[40 * 8 + 2, 30 * 8 + 2], [20 * 8 + 1, 10 * 8 + 1]]
    assert scores[0] > scores[1] > accuracy.KEYPOINT_THRESHOLD
    np.testing.assert_allclose(np.linalg.norm(sampled, axis=1), 1.0, rtol=1e-5)


def test_intrinsics_follow_the_resize_and_a_quarter_turn(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    # A 2000x1000 image resized to 1600x800, then rotated a quarter turn to 800x1600, as match_pairs.py does.
    data = {
        "image_sizes": np.array([[800, 1600]]), "rotations": np.array([1]), "original_sizes": np.array([[2000, 1000]]),
    }
    K = np.array([[1000.0, 0.0, 999.5], [0.0, 1000.0, 499.5], [0.0, 0.0, 1.0]])

    mapped = accuracy.image_intrinsics(K, data, 0)

    # Scaled by 0.8: f = 800, cx = 799.6, cy = 399.6; the quarter turn swaps the axes and mirrors x in the width.
    np.testing.assert_allclose(mapped, [[800.0, 0.0, 399.6], [0.0, 800.0, 1600 - 1 - 799.6], [0.0, 0.0, 1.0]])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
