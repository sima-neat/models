from __future__ import annotations

import json
import sys
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


@pytest.mark.parametrize("wrong", [0, 60])
def test_score_pair_follows_the_homography(wrong: int, load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    features0, (keypoints1, scores1, descriptors1) = _features(accuracy)
    # Swap the first `wrong` image-1 descriptors pairwise: their nearest neighbours land on the wrong keypoints, and
    # RANSAC still finds the homography from the rest.
    descriptors1 = descriptors1.copy()
    descriptors1[:wrong] = descriptors1[:wrong].reshape(-1, 2, 256)[:, ::-1].reshape(-1, 256)

    scores = accuracy.score_pair(features0, (keypoints1, scores1, descriptors1), SHIFT)

    assert scores == {"keypoint_repeatability": 1.0, "homography_accuracy": 1.0, "keypoints": 300}


def test_homography_must_put_the_corners_within_the_tolerance(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    (keypoints0, _, _), (keypoints1, _, _) = _features(accuracy)

    assert accuracy.homography_correct(keypoints0, keypoints1, SHIFT)
    assert not accuracy.homography_correct(keypoints0, keypoints1 + [5, 0], SHIFT)
    assert not accuracy.homography_correct(keypoints0[:3], keypoints1[:3], SHIFT)


def _write_sequence(directory: Path, size: tuple[int, int], dot: tuple[int, int], shift: tuple[int, int]) -> None:
    """An HPatches-style sequence: a white dot at `dot` in image 1, moved by `shift` in images 2 to 6, and H_1_k."""
    cv2 = pytest.importorskip("cv2")
    directory.mkdir(parents=True)
    for index in range(1, 7):
        image = np.zeros((*size, 3), np.uint8)
        offset = (0, 0) if index == 1 else shift
        cv2.circle(image, (dot[0] + offset[0], dot[1] + offset[1]), 6, (255, 255, 255), -1)
        assert cv2.imwrite(str(directory / f"{index}.ppm"), image)
        if index > 1:
            matrix = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], float)
            np.savetxt(directory / f"H_1_{index}", matrix)


def test_load_sequence_rescales_each_homography_to_the_model_input(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    _write_sequence(tmp_path / "v_dot", size=(300, 500), dot=(200, 120), shift=(40, -20))

    reference, targets = accuracy.load_sequence(tmp_path / "v_dot")

    def centre(values: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(values > 0.5 * values.max())
        return np.array([[xs.mean(), ys.mean()]])

    assert reference.shape == (accuracy.HEIGHT, accuracy.WIDTH) and len(targets) == 5
    for target, matrix in targets:
        np.testing.assert_allclose(accuracy.project(centre(reference), matrix), centre(target), atol=1.5)


def test_sequence_dirs_needs_every_val_sequence(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    (tmp_path / "val" / "i_only").mkdir(parents=True)

    with pytest.raises(SystemExit, match="expected 106 val sequences"):
        accuracy.sequence_dirs(tmp_path)


def test_neat_decodes_with_the_magic_leap_demo_settings(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    pyneat = sys.modules["pyneat"]

    options = accuracy.tensor_options()

    assert options.decode_type is pyneat.BoxDecodeType.SuperPoint
    assert options.superpoint.profile is pyneat.SuperPointProfile.MagicLeapDemoV1
    assert options.top_k == 1000


def test_evaluate_scores_each_target_against_image_1(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    features0, features1 = _features(accuracy)
    empty = (np.zeros((0, 2), np.float32), np.zeros(0, np.float32), np.zeros((0, 256), np.float32))
    outputs = {"reference": features0, "match": features1, "empty": empty}
    sequences = [("reference", [("match", SHIFT), ("empty", SHIFT)])]

    metrics, keypoints = accuracy.evaluate(iter(sequences), 1, lambda name: outputs[name])

    assert metrics == {"keypoint_repeatability": 0.5, "homography_accuracy": 0.5}
    assert keypoints == 225.0


def _heads(points: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Raw heads whose demo score map peaks at each (x, y) with the given probability, and random descriptors."""
    logits = np.full((65, 60, 80), -20.0, np.float32)
    logits[64] = 0.0
    for x, y, probability in points:
        logits[(y % 8) * 8 + x % 8, y // 8, x // 8] = np.log(probability / (1 - probability))
    return logits, np.random.default_rng(0).normal(size=(256, 60, 80)).astype(np.float32)


def test_fp32_baseline_decodes_like_the_magic_leap_demo(load_script: Callable[[str], ModuleType]) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    # Two strong points 4 pixels apart in neighbouring cells, a weak one below the threshold, and one inside the
    # 4-pixel border.
    logits, descriptors = _heads([(100, 100, 0.9), (104, 100, 0.6), (300, 200, 0.01), (2, 50, 0.9)])

    keypoints, scores, unit = baseline.decode(logits, descriptors)

    assert keypoints.tolist() == [[100, 100]]
    assert scores[0] == pytest.approx(0.9, abs=1e-3)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=1e-5)


def test_fp32_baseline_scores_the_onnx_as_accuracy_scores_the_mpk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    pytest.importorskip("cv2")
    baseline = load_script("fp32_baseline_accuracy")
    accuracy = baseline.accuracy
    heads = _heads([(100 + 40 * column, 80 + 40 * row, 0.9) for row in range(8) for column in range(10)])
    feeds = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def get_inputs(self) -> list[Any]:
            return [type("Input", (), {"name": "image"})]

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            feeds.append({name: value.shape for name, value in inputs.items()})
            return [heads[0][None], heads[1][None]]

    reference = np.zeros((accuracy.HEIGHT, accuracy.WIDTH), np.float32)
    identity = np.eye(3)
    monkeypatch.setitem(sys.modules, "onnxruntime", type("onnxruntime", (), {"InferenceSession": Session}))
    monkeypatch.setattr(accuracy, "sequence_dirs", lambda directory: [Path("v_a")])
    monkeypatch.setattr(accuracy, "load_sequence", lambda path: (reference, [(reference, identity)] * 5))
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", f"hpatches-val={tmp_path}", "--output", str(output)])

    expected, keypoints = accuracy.evaluate(
        iter([(reference, [(reference, identity)] * 5)]), 1, lambda image: baseline.decode(*heads)
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result == {"dataset": "hpatches-val", "pairs": 5, "keypoints_per_image": keypoints,
                      "metrics": {name: round(value, 4) for name, value in expected.items()}}
    assert expected == {"keypoint_repeatability": 1.0, "homography_accuracy": 1.0}
    assert feeds == [{"image": (1, 1, accuracy.HEIGHT, accuracy.WIDTH)}] * 6


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
