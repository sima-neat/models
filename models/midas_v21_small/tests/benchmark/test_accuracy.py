from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _depth(rng: np.random.Generator) -> np.ndarray:
    depth = rng.uniform(0.5, 9.5, (48, 64)).astype(np.float32)
    depth[:4] = 0.0  # Rows the sensor did not measure.
    return depth


def test_a_scaled_and_shifted_inverse_depth_scores_perfectly(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    depth = _depth(np.random.default_rng(0))
    # Relative disparity at the ground truth's resolution, any scale and shift, with junk where depth is missing.
    disparity = np.where(depth > 0, 3.0 / np.maximum(depth.astype(np.float64), 1e-6) + 0.7, 123.0)

    abs_rel, delta1 = accuracy.score(disparity, depth)

    assert abs_rel == pytest.approx(0.0, abs=1e-5)
    assert delta1 == 1.0


def test_score_caps_predictions_at_ten_metres_and_counts_delta1_per_pixel(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    depth = np.full((10, 10), 2.0, np.float32)
    depth[:, 5:] = 4.0
    # A constant prediction aligns to the mean inverse depth, (1/2 + 1/4) / 2, so every pixel predicts 8/3 m.
    abs_rel, delta1 = accuracy.score(np.ones((10, 10)), depth)

    predicted = 8 / 3
    assert abs_rel == pytest.approx((abs(predicted - 2) / 2 + abs(predicted - 4) / 4) / 2)
    # 8/3 is within 1.25x of neither 2 nor 4.
    assert delta1 == 0.0


def test_the_prediction_is_resized_bicubically_to_the_ground_truth(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    # The model's 256x256 map of a left-to-right ramp, resized to NYU's 640x480.
    ramp = np.tile(np.linspace(1.0, 2.0, 256, dtype=np.float32), (256, 1))

    resized = accuracy.resize_bicubic(ramp, 480, 640)

    assert resized.shape == (480, 640)
    assert np.all(np.diff(resized[240]) >= -1e-6)
    np.testing.assert_allclose(resized[:, 320], resized[0, 320], atol=1e-6)
    np.testing.assert_allclose(accuracy.resize_bicubic(np.full((256, 256), 5.0), 480, 640), 5.0, atol=1e-5)


def test_score_rejects_ground_truth_without_depth(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    with pytest.raises(ValueError, match="no valid depth"):
        accuracy.score(np.ones((4, 4)), np.zeros((4, 4), np.float32))


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/depth_mpk.tar.gz"]) == Path("/data/depth_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        accuracy.parse_mpks(["encoder=/data/depth_mpk.tar.gz"])


def test_the_fp32_baseline_takes_only_the_validation_dataset(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    arguments = ["--onnx", str(tmp_path / "model.onnx"), "--output", str(tmp_path / "baseline.json")]

    with pytest.raises(SystemExit, match="--dataset must be nyu-depth-v2-val"):
        baseline.main([*arguments, "--dataset", f"nyu-depth-v2-cal={tmp_path}"])
    with pytest.raises(SystemExit, match="has no images under test/"):
        baseline.main([*arguments, "--dataset", f"nyu-depth-v2-val={tmp_path}"])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
