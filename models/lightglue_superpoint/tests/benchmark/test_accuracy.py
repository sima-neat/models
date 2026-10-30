from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/lightglue_mpk.tar.gz"]) == Path("/data/lightglue_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["matcher=/data/lightglue_mpk.tar.gz"])


def test_pose_auc_integrates_the_recall_of_the_pose_errors(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    # Sorted errors 1, 2, 15 and a failure: recall 1/4, 1/2, 3/4, and 1 only at infinity.
    np.testing.assert_allclose(script.pose_auc([1.0, 2.0, np.inf, 15.0]), [0.4, 0.45, 0.61875])
    assert script.pose_auc([np.inf, np.inf]) == [0.0, 0.0, 0.0]


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def _rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def test_pose_error_takes_the_larger_angle_and_either_translation_sign(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    truth = np.eye(4)
    truth[:3, 3] = [2.0, 0.0, 0.0]

    assert script.pose_error(truth, np.eye(3), np.asarray([1.0, 0.0, 0.0])) == pytest.approx(0, abs=1e-6)
    assert script.pose_error(truth, np.eye(3), np.asarray([-1.0, 0.0, 0.0])) == pytest.approx(0, abs=1e-6)
    assert script.pose_error(truth, _rotation_z(10), np.asarray([1.0, 0.0, 0.0])) == pytest.approx(10)
    assert script.pose_error(truth, np.eye(3), np.asarray([1.0, 1.0, 0.0])) == pytest.approx(45)


def test_the_pose_estimate_recovers_a_known_relative_pose(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    rng = np.random.default_rng(0)
    intrinsics = np.asarray([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    rotation, translation = _rotation_z(5) @ _rotation_x(5), np.asarray([0.4, 0.1, 0.05])
    points = rng.uniform([-2, -2, 4], [2, 2, 8], (200, 3))
    projected0 = points @ intrinsics.T
    projected1 = (points @ rotation.T + translation) @ intrinsics.T
    pixels0, pixels1 = projected0[:, :2] / projected0[:, 2:], projected1[:, :2] / projected1[:, 2:]
    truth = np.eye(4)
    truth[:3, :3], truth[:3, 3] = rotation, translation

    estimate = script.estimate_pose(pixels0, pixels1, intrinsics, intrinsics, 1 / 500)

    assert estimate is not None and script.pose_error(truth, *estimate) < 1
    assert script.estimate_pose(pixels0[:4], pixels1[:4], intrinsics, intrinsics, 1 / 500) is None


def test_the_ransac_threshold_is_a_pixel_of_the_stretched_input(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    intrinsics = np.asarray([[1000.0, 0, 640], [0, 1000.0, 480], [0, 0, 1]])
    pair = script.Pair("a.jpg", "b.jpg", intrinsics, intrinsics, np.eye(4))
    # A 1280x960 image stretches to 640x480 at half its size, so its focal lengths halve; a turned 960x1280 image
    # stretches to 480x640, also at half its size.
    upright, turned = script.Image(1280, 960, 0, 512), script.Image(960, 1280, 1, 512)

    assert script.ransac_threshold(pair, upright, upright) == pytest.approx(1 / 500)
    assert script.ransac_threshold(pair, upright, turned) == pytest.approx(1 / 500)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_keypoints_return_to_the_original_image_pixels(rotation: int, load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    # The stretched image before its turns, each pixel holding its own (x, y), turned as the dataset turns it.
    width, height = (480, 640) if rotation % 2 else (640, 480)
    ys, xs = np.mgrid[0:height, 0:width]
    turned = np.rot90(np.stack([xs, ys], -1), k=rotation)
    assert turned.shape[:2] == (480, 640)
    points = np.asarray([[0, 0], [639, 0], [17, 400], [600, 479]])  # (x, y) in SuperPoint's turned input
    stretched = turned[points[:, 1], points[:, 0]]
    image = script.Image(width=width * 2, height=height * 3, rotation=rotation, keypoints=512)

    np.testing.assert_allclose(script.original_pixels(points.astype(np.float64), image), stretched * [2, 3])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
