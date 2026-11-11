from pathlib import Path

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_relative_depth_alignment_recovers_an_inverse_affine_transform(load_script):
    accuracy = load_script("accuracy")
    rows, columns = np.indices((accuracy.HEIGHT, accuracy.WIDTH))
    depth = 1 + rows * 0.001 + columns * 0.0005
    _, (width, height), (left, top) = accuracy.cover_geometry(640, 480)
    ys, xs = np.indices((480, 640))
    u = np.clip(np.floor((xs + 0.5) * width / 640 - left).astype(int), 0, accuracy.WIDTH - 1)
    v = np.clip(np.floor((ys + 0.5) * height / 480 - top).astype(int), 0, accuracy.HEIGHT - 1)
    truth = 1 / (0.8 / depth[v, u] + 0.1)
    assert accuracy.score(depth, truth) == pytest.approx({"abs_rel": 0, "delta1": 1}, abs=1e-10)
    truth[:45] = 9
    truth[471:] = 9
    truth[:, :41] = 9
    truth[:, 601:] = 9
    assert accuracy.score(depth, truth) == pytest.approx({"abs_rel": 0, "delta1": 1}, abs=1e-10)


def test_accuracy_rejects_invalid_depth_and_missing_ground_truth(load_script):
    accuracy = load_script("accuracy")
    depth = np.ones((accuracy.HEIGHT, accuracy.WIDTH))
    with pytest.raises(ValueError, match="no valid"):
        accuracy.score(depth, np.zeros((480, 640)))
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        accuracy.score(depth, np.ones((480, 640)))


def test_test_split_excludes_train_and_rejects_duplicates(tmp_path: Path, load_script):
    accuracy = load_script("accuracy")
    for split in (*accuracy.SPLITS, "train"):
        (tmp_path / split).mkdir()
    for index in range(654):
        (tmp_path / accuracy.SPLITS[index % 3] / f"rgb_{index:04}.jpg").touch()
    (tmp_path / "train/rgb_9999.jpg").touch()
    images = accuracy.frames(tmp_path)
    assert len(images) == 654 and all(image.parent.name != "train" for image in images)
    assert accuracy.selected_frames(images, 32) == accuracy.selected_frames(images, 32)
    (tmp_path / "cal/rgb_0000.jpg").touch()
    with pytest.raises(ValueError, match="distinct"):
        accuracy.frames(tmp_path)


def test_accuracy_meets_expected_values(accuracy, model_dir, build_profile):
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
    result = accuracy["datasets"]["nyu-depth-v2-default"]
    assert result["evaluated_samples"] == result["available_samples"] == 654
