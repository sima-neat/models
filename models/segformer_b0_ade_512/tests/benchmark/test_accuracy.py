from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def _ramp_logits(accuracy: ModuleType) -> np.ndarray:
    """Class 1 wins on the left half of the logits and class 0 on the right; the rest never win."""
    logits = np.full((accuracy.CLASS_COUNT, accuracy.LOGITS_SIZE, accuracy.LOGITS_SIZE), -1e3, np.float32)
    column = np.arange(accuracy.LOGITS_SIZE, dtype=np.float32)
    logits[0] = column
    logits[1] = accuracy.LOGITS_SIZE - 1 - column
    return logits


def test_segment_interpolates_the_logits_before_the_argmax(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    classes = accuracy.segment(_ramp_logits(accuracy), 40, 256)

    # At twice the width, output column i samples logit column (i + 0.5) / 2 - 0.5; the classes tie at logit column
    # 63.5, which output column 127.5 samples.
    assert classes.shape == (40, 256) and classes.dtype == np.uint8
    assert (classes[:, :128] == 1).all() and (classes[:, 128:] == 0).all()


def _every_class_argmax(accuracy: ModuleType, logits: np.ndarray, height: int, width: int) -> np.ndarray:
    """The argmax of all 150 classes interpolated to height x width, without pruning any class."""
    left, right, across = accuracy._interpolation(accuracy.LOGITS_SIZE, width)
    top, bottom, down = accuracy._interpolation(accuracy.LOGITS_SIZE, height)
    columns = logits[:, :, left] * (1 - across) + logits[:, :, right] * across
    down = down[None, :, None]
    return (columns[:, top] * (1 - down) + columns[:, bottom] * down).argmax(axis=0)


def test_segment_interpolates_only_classes_that_can_win_and_gets_the_same_argmax(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(0)
    size = (accuracy.CLASS_COUNT, accuracy.LOGITS_SIZE, accuracy.LOGITS_SIZE)
    noisy = rng.standard_normal(size).astype(np.float32)
    # Like a trained model's: four classes own regions, and the classes absent from the image sit far below them.
    peaked = rng.standard_normal(size).astype(np.float32) - 20
    regions = rng.integers(0, 4, (accuracy.LOGITS_SIZE // 16, accuracy.LOGITS_SIZE // 16)).repeat(16, 0).repeat(16, 1)
    owners = [3, 40, 77, 149]
    for index, owner in enumerate(owners):
        peaked[owner] = rng.standard_normal(size[1:]) + 10 * (regions == index)

    assert len(accuracy.candidate_classes(noisy)) == accuracy.CLASS_COUNT
    assert list(accuracy.candidate_classes(peaked)) == owners
    for logits in (noisy, peaked):
        for height, width in ((70, 90), (300, 201)):
            np.testing.assert_array_equal(
                accuracy.segment(logits, height, width), _every_class_argmax(accuracy, logits, height, width)
            )


def test_segment_rejects_unexpected_logits(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    logits = _ramp_logits(accuracy)

    with pytest.raises(ValueError, match="logits"):
        accuracy.segment(logits[:, :64], 10, 10)
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        accuracy.segment(logits, 10, 10)


def test_miou_ignores_unlabelled_pixels_and_averages_the_present_classes(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    # Label 0 is unlabelled; labels 1 and 2 are classes 0 and 1.
    label = np.array([[0, 1, 1], [2, 2, 1]], np.uint8)
    classes = np.array([[7, 0, 1], [1, 1, 0]], np.uint8)

    counts = accuracy.confusion(classes, label)
    measured = accuracy.scores(counts)

    # Class 0: 2 hits of 3 labelled, plus no false hits (union 3). Class 1: 2 hits of 2, plus 1 false hit (union 3).
    assert counts.sum() == 5
    assert measured["miou"] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert measured["pixel_accuracy"] == pytest.approx(4 / 5)
    assert measured["classes"] == 2


def test_confusion_rejects_mismatched_or_out_of_range_labels(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    with pytest.raises(ValueError, match="differ in size"):
        accuracy.confusion(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))
    with pytest.raises(ValueError, match="label values"):
        accuracy.confusion(np.zeros((1, 1), np.uint8), np.full((1, 1), 151, np.uint8))


def test_logits_come_back_channels_first_from_either_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    logits = _ramp_logits(accuracy)

    np.testing.assert_array_equal(accuracy.channels_first(logits[None]), logits)
    np.testing.assert_array_equal(accuracy.channels_first(logits.transpose(1, 2, 0)[None]), logits)
    with pytest.raises(ValueError, match="unexpected logits shape"):
        accuracy.channels_first(logits.reshape(-1))


def test_samples_pair_each_image_with_its_label_map(load_script: Callable[[str], ModuleType], tmp_path: Path) -> None:
    # The DevKit's test environment has no Pillow; the scoring tests above do not need it.
    image = pytest.importorskip("PIL.Image")
    accuracy = load_script("accuracy")
    root = tmp_path / accuracy.DATASET
    for folder, suffix in (("images", "jpg"), ("annotations", "png")):
        (root / folder / accuracy.SPLIT).mkdir(parents=True)
        for name in ("ADE_val_00000002", "ADE_val_00000001"):
            image.new("L", (4, 3)).save(root / folder / accuracy.SPLIT / f"{name}.{suffix}")

    # The archive's top folder or the folder itself.
    for directory in (tmp_path, root):
        found_root, images = accuracy.samples(directory)
        assert found_root == root
        assert [image.stem for image in images] == ["ADE_val_00000001", "ADE_val_00000002"]
    assert accuracy.label_map(root, images[0]).shape == (3, 4)
    (root / "annotations" / accuracy.SPLIT / "ADE_val_00000002.png").unlink()
    with pytest.raises(SystemExit, match="label map"):
        accuracy.samples(tmp_path)


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/segformer_mpk.tar.gz"]) == Path("/data/segformer_mpk.tar.gz")
    with pytest.raises(SystemExit, match="model=<path>"):
        accuracy.parse_mpks(["model=/a", "model=/b"])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
    result = accuracy["datasets"]["ade20k-val"]
    assert result["evaluated_samples"] == result["available_samples"] == 2000
