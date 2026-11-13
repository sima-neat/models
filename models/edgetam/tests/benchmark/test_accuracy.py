"""Checks of EdgeTAM's DAVIS scoring and first-frame handling, and of a measured accuracy report."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    return load_script("accuracy")


def _square(top: int, left: int, side: int, shape: tuple[int, int] = (480, 854)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_preprocessing_matches_the_calibration_trackers(
    script: ModuleType, load_script: Callable[[str], ModuleType], tmp_path: Path
) -> None:
    pytest.importorskip("onnx")
    from PIL import Image

    calibration = load_script("calibration")
    pixels = np.random.default_rng(0).integers(0, 256, (480, 854, 3), np.uint8)
    Image.fromarray(pixels).save(tmp_path / "frame.png")

    frame = script.preprocess((tmp_path / "frame.png").read_bytes())

    expected = calibration.preprocess(Image.open(tmp_path / "frame.png")).transpose(0, 2, 3, 1)
    assert frame.shape == (1, script.IMAGE_SIZE, script.IMAGE_SIZE, 3)
    np.testing.assert_allclose(frame, expected, atol=1e-5)


def test_the_first_frame_box_matches_the_calibration_trackers(
    script: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    pytest.importorskip("onnx")
    import random

    calibration = load_script("calibration")
    mask = _square(100, 200, 300, (script.IMAGE_SIZE, script.IMAGE_SIZE))
    expected = calibration.first_frame_prompt(mask, "box", random.Random(0)).inputs()

    coords, types = script.box_prompt(mask)

    np.testing.assert_allclose(coords, expected["point_coords"].transpose(0, 2, 3, 1), atol=1e-6)
    np.testing.assert_array_equal(types, expected["point_types"].transpose(0, 2, 3, 1))


def test_each_annotated_object_gets_its_model_sized_mask(script: ModuleType) -> None:
    object_ids = np.zeros((480, 854), np.uint8)
    object_ids[:240] = 1
    object_ids[240:, :427] = 3

    masks = script.object_masks(object_ids)

    assert list(masks) == [1, 3]
    assert masks[1].shape == (script.IMAGE_SIZE, script.IMAGE_SIZE) and masks[1].dtype == bool
    assert masks[1][:500].all() and not masks[1][524:].any()
    assert masks[3][524:, :500].all() and not masks[3][:, 524:].any()


def test_the_highest_logit_wins_an_overlap_and_no_positive_logit_is_background(script: ModuleType) -> None:
    size = script.MASK_SIZE
    logits = {1: np.full((size, size), -5, np.float32), 2: np.full((size, size), 3, np.float32)}
    logits[1][:, : size // 2] = 5
    logits[2][: size // 2] = -5

    labels = script.label_map(logits, 480, 854)

    assert labels.shape == (480, 854)
    assert (labels[:, :400] == 1).all() and (labels[260:, 454:] == 2).all() and not labels[:220, 454:].any()


def test_scores_follow_the_davis_evaluation(script: ModuleType) -> None:
    truth, empty = _square(100, 100, 200), np.zeros((480, 854), bool)

    assert script.region_overlap(truth, truth) == script.boundary_f_measure(truth, truth) == 1.0
    assert script.region_overlap(empty, empty) == script.boundary_f_measure(empty, empty) == 1.0
    assert script.region_overlap(empty, truth) == script.boundary_f_measure(empty, truth) == 0.0
    assert script.region_overlap(_square(100, 200, 200), truth) == pytest.approx(1 / 3)
    # Boundaries 5 pixels apart match within the 8-pixel tolerance of a 480x854 frame; 50 pixels apart they do not.
    assert script.boundary_f_measure(_square(105, 105, 200), truth) == pytest.approx(1.0, abs=0.02)
    assert script.boundary_f_measure(_square(150, 150, 200), truth) < 0.3


def test_the_boundary_is_one_pixel_wide(script: ModuleType) -> None:
    edges = script.boundary(_square(2, 2, 4, (8, 8)))

    assert edges.sum() == 16 and not edges[3:5, 3:5].any()


def test_a_bank_keeps_the_first_frame_and_the_latest_frames(script: ModuleType) -> None:
    bank = script.MemoryBank(np.zeros((script.MEMORY_TOKENS, script.MEMORY_DIM), np.float32), np.zeros(script.HIDDEN))
    for frame in range(1, 21):
        bank.add(np.full((script.MEMORY_TOKENS, script.MEMORY_DIM), frame, np.float32), np.full(script.HIDDEN, frame))

    memories, used_memories, pointers, used_pointers = bank.arrays()

    assert used_memories.all() and used_pointers.all()
    assert memories[0, :: script.MEMORY_TOKENS, 0].tolist() == [0, 20, 19, 18, 17, 16, 15]
    assert pointers[0, :, 0].tolist() == [0, *range(20, 5, -1)]


def test_every_component_needs_its_mpk(script: ModuleType) -> None:
    mpks = [f"{component}={component}.tar.gz" for component in script.COMPONENTS]

    assert list(script.parse_mpks(mpks)) == script.COMPONENTS
    with pytest.raises(SystemExit, match="memory_encoder"):
        script.parse_mpks(mpks[:-1])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
