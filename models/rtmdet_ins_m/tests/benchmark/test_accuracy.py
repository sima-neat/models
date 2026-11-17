from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed column-major RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "encode", _encode)
    return module


def _encode(binary: np.ndarray) -> list[dict]:
    rles = []
    for index in range(binary.shape[2]):
        flat = binary[:, :, index].ravel(order="F").astype(np.int64)
        changes = np.flatnonzero(np.diff(np.concatenate([[0], flat, [1 - flat[-1]]])))
        rles.append({"size": list(binary.shape[:2]), "counts": np.diff(np.concatenate([[0], changes])).tolist()})
    return rles


def _outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for scores in outputs[:3]:
        scores[:] = -20.0
    return outputs


def test_nms_suppresses_overlaps_within_a_class_only(accuracy_script: ModuleType) -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]], np.float64)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([0, 0, 1, 0])

    keep = accuracy_script.batched_nms(boxes, scores, labels)

    assert keep.tolist() == [0, 2, 3]


def test_candidates_take_points_and_clip_boxes_to_the_resized_image(accuracy_script: ModuleType) -> None:
    outputs = _outputs(accuracy_script)
    # Stride 16, row 2, column 3: the prior is (48, 32).
    outputs[1][0, 2, 3, 5] = 4.0
    outputs[4][0, 2, 3] = [10, 20, 600, 5]

    boxes, scores, labels, kernels, priors = accuracy_script.candidates(outputs, 640, 480)

    assert labels.tolist() == [5]
    np.testing.assert_allclose(boxes, [[38, 12, 640, 37]])
    np.testing.assert_allclose(priors, [[48, 32, 16]])
    np.testing.assert_allclose(scores, 1 / (1 + np.exp(-4.0)))


def test_mask_logits_match_the_dynamic_convolutions(accuracy_script: ModuleType) -> None:
    rng = np.random.RandomState(0)
    features = rng.randn(1, 80, 80, 8).astype(np.float32)
    kernels = rng.randn(2, accuracy_script.KERNEL_PARAMETERS).astype(np.float32)
    priors = np.array([[96, 160, 8], [320, 64, 32]], np.float32)

    logits = accuracy_script.mask_logits(features, kernels, priors)

    for index in range(2):
        w0, w1, w2, b0, b1, b2 = np.split(kernels[index], np.cumsum(accuracy_script.KERNEL_SPLITS)[:-1])
        ys, xs = np.mgrid[0:80, 0:80] * 8.0
        scale = priors[index, 2] * 8
        coords = np.stack([(priors[index, 0] - xs) / scale, (priors[index, 1] - ys) / scale])
        x = np.concatenate([coords, features[0].transpose(2, 0, 1)]).reshape(10, -1)
        x = np.maximum(w0.reshape(8, 10) @ x + b0[:, None], 0)
        x = np.maximum(w1.reshape(8, 8) @ x + b1[:, None], 0)
        expected = (w2.reshape(1, 8) @ x + b2[:, None]).reshape(80, 80)
        np.testing.assert_allclose(logits[index], expected, rtol=1e-4, atol=1e-4)


def test_image_masks_resize_the_padded_logits_and_crop_the_image(accuracy_script: ModuleType) -> None:
    logits = np.full((1, 80, 80), -1.0, np.float32)
    logits[0, :, :40] = 1.0  # the left half of the padded 640x640 input

    masks = accuracy_script.image_masks(logits, 500, 375, 1.28, 1.28)

    assert masks.shape == (375, 500, 1)
    assert masks[:, :245, 0].all() and not masks[:, 255:, 0].any()



def test_image_masks_cover_an_image_whose_sides_round_apart(accuracy_script: ModuleType) -> None:
    # 200x513 resizes to 250x640: the x scale 1.25 rescales the logits to 512 rows, one short of the image.
    logits = np.ones((1, 80, 80), np.float32)

    masks = accuracy_script.image_masks(logits, 200, 513, 250 / 200, 640 / 513)

    assert masks.shape == (513, 200, 1)
    assert masks[:512, :, 0].all() and not masks[512:, :, 0].any()

def test_decode_returns_separate_box_and_mask_results(accuracy_script: ModuleType) -> None:
    outputs = _outputs(accuracy_script)
    outputs[0][0, 10, 10, 2] = 5.0
    outputs[3][0, 10, 10] = [20, 20, 20, 20]
    outputs[6][0, 10, 10, -1] = 3.0  # the last layer's bias: every mask pixel is positive
    categories = list(range(1, 81))

    results = accuracy_script.decode(outputs, 7, 640, 480, categories)

    (box,) = results["bbox"]
    (mask,) = results["segm"]
    assert box["category_id"] == mask["category_id"] == 3 and box["image_id"] == 7
    np.testing.assert_allclose(box["bbox"], [60, 60, 40, 40])
    assert "bbox" not in mask and mask["segmentation"]["size"] == [480, 640]
    with pytest.raises(ValueError, match="non-finite"):
        outputs[0][0, 0, 0, 0] = np.nan
        accuracy_script.decode(outputs, 7, 640, 480, categories)


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
