from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

QUERY = 3
CATEGORY = 18
# The tests' stand-in for a missing faster_coco_eval would pass importorskip, so its absence is read first.
NEEDS_DECODE = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("cv2", "faster_coco_eval")),
    reason="needs cv2 and faster_coco_eval",
)


def _outputs(accuracy: ModuleType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decoder outputs in the MPK layout where only QUERY finds CATEGORY, with its mask over the top half."""
    queries, size = accuracy.QUERY_COUNT, accuracy.MASK_SIZE
    dets = np.full((1, queries, 4), 0.1, np.float32)
    dets[0, QUERY] = [0.5, 0.25, 0.5, 0.5]
    labels = np.full((1, queries, accuracy.CLASS_COUNT), -20.0, np.float32)
    labels[0, QUERY, CATEGORY] = 4.0
    masks = np.full((size, size, queries), -5.0, np.float32)
    masks[: size // 2, :, QUERY] = 5.0
    return dets, labels, masks


@NEEDS_DECODE
def test_decode_scales_the_box_and_encodes_the_resized_mask(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    valid = np.ones(accuracy.CLASS_COUNT, bool)

    (result,) = accuracy.decode(*_outputs(accuracy), record={"id": 7, "width": 200, "height": 100}, valid=valid)

    assert result["image_id"] == 7
    assert result["category_id"] == CATEGORY
    assert result["score"] == pytest.approx(1 / (1 + np.exp(-4.0)))
    assert result["bbox"] == pytest.approx([50.0, 0.0, 100.0, 50.0])
    expected = np.zeros((100, 200), np.uint8)
    expected[:50] = 1
    np.testing.assert_array_equal(accuracy.mask_util.decode(result["segmentation"]), expected)


@NEEDS_DECODE
def test_decode_drops_unused_coco_slots_and_rejects_another_mask_layout(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    dets, labels, masks = _outputs(accuracy)
    valid = np.ones(accuracy.CLASS_COUNT, bool)
    valid[CATEGORY] = False

    assert accuracy.decode(dets, labels, masks, record={"id": 7, "width": 200, "height": 100}, valid=valid) == []
    with pytest.raises(ValueError, match="unexpected mask logits"):
        accuracy.decode(
            dets, labels, masks.transpose(2, 0, 1), record={"id": 7, "width": 200, "height": 100}, valid=valid
        )


def test_mask_results_carry_no_box_so_coco_measures_the_mask_area(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    detections = [{"image_id": 7, "category_id": 18, "bbox": [0, 0, 10, 10], "score": 0.9, "segmentation": {}}]

    assert accuracy.coco_results(detections, "segm") == [
        {"image_id": 7, "category_id": 18, "score": 0.9, "segmentation": {}}
    ]
    assert accuracy.coco_results(detections, "bbox") == [
        {"image_id": 7, "category_id": 18, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    assert "bbox" in detections[0] and "segmentation" in detections[0]


def test_fp32_baseline_feeds_nchw_and_returns_masks_in_the_mpk_layout(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    accuracy = baseline.accuracy
    dets, labels, masks = _outputs(accuracy)
    feeds = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            feeds.append(inputs["input"])
            return [dets, labels, masks.transpose(2, 0, 1)[None]]

    monkeypatch.setitem(sys.modules, "onnxruntime", type("onnxruntime", (), {"InferenceSession": Session}))
    image = np.random.default_rng(0).random((accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, 3), dtype=np.float32)

    result = baseline.onnx_pipeline(Path("model.onnx"))(image)

    np.testing.assert_array_equal(feeds[0], image.transpose(2, 0, 1)[None])
    for actual, expected in zip(result, (dets, labels, masks), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
