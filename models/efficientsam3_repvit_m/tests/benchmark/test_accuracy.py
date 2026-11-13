from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


@pytest.fixture
def script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_onnx_baseline_maps_pixels_prompt_features_and_masks(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("onnxruntime")
    baseline = load_script("fp32_baseline_accuracy")
    names = ["person", "traffic light"]
    prompt_tokens = [[49406, 2533, 49407], [49406, 3399, 1395, 49407]]
    monkeypatch.setattr(baseline.accuracy, "tokens", lambda name, tokenizer: prompt_tokens[names.index(name)])
    pixels = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    calls = []

    class Session:
        def run(self, outputs, feeds, options):
            assert set(feeds) == {"token_one_hot", "pixel_values", "text"}
            assert options.only_execute_path_to_fetches
            index = len(calls) % len(names)
            calls.append(outputs)
            features = np.full((1, 16, 256), index + 1, np.float32)
            if outputs == ["text_features"]:
                np.testing.assert_array_equal(feeds["token_one_hot"], baseline.accuracy.one_hot(prompt_tokens[index]))
                return [features]
            assert outputs == ["detections", "mask_logits"]
            np.testing.assert_array_equal(feeds["pixel_values"], pixels.transpose(2, 0, 1)[None])
            assert feeds["pixel_values"].flags.c_contiguous
            np.testing.assert_array_equal(feeds["text"], baseline.accuracy.packed(features, prompt_tokens[index]))
            return [np.full((1, 200, 6), index, np.float32), np.full((1, 200, 2, 3), index, np.float32)]

    infer = baseline.onnx_pipeline(Session(), names, tokenizer=None)
    assert calls == [["text_features"], ["text_features"]]
    for index, (detections, masks) in enumerate(infer(pixels)):
        assert detections.shape == (1, 200, 6)
        assert masks.shape == (200, 2, 3)
        assert np.all(detections == index) and np.all(masks == index)
    assert calls == [["text_features"], ["text_features"], ["detections", "mask_logits"], ["detections", "mask_logits"]]


def test_the_image_model_text_holds_the_features_and_the_token_mask(script: ModuleType) -> None:
    features = np.arange(16 * 256, dtype=np.float32).reshape(1, 16, 256)

    text = script.packed(features, [script.START, 3399, 1395, script.END])

    assert text.shape == (1, 17, 256)
    np.testing.assert_array_equal(text[0, :16], features[0])
    assert text[0, 16].tolist() == [1.0] * 4 + [0.0] * 252


def test_each_image_keeps_the_best_queries_of_all_prompts(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(script, "TOP_K", 3)
    kept = (np.zeros((0, 6), np.float32), np.zeros(0, np.int64), np.zeros((0, 2, 2), np.float32))

    for category, scores in ((1, [0.1, 0.9, 0.3]), (7, [0.5, 0.05, 0.95]), (9, [0.5, 0.2, 0.1])):
        detections = np.zeros((3, 6), np.float32)
        detections[:, 4] = scores
        masks = np.stack([np.full((2, 2), score, np.float32) for score in scores])
        kept = script.best_queries(kept, detections, masks, category)

    assert kept[0][:, 4].tolist() == pytest.approx([0.95, 0.9, 0.5])
    # The earlier prompt keeps a tied score, and each kept query keeps its own mask.
    assert kept[1].tolist() == [7, 1, 7]
    assert kept[2][:, 0, 0].tolist() == pytest.approx([0.95, 0.9, 0.5])


def test_boxes_and_masks_map_to_the_original_image(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = []
    monkeypatch.setattr(
        script.mask_util, "encode", lambda masks: encoded.append(masks) or [{"mask": i} for i in range(masks.shape[-1])]
    )
    detections = np.array([[-5.04, 504, 1008, 1058.4, 0.8, 0]], np.float32)
    # The image model returns HWC mask logits; query 0 is foreground in the right half.
    mask_logits = np.full((192, 192, 200), -1, np.float32)
    mask_logits[:, 96:, 0] = 1

    boxes, segments = script.coco_results(
        42, 200, 100, (detections, np.array([18]), script.query_masks(mask_logits)[:1])
    )

    # Boxes scale from 1008x1008 input pixels and are not clipped.
    assert boxes == [
        {"image_id": 42, "category_id": 18, "score": pytest.approx(0.8), "bbox": pytest.approx([-1, 50, 201, 55])}
    ]
    assert segments == [{"image_id": 42, "category_id": 18, "score": pytest.approx(0.8), "segmentation": {"mask": 0}}]
    (binary,) = encoded
    assert binary.shape == (100, 200, 1) and binary.flags.f_contiguous
    assert not binary[:, :99].any() and binary[:, 101:].all()
