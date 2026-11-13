"""Host checks of the accuracy script's query handoff and COCO decode; the board run is the measurement."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(load_script):
    return load_script("accuracy")


def tensor(array, meta="geometry"):
    return SimpleNamespace(to_numpy=lambda **kw: array, semantic=SimpleNamespace(preprocess=meta))


@pytest.mark.parametrize("case", ["normal", "ties", "strided"])
def test_query_gather_preserves_stable_order_and_owned_lifetime(accuracy_script, monkeypatch, case):
    content = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
    if case == "strided":
        content = np.ascontiguousarray(content.transpose(0, 2, 1)).transpose(0, 2, 1)
    scores = np.arange(2000, dtype=np.float32)[None]
    if case == "ties":
        scores.fill(0)
    refs = np.arange(2000 * 4, dtype=np.float32).reshape(1, 2000, 4)
    copies = []

    def create(array, **options):
        assert options == {"copy": True, "layout": "Unknown", "memory": "EV74"}
        copies.append(array.copy())
        return SimpleNamespace(semantic=SimpleNamespace())

    monkeypatch.setattr(accuracy_script.pyneat, "Tensor", SimpleNamespace(from_numpy=create), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "TensorLayout", SimpleNamespace(Unknown="Unknown"), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "TensorMemory", SimpleNamespace(EV74="EV74"), raising=False)
    result = accuracy_script.select_queries(*map(tensor, [content, scores, refs]))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :300]
    for copied, source in zip(copies, [content, refs], strict=True):
        np.testing.assert_array_equal(copied, np.take_along_axis(source, order[..., None], axis=1))
        assert not np.shares_memory(copied, source)
    assert all(t.semantic.preprocess == "geometry" for t in result)


@pytest.mark.parametrize("case", ["shape", "nan", "inf", "dtype"])
def test_bad_query_inputs_are_rejected_before_device_copies(accuracy_script, case):
    arrays = [np.zeros((1, 2000, 128), np.float32), np.zeros((1, 2000), np.float32), np.zeros((1, 2000, 4), np.float32)]
    if case == "shape":
        arrays[0] = arrays[0][:, :1]
    elif case == "dtype":
        arrays[0] = arrays[0].astype(np.float64)
    else:
        arrays[0][0, 0, 0] = float(case)
    with pytest.raises(ValueError):
        accuracy_script.select_queries(*map(tensor, arrays))


def test_decode_ranks_query_class_pairs_and_scales_float_boxes_to_the_image(accuracy_script):
    boxes = np.zeros((1, 300, 4), np.float32)
    logits = np.full((1, 300, 80), -20.0, np.float32)
    # Query 0: a centered box with two confident classes; query 1: a box crossing the right edge.
    boxes[0, 0] = [0.5, 0.5, 0.5, 0.25]
    logits[0, 0, [3, 7]] = [2.0, 1.0]
    boxes[0, 1] = [0.9, 0.5, 0.4, 0.2]
    logits[0, 1, 79] = 0.0
    categories = np.arange(1, 81, dtype=np.int64) * 2
    rows = accuracy_script.decode(boxes, logits, 139, 640, 480, categories)
    assert rows.shape[1] == 7 and rows.shape[0] <= 300
    assert rows[:, 0].tolist() == [139] * len(rows)
    sigmoid = 1 / (1 + np.exp(-np.array([2.0, 1.0, 0.0])))
    np.testing.assert_allclose(rows[0], [139, 160, 180, 320, 120, sigmoid[0], 8], rtol=1e-6)
    np.testing.assert_allclose(rows[1], [139, 160, 180, 320, 120, sigmoid[1], 16], rtol=1e-6)
    np.testing.assert_allclose(rows[2], [139, 448, 192, 192, 96, sigmoid[2], 160], rtol=1e-6)
    assert (np.diff(rows[:, 5]) <= 1e-6).all()
    assert (rows[:, 1] + rows[:, 3] <= 640).all() and (rows[:, 2] + rows[:, 4] <= 480).all()


def test_decode_keeps_the_best_300_pairs_and_drops_empty_boxes(accuracy_script):
    rng = np.random.default_rng(3)
    boxes = rng.random((1, 300, 4), dtype=np.float32) * 0.5 + 0.25
    logits = rng.random((1, 300, 80), dtype=np.float32) * 10 - 5
    boxes[0, 5, 2:] = 0  # an empty box never scores
    categories = np.arange(1, 81, dtype=np.int64)
    rows = accuracy_script.decode(boxes, logits, 1, 100, 100, categories)
    scores = (1 / (1 + np.exp(-logits[0]))).ravel()
    order = np.argsort(-scores, kind="stable")[:300]
    kept = order // 80 != 5
    assert len(rows) == np.count_nonzero(kept) < 300
    np.testing.assert_allclose(rows[:, 5], scores[order][kept], rtol=1e-6)
    assert (rows[:, 3] > 0).all() and (rows[:, 4] > 0).all()


@pytest.mark.parametrize("case", ["boxes", "logits", "nan"])
def test_decode_rejects_unexpected_heads(accuracy_script, case):
    boxes, logits = np.zeros((1, 300, 4), np.float32), np.zeros((1, 300, 80), np.float32)
    if case == "boxes":
        boxes = boxes[:, :299]
    elif case == "logits":
        logits = logits[:, :, :79]
    else:
        logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        accuracy_script.decode(boxes, logits, 1, 10, 10, np.arange(1, 81))


def test_inference_builds_each_runner_once_and_forwards_attention_memory(accuracy_script, monkeypatch):
    records = [{"id": 1, "width": 640, "height": 426}, {"id": 2, "width": 586, "height": 640}]
    memories, builds, closed, sizes = [], [], [], []

    class Backbone:
        def run(self, inputs, timeout_ms):
            memory = SimpleNamespace(image=inputs[0].image)
            memories.append(memory)
            return [memory, "content", "scores", "references"]

        def close(self):
            closed.append("backbone")

    class Decoder:
        def run(self, inputs, timeout_ms):
            assert inputs[0] is memories[-1]
            assert inputs[1:] == ["selected_content", "selected_refs"]
            height, width = inputs[0].image.shape[:2]
            return [tensor(np.full((1, 300, 4), width, np.float32)), tensor(np.full((1, 300, 80), height, np.float32))]

        def close(self):
            closed.append("decoder")

    class Model:
        def __init__(self, name, options):
            self.name = name
            sizes.append(options)

        def build(self, inputs, run_options):
            builds.append(self.name)
            if self.name == "backbone":
                return Backbone()
            assert inputs[0] is memories[-1]
            return Decoder()

    monkeypatch.setattr(accuracy_script.pyneat, "RunOptions", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "OverflowPolicy", SimpleNamespace(Block="block"), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "OutputMemory", SimpleNamespace(ZeroCopy="zero"), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "Model", Model, raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "PixelFormat", SimpleNamespace(RGB="rgb"), raising=False)
    monkeypatch.setattr(accuracy_script.pyneat, "TensorMemory", SimpleNamespace(EV74="ev"), raising=False)
    monkeypatch.setattr(
        accuracy_script.pyneat,
        "Tensor",
        SimpleNamespace(from_numpy=lambda image, **kw: SimpleNamespace(image=image)),
        raising=False,
    )
    monkeypatch.setattr(accuracy_script, "image_options", lambda w, h: (w, h))
    monkeypatch.setattr(accuracy_script, "tensor_options", lambda: "tensor")
    monkeypatch.setattr(accuracy_script, "select_queries", lambda *args: ["selected_content", "selected_refs"])
    monkeypatch.setattr(
        accuracy_script,
        "prefetched",
        lambda load, items: [np.zeros((r["height"], r["width"], 3), np.uint8) for r in items],
    )
    results = list(accuracy_script.iter_inference({"backbone": "backbone", "decoder": "decoder"}, records, None))
    assert [r[0]["id"] for r in results] == [1, 2]
    assert [(int(boxes[0, 0, 0]), int(logits[0, 0, 0])) for _, (boxes, logits) in results] == [(640, 426), (586, 640)]
    assert sizes == [(640, 640), "tensor"]
    assert builds == ["backbone", "decoder"]
    assert closed == ["decoder", "backbone"]


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
