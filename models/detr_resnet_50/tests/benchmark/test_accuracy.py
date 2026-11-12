from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def scorer(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_each_query_keeps_its_best_class_other_than_no_object(scorer: ModuleType) -> None:
    logits = np.zeros((1, scorer.QUERY_COUNT, scorer.CLASSES), dtype=np.float32)
    logits[0, 0, 17] = 4.0
    # "No object" wins this query, so the detection takes the best real class and a low score.
    logits[0, 1, -1], logits[0, 1, 3] = 9.0, 2.0
    boxes = np.full((1, scorer.QUERY_COUNT, 4), 0.5, dtype=np.float32)
    rows = scorer.decode(logits, boxes, image_id=7, width=640, height=480)
    assert rows.shape == (scorer.QUERY_COUNT, 7)
    assert rows[0, 6] == 17 and rows[1, 6] == 3
    probabilities = np.exp(logits[0, 1]) / np.exp(logits[0, 1]).sum()
    assert rows[1, 5] == pytest.approx(probabilities[3])
    assert rows[0, 5] > rows[1, 5]


def test_boxes_map_from_fractions_of_the_stretched_input_to_image_pixels(scorer: ModuleType) -> None:
    logits = np.zeros((1, scorer.QUERY_COUNT, scorer.CLASSES), dtype=np.float32)
    boxes = np.zeros((1, scorer.QUERY_COUNT, 4), dtype=np.float32)
    # The box (100, 150) to (500, 270) in pixels of an 800x600 image, as fractions of the image, which the
    # stretched input covers exactly.
    x1, y1, x2, y2 = 100 / 800, 150 / 600, 500 / 800, 270 / 600
    boxes[0, 0] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    (x, y, w, h) = scorer.decode(logits, boxes, image_id=1, width=800, height=600)[0, 1:5]
    assert (x, y, w, h) == pytest.approx((100.0, 150.0, 400.0, 120.0), abs=1e-3)


@pytest.mark.parametrize("shape", [(1, 100, 91), (1, 300, 92)])
def test_decode_rejects_other_output_shapes(scorer: ModuleType, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        scorer.decode(np.zeros(shape, np.float32), np.zeros((1, shape[1], 4), np.float32), 1, 10, 10)


def test_the_mpk_box_logits_pass_through_the_sigmoid_neats_decoder_applies(
    scorer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    logits = np.zeros((1, scorer.QUERY_COUNT, scorer.CLASSES), np.float32)
    box_logits = np.zeros((1, scorer.QUERY_COUNT, 4), np.float32)
    box_logits[0, 0] = [0.0, 2.0, -2.0, 0.5]

    class Runner:
        def run(self, tensors: list, timeout_ms: int) -> list:
            return [SimpleNamespace(to_numpy=lambda copy, value=value: value) for value in (logits, box_logits)]

    model = SimpleNamespace(build=lambda tensors: Runner())
    pyneat = SimpleNamespace(Model=lambda path: model, Tensor=SimpleNamespace(from_numpy=lambda array, copy: array))
    monkeypatch.setitem(sys.modules, "pyneat", pyneat)

    returned_logits, boxes = scorer.mpk_inference(Path("model_mpk.tar.gz"))(np.zeros((640, 640, 3), np.float32))

    assert returned_logits is logits
    np.testing.assert_allclose(boxes, 1 / (1 + np.exp(-box_logits)), rtol=1e-6)


def test_mpk_argument_names_the_model_component(scorer: ModuleType) -> None:
    # run_model_accuracy.sh passes every MPK as <component>=<path>.
    assert scorer.parse_mpks(["model=/compiled/model_mpk.tar.gz"]) == Path("/compiled/model_mpk.tar.gz")
    with pytest.raises(SystemExit, match="model=<path>"):
        scorer.parse_mpks(["backbone=/compiled/model_mpk.tar.gz"])


def test_fp32_baseline_feeds_the_onnx_the_nchw_image_and_reads_both_outputs(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    # The DevKit runs these tests in PyNeat, which has neither package; the host checks run this one.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    baseline = load_script("fp32_baseline_accuracy")
    helper = onnx.helper
    # logits echo the NCHW input and pred_boxes undo the transpose, so each output shows the layout it received.
    nodes = [
        helper.make_node("Identity", ["input"], ["logits"]),
        helper.make_node("Transpose", ["input"], ["pred_boxes"], perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "layout",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 2, 4])],
        [
            helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 3, 2, 4]),
            helper.make_tensor_value_info("pred_boxes", onnx.TensorProto.FLOAT, [1, 2, 4, 3]),
        ],
    )
    model = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), model)
    image = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)

    logits, boxes = baseline.onnx_inference(model)(image)

    np.testing.assert_array_equal(logits, image.transpose(2, 0, 1)[None])
    np.testing.assert_array_equal(boxes, image[None])


def test_fp32_baseline_scores_with_the_accuracy_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    baseline = load_script("fp32_baseline_accuracy")
    calls = []

    def evaluate(dataset: Path, infer: Callable, limit: int | None) -> dict[str, Any]:
        calls.append((dataset, infer, limit))
        return {"metrics": {"mAP": 0.412345, "AP50": 0.61}, "evaluated_samples": 5000}

    monkeypatch.setattr(baseline.accuracy, "evaluate", evaluate)
    monkeypatch.setattr(baseline, "onnx_inference", lambda onnx: f"session for {onnx.name}")
    output = tmp_path / "fp32-baseline.json"

    baseline.main(["--onnx", "model.onnx", "--dataset", "coco-2017-val=/data/coco", "--output", str(output)])

    assert calls == [(Path("/data/coco"), "session for model.onnx", None)]
    assert json.loads(output.read_text()) == {
        "dataset": "coco-2017-val",
        "images": 5000,
        "metrics": {"mAP": 0.4123, "AP50": 0.61},
    }
