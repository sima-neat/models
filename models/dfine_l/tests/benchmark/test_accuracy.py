"""Native accuracy uses the same checked handoff, without a replay helper or CPU decode."""

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def a(load_script):
    return load_script("accuracy")


def output():
    return NS(semantic=NS(detection=NS(format="BBOX")))


def tensor(array, meta="geometry"):
    return NS(to_numpy=lambda **kw: array, semantic=NS(preprocess=meta))


@pytest.mark.parametrize("case", ["normal", "ties", "strided"])
def test_query_gather_preserves_stable_order_and_owned_lifetime(a, monkeypatch, case):
    content = np.arange(8400 * 256, dtype=np.float32).reshape(1, 8400, 256)
    if case == "strided":
        content = np.ascontiguousarray(content.transpose(0, 2, 1)).transpose(0, 2, 1)
    scores = np.arange(8400, dtype=np.float32)[None]
    if case == "ties":
        scores.fill(0)
    refs = np.arange(8400 * 4, dtype=np.float32).reshape(1, 8400, 4)
    copies = []

    def create(array, **options):
        assert options == {"copy": True, "layout": "Unknown", "memory": "EV74"}
        copies.append(array.copy())
        return NS(semantic=NS())

    monkeypatch.setattr(a.pyneat, "Tensor", NS(from_numpy=create), raising=False)
    monkeypatch.setattr(a.pyneat, "TensorLayout", NS(Unknown="Unknown"), raising=False)
    monkeypatch.setattr(a.pyneat, "TensorMemory", NS(EV74="EV74"), raising=False)
    result = a.select_queries(*map(tensor, [content, scores, refs]))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :300]
    for copied, source in zip(copies, [content, refs], strict=True):
        np.testing.assert_array_equal(copied, np.take_along_axis(source, order[..., None], axis=1))
        assert not np.shares_memory(copied, source)
    assert all(t.semantic.preprocess == "geometry" for t in result)


@pytest.mark.parametrize("case", ["shape", "nan", "inf", "dtype"])
def test_bad_query_inputs_are_rejected_before_device_copies(a, case):
    arrays = [np.zeros((1, 8400, 256), np.float32), np.zeros((1, 8400), np.float32), np.zeros((1, 8400, 4), np.float32)]
    if case == "shape":
        arrays[0] = arrays[0][:, :1]
    elif case == "dtype":
        arrays[0] = arrays[0].astype(np.float64)
    else:
        arrays[0][0, 0, 0] = float(case)
    with pytest.raises(ValueError):
        a.select_queries(*map(tensor, arrays))


def test_native_rows_preserve_zero_score_empty_boxes_and_repeated_classes(a, monkeypatch):
    boxes = np.zeros((300, 6), np.float32)
    boxes[1] = [1, 2, 4, 6, 0.75, 2]
    boxes[2] = [3, 4, 7, 8, 0.5, 2]
    monkeypatch.setattr(a.pyneat, "decode_bbox", lambda values, strict: [tensor(boxes)], raising=False)
    categories = np.arange(1, 81, dtype=np.int64) * 2
    rows = a.native_rows([output()], {"id": 139, "width": 13, "height": 9}, categories)
    assert rows.shape == (300, 7) and rows.dtype == np.float64
    np.testing.assert_array_equal(rows[0], [139, 0, 0, 0, 0, 0, 2])
    np.testing.assert_array_equal(rows[1], [139, 1, 2, 3, 4, 0.75, 6])
    assert rows[2, 6] == rows[1, 6] == 6


@pytest.mark.parametrize("case", ["missing", "raw", "count", "nan", "class", "fractional", "score", "extent", "x", "y"])
def test_native_rows_fail_closed_without_filtering(a, monkeypatch, case):
    boxes = np.zeros((300, 6), np.float32)
    outputs = [output()]
    if case == "missing":
        outputs = []
    elif case == "raw":
        outputs[0].semantic.detection.format = "RAW"
    elif case == "count":
        boxes = boxes[:299]
    elif case == "nan":
        boxes[0, 0] = np.nan
    elif case == "class":
        boxes[0, 5] = 80
    elif case == "fractional":
        boxes[0, 5] = 0.5
    elif case == "score":
        boxes[0, 4] = 1.1
    elif case == "extent":
        boxes[0, :4] = [4, 4, 3, 3]
    elif case == "x":
        boxes[0, 2] = 15
    else:
        boxes[0, 3] = 11
    monkeypatch.setattr(a.pyneat, "decode_bbox", lambda values, strict: [tensor(boxes)], raising=False)
    with pytest.raises((ValueError, RuntimeError)):
        a.native_rows(outputs, {"id": 139, "width": 13, "height": 9}, np.arange(1, 81))


def test_decoder_keeps_mpk_adapters_and_uses_per_frame_geometry(a, monkeypatch):
    added = []

    class Graph:
        def add(self, node):
            added.append(node)

    def boxdecode(kind, **options):
        assert kind == "focal"
        assert options == {
            "original_width": 0,
            "original_height": 0,
            "model_width": 640,
            "model_height": 640,
            "detection_threshold": 0.0,
            "nms_iou_threshold": 0.0,
            "top_k": 300,
            "decode_type_option": "logits",
            "source_storage": "dense",
            "detess": False,
            "dequant": False,
            "resize_mode": "stretch",
        }
        return "native"

    class Model:
        def graph(self, route):
            assert route.include_input and not route.include_output
            return Graph()

    monkeypatch.setattr(a.pyneat, "ModelRouteOptions", lambda: NS(), raising=False)
    monkeypatch.setattr(a.pyneat, "BoxDecodeType", NS(DetrFocal="focal"), raising=False)
    monkeypatch.setattr(a.pyneat, "BoxDecodeTypeOption", NS(InterleavedByHeadLogit="logits"), raising=False)
    monkeypatch.setattr(a.pyneat, "BoxDecodeSourceStorage", NS(DenseHwc="dense"), raising=False)
    monkeypatch.setattr(a.pyneat, "ResizeMode", NS(Stretch="stretch"), raising=False)
    monkeypatch.setattr(a.pyneat, "nodes", NS(sima_box_decode=boxdecode, output=lambda: "output"), raising=False)
    a.decoder_graph(Model())
    assert added == ["native", "output"]


def test_runtime_scripts_have_no_research_cli_or_cross_fingerprint_imports():
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    for name in ("accuracy.py", "performance.py"):
        source = (scripts / name).read_text()
        assert "cpu-reference-bridge" not in source and "native-boxdecode-helper" not in source
        assert "import performance" not in source and "import accuracy" not in source
        assert "/workspace/" not in source and "same-inference-heads" not in source


def test_native_bbox_independent_origin_span_rounding_is_not_clipped_again(a, monkeypatch):
    boxes = np.zeros((300, 6), np.float32)
    boxes[0] = [147, 69, 587, 632, 0.25, 21]
    monkeypatch.setattr(a.pyneat, "decode_bbox", lambda values, strict: [tensor(boxes)], raising=False)
    rows = a.native_rows([output()], {"id": 285, "width": 586, "height": 640}, np.arange(1, 81))
    np.testing.assert_array_equal(rows[0], [285, 147, 69, 440, 563, 0.25, 22])


def test_inference_reuses_full_decoder_and_forwards_attention_identity(a, monkeypatch):
    records = [{"id": 1, "width": 640, "height": 426}, {"id": 2, "width": 586, "height": 640}]
    memories, builds, closed = [], [], []

    class Backbone:
        def run(self, inputs, timeout_ms):
            height, width = inputs[0].image.shape[:2]
            meta = NS(
                original_width=width,
                original_height=height,
                resized_width=640,
                resized_height=640,
                resize_mode="stretch",
            )
            memory = NS(semantic=NS(preprocess=meta))
            memories.append(memory)
            return [memory, "content", "scores", "references"]

        def close(self):
            closed.append("backbone")

    class Decoder:
        def push(self, inputs):
            assert inputs[0] is memories[-1]
            assert inputs[1:] == ["selected_content", "selected_refs"]
            self.current = inputs[0].semantic.preprocess
            return True

        def pull(self, timeout):
            return NS(tensors=[(self.current.original_width, self.current.original_height)])

        def close(self):
            closed.append("decoder")

    class Model:
        def __init__(self, name, options):
            self.name = name

        def build(self, inputs, run_options):
            assert self.name == "backbone"
            builds.append("backbone")
            return Backbone()

    def graph(model):
        assert model.name == "decoder"

        def build(inputs, options):
            assert inputs[0] is memories[-1]
            builds.append("decoder")
            return Decoder()

        return NS(build=build)

    monkeypatch.setattr(a.pyneat, "RunOptions", lambda: NS(), raising=False)
    monkeypatch.setattr(a.pyneat, "OverflowPolicy", NS(Block="block"), raising=False)
    monkeypatch.setattr(a.pyneat, "OutputMemory", NS(ZeroCopy="zero"), raising=False)
    monkeypatch.setattr(a.pyneat, "Model", Model, raising=False)
    monkeypatch.setattr(a.pyneat, "PixelFormat", NS(RGB="rgb"), raising=False)
    monkeypatch.setattr(a.pyneat, "TensorMemory", NS(EV74="ev"), raising=False)
    monkeypatch.setattr(a.pyneat, "Tensor", NS(from_numpy=lambda image, **kw: NS(image=image)), raising=False)
    monkeypatch.setattr(a, "image_options", lambda w, h: None)
    monkeypatch.setattr(a, "tensor_options", lambda: None)
    monkeypatch.setattr(a, "decoder_graph", graph)
    monkeypatch.setattr(a, "select_queries", lambda *args: ["selected_content", "selected_refs"])
    monkeypatch.setattr(
        a, "prefetched", lambda load, items: [np.zeros((r["height"], r["width"], 3), np.uint8) for r in items]
    )
    results = list(a.iter_inference({"backbone": "backbone", "decoder": "decoder"}, records, None))
    assert [r[0]["id"] for r in results] == [1, 2]
    assert [r[1] for r in results] == [[(640, 426)], [(586, 640)]]
    assert builds == ["backbone", "decoder"]
    assert closed == ["decoder", "backbone"]


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def test_category_partition_matches_official_batch_including_empty_categories(load_script):
    real = pytest.importorskip("faster_coco_eval")
    COCO, COCOeval_faster = real.COCO, real.COCOeval_faster
    a = load_script("accuracy")
    gt = COCO(print_function=lambda *x: None)
    gt.dataset = {
        "images": [
            {"id": 1, "width": 640, "height": 480},
            {"id": 2, "width": 640, "height": 480},
        ],
        "categories": [{"id": 1, "name": "one"}, {"id": 90, "name": "ninety"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 100, 100],
                "area": 10000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 90,
                "bbox": [10, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
    }
    gt.createIndex()
    rows = np.array([[1, 0, 0, 100, 100, 0.9, 1]], dtype=np.float64)
    merged = a.score_coco(gt, rows, [1, 2], [1, 90])
    ordinary = COCOeval_faster(gt, gt.loadRes(rows), "bbox", print_function=lambda *x: None)
    ordinary.params.imgIds = [1, 2]
    ordinary.params.catIds = [1, 90]
    ordinary.params.maxDets = [1, 10, 100]
    ordinary.evaluate()
    ordinary.accumulate()
    ordinary.summarize()
    for name in ["precision", "recall", "scores"]:
        np.testing.assert_array_equal(merged.eval[name], ordinary.eval[name])
    np.testing.assert_array_equal(merged.stats, ordinary.stats)


def test_accuracy_records_the_complete_coco_validation_set(accuracy):
    result = accuracy["datasets"]["coco-2017-val"]
    assert result["split"] == "val2017"
    assert result["evaluated_samples"] == result["available_samples"] == 5000
    assert result["details"]["detections"] == 5000 * 300
    assert len(result["metrics"]) == 12
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in result["metrics"].values())
