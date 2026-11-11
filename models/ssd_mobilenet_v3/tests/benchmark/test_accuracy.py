"""V3 source geometry, decoding and calibration regressions (not target qualification)."""

import io
import math
import random
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class SourceReference:
    """Independent NumPy source decoder, used only by these tests."""

    IMAGE_SIZE = 320

    CLASSES = 91

    GRIDS = (20, 10, 5, 3, 2, 1)

    ANCHORS_PER_CELL = 6

    ANCHOR_COUNT = 3234

    SCORE_THRESHOLD = 0.001

    IOU_THRESHOLD = 0.55

    PRE_NMS_TOPK = 300

    MAX_DETECTIONS = 300

    def anchors_numpy(self) -> np.ndarray:
        """Return [3234,4] xyxy priors in320-pixel coordinates, cell/anchor order."""
        scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
        levels = []
        for level, size in enumerate(self.GRIDS):
            scale = scales[level]
            pairs = [[scale, scale], [math.sqrt(scale * scales[level + 1])] * 2]
            for ratio in (2, 3):
                pairs.extend(
                    [
                        [scale * math.sqrt(ratio), scale / math.sqrt(ratio)],
                        [scale / math.sqrt(ratio), scale * math.sqrt(ratio)],
                    ]
                )
            widths_heights = np.clip(np.asarray(pairs, np.float32), 0, 1)
            xy = (np.arange(size, dtype=np.float32) + np.float32(0.5)) / size
            centers = np.stack(np.meshgrid(xy, xy), axis=-1).reshape(-1, 1, 2)
            centers = np.broadcast_to(centers, (size * size, self.ANCHORS_PER_CELL, 2)).reshape(-1, 2)
            widths_heights = np.tile(widths_heights, (size * size, 1))
            levels.append(
                np.concatenate([centers - 0.5 * widths_heights, centers + 0.5 * widths_heights], axis=1)
                * self.IMAGE_SIZE
            )
        return np.concatenate(levels)

    def decode_numpy(self, reg: np.ndarray, anc: np.ndarray | None = None) -> np.ndarray:
        """Apply Tx/Ty/Tw/Th with10/10/5/5 scales and source exp cap; clip xyxy to320."""
        if anc is None:
            anc = self.ANCHORS
        widths_heights = anc[:, 2:] - anc[:, :2]
        center = anc[:, :2] + widths_heights * 0.5
        xy = reg[:, :2] / np.float32(10) * widths_heights + center
        wh = np.exp(np.minimum(reg[:, 2:] / np.float32(5), np.float32(math.log(1000 / 16)))) * widths_heights
        return np.clip(np.concatenate([xy - wh * 0.5, xy + wh * 0.5], axis=1), 0, self.IMAGE_SIZE)

    def post_numpy(self, reg: np.ndarray, logits: np.ndarray, anc: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Softmax91 including background; per-class top300/NMS then global top300.

        Preserve source degenerate-box behavior: clipping does not add a positive-area filter.
        """
        if anc is None:
            anc = self.ANCHORS
        if reg.shape != (len(anc), 4) or logits.shape != (len(anc), self.CLASSES):
            raise ValueError("unexpected localization, confidence or anchor shape")
        if not (np.isfinite(reg).all() and np.isfinite(logits).all() and np.isfinite(anc).all()):
            raise ValueError("MPK output or anchors contain non-finite values")
        boxes = self.decode_numpy(reg, anc)
        exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = exponential / exponential.sum(axis=1, keepdims=True)
        survivors = []
        for label in range(1, self.CLASSES):
            indices = np.flatnonzero(scores[:, label] > self.SCORE_THRESHOLD)
            indices = indices[np.argsort(-scores[indices, label], kind="stable")[: self.PRE_NMS_TOPK]]
            while len(indices):
                index = indices[0]
                survivors.append((float(scores[index, label]), label, int(index)))
                others = indices[1:]
                low, high = (
                    np.maximum(boxes[index, :2], boxes[others, :2]),
                    np.minimum(boxes[index, 2:], boxes[others, 2:]),
                )
                intersection = np.maximum(0, high - low).prod(axis=1)
                area = np.maximum(0, boxes[index, 2:] - boxes[index, :2]).prod()
                areas = np.maximum(0, boxes[others, 2:] - boxes[others, :2]).prod(axis=1)
                union = area + areas - intersection
                iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
                indices = others[iou <= self.IOU_THRESHOLD]
        survivors = sorted(survivors, key=lambda item: -item[0])[: self.MAX_DETECTIONS]
        return {
            "scores": np.array([item[0] for item in survivors], np.float32),
            "labels": np.array([item[1] for item in survivors], np.int64),
            "boxes": np.array([boxes[item[2]] for item in survivors], np.float32).reshape(-1, 4),
        }

    def decode(self, outputs: Sequence[np.ndarray], image_id: int, width: int, height: int) -> np.ndarray:
        """Map12 raw NHWC heads to COCO rows [image_id,x,y,w,h,score,category_id]."""
        if any(not isinstance(output, np.ndarray) or output.dtype != np.float32 for output in outputs):
            raise ValueError("MPK outputs must be float32 NumPy arrays")
        shapes = [tuple(output.shape) for output in outputs]
        if shapes != self.OUTPUT_SHAPES:
            raise ValueError(f"unexpected MPK outputs: {shapes}")
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        reg = np.concatenate([output.reshape(-1, 4) for output in outputs[: len(self.GRIDS)]]).astype(
            np.float32, copy=False
        )
        logits = np.concatenate([output.reshape(-1, self.CLASSES) for output in outputs[len(self.GRIDS) :]]).astype(
            np.float32, copy=False
        )
        result = self.post_numpy(reg, logits)
        # TorchVision transform.postprocess rescales xyxy to the original image before COCO xywh conversion.
        boxes = result["boxes"] * np.asarray([width / self.IMAGE_SIZE, height / self.IMAGE_SIZE] * 2, np.float32)
        ids = np.full(len(boxes), image_id)
        return np.column_stack([ids, boxes[:, :2], boxes[:, 2:] - boxes[:, :2], result["scores"], result["labels"]])

    def __init__(self):
        self.OUTPUT_SHAPES = [
            (1, size, size, self.ANCHORS_PER_CELL * channels) for channels in (4, self.CLASSES) for size in self.GRIDS
        ]
        self.ANCHORS = self.anchors_numpy()


def _heads(reference):
    heads = [np.zeros(shape, np.float32) for shape in reference.OUTPUT_SHAPES]
    for head in heads[6:]:
        head.reshape(-1, 91)[:, 0] = 100
    return heads


def _png(pixels):
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def test_geometry_and_prior_order(reference):
    anchors = reference.anchors_numpy()
    assert anchors.shape == (3234, 4) and anchors.dtype == np.float32
    assert reference.GRIDS == (20, 10, 5, 3, 2, 1)
    assert [shape[3] for shape in reference.OUTPUT_SHAPES] == [24] * 6 + [546] * 6
    np.testing.assert_allclose(anchors[0], [-24, -24, 40, 40])
    np.testing.assert_allclose((anchors[6, :2] + anchors[6, 2:]) / 2, [24, 8])
    np.testing.assert_allclose((anchors[120, :2] + anchors[120, 2:]) / 2, [8, 24])


def test_box_order_scales_and_exp_clamp(reference):
    anchors = np.array([[100, 100, 140, 160]], np.float32)
    reg = np.array([[10, -10, 5 * np.log(2), 5 * np.log(3)]], np.float32)
    np.testing.assert_allclose(reference.decode_numpy(reg, anchors), [[120, 0, 200, 160]], atol=2e-5)
    reg[0, 2:] = 10000
    decoded = reference.decode_numpy(reg, anchors)
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded, [[0, 0, 320, 320]])


def test_softmax_includes_background_and_empty_shape(reference):
    heads = _heads(reference)
    rows = reference.decode(heads, 123, 640, 480)
    assert rows.shape == (0, 7)
    result = reference.post_numpy(
        np.zeros((1, 4), np.float32), np.array([[100] + [0] * 90], np.float32), reference.ANCHORS[:1]
    )
    assert result["boxes"].shape == (0, 4) and result["boxes"].dtype == np.float32
    assert result["labels"].shape == (0,) and result["labels"].dtype == np.int64
    assert result["scores"].shape == (0,) and result["scores"].dtype == np.float32


def test_multiclass_same_anchor_is_not_argmax_only(reference):
    logits = np.full((1, 91), -100, np.float32)
    logits[0, 0:3] = [0, 10, 9]
    result = reference.post_numpy(np.zeros((1, 4), np.float32), logits, reference.ANCHORS[:1])
    np.testing.assert_array_equal(result["labels"], [1, 2])
    np.testing.assert_allclose(result["scores"].sum(), 1, atol=4e-5)


def test_per_class_top300_precedes_nms(reference):
    anchors = np.repeat(np.array([[10, 10, 20, 20]], np.float32), 301, axis=0)
    anchors[-1] = [100, 100, 110, 110]
    logits = np.full((301, 91), -100, np.float32)
    logits[:, 0] = 0
    logits[:, 1] = np.linspace(10, 5, 301)
    result = reference.post_numpy(np.zeros((301, 4), np.float32), logits, anchors)
    # All top300 are overlapping. Candidate301 must not be refilled after their NMS.
    assert result["labels"].tolist() == [1]
    np.testing.assert_array_equal(result["boxes"], [[10, 10, 20, 20]])


def test_final_global_cap_and_degenerate_source_policy(reference):
    anchors = np.array([[i % 20 * 10, i // 20 * 10, i % 20 * 10 + 1, i // 20 * 10 + 1] for i in range(400)], np.float32)
    logits = np.full((400, 91), -100, np.float32)
    logits[:, 0] = 0
    logits[:200, 1] = np.linspace(10, 8, 200)
    logits[200:, 2] = np.linspace(7, 5, 200)
    result = reference.post_numpy(np.zeros((400, 4), np.float32), logits, anchors)
    assert len(result["scores"]) == 300
    assert np.all(result["scores"][:-1] >= result["scores"][1:])
    # TorchVision clips but does not remove a zero-area box in SSD postprocessing.
    result = reference.post_numpy(np.array([[10000, 0, 0, 0]], np.float32), logits[:1], anchors[:1])
    assert len(result["scores"]) == 1 and result["boxes"][0, 0] == result["boxes"][0, 2]


def test_coco_xywh_scaling_and_category90(reference):
    heads = _heads(reference)
    logits = heads[6].reshape(-1, 91)
    logits[0, 0], logits[0, 90] = 0, 20
    result = reference.decode(heads, 123, 640, 160)
    assert result.shape == (1, 7)
    np.testing.assert_array_equal(result[0, [0, 1, 2, 3, 4, 6]], [123, 0, 0, 80, 20, 90])
    expected_score = np.float32(1) / (np.float32(1) + np.float32(90) * np.exp(np.float32(-20)))
    np.testing.assert_allclose(result[0, 5], expected_score, atol=2e-7, rtol=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("head", [0, 6])
def test_nonfinite_is_rejected(reference, bad, head):
    heads = _heads(reference)
    heads[head].flat[0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        reference.decode(heads, 1, 100, 100)


def test_wrong_heads_and_dimensions_rejected(reference):
    heads = _heads(reference)
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        reference.decode(heads[:-1], 1, 10, 10)
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        reference.decode([heads[6], *heads[1:]], 1, 10, 10)
    with pytest.raises(ValueError, match="positive"):
        reference.decode(heads, 1, 0, 10)


def test_seeded_calibration_uses_only_cal2017_and_full_archive(tmp_path, calibration):
    archive = tmp_path / "cal.tar.gz"
    paths = [f"coco/cal2017/{i:012}.jpg" for i in range(8)]
    with tarfile.open(archive, "w:gz") as stream:
        for name in [*reversed(paths), "coco/val2017/999.jpg"]:
            content = b"bytes are not decoded by the selection test"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))
    selected = calibration.calibration_names(archive, 4)
    assert selected == random.Random(20261028).sample(sorted(paths), 4)
    assert selected == calibration.calibration_names(archive, 4)
    with pytest.raises(ValueError, match="cannot select"):
        calibration.calibration_names(archive, 9)


@pytest.mark.parametrize("name", ["/absolute.jpg", "../escape.jpg"])
def test_unsafe_archive_rejected(tmp_path, calibration, name):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(name)
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        calibration.calibration_names(archive, 1)


@pytest.mark.parametrize("dtype", [np.float64, np.int8, np.uint8])
def test_wrong_head_dtype_rejected(reference, dtype):
    heads = _heads(reference)
    heads[0] = heads[0].astype(dtype)
    with pytest.raises(ValueError, match="float32"):
        reference.decode(heads, 1, 100, 100)


@pytest.fixture
def reference():
    return SourceReference()


@pytest.fixture
def calibration(load_script):
    return load_script("calibration")


def test_accuracy_meets_the_source_relative_gate(accuracy, model_dir, build_profile):
    from model_registry.model_tests import measured_shortfalls

    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


@pytest.mark.parametrize("profile", ["w16a16", "w8a8"])
def test_accuracy_regression_gates_cover_all_twelve_coco_metrics(model_dir, load_script, profile):
    from model_registry.model_tests import load_model_tests, measured_shortfalls

    expected = load_model_tests(model_dir)["accuracy"]["expected"][profile]["coco-2017-val"]
    assert set(expected) == set(load_script("accuracy").METRICS)
    report = {"datasets": {"coco-2017-val": {"metrics": dict(expected)}}}
    assert measured_shortfalls(model_dir, "accuracy", profile, report) == []
    for metric, value in expected.items():
        report["datasets"]["coco-2017-val"]["metrics"] = {**expected, metric: value * 0.94}
        shortfalls = measured_shortfalls(model_dir, "accuracy", profile, report)
        assert len(shortfalls) == 1 and metric in shortfalls[0]


def _scorer(accuracy: ModuleType, **overrides: float) -> SimpleNamespace:
    stats = dict.fromkeys(accuracy.METRICS.values(), 0.5)
    stats.update(overrides)
    return SimpleNamespace(params=SimpleNamespace(maxDets=[1, 10, 100]), stats_as_dict=stats)


def test_local_subset_omits_undefined_size_metrics_without_inventing_zero(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = load_script("accuracy")
    scorer = _scorer(accuracy, AP_small=-1, AR_small=-1)

    measured = accuracy.coco_stats(scorer, allow_missing_size_buckets=True)

    assert measured["coco_map"] == 0.5
    assert set(accuracy.METRICS) - measured.keys() == {"coco_ap_small", "coco_ar_small"}
    assert all(value == 0.5 for value in measured.values())
    with pytest.raises(SystemExit, match="no ground-truth objects"):
        accuracy.coco_stats(scorer)


@pytest.mark.parametrize("metric", ["AP_all", "AP_50", "AR_third"])
def test_subset_cannot_omit_global_metrics(metric: str, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    with pytest.raises(SystemExit, match="no ground-truth objects"):
        accuracy.coco_stats(_scorer(accuracy, **{metric: -1}), allow_missing_size_buckets=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1.1, -2, -0.5])
@pytest.mark.parametrize("metric", ["AP_all", "AP_small"])
def test_subset_rejects_invalid_metrics_instead_of_hiding_them(
    value: float, metric: str, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = load_script("accuracy")
    with pytest.raises(SystemExit):
        accuracy.coco_stats(_scorer(accuracy, **{metric: value}), allow_missing_size_buckets=True)


def test_accuracy_rejects_missing_required_coco_map(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    scorer = _scorer(accuracy)
    del scorer.stats_as_dict["AP_all"]
    with pytest.raises(KeyError, match="AP_all"):
        accuracy.coco_stats(scorer, allow_missing_size_buckets=True)


def test_performance_invalid_dataset_fails_before_model_loading_without_a_report(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts/performance.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mpk",
            "unused.tar.gz",
            "--dataset",
            "other=unused",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--dataset" in result.stderr
    assert not (tmp_path / "model-report.json").exists()


def test_performance_declaration_and_whole_detector_references_are_enabled_together():
    import yaml

    model = Path(__file__).resolve().parents[2]
    definition = yaml.safe_load((model / "model.yaml").read_text())
    tests = yaml.safe_load((model / "tests/tests.yaml").read_text())
    assert definition["performance"] == {}
    assert tests["performance"]["run_tests"] is True
    assert set(tests["performance"]["expected"]) == {"w16a16", "w8a8"}
    for profile in tests["performance"]["expected"].values():
        assert set(profile) == {"throughput_fps"}
        assert np.isfinite(profile["throughput_fps"]) and profile["throughput_fps"] > 0


# Native-route adapter tests deliberately retain the independent source-oracle tests above.
def _native_parser(module, monkeypatch, boxes):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(module.pyneat.detections, "read_detection_format", lambda tensor: "BBOX", raising=False)
    monkeypatch.setattr(module.pyneat.detections, "format_is_bbox", lambda value: value == "BBOX", raising=False)

    def parse(tensor, width, height, *, expected_topk, strict):
        calls.append((width, height, expected_topk, strict))
        return SimpleNamespace(boxes=boxes)

    monkeypatch.setattr(module.pyneat.detections, "decode_bbox_tensor", parse, raising=False)
    return calls


def _native_box(**values):
    from types import SimpleNamespace

    return SimpleNamespace(**dict(dict(x1=2, y1=3, x2=8, y2=9, score=0.8, class_id=90), **values))


def test_native_policy_is_explicit_and_uses_per_frame_image_geometry(load_script):
    module = load_script("accuracy")
    options = module.native_options()
    assert options.decode_type == module.pyneat.BoxDecodeType.Ssd
    assert options.num_classes == module.CLASSES == 91
    assert options.preprocess.kind == module.pyneat.InputKind.Image
    assert options.preprocess.resize.mode == module.pyneat.ResizeMode.Stretch
    assert options.preprocess.normalize.mean == [0.5, 0.5, 0.5]
    assert options.preprocess.normalize.stddev == [0.5, 0.5, 0.5]
    assert options.preprocess.normalize.has_explicit_stats is True
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)
    assert (options.boxdecode_original_width, options.boxdecode_original_height) == (0, 0)
    assert (options.score_threshold, options.nms_iou_threshold, options.top_k) == (
        module.SCORE_THRESHOLD,
        module.IOU_THRESHOLD,
        module.MAX_DETECTIONS,
    )


def test_native_rows_preserve_category_ids_and_original_geometry(load_script, monkeypatch):
    module = load_script("accuracy")
    calls = _native_parser(module, monkeypatch, [_native_box(class_id=18), _native_box(class_id=90)])
    rows = module.native_rows([object()], 139, 640, 480)
    np.testing.assert_array_equal(rows[:, 6], [18, 90])
    np.testing.assert_array_equal(rows[0, :5], [139, 2, 3, 6, 6])
    assert calls == [(0, 0, module.MAX_DETECTIONS, True)]


def test_native_empty_and_zero_area_or_one_pixel_overhang_are_not_repaired(load_script, monkeypatch):
    module = load_script("accuracy")
    _native_parser(module, monkeypatch, [])
    assert module.native_rows([object()], 1, 10, 10).shape == (0, 7)
    _native_parser(module, monkeypatch, [_native_box(x1=10, x2=11), _native_box(x1=4, x2=4)])
    rows = module.native_rows([object()], 1, 10, 10)
    assert rows[0, 1] + rows[0, 3] == 11
    assert rows[1, 3] == 0


@pytest.mark.parametrize(
    "values",
    [
        {"score": float("nan")},
        {"x1": float("inf")},
        {"class_id": 0},
        {"class_id": 91},
        {"class_id": 18.5},
        {"score": -1},
        {"score": 1.1},
        {"x1": -1},
        {"x2": 12},
        {"x2": 1},
    ],
)
def test_native_rejects_invalid_values_instead_of_fixing_them(load_script, monkeypatch, values):
    module = load_script("accuracy")
    _native_parser(module, monkeypatch, [_native_box(**values)])
    with pytest.raises(ValueError):
        module.native_rows([object()], 1, 10, 10)


def test_native_rejects_raw_heads_wrong_format_and_count(load_script, monkeypatch):
    module = load_script("accuracy")
    _native_parser(module, monkeypatch, [])
    for outputs in [[], [object()] * 12]:
        with pytest.raises(ValueError, match="typed BBOX"):
            module.native_rows(outputs, 1, 10, 10)
    monkeypatch.setattr(module.pyneat.detections, "read_detection_format", lambda tensor: "POSE")
    with pytest.raises(ValueError, match="typed BBOX"):
        module.native_rows([object()], 1, 10, 10)
    _native_parser(module, monkeypatch, [_native_box()] * (module.MAX_DETECTIONS + 1))
    with pytest.raises(ValueError, match="topK"):
        module.native_rows([object()], 1, 10, 10)


def test_native_a_b_a_geometry_and_loans_are_per_frame(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = load_script("accuracy")
    records = [dict(id=i + 1, width=w, height=h) for i, (w, h) in enumerate([(640, 480), (320, 240), (640, 480)])]
    monkeypatch.setattr(module, "read_rgb_record", lambda r, _: np.zeros((r["height"], r["width"], 3), np.uint8))
    monkeypatch.setattr(module.pyneat.Tensor, "from_numpy", lambda image, **kw: SimpleNamespace(image=image, kwargs=kw))
    outputs, seeds, shapes, closed = [], [], [], []

    class Runner:
        def run(self, inputs, timeout_ms):
            assert not outputs  # Prior producer loans were released before this push.
            (tensor,) = inputs
            h, w, _ = tensor.image.shape
            shapes.append((w, h))
            outputs.append(SimpleNamespace(boxes=[_native_box(x2=w, y2=h)]))
            return outputs

        def close(self):
            closed.append(True)

    def build(inputs, *, run_options):
        (seed,) = inputs
        seeds.append(seed.image.shape)
        assert run_options.startup_preflight is False
        return Runner()

    monkeypatch.setattr(module.pyneat.detections, "read_detection_format", lambda _: "BBOX", raising=False)
    monkeypatch.setattr(module.pyneat.detections, "format_is_bbox", lambda _: True, raising=False)
    monkeypatch.setattr(module.pyneat.detections, "decode_bbox_tensor", lambda output, *a, **kw: output, raising=False)
    rows = module.native_predictions(SimpleNamespace(build=build), records, tmp_path)
    assert seeds == [(720, 1280, 3)] and closed == [True] and not outputs
    assert shapes == [(640, 480), (320, 240), (640, 480)]
    assert [list(r[0, 1:5]) for r in rows] == [[2, 3, 638, 477], [2, 3, 318, 237], [2, 3, 638, 477]]


def test_native_rejects_capacity_overflow_before_build(load_script, tmp_path):
    from unittest.mock import Mock

    module = load_script("accuracy")
    model = Mock()
    with pytest.raises(ValueError, match="capacity"):
        module.native_predictions(model, [dict(id=1, width=1281, height=720)], tmp_path)
    model.build.assert_not_called()


def test_native_rgb_decode_checks_original_annotation_geometry(load_script, tmp_path):
    import cv2

    module = load_script("accuracy")
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), np.full((8, 12, 3), [10, 20, 30], np.uint8))
    record = dict(file_name=path.name, width=12, height=8)
    image = module.read_rgb_record(record, tmp_path)
    assert image.shape == (8, 12, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], [30, 20, 10])
    with pytest.raises(ValueError, match="geometry"):
        module.read_rgb_record(dict(record, width=13), tmp_path)


def test_native_all_empty_detector_constructs_real_empty_coco_result(load_script, monkeypatch):
    from types import SimpleNamespace

    module = load_script("accuracy")
    results = SimpleNamespace(createIndex=lambda: None)
    monkeypatch.setattr(module, "COCO", lambda: results)
    gt = SimpleNamespace(dataset={"images": [{"id": 1}], "categories": [{"id": 90}]})
    assert module.coco_results(gt, np.empty((0, 7))) is results
    assert results.dataset == dict(gt.dataset, annotations=[])


def test_native_runner_closes_on_inference_failure(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = load_script("accuracy")
    record = dict(id=1, width=16, height=8)
    monkeypatch.setattr(module, "read_rgb_record", lambda *args: np.zeros((8, 16, 3), np.uint8))
    monkeypatch.setattr(module.pyneat.Tensor, "from_numpy", lambda *args, **kwargs: object())
    closed = []

    def fail(*args, **kwargs):
        raise RuntimeError("inference failure")

    runner = SimpleNamespace(run=fail, close=lambda: closed.append(True))
    model = SimpleNamespace(build=lambda *args, **kwargs: runner)
    with pytest.raises(RuntimeError, match="inference failure"):
        module.native_predictions(model, [record], tmp_path)
    assert closed == [True]


@pytest.mark.parametrize("value", ["", "=", "coco-2017-val=", "coco-2017-val=   ", "other=/data", "coco-2017-val"])
def test_dataset_argument_is_one_named_nonempty_source(load_script, value):
    script = load_script("accuracy")
    with pytest.raises(SystemExit, match="--dataset"):
        script.parse_dataset([value])


def test_dataset_argument_rejects_repeats_instead_of_taking_the_last(load_script):
    script = load_script("accuracy")
    with pytest.raises(SystemExit, match="--dataset"):
        script.parse_dataset(["coco-2017-val=/first", "coco-2017-val=/second"])
    assert script.parse_dataset(["coco-2017-val=/data"]) == ("coco-2017-val", Path("/data"))


@pytest.mark.parametrize(
    "values", [[], [""], ["model="], ["model=  "], ["model"], ["other=/data"], ["model=/a", "model=/b"]]
)
def test_accuracy_requires_exactly_one_nonempty_component_path(load_script, values):
    script = load_script("accuracy")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(values)


@pytest.mark.parametrize("values", [[], [""], ["   "], ["/a", "/b"]])
def test_output_argument_rejects_empty_or_duplicate_values(load_script, values):
    script = load_script("accuracy")
    with pytest.raises(SystemExit, match="--output"):
        script.one_argument(values, "--output")


@pytest.mark.parametrize("flag,value", [("--dataset", "coco-2017-val=/again"), ("--output", "/again")])
def test_actual_parser_rejects_duplicate_options_before_model(load_script, monkeypatch, flag, value):
    script = load_script("accuracy")

    def no_model(*args, **kwargs):
        raise AssertionError("invalid command line must not construct a Model")

    monkeypatch.setattr(script.pyneat, "Model", no_model)
    mpk = "model=/model.tar.gz"
    with pytest.raises(SystemExit, match=flag):
        script.main(["--mpk", mpk, "--dataset", "coco-2017-val=/data", "--output", "/out", flag, value])


def test_native_multiclass_policy_matches_all_encoded_confidence_heads(load_script):
    module = load_script("accuracy")
    options = module.native_options()
    assert options.decode_type != module.pyneat.BoxDecodeType.YoloV26
    shapes = SourceReference().OUTPUT_SHAPES
    assert len(shapes) == 12
    localization, confidence = shapes[:6], shapes[6:]
    for loc, scores in zip(localization, confidence):
        assert loc[:3] == scores[:3]
        assert loc[-1] % 4 == 0
        assert scores[-1] == (loc[-1] // 4) * options.num_classes
