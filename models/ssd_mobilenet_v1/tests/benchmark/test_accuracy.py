from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class SourceReference:
    """Independent NumPy source decoder, used only by these tests."""

    CLASSES = 91

    GRIDS = (19, 10, 5, 3, 2, 1)

    BOXES_PER_CELL = (3, 6, 6, 6, 6, 6)

    PRIOR_WIDTHS = (
        (0.10000000149011612, 0.2828427255153656, 0.1414213627576828),
        (
            0.3499999940395355,
            0.4949747323989868,
            0.2474873661994934,
            0.6062177419662476,
            0.2020624876022339,
            0.41833001375198364,
        ),
        (0.5, 0.7071067690849304, 0.3535533845424652, 0.8660253882408142, 0.2886607050895691, 0.570087730884552),
        (
            0.6499999761581421,
            0.9192387461662292,
            0.4596193730831146,
            1.1258329153060913,
            0.3752588927745819,
            0.7211102247238159,
        ),
        (
            0.800000011920929,
            1.1313709020614624,
            0.5656854510307312,
            1.3856406211853027,
            0.4618571400642395,
            0.8717797994613647,
        ),
        (
            0.949999988079071,
            1.3435028791427612,
            0.6717514395713806,
            1.6454482078552246,
            0.5484553575515747,
            0.9746794104576111,
        ),
    )

    PRIOR_HEIGHTS = (
        (0.10000000149011612, 0.1414213627576828, 0.2828427255153656),
        (
            0.3499999940395355,
            0.2474873661994934,
            0.4949747323989868,
            0.20207259058952332,
            0.6062480807304382,
            0.41833001375198364,
        ),
        (0.5, 0.3535533845424652, 0.7071067690849304, 0.28867512941360474, 0.8660687208175659, 0.570087730884552),
        (
            0.6499999761581421,
            0.459619402885437,
            0.919238805770874,
            0.37527766823768616,
            1.1258893013000488,
            0.7211102247238159,
        ),
        (
            0.800000011920929,
            0.5656854510307312,
            1.1313709020614624,
            0.46188023686408997,
            1.3857098817825317,
            0.8717797994613647,
        ),
        (
            0.949999988079071,
            0.6717514395713806,
            1.3435028791427612,
            0.5484827756881714,
            1.6455304622650146,
            0.9746794104576111,
        ),
    )

    BOX_RECIPROCALS = np.asarray([0.1, 0.1, 0.2, 0.2], np.float32)

    SCORE_THRESHOLD = 1e-8

    IOU_THRESHOLD = 0.6

    MAX_DETECTIONS = 100

    RANKED_PAIRS = 1024

    def make_anchors(self) -> np.ndarray:
        """[1917, 4] (ymin, xmin, ymax, xmax) anchors in normalized coordinates, in the heads' cell-major order."""
        levels = []
        for size, widths, heights in zip(self.GRIDS, self.PRIOR_WIDTHS, self.PRIOR_HEIGHTS):
            widths, heights = np.asarray(widths, np.float32), np.asarray(heights, np.float32)
            step = np.float32(1.0) / np.float32(size)
            centers = np.arange(size, dtype=np.float32) * step + np.float32(0.5) * step
            center_y, center_x = (axis[..., None] for axis in np.meshgrid(centers, centers, indexing="ij"))
            boxes = np.stack(
                [center_y - heights / 2, center_x - widths / 2, center_y + heights / 2, center_x + widths / 2], axis=-1
            )
            levels.append(boxes.reshape(-1, 4))
        return np.concatenate(levels).astype(np.float32)

    def decode_boxes(self, encodings: np.ndarray) -> np.ndarray:
        """Decode [1917, 4] (ty, tx, th, tw) encodings to clipped normalized (ymin, xmin, ymax, xmax)."""
        anchor_height, anchor_width = self.ANCHORS[:, 2] - self.ANCHORS[:, 0], self.ANCHORS[:, 3] - self.ANCHORS[:, 1]
        anchor_y, anchor_x = self.ANCHORS[:, 0] + anchor_height / 2, self.ANCHORS[:, 1] + anchor_width / 2
        ty, tx, th, tw = (encodings * self.BOX_RECIPROCALS).T
        center_y, center_x = ty * anchor_height + anchor_y, tx * anchor_width + anchor_x
        height, width = np.exp(th) * anchor_height, np.exp(tw) * anchor_width
        boxes = np.stack([center_y - height / 2, center_x - width / 2, center_y + height / 2, center_x + width / 2], -1)
        return boxes.clip(0.0, 1.0)

    def _iou(self, box: np.ndarray, others: np.ndarray) -> np.ndarray:
        top, left = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
        bottom, right = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
        inter = np.maximum(bottom - top, 0) * np.maximum(right - left, 0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = area + areas - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    def ranked(self, values: np.ndarray, count: int) -> np.ndarray:
        """Indices of the count largest values, largest first, ties in index order, like a stable descending sort."""
        chosen = np.arange(len(values))
        if len(values) > count:
            cutoff = np.partition(values, len(values) - count)[len(values) - count]
            above = np.flatnonzero(values > cutoff)
            chosen = np.sort(np.concatenate([above, np.flatnonzero(values == cutoff)[: count - len(above)]]))
        return chosen[np.argsort(-values[chosen], kind="stable")]

    def select(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-class greedy NMS keeping the MAX_DETECTIONS best detections overall: (anchor indices, class indices).

        Visiting every anchor/class pair in descending score and suppressing it against the kept boxes of its own
        class gives TensorFlow's per-class NMS followed by its top-100 merge, without running NMS on every class.
        The visit usually ends within a few hundred pairs, so only the best RANKED_PAIRS pairs are sorted; when they
        cannot fill MAX_DETECTIONS, the walk restarts over four times as many, up to every pair.

        TensorFlow concatenates the classes' NMS results in class order before its stable top-100, so equal scores,
        common with INT8 logits, rank by class and then by anchor. Pairs are therefore flattened class-major.
        """
        anchor_count = scores.shape[0]
        # decode_boxes() has already clipped to the image. Boxes outside it can collapse to a line or point.
        positive_area = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
        flat = scores.T.ravel()
        count = self.RANKED_PAIRS
        while True:
            kept: dict[int, list[int]] = {}
            anchors, classes = [], []
            # Later pairs score no higher, so reaching the threshold also ends the walk.
            finished = count >= flat.size
            for pair in self.ranked(flat, count):
                if flat[pair] <= self.SCORE_THRESHOLD:
                    finished = True
                    break
                label, anchor = divmod(int(pair), anchor_count)
                if not positive_area[anchor]:
                    continue
                same_class = kept.setdefault(label, [])
                if same_class and (self._iou(boxes[anchor], boxes[same_class]) > self.IOU_THRESHOLD).any():
                    continue
                same_class.append(anchor)
                anchors.append(anchor)
                classes.append(label)
                if len(anchors) == self.MAX_DETECTIONS:
                    finished = True
                    break
            if finished:
                return np.asarray(anchors, np.int64), np.asarray(classes, np.int64)
            count *= 4

    def decode(self, outputs: Sequence[np.ndarray], image_id: int, width: int, height: int) -> np.ndarray:
        """Return COCO result rows [image_id, x, y, w, h, score, category_id] for one image."""
        shapes = [tuple(output.shape) for output in outputs]
        if shapes != self.OUTPUT_SHAPES:
            raise ValueError(f"unexpected MPK outputs: {shapes}")
        encodings = np.concatenate([output.reshape(-1, 4) for output in outputs[: len(self.GRIDS)]])
        logits = np.concatenate([output.reshape(-1, self.CLASSES) for output in outputs[len(self.GRIDS) :]])
        if not (np.isfinite(encodings).all() and np.isfinite(logits).all()):
            raise ValueError("MPK output contains non-finite values")
        boxes = self.decode_boxes(encodings)
        # Class 0 is the background; scores cover the 90 label-map ids.
        # Source sigmoid has no clipping; overflow of exp for a finite very
        # negative logit correctly produces zero probability.
        with np.errstate(over="ignore"):
            scores = 1.0 / (1.0 + np.exp(-logits[:, 1:]))
        anchor, label = self.select(boxes, scores)
        top, left, bottom, right = (boxes[anchor] * [height, width, height, width]).T
        ids = np.full(anchor.shape, image_id)
        return np.stack([ids, left, top, right - left, bottom - top, scores[anchor, label], label + 1], axis=1)

    def __init__(self):
        self.OUTPUT_SHAPES = [
            (1, size, size, boxes * channels)
            for channels in (4, self.CLASSES)
            for size, boxes in zip(self.GRIDS, self.BOXES_PER_CELL)
        ]
        self.ANCHORS = self.make_anchors()


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    """All-zero box encodings and class logits low enough that no anchor scores."""
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[len(accuracy.GRIDS) :]:
        logits[:] = -30.0
    return outputs


def test_anchors_follow_the_tensorflow_ssd_generator(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    anchors = accuracy.ANCHORS

    assert anchors.shape == (1917, 4)
    # The first anchor is the 0.1 square box centred on the first 19x19 cell.
    centre = 0.5 / 19
    np.testing.assert_allclose(anchors[0], [centre - 0.05, centre - 0.05, centre + 0.05, centre + 0.05], atol=1e-6)
    # The last anchor is the 1x1 cell's interpolated square box: sqrt(0.95 * 1.0) on each side.
    side = np.sqrt(0.95)
    np.testing.assert_allclose(anchors[-1], [0.5 - side / 2, 0.5 - side / 2, 0.5 + side / 2, 0.5 + side / 2], atol=1e-6)


def test_priors_match_original_export_bits(load_script: Callable[[str], ModuleType]) -> None:
    import hashlib

    accuracy = SourceReference()
    # Original V1 constant-only anchor graph; independently identical V2 priors.
    # This hash is not generated by the NumPy reconstruction under test.
    assert hashlib.sha256(accuracy.ANCHORS.astype("<f4").tobytes()).hexdigest() == (
        "1475439db129cb712b43dadf0c88aa53060614fed4509717cd4e306442eb8e27"
    )


def test_coder_matches_original_reciprocal_multiply(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    regression = np.zeros((1917, 4), np.float32)
    regression[543, 1] = np.asarray(0x3EA7E282, np.uint32).view(np.float32)
    actual = accuracy.decode_boxes(regression)[543].view(np.uint32)
    # Captured directly from the pinned original ONNX coder, in YXYX order.
    np.testing.assert_array_equal(actual, np.asarray([0x3EE66666, 0x3F01835C, 0x3F0CCCCD, 0x3F1B1CF6], np.uint32))


def test_nms_preserves_tiny_positive_union(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    boxes = np.tile(np.asarray([0, 0, 2**-22, 2**-22], np.float32), (2, 1))
    np.testing.assert_array_equal(accuracy._iou(boxes[0], boxes[1:]), [1.0])
    anchors, labels = accuracy.select(boxes, np.asarray([[0.9], [0.8]], np.float32))
    assert anchors.tolist() == [0] and labels.tolist() == [0]


def test_sigmoid_has_no_non_source_clipping(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    outputs = _board_outputs(accuracy)
    for head in outputs[6:]:
        head.fill(-1000)
    outputs[-1][0, 0, 0, 5 * 91 + 1] = -85
    accuracy.SCORE_THRESHOLD = 1e-36
    assert accuracy.decode(outputs, 1, 640, 480).shape == (0, 7)


def test_zero_encodings_decode_to_the_anchors_clipped_to_the_image(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()

    boxes = accuracy.decode_boxes(np.zeros((1917, 4), np.float32))

    np.testing.assert_allclose(boxes, accuracy.ANCHORS.clip(0, 1), atol=1e-6)


def test_decode_scales_boxes_to_the_image_and_keeps_label_map_ids(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    outputs = _board_outputs(accuracy)
    # The 1x1 level's square box, scored for class index 18, which is COCO category 18 (dog).
    outputs[11][0, 0, 0, 5 * 91 + 18] = 4.0

    rows = accuracy.decode(outputs, 7, 640, 480)

    (row,) = rows
    top, left, bottom, right = accuracy.ANCHORS[-1].clip(0, 1) * [480, 640, 480, 640]
    np.testing.assert_allclose(row, [7, left, top, right - left, bottom - top, 1 / (1 + np.exp(-4.0)), 18], rtol=1e-5)


def _per_class_nms(accuracy: ModuleType, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[int, int]]:
    """TensorFlow's form: greedy NMS within each class, then the best MAX_DETECTIONS over all classes."""
    found = []
    for label in range(scores.shape[1]):
        order = np.argsort(-scores[:, label], kind="stable")
        order = order[scores[order, label] > accuracy.SCORE_THRESHOLD]
        kept: list[int] = []
        for anchor in order:
            if kept and (accuracy._iou(boxes[anchor], boxes[kept]) > accuracy.IOU_THRESHOLD).any():
                continue
            kept.append(int(anchor))
        found += [(scores[anchor, label], anchor, label) for anchor in kept[:100]]
    found.sort(key=lambda item: -item[0])
    return [(anchor, label) for _, anchor, label in found[: accuracy.MAX_DETECTIONS]]


@pytest.mark.parametrize("seed", range(3))
def test_select_matches_per_class_nms(seed: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    rng = np.random.default_rng(seed)
    corners = rng.uniform(0, 0.8, (300, 2))
    boxes = np.hstack([corners, corners + rng.uniform(0.05, 0.3, (300, 2))]).astype(np.float32)
    scores = rng.uniform(0, 1, (300, 6)).astype(np.float32) ** 4

    anchors, labels = accuracy.select(boxes, scores)

    assert list(zip(anchors.tolist(), labels.tolist())) == _per_class_nms(accuracy, boxes, scores)


def test_select_ranks_tied_scores_by_class_as_tensorflow_merges_them(
    load_script: Callable[[str], ModuleType],
) -> None:
    accuracy = SourceReference()
    rng = np.random.default_rng(7)
    # Small, scattered boxes rarely overlap, and three score levels tie across the 100-detection cutoff.
    corners = rng.uniform(0, 0.95, (300, 2))
    boxes = np.hstack([corners, corners + 0.02]).astype(np.float32)
    scores = rng.choice(np.asarray([0.2, 0.5, 0.8], np.float32), (300, 6))

    anchors, labels = accuracy.select(boxes, scores)

    assert list(zip(anchors.tolist(), labels.tolist())) == _per_class_nms(accuracy, boxes, scores)
    assert len(anchors) == accuracy.MAX_DETECTIONS


def test_ranked_cuts_a_stable_descending_sort(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.ranked(values, count), np.argsort(-values, kind="stable")[:count])


@pytest.mark.parametrize("ranked_pairs", [1, 64, 1024])
def test_select_widens_the_ranking_until_it_fills_or_runs_out(
    ranked_pairs: int, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = SourceReference()
    monkeypatch.setattr(accuracy, "RANKED_PAIRS", ranked_pairs)
    rng = np.random.default_rng(5)
    # Two stacks of identical boxes: each class keeps one box per stack, so 12 survive out of 1800 pairs.
    boxes = np.repeat(np.asarray([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]], np.float32), 150, axis=0)
    scores = rng.uniform(0, 1, (300, 6)).astype(np.float32)
    scores[:, 5] = 0.0  # Never above the threshold.

    anchors, labels = accuracy.select(boxes, scores)

    assert list(zip(anchors.tolist(), labels.tolist())) == _per_class_nms(accuracy, boxes, scores)
    assert len(anchors) == 10


@pytest.mark.parametrize("output", [0, 6])
def test_decode_rejects_non_finite_outputs(output: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    outputs = _board_outputs(accuracy)
    outputs[output][0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 480)


def test_decode_rejects_another_output_layout(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = SourceReference()
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 480)


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")

    assert accuracy.parse_mpks(["model=/data/ssd_mpk.tar.gz"]) == Path("/data/ssd_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        accuracy.parse_mpks(["backbone=/data/ssd_mpk.tar.gz"])


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


@pytest.mark.parametrize("ranked_pairs", [1, 1024])
def test_select_discards_degenerate_boxes_before_the_topk_limit(
    ranked_pairs: int, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = SourceReference()
    monkeypatch.setattr(accuracy, "RANKED_PAIRS", ranked_pairs)
    monkeypatch.setattr(accuracy, "MAX_DETECTIONS", 2)
    # The first three high scores are a point, a horizontal line and a vertical line after clipping.
    # Reversed corners are invalid too, even though multiplying two negative lengths gives positive area.
    boxes = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0.5],
            [0, 0, 0.5, 0],
            [0.8, 0.8, 0.2, 0.2],
            [0.1, 0.1, 0.3, 0.3],
            [0.6, 0.6, 0.9, 0.9],
        ],
        np.float32,
    )
    scores = np.asarray([[0.99], [0.98], [0.97], [0.96], [0.8], [0.7]], np.float32)

    anchors, labels = accuracy.select(boxes, scores)

    np.testing.assert_array_equal(anchors, [4, 5])
    np.testing.assert_array_equal(labels, [0, 0])


@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_decode_prunes_high_score_boxes_that_collapse_after_clipping(
    offset: float, load_script: Callable[[str], ModuleType]
) -> None:
    accuracy = SourceReference()
    outputs = _board_outputs(accuracy)
    outputs[0][0, 0, 0, :4] = [offset, offset, 0, 0]
    outputs[6][0, 0, 0, 1] = 10.0
    outputs[11][0, 0, 0, 5 * 91 + 18] = 4.0

    rows = accuracy.decode(outputs, 7, 640, 480)

    assert rows.shape == (1, 7)
    assert rows[0, -1] == 18  # The in-frame dog, not the higher-scoring collapsed box.
    assert rows[0, 3] > 0 and rows[0, 4] > 0


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
