from __future__ import annotations

import importlib.util
import io
import itertools
import json
import math
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "yolo26x_obb"
SCRIPTS_DIR = MODEL_DIR / "scripts"
# Compiler and board modules that compile.py and accuracy.py import but these tests never call.
COMPILER_MODULES = [
    "afe",
    "afe.apis",
    "afe.apis.defines",
    "afe.apis.loaded_net",
    "afe.apis.model",
    "afe.core",
    "afe.core.utils",
    "afe.load",
    "afe.load.importers",
    "afe.load.importers.general_importer",
    "sima_utils",
    "sima_utils.data",
    "sima_utils.data.data_generator",
]
# CI installs no shapely; only the polygon-overlap tests need the real one, and they skip without it.
BOARD_MODULES = ["pyneat"] if importlib.util.find_spec("shapely") else ["pyneat", "shapely"]


def load_module(name: str, monkeypatch: pytest.MonkeyPatch | None = None, stubs: list[str] | None = None) -> ModuleType:
    for stub in stubs or []:
        monkeypatch.setitem(sys.modules, stub, MagicMock())
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"yolo26x_obb_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Sibling scripts are imported by name, so another model's calibration or surgery must not stand in for ours.
    shadowed = {dependency: sys.modules.pop(dependency, None) for dependency in ("calibration", "surgery")}
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        for dependency, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(dependency, None)
            else:
                sys.modules[dependency] = previous
    return module


def test_letterbox_geometry_matches_ultralytics_rounding() -> None:
    calibration = load_module("calibration")

    # An edge tile narrower than the input is padded, not scaled.
    assert calibration.letterbox_geometry(824, 1024) == (1.0, (824, 1024), (100, 0, 100, 0))
    # A whole image smaller than the input scales up.
    assert calibration.letterbox_geometry(800, 600) == (1.28, (1024, 768), (0, 128, 0, 128))
    # An odd padding puts the extra row at the bottom.
    assert calibration.letterbox_geometry(1024, 1023) == (1.0, (1024, 1023), (0, 0, 0, 1))


def _ultralytics_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Ultralytics' get_windows for one 1024-pixel crop with a 200-pixel gap, coverage filter included."""
    size, step = 1024, 824
    starts = []
    for side in (width, height):
        count = 1 if side <= size else math.ceil((side - size) / step + 1)
        values = [step * index for index in range(count)]
        if len(values) > 1 and values[-1] + size > side:
            values[-1] = side - size
        starts.append(values)
    windows = np.array([(x, y, x + size, y + size) for x, y in itertools.product(*starts)], dtype=np.float64)
    inside = windows.copy()
    inside[:, 0::2] = np.clip(inside[:, 0::2], 0, width)
    inside[:, 1::2] = np.clip(inside[:, 1::2], 0, height)
    rates = (inside[:, 2] - inside[:, 0]) * (inside[:, 3] - inside[:, 1]) / (size * size)
    if not (rates > 0.6).any():
        rates[abs(rates - rates.max()) < 0.01] = 1
    return [tuple(int(value) for value in window) for window in windows[rates > 0.6]]


def test_tile_windows_follow_the_ultralytics_dota_split() -> None:
    calibration = load_module("calibration")

    assert calibration.tile_windows(800, 600) == [(0, 0, 1024, 1024)]
    assert calibration.tile_windows(1147, 1023) == [(0, 0, 1024, 1024), (123, 0, 1147, 1024)]
    assert calibration.tile_windows(1490, 2137) == [
        (0, 0, 1024, 1024),
        (0, 824, 1024, 1848),
        (0, 1113, 1024, 2137),
        (466, 0, 1490, 1024),
        (466, 824, 1490, 1848),
        (466, 1113, 1490, 2137),
    ]
    # The coverage filter never drops a window, so tile_windows leaves it out.
    rng = np.random.default_rng(0)
    for width, height in [*rng.integers(200, 8000, (300, 2)), (1024, 1024), (1025, 300), (300, 1848)]:
        assert calibration.tile_windows(int(width), int(height)) == _ultralytics_windows(int(width), int(height))


LABELS = b"""imagesource:GoogleEarth
gsd:0.146
1100 100 1120 100 1120 120 1100 120 plane 0
1150 200 1170 200 1170 220 1150 220 ship 1
100 100 120 100 120 120 100 120 plane 0
"""


def test_label_centres_skip_the_header_lines() -> None:
    calibration = load_module("calibration")

    np.testing.assert_allclose(calibration.label_centres(LABELS), [[1110, 110], [1160, 210], [110, 110]])


def test_busiest_window_holds_the_most_object_centres() -> None:
    calibration = load_module("calibration")
    centres = calibration.label_centres(LABELS)

    assert calibration.busiest_window(2000, 1000, centres) == (824, 0, 1848, 1024)
    # Windows are half-open: a centre on the first window's right edge belongs to the next one.
    assert calibration.busiest_window(2000, 1000, np.array([[1024.0, 10.0]])) == (824, 0, 1848, 1024)
    # A tie goes to the first window.
    assert calibration.busiest_window(2000, 1000, np.array([[10.0, 10.0], [1900.0, 10.0]])) == (0, 0, 1024, 1024)


def _cal_archive(path: Path, images: dict[str, bytes], labels: dict[str, bytes], listing: list[str]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        files = {"dota/cal.txt": "\n".join(listing).encode() + b"\n"}
        files |= {f"dota/cal/images/{name}": content for name, content in images.items()}
        files |= {f"dota/cal/labelTxt/{name}": content for name, content in labels.items()}
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def test_calibration_names_are_the_first_listed_images(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    archive = tmp_path / "dota.tgz"
    _cal_archive(archive, {}, {}, ["P0002.jpg", "P0058.jpg", "P0066.jpg"])

    assert calibration.calibration_names(archive, 2) == ["P0002.jpg", "P0058.jpg"]
    with pytest.raises(ValueError, match="cannot select 4"):
        calibration.calibration_names(archive, 4)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no cal.txt"):
        calibration.calibration_names(empty, 1)


def test_calibration_tiles_crop_the_busiest_window(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    # The right window of a 2000x1000 image is red and holds two of the three objects.
    image = np.zeros((1000, 2000, 3), np.uint8)
    image[:, 1024:, 2] = 255
    _, content = cv2.imencode(".png", image)
    archive = tmp_path / "dota.tgz"
    _cal_archive(archive, {"P0001.png": content.tobytes()}, {"P0001.txt": LABELS}, ["P0001.png"])

    (tile,) = calibration.calibration_tiles(archive, 1)

    assert tile.shape == (1024, 1024, 3) and tile.dtype == np.float32
    # The 1024x1000 crop is padded by 12 rows above and turns red, in RGB order, from its column 200.
    np.testing.assert_array_equal(tile[11, 1000], [np.float32(114) / np.float32(255)] * 3)
    np.testing.assert_array_equal(tile[500, 1000], [1, 0, 0])
    np.testing.assert_array_equal(tile[500, 100], [0, 0, 0])


def _source_graph(constant_weight: bool = False) -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names, plus an unrelated output0 the cut drops."""
    surgery = load_module("surgery")
    helper = onnx.helper
    nodes, initializers = [], []
    for (name, head), shape in zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values()):
        stride = 1024 // shape[2]
        weight = np.zeros((shape[1], 3, stride, stride), np.float32)
        if constant_weight and name == "bbox_0":
            nodes.append(helper.make_node("Constant", [], [f"{name}_w"], value=onnx.numpy_helper.from_array(weight)))
        else:
            initializers.append(onnx.numpy_helper.from_array(weight, f"{name}_w"))
        nodes.append(helper.make_node("Conv", ["images", f"{name}_w"], [head], strides=[stride, stride]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 7), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 1024, 1024])],
        [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 7])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_surgery_cuts_at_the_nine_raw_heads_in_order(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    expected = [f"{name}_{i}" for name in ("bbox", "class_logit", "angle") for i in range(3)]
    assert list(outputs) == surgery.OUTPUTS == expected
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["bbox_0"] == [1, 4, 128, 128] and outputs["class_logit_1"] == [1, 15, 64, 64]
    assert outputs["angle_2"] == [1, 1, 32, 32]
    assert surgery.HEADS["angle_2"] == "/model.23/one2one_cv4.2/one2one_cv4.2.2/Conv_output_0"
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 9


def test_surgery_requires_a_folded_source(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(constant_weight=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    sizes = compile_module.OUTPUT_BYTES
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "images", "size": 1 * 3 * 1024 * 1024 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in sizes],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[3], outputs[6] = outputs[6], outputs[3]
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


def test_compile_reads_the_declared_build_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)

    assert compile_module._parameters("modalix_bf16") == {"precision": "bf16"}
    assert compile_module._parameters("modalix_int8") == {
        "precision": "int8",
        "calibration_bins": 2048,
        "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("modalix_fp32")


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:6]:
        logits[:] = -20.0
    return outputs


@pytest.mark.parametrize("angle", [0.0, 0.4, -1.2, math.pi / 2])
def test_decode_places_the_anchor_at_its_side_distances(monkeypatch: pytest.MonkeyPatch, angle: float) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    # Stride 16, row 15, column 20: the anchor centre is (20.5, 15.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 15, 20] = [1, 2, 3, 4]
    outputs[4][0, 15, 20, 9] = 3.0
    outputs[7][0, 15, 20, 0] = angle

    ((x, y, w, h, theta, score, label),) = accuracy.decode_tile(outputs)

    assert (theta, label) == (pytest.approx(angle), 9)
    np.testing.assert_allclose(score, 1 / (1 + np.exp(-3.0)), rtol=1e-6)
    np.testing.assert_allclose([w, h], [4 * 16, 6 * 16], rtol=1e-6)
    # In the box's own frame the anchor lies 1, 2, 3 and 4 cells from its left, top, right and bottom sides.
    cos, sin = math.cos(angle), math.sin(angle)
    dx, dy = 20.5 * 16 - x, 15.5 * 16 - y
    along, across = dx * cos + dy * sin, -dx * sin + dy * cos
    np.testing.assert_allclose(
        [w / 2 + along, h / 2 + across, w / 2 - along, h / 2 - across], [16, 32, 48, 64], atol=1e-4
    )


def test_decode_keeps_at_most_the_top_300_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, :2] = 2.0

    assert len(accuracy.decode_tile(outputs)) == accuracy.TOP_K == 300


def test_decode_returns_nothing_when_no_score_clears_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    assert accuracy.decode_tile(outputs).shape == (0, 7)
    # sigmoid(-6.95) is just below SCORE_THRESHOLD.
    outputs[5][0, :2, :2, 3] = -6.95

    assert accuracy.decode_tile(outputs).shape == (0, 7)


def _full_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultralytics' get_topk_index with a sigmoid over every logit and stable sorts, then the score threshold."""
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 15) for output in outputs[3:6]]))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > 0.001]
    return anchors[order // 15], order % 15, pairs[order]


@pytest.mark.parametrize("seed", range(4))
def test_decode_matches_the_full_sigmoid_selection(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape) < 0.01
        logits[hot] += rng.uniform(3, 30, hot.sum())
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 1, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[6:]]
    anchors, labels, scores = _full_selection(accuracy, outputs)
    angles = np.concatenate([output.reshape(-1) for output in outputs[6:]])

    detections = accuracy.decode_tile(outputs)

    np.testing.assert_array_equal(detections[:, 5], scores)
    np.testing.assert_array_equal(detections[:, 6], labels)
    np.testing.assert_array_equal(detections[:, 4], angles[anchors])


def test_top_cuts_a_stable_descending_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top(values, count), np.argsort(-values, kind="stable")[:count])


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (4, np.nan), (4, np.inf), (7, np.nan)])
def test_decode_rejects_non_finite_outputs(monkeypatch: pytest.MonkeyPatch, output: int, value: float) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 0] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode_tile(outputs)


def test_decode_rejects_another_output_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode_tile(outputs)


def test_image_detections_undo_the_letterbox_and_add_the_window_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    # An 800x600 tile is scaled by 1.28 and padded by 128 rows above; a 824x1024 edge tile is padded by 100 columns.
    tile = np.array([[512.0, 400.0, 128.0, 64.0, 0.3, 0.9, 2.0]])

    scaled = accuracy.image_detections(tile, (0, 0, 1024, 1024), 800, 600)
    padded = accuracy.image_detections(tile, (466, 1113, 1490, 2137), 824, 1024)

    np.testing.assert_allclose(scaled, [[400, 212.5, 100, 50, 0.3, 0.9, 2]])
    np.testing.assert_allclose(padded, [[878, 1513, 128, 64, 0.3, 0.9, 2]])


def test_probiou_matches_the_bhattacharyya_closed_form(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    box = np.array([[50.0, 40.0, 12.0, 6.0, 0.0]])
    # Equal covariances leave only the mean term: D = dx^2 / (8 * w^2 / 12) = 1/6 for dx = 4.
    shifted = box + [4.0, 0, 0, 0, 0]

    np.testing.assert_allclose(accuracy.probiou(box, shifted), [[1 - math.sqrt(1 - math.exp(-1 / 6))]], atol=1e-6)
    assert accuracy.probiou(box, box)[0, 0] > 0.999
    assert accuracy.probiou(box, box + [500.0, 0, 0, 0, 0])[0, 0] < 1e-3


def test_probiou_is_symmetric_and_turns_with_the_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(1)
    first = np.column_stack([rng.uniform(0, 50, (6, 2)), rng.uniform(2, 20, (6, 2)), rng.uniform(-1.5, 1.5, 6)])
    second = np.column_stack([rng.uniform(0, 50, (4, 2)), rng.uniform(2, 20, (4, 2)), rng.uniform(-1.5, 1.5, 4)])

    def turned(boxes: np.ndarray, angle: float) -> np.ndarray:
        cos, sin = math.cos(angle), math.sin(angle)
        x, y = boxes[:, 0], boxes[:, 1]
        return np.column_stack([x * cos - y * sin, x * sin + y * cos, boxes[:, 2:4], boxes[:, 4] + angle])

    overlaps = accuracy.probiou(first, second)
    np.testing.assert_allclose(overlaps, accuracy.probiou(second, first).T, atol=1e-9)
    np.testing.assert_allclose(overlaps, accuracy.probiou(turned(first, 0.7), turned(second, 0.7)), atol=1e-6)


def test_fast_nms_lets_a_dropped_box_still_drop_others(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    # Box 2 overlaps box 1, which overlaps box 0, but box 0 does not reach box 2. Greedy NMS would keep 1, 0 and 3.
    boxes = np.array([[8.0, 0, 10, 10, 0], [0, 0, 10, 10, 0], [4, 0, 10, 10, 0], [40, 0, 10, 10, 0]])
    overlaps = accuracy.probiou(boxes, boxes)
    assert overlaps[1, 0] < accuracy.MERGE_IOU <= min(overlaps[1, 2], overlaps[2, 0])

    assert accuracy.fast_nms(boxes, np.array([0.7, 0.9, 0.8, 0.1])).tolist() == [1, 3]


def test_fast_nms_blocks_match_one_full_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(2)
    boxes = np.column_stack([rng.uniform(0, 200, (500, 2)), rng.uniform(5, 40, (500, 2)), rng.uniform(-1.5, 1.5, 500)])
    scores = rng.random(500)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.triu(accuracy.probiou(boxes[order], boxes[order]) >= accuracy.MERGE_IOU, k=1).any(axis=0)

    monkeypatch.setattr(accuracy, "MERGE_BLOCK", 64)

    np.testing.assert_array_equal(accuracy.fast_nms(boxes, scores), order[~suppressed])


def test_merge_suppresses_overlaps_within_a_class_only(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    detections = np.array(
        [
            [0.0, 0, 10, 10, 0, 0.9, 1],
            [1.0, 0, 10, 10, 0, 0.8, 1],
            [1.0, 0, 10, 10, 0, 0.7, 4],
        ]
    )

    np.testing.assert_array_equal(accuracy.merge(detections), detections[[0, 2]])
    assert accuracy.merge(detections[:0]).shape == (0, 7)


def test_corners_follow_ultralytics_point_order(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    boxes = np.array([[10.0, 20.0, 4.0, 2.0, 0.0], [0.0, 0.0, 4.0, 2.0, math.pi / 2]])

    np.testing.assert_allclose(
        accuracy.corners(boxes),
        [[[12, 21], [12, 19], [8, 19], [8, 21]], [[-1, 2], [1, 2], [1, -2], [-1, -2]]],
        atol=1e-12,
    )


def test_ground_truth_reads_classes_and_difficult_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    content = b"imagesource:GoogleEarth\ngsd:0.1\n0 0 4 0 4 2 0 2 ship 1\n1 1 3 1 3 3 1 3 swimming-pool 0\n"

    polygons, labels, difficult = accuracy.ground_truth(content)

    np.testing.assert_array_equal(polygons[0], [[0, 0], [4, 0], [4, 2], [0, 2]])
    assert labels.tolist() == [1, 14] and difficult.tolist() == [True, False]


def test_best_overlaps_pick_the_first_of_equal_truths(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("shapely")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    square = np.array([[0.0, 0], [2, 0], [2, 2], [0, 2]])
    truths = np.stack([square + [1, 0], square - [1, 0], square + [10, 0]])
    detections = np.stack([square, square + [10, 0], square + [50, 0]])

    best, index = accuracy.best_overlaps(detections, truths)

    np.testing.assert_allclose(best, [1 / 3, 1, 0])
    assert index.tolist() == [0, 2, -1]


def test_class_ap_counts_dota_task1_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    # In score order: a hit on truth 0, a duplicate of it, a difficult truth 2, no overlap, then truth 1.
    best = np.array([0.8, 0.7, 0.9, 0.0, 0.6])
    truth = np.array([0, 0, 2, -1, 1])
    difficult = np.array([False, False, True])

    def ap(threshold: float) -> float:
        return accuracy.class_ap(best, truth, difficult, 2, threshold)

    # Recall reaches 0.5 at precision 1 and 1.0 at precision 0.5; the difficult match counts neither way.
    assert ap(0.5) == pytest.approx((6 * 1 + 5 * 0.5) / 11)
    assert ap(0.65) == pytest.approx(6 / 11)
    assert ap(0.85) == 0
    # An overlap must exceed the threshold.
    assert ap(0.6) == pytest.approx(6 / 11)


def test_score_averages_ap_over_labelled_classes_and_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("shapely")
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    (tmp_path / "P0001.txt").write_text("gsd:0.1\n100 100 130 100 130 110 100 110 plane 0\n")
    (tmp_path / "P0002.txt").write_text("200 200 230 200 230 210 200 210 ship 0\n")
    detections = {
        # The plane is found exactly; the ship box is shifted 6 pixels, IoU 240 / 360.
        "P0001.jpg": np.array([[115.0, 105, 30, 10, 0, 0.9, 0]]),
        "P0002.jpg": np.array([[221.0, 205, 30, 10, 0, 0.8, 1], [221.0, 205, 30, 10, 0, 0.7, 2]]),
    }

    metrics, class_ap50 = accuracy.score(detections, tmp_path)

    # The ship clears 0.50 to 0.65, four of the ten thresholds; the unlabelled third class is not averaged.
    assert class_ap50 == {"plane": 1.0, "ship": 1.0}
    assert metrics == {"dota_obb_map50": 1.0, "dota_obb_map": pytest.approx((1 + 0.4) / 2)}


def test_score_finds_each_truth_across_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)

    def axis_aligned_overlaps(detections: np.ndarray, truths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Every box here has angle 0, so axis-aligned IoU equals polygon IoU and CI needs no shapely.
        if not len(detections) or not len(truths):
            return np.zeros(len(detections)), np.full(len(detections), -1)
        found_low, found_high = detections.min(1)[:, None], detections.max(1)[:, None]
        truth_low, truth_high = truths.min(1), truths.max(1)
        overlap = (np.minimum(found_high, truth_high) - np.maximum(found_low, truth_low)).clip(0).prod(2)
        iou = overlap / ((found_high - found_low).prod(2) + (truth_high - truth_low).prod(1) - overlap)
        best = iou.max(1)
        return best, np.where(best > 0, iou.argmax(1), -1)

    monkeypatch.setattr(accuracy, "best_overlaps", axis_aligned_overlaps)
    (tmp_path / "P0001.txt").write_text("100 100 130 100 130 110 100 110 plane 0\n")
    (tmp_path / "P0002.txt").write_text(
        "200 200 230 200 230 210 200 210 plane 1\n0 0 30 0 30 10 0 10 ship 0\n300 300 330 300 330 310 300 310 plane 0\n"
    )
    detections = {
        # A ship where P0001 labels none ranks first, so it is the ship's only miss.
        "P0001.jpg": np.array([[115.0, 105, 30, 10, 0, 0.9, 0], [500.0, 500, 30, 10, 0, 0.95, 1]]),
        # P0002's truths follow P0001's, its difficult plane first: a truth taken from the wrong image or line
        # turns the ignored match into a duplicate and the found ship into a difficult one.
        "P0002.jpg": np.array(
            [[215.0, 205, 30, 10, 0, 0.8, 0], [315.0, 305, 30, 10, 0, 0.7, 0], [15.0, 5, 30, 10, 0, 0.6, 1]]
        ),
    }

    metrics, class_ap50 = accuracy.score(detections, tmp_path)

    assert class_ap50 == {"plane": 1.0, "ship": 0.5}
    assert metrics == {"dota_obb_map50": 0.75, "dota_obb_map": 0.75}


def test_model_declares_the_cut_interface_and_both_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    surgery = load_module("surgery")
    component = model["components"][0]
    entrypoint = component["interface"]["entrypoints"][0]

    assert [(item["name"], item["shape"]) for item in entrypoint["inputs"]] == [("images", surgery.INPUT_SHAPE)]
    assert {item["name"]: item["shape"] for item in entrypoint["outputs"]} == surgery.OUTPUT_SHAPES
    assert [item["name"] for item in entrypoint["outputs"]] == surgery.OUTPUTS
    assert component["surgery"]["entrypoint"] == "scripts/surgery.py:prepare_model"
    assert component["recipe"] == "scripts/compile.py"
    assert {profile["id"]: profile["components"][0]["artifact_suffix"] for profile in model["build_profiles"]} == {
        "modalix_bf16": "bf16",
        "modalix_int8": "int8_mse_h2048_bcr",
    }
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    entry = model["accuracy"][0]
    assert entry["metrics"] == list(load_module("accuracy", monkeypatch, BOARD_MODULES).METRICS)
    assert entry["required_metrics"] == ["dota_obb_map50"]
    assert "reference" not in entry
    assert (entry["dataset"]["id"], entry["dataset"]["variant"]) == ("dota-v1-0", "val")
    assert model["calibrations"][0]["dataset"]["variant"] == "cal"
