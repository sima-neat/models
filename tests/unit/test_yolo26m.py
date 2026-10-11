from __future__ import annotations

import importlib.util
import io
import json
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
MODEL_DIR = ROOT / "models" / "yolo26m"
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
BOARD_MODULES = ["pyneat", "faster_coco_eval"]


def load_module(name: str, monkeypatch: pytest.MonkeyPatch | None = None, stubs: list[str] | None = None) -> ModuleType:
    for stub in stubs or []:
        monkeypatch.setitem(sys.modules, stub, MagicMock())
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"yolo26m_{name}", path)
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

    # 500x375 scales up by 1.28 to 640x480 and pads 80 rows above and below.
    assert calibration.letterbox_geometry(500, 375) == (1.28, (640, 480), (0, 80, 0, 80))
    # An odd padding puts the extra row at the bottom.
    assert calibration.letterbox_geometry(640, 427) == (1.0, (640, 427), (0, 106, 0, 107))
    assert calibration.letterbox_geometry(1280, 960) == (0.5, (640, 480), (0, 80, 0, 80))


def test_image_boxes_invert_the_letterbox_and_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(500, 375)
    original = np.array([[10.0, 20.0, 110.0, 220.0], [-5.0, -1.0, 600.0, 400.0]])

    mapped = accuracy.image_boxes(original * gain + [left, top, left, top], 500, 375)

    np.testing.assert_allclose(mapped, [[10, 20, 110, 220], [0, 0, 500, 375]])


def test_letterbox_centres_the_image_on_114_padding() -> None:
    pytest.importorskip("cv2")
    calibration = load_module("calibration")

    result = calibration.letterbox(np.full((427, 640, 3), 200, np.uint8))

    assert result.shape == (640, 640, 3)
    assert (result[:106] == 114).all() and (result[106:533] == 200).all() and (result[533:] == 114).all()


def test_preprocessing_returns_rgb_hwc_in_unit_range() -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    blue = np.zeros((320, 320, 3), np.uint8)
    blue[..., 0] = 255
    _, content = cv2.imencode(".png", blue)

    result = calibration.preprocess_image(content.tobytes())

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[320, 320], [0, 0, 1])


def test_calibration_draws_a_seeded_sample_from_cal2017_only(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"mini/{split}/{index:012d}.webp")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _source_graph(constant_weight: bool = False) -> onnx.ModelProto:
    """Six strided Convs at the Ultralytics head names, plus an unrelated output0 the cut drops."""
    surgery = load_module("surgery")
    helper = onnx.helper
    nodes, initializers = [], []
    for (name, head), shape in zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values()):
        stride = 640 // shape[2]
        weight = np.zeros((shape[1], 3, stride, stride), np.float32)
        if constant_weight and name == "bbox_0":
            nodes.append(helper.make_node("Constant", [], [f"{name}_w"], value=onnx.numpy_helper.from_array(weight)))
        else:
            initializers.append(onnx.numpy_helper.from_array(weight, f"{name}_w"))
        nodes.append(helper.make_node("Conv", ["images", f"{name}_w"], [head], strides=[stride, stride]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 6), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 6])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_surgery_cuts_at_the_six_raw_heads_in_order(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    assert list(outputs) == surgery.OUTPUTS == [f"bbox_{i}" for i in range(3)] + [f"class_logit_{i}" for i in range(3)]
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["bbox_0"] == [1, 4, 80, 80] and outputs["class_logit_2"] == [1, 80, 20, 20]
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 6


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
        "input_nodes": [{"name": "images", "size": 1 * 3 * 640 * 640 * 4}],
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
        outputs[0], outputs[3] = outputs[3], outputs[0]
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
    for logits in outputs[3:]:
        logits[:] = -20.0
    return outputs


def test_decode_selects_pairs_and_maps_boxes_like_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = [1, 2, 3, 4]
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    np.testing.assert_allclose(rows[:, :5], [[42, *expected_box], [42, *expected_box]])
    np.testing.assert_allclose(rows[:, 5], [1 / (1 + np.exp(-3.0)), 0.5])
    np.testing.assert_array_equal(rows[:, 6], [3, 8])


def test_decode_keeps_at_most_the_top_300_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, 0] = 2.0

    rows = accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert len(rows) == accuracy.TOP_K == 300


def _full_selection_rows(accuracy: ModuleType, outputs: list[np.ndarray], categories: np.ndarray) -> np.ndarray:
    """Decode with a sigmoid over every logit and two stable sorts, the straightforward form of the selection."""
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:]]))
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > 0.001]
    anchor, label = anchors[order // 80], order % 80
    centres = accuracy.ANCHORS[anchor]
    corners = np.hstack([centres - distances[anchor, :2], centres + distances[anchor, 2:]])
    x1, y1, x2, y2 = accuracy.image_boxes(corners * accuracy.ANCHOR_STRIDES[anchor], 640, 427).T
    return np.stack([np.full(order.shape, 9), x1, y1, x2 - x1, y2 - y1, pairs[order], categories[label]], axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_decode_matches_the_full_sigmoid_selection(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape[:-1]) < 0.03
        logits[hot] += rng.uniform(3, 30, (hot.sum(), 80))
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    categories = np.arange(80) + 1

    rows = accuracy.decode(outputs, 9, 640, 427, categories)

    np.testing.assert_array_equal(rows, _full_selection_rows(accuracy, outputs, categories))


def test_top_cuts_a_stable_descending_sort(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top(values, count), np.argsort(-values, kind="stable")[:count])


def test_logit_floor_cannot_reach_the_score_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)

    assert accuracy.sigmoid(np.float32(accuracy.LOGIT_FLOOR)) < accuracy.SCORE_THRESHOLD


@pytest.mark.parametrize(("output", "value"), [(0, np.nan), (0, np.inf), (4, np.nan), (4, np.inf)])
def test_decode_rejects_non_finite_outputs(monkeypatch: pytest.MonkeyPatch, output: int, value: float) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 3] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_decode_rejects_another_output_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_model_declares_the_cut_interface_and_both_builds() -> None:
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
    assert model["accuracy"][0]["required_metrics"] == ["coco_map"]
