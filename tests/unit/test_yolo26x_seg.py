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
MODEL_DIR = ROOT / "models" / "yolo26x_seg"
SCRIPTS_DIR = MODEL_DIR / "scripts"
MASK_FIXTURE = ROOT / "tests" / "fixtures" / "yolo26x_seg_masks.json"
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
BOARD_MODULES = ["pyneat", "faster_coco_eval", "faster_coco_eval.core"]


def load_module(name: str, monkeypatch: pytest.MonkeyPatch | None = None, stubs: list[str] | None = None) -> ModuleType:
    for stub in stubs or []:
        monkeypatch.setitem(sys.modules, stub, MagicMock())
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"yolo26x_seg_{name}", path)
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


def test_image_boxes_invert_the_letterbox_and_clip(accuracy: ModuleType) -> None:
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


def _source_graph(constant_weight: bool = False, proto_op: str = "Mul") -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names and a SiLU-closed prototype output1, plus an unrelated
    output0 the cut drops."""
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
    initializers.append(onnx.numpy_helper.from_array(np.zeros((32, 3, 4, 4), np.float32), "proto_w"))
    nodes.append(helper.make_node("Conv", ["images", "proto_w"], ["proto_conv"], strides=[4, 4]))
    if proto_op == "Mul":
        nodes.append(helper.make_node("Sigmoid", ["proto_conv"], ["proto_gate"]))
        nodes.append(helper.make_node("Mul", ["proto_conv", "proto_gate"], ["output1"]))
    else:
        nodes.append(helper.make_node(proto_op, ["proto_conv"], ["output1"]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 38), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 38]),
            helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [1, 32, 160, 160]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_surgery_cuts_at_the_nine_raw_heads_and_the_prototypes_in_order(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    heads = [f"{name}_{level}" for name in ("bbox", "class_logit", "mask_coeff") for level in range(3)]
    assert list(outputs) == surgery.OUTPUTS == [*heads, "mask_proto"]
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["mask_coeff_2"] == [1, 32, 20, 20] and outputs["mask_proto"] == [1, 32, 160, 160]
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 10 + ["Sigmoid", "Mul"]


def test_surgery_requires_a_folded_source(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(constant_weight=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_requires_the_prototype_silu(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(proto_op="Relu"), source)

    with pytest.raises(ValueError, match="no SiLU Mul produces output1"):
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
        "calibration_bins": 1024,
        "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("modalix_fp32")


def _board_outputs(accuracy: ModuleType) -> list[np.ndarray]:
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for logits in outputs[3:6]:
        logits[:] = -20.0
    return outputs


@pytest.fixture
def accuracy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_module("accuracy", monkeypatch, BOARD_MODULES)
    # Uncompressed RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "frPyObjects", lambda rles, height, width: rles)
    return module


def _rle_image(rle: dict) -> np.ndarray:
    """Expand an uncompressed column-major COCO RLE, background first."""
    height, width = rle["size"]
    values = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).astype(bool)
    return values.reshape(width, height).T


def test_decode_selects_pairs_and_maps_boxes_like_ultralytics(accuracy: ModuleType) -> None:
    outputs = _board_outputs(accuracy)
    # Stride 16, row 3, column 5: the anchor centre is (5.5, 3.5) cells. Distances are left, top, right, bottom.
    outputs[1][0, 3, 5] = [1, 2, 3, 4]
    outputs[4][0, 3, 5, [2, 7]] = [3.0, 0.0]
    categories = np.arange(80) + 1

    results = accuracy.decode(outputs, 42, 640, 640, categories)

    expected_box = [(5.5 - 1) * 16, (3.5 - 2) * 16, 4 * 16, 6 * 16]
    assert [item["image_id"] for item in results] == [42, 42]
    np.testing.assert_allclose([item["bbox"] for item in results], [expected_box, expected_box])
    np.testing.assert_allclose([item["score"] for item in results], [1 / (1 + np.exp(-3.0)), 0.5])
    assert [item["category_id"] for item in results] == [3, 8]


def test_decode_keeps_at_most_the_top_300_pairs(accuracy: ModuleType) -> None:
    outputs = _board_outputs(accuracy)
    outputs[3][0, :5, :, 0] = 2.0

    results = accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)

    assert len(results) == accuracy.TOP_K == 300


def _full_selection(accuracy: ModuleType, outputs: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Select with a sigmoid over every logit and two stable sorts, the straightforward form of the selection."""
    distances = np.concatenate([output.reshape(-1, 4) for output in outputs[:3]])
    scores = accuracy.sigmoid(np.concatenate([output.reshape(-1, 80) for output in outputs[3:6]]))
    coefficients = np.concatenate([output.reshape(-1, 32) for output in outputs[6:9]])
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
    pairs = scores[anchors].ravel()
    order = np.argsort(-pairs, kind="stable")[:300]
    order = order[pairs[order] > 0.001]
    anchor, label = anchors[order // 80], order % 80
    centres = accuracy.ANCHORS[anchor]
    corners = np.hstack([centres - distances[anchor, :2], centres + distances[anchor, 2:]])
    return corners * accuracy.ANCHOR_STRIDES[anchor], pairs[order], label, coefficients[anchor]


@pytest.mark.parametrize("seed", range(4))
def test_select_matches_the_full_sigmoid_selection(accuracy: ModuleType, seed: int) -> None:
    rng = np.random.default_rng(seed)
    outputs = [rng.uniform(0, 6, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[:3]]
    for shape in accuracy.OUTPUT_SHAPES[3:6]:
        logits = rng.normal(-10, 3, shape)
        hot = rng.random(shape[:-1]) < 0.03
        logits[hot] += rng.uniform(3, 30, (hot.sum(), 80))
        # INT8 steps make ties; logits above about 17 all score exactly 1.0 in float32.
        outputs.append((np.round(logits / 0.15) * 0.15).astype(np.float32))
    outputs += [rng.normal(0, 1, shape).astype(np.float32) for shape in accuracy.OUTPUT_SHAPES[6:]]

    for actual, expected in zip(accuracy.select(outputs), _full_selection(accuracy, outputs)):
        np.testing.assert_array_equal(actual, expected)


def test_top_cuts_a_stable_descending_sort(accuracy: ModuleType) -> None:
    rng = np.random.default_rng(0)
    for size, count in [(0, 5), (4, 5), (5, 5), (1000, 300), (1000, 1)]:
        values = rng.integers(0, 7, size).astype(np.float32)
        np.testing.assert_array_equal(accuracy.top(values, count), np.argsort(-values, kind="stable")[:count])


def test_logit_floor_cannot_reach_the_score_threshold(accuracy: ModuleType) -> None:
    assert accuracy.sigmoid(np.float32(accuracy.LOGIT_FLOOR)) < accuracy.SCORE_THRESHOLD


@pytest.mark.parametrize("output", [0, 4, 7, 9])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_decode_rejects_non_finite_outputs(accuracy: ModuleType, output: int, value: float) -> None:
    outputs = _board_outputs(accuracy)
    outputs[output][0, 1, 2, 3] = value

    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_decode_rejects_another_output_layout(accuracy: ModuleType) -> None:
    outputs = [output.transpose(0, 3, 1, 2) for output in _board_outputs(accuracy)]

    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80) + 1)


def test_resize_sources_follow_torch_bilinear_without_corner_alignment(accuracy: ModuleType) -> None:
    lower, upper, weight = accuracy.resize_sources(160, 640)

    # Target pixel i samples source position (i + 0.5) / 4 - 0.5, clamped to the first and last pixel.
    position = np.clip((np.arange(640) + 0.5) / 4 - 0.5, 0, 159)
    np.testing.assert_allclose((1 - weight) * lower + weight * upper, position)
    assert (upper - lower <= 1).all() and (upper[weight == 0] == lower[weight == 0]).all()
    np.testing.assert_array_equal(accuracy.PROTO_UPSAMPLE.sum(axis=1), 1)
    lower, upper, weight = accuracy.resize_sources(426, 333)
    position = np.clip((np.arange(333) + 0.5) * 426 / 333 - 0.5, 0, 425)
    np.testing.assert_allclose((1 - weight) * lower + weight * upper, position, atol=1e-4)


def test_crop_bounds_round_and_wrap_like_slices_below_fifty_masks_and_use_exact_bounds_above(
    accuracy: ModuleType,
) -> None:
    boxes = np.array([[10.4, 10.6, 100.5, 639.7], [-0.6, -0.4, 700.0, 50.2]])

    # Rounded slice ends: -0.6 rounds to -1, which a slice reads as 639.
    np.testing.assert_array_equal(accuracy.crop_bounds(boxes), [[10, 11, 100, 640], [639, 0, 640, 50]])
    many = np.repeat(boxes, 25, axis=0)
    np.testing.assert_array_equal(accuracy.crop_bounds(many)[[0, -1]], [[11, 11, 101, 640], [0, 0, 640, 51]])


@pytest.mark.parametrize(("rows", "columns"), [((2, 6), (3, 8)), ((3, 7), (2, 6)), ((0, 7), (0, 9)), ((3, 7), (6, 9))])
def test_run_lengths_encode_column_major_with_background_first(
    accuracy: ModuleType, rows: tuple[int, int], columns: tuple[int, int]
) -> None:
    rng = np.random.default_rng(0)
    image = np.zeros((7, 9), dtype=bool)
    region = (slice(*rows), slice(*columns))
    image[region] = rng.random((rows[1] - rows[0], columns[1] - columns[0])) > 0.4
    # Runs that start the region and reach its end: the bottom row, before or at the image's last pixel.
    image[rows[0], columns[0]] = image[rows[1] - 1, columns[1] - 1] = True

    counts = accuracy.run_lengths(image[region], rows[0], columns[0], 7, 9)

    assert sum(counts) == 63 and all(counts[1:])
    np.testing.assert_array_equal(_rle_image({"size": [7, 9], "counts": counts}), image)


@pytest.mark.parametrize("case", [0, 1, 2])
def test_instance_masks_reproduce_the_stored_ultralytics_masks(accuracy: ModuleType, case: int) -> None:
    """The fixture holds Ultralytics 8.4.0's process_mask_native and scale_masks(...).byte() output for an
    upscaled and a downscaled image with rounded crops, and for 60 masks with exact crop bounds. Its integer
    prototypes keep every value exact before the threshold."""
    fixture = json.loads(MASK_FIXTURE.read_text(encoding="utf-8"))
    blocks = np.array(fixture["proto_blocks"], np.float32)
    proto = np.repeat(np.repeat(blocks, 32, axis=1), 32, axis=2).transpose(1, 2, 0)
    item = fixture["cases"][case]
    boxes = np.array(item["boxes"], np.float32)
    coefficients = np.array(item["coefficients"], np.float32)

    rles = accuracy.instance_masks(proto, boxes, coefficients, item["width"], item["height"])

    assert [rle["counts"] for rle in rles] == item["counts"]


def test_a_positive_mask_over_the_whole_frame_sets_every_image_pixel(accuracy: ModuleType) -> None:
    proto = np.ones((160, 160, 32), np.float32)

    (rle,) = accuracy.instance_masks(proto, np.array([[0.0, 0.0, 640.0, 640.0]]), np.ones((1, 32)), 640, 427)

    assert rle["counts"] == [0, 640 * 427]


def test_model_declares_the_cut_interface_and_both_builds(accuracy: ModuleType) -> None:
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
        "modalix_int8": "int8_mse_h1024_bcr",
    }
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    declared = model["accuracy"][0]
    assert declared["metrics"] == [name for metrics in accuracy.METRICS.values() for name in metrics]
    assert declared["required_metrics"] == ["coco_mask_map"]
    assert declared["reference"]["metrics"] == {"coco_mask_map": 0.470, "coco_box_map": 0.565}
