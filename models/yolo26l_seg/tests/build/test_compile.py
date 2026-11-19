from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata
from model_registry.model_tests import load_model_tests

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "yolo26l_seg"


def test_letterbox_geometry_matches_ultralytics_rounding(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    # 500x375 scales up by 1.28 to 640x480 and pads 80 rows above and below.
    assert calibration.letterbox_geometry(500, 375) == (1.28, (640, 480), (0, 80, 0, 80))
    # An odd padding puts the extra row at the bottom.
    assert calibration.letterbox_geometry(640, 427) == (1.0, (640, 427), (0, 106, 0, 107))
    assert calibration.letterbox_geometry(1280, 960) == (0.5, (640, 480), (0, 80, 0, 80))


def test_letterbox_centres_the_image_on_114_padding(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    calibration = load_script("calibration")

    result = calibration.letterbox(np.full((427, 640, 3), 200, np.uint8))

    assert result.shape == (640, 640, 3)
    assert (result[:106] == 114).all() and (result[106:533] == 200).all() and (result[533:] == 114).all()


def test_preprocessing_returns_rgb_hwc_in_unit_range(load_script: Callable[[str], ModuleType]) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_script("calibration")
    blue = np.zeros((320, 320, 3), np.uint8)
    blue[..., 0] = 255
    _, content = cv2.imencode(".png", blue)

    result = calibration.preprocess_image(content.tobytes())

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[320, 320], [0, 0, 1])


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
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


ATTENTION_HEADS, KEY_DIM, VALUE_DIM = 2, 2, 4
ATTENTION_CHANNELS = ATTENTION_HEADS * VALUE_DIM


def _attention_blocks(count: int) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """A stride-32 Conv on images into after_0, then `count` Attention blocks as Ultralytics exports them, block
    `index` reading after_{index} and writing after_{index + 1}."""
    helper, rng = onnx.helper, np.random.default_rng(1)
    nodes, initializers = [], []
    head_channels = 2 * KEY_DIM + VALUE_DIM

    def constant(name: str, value: np.ndarray) -> str:
        initializers.append(onnx.numpy_helper.from_array(value, name))
        return name

    def conv(name: str, source: str, output: str, weight_shape: tuple[int, ...], **attributes: object) -> None:
        # Weights scaled by their fan-in keep every activation near one, where float32 rounding stays small.
        fan_in = np.prod(weight_shape[1:])
        weight = constant(f"{name}_w", (rng.standard_normal(weight_shape) / np.sqrt(fan_in)).astype(np.float32))
        bias = constant(f"{name}_b", rng.standard_normal(weight_shape[0]).astype(np.float32))
        nodes.append(helper.make_node("Conv", [source, weight, bias], [output], name=name, **attributes))

    conv("stem", "images", "after_0", (ATTENTION_CHANNELS, 3, 32, 32), strides=[32, 32])
    for index in range(count):
        at = f"attn{index}/"
        heads_shape = constant(at + "heads_shape", np.array([1, ATTENTION_HEADS, head_channels, 400]))
        image_shape = constant(at + "image_shape", np.array([1, ATTENTION_CHANNELS, 20, 20]))
        scale = constant(at + "scale", np.array(KEY_DIM**-0.5, np.float32))
        conv(at + "qkv", f"after_{index}", at + "qkv", (ATTENTION_HEADS * head_channels, ATTENTION_CHANNELS, 1, 1))
        nodes += [
            helper.make_node("Reshape", [at + "qkv", heads_shape], [at + "heads"], name=at + "Reshape"),
            helper.make_node(
                "Split",
                [at + "heads"],
                [at + "q", at + "k", at + "v"],
                name=at + "Split",
                axis=2,
                split=[KEY_DIM, KEY_DIM, VALUE_DIM],
            ),
            helper.make_node("Transpose", [at + "q"], [at + "q_rows"], name=at + "Transpose", perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", [at + "q_rows", at + "k"], [at + "scores"], name=at + "MatMul"),
            helper.make_node("Mul", [at + "scores", scale], [at + "scaled"], name=at + "Mul"),
            helper.make_node("Softmax", [at + "scaled"], [at + "weights"], name=at + "Softmax", axis=3),
            helper.make_node(
                "Transpose", [at + "weights"], [at + "weight_columns"], name=at + "Transpose_1", perm=[0, 1, 3, 2]
            ),
            helper.make_node("MatMul", [at + "v", at + "weight_columns"], [at + "values"], name=at + "MatMul_1"),
            helper.make_node("Reshape", [at + "values", image_shape], [at + "image"], name=at + "Reshape_1"),
            helper.make_node("Reshape", [at + "v", image_shape], [at + "v_image"], name=at + "Reshape_2"),
        ]
        conv(
            at + "pe",
            at + "v_image",
            at + "position",
            (ATTENTION_CHANNELS, 1, 3, 3),
            group=ATTENTION_CHANNELS,
            pads=[1] * 4,
        )
        nodes.append(helper.make_node("Add", [at + "image", at + "position"], [at + "sum"], name=at + "Add"))
        conv(at + "proj", at + "sum", f"after_{index + 1}", (ATTENTION_CHANNELS, ATTENTION_CHANNELS, 1, 1))
    return nodes, initializers


def _source_graph(
    load_script: Callable[[str], ModuleType],
    constant_weight: bool = False,
    attention_blocks: int | None = None,
    proto_op: str = "Mul",
) -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names and a SiLU-closed prototype output1, plus an unrelated
    output0 the cut drops."""
    surgery = load_script("surgery")
    helper = onnx.helper
    rng = np.random.default_rng(0)
    block_count = surgery.ATTENTION_BLOCKS if attention_blocks is None else attention_blocks
    nodes, initializers = _attention_blocks(block_count)
    for (name, head), shape in zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values()):
        # The last box head reads the attention blocks, so the cut keeps them.
        source, channels, stride = (
            (f"after_{block_count}", ATTENTION_CHANNELS, 1) if name == "bbox_2" else ("images", 3, 640 // shape[2])
        )
        weight = rng.standard_normal((shape[1], channels, stride, stride)).astype(np.float32) / (channels * stride**2)
        if constant_weight and name == "bbox_0":
            nodes.append(helper.make_node("Constant", [], [f"{name}_w"], value=onnx.numpy_helper.from_array(weight)))
        else:
            initializers.append(onnx.numpy_helper.from_array(weight, f"{name}_w"))
        nodes.append(helper.make_node("Conv", [source, f"{name}_w"], [head], strides=[stride, stride]))
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
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=7)


def test_surgery_cuts_at_the_nine_raw_heads_and_the_prototypes_in_order(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    pytest.importorskip("onnxruntime")
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    heads = [f"{name}_{level}" for name in ("bbox", "class_logit", "mask_coeff") for level in range(3)]
    assert list(outputs) == surgery.OUTPUTS == [*heads, "mask_proto"]
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["mask_coeff_2"] == [1, 32, 20, 20] and outputs["mask_proto"] == [1, 32, 160, 160]
    assert "Constant" not in {node.op_type for node in cut.graph.node}


def test_surgery_requires_a_folded_source(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, constant_weight=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_requires_the_prototype_silu(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, proto_op="Relu"), source)

    with pytest.raises(ValueError, match="no SiLU Mul produces output1"):
        surgery.prepare_model(source, tmp_path / "out")


def test_attention_rewrite_keeps_the_heads_with_the_softmax_over_channels(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    ort = pytest.importorskip("onnxruntime")
    surgery = load_script("surgery")
    source = _source_graph(load_script)
    onnx.save(source, tmp_path / "source.onnx")

    rewritten = onnx.load(surgery.prepare_model(tmp_path / "source.onnx", tmp_path / "out"))

    operations = [node.op_type for node in rewritten.graph.node]
    assert operations.count("Einsum") == 2 * surgery.ATTENTION_BLOCKS
    assert not {"MatMul", "Transpose"} & set(operations)
    softmax_axes = [
        onnx.helper.get_attribute_value(node.attribute[0]) for node in rewritten.graph.node if node.op_type == "Softmax"
    ]
    assert softmax_axes == [1] * surgery.ATTENTION_BLOCKS
    # The fixture's last box head is the output that reads the attention blocks.
    source.graph.output.append(onnx.helper.make_empty_tensor_value_info(surgery.HEADS["bbox_2"]))
    image = np.random.default_rng(2).random((1, 3, 640, 640), dtype=np.float32)
    expected, actual = (
        ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            [name], {"images": image}
        )[0]
        for model, name in ((source, surgery.HEADS["bbox_2"]), (rewritten, "bbox_2"))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_surgery_requires_every_attention_block(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, attention_blocks=surgery.ATTENTION_BLOCKS - 1), source)

    with pytest.raises(ValueError, match=f"expected {surgery.ATTENTION_BLOCKS} attention blocks"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    sizes = compile_module.OUTPUT_BYTES
    return {
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


@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[3] = outputs[3], outputs[0]
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a16"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    expected = {"precision": profile_id}
    if profile_id == "w8a8":
        expected |= {"calibration_bins": 1024, "calibration_samples": 50}

    assert load_script("compile")._parameters(profile_id) == expected


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w32a32")


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "frPyObjects", lambda rles, height, width: rles)
    return module


def test_model_preserves_compilation_and_quality_settings(accuracy_script: ModuleType) -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    declared = model["accuracy"]
    metric_names = {name for metrics in accuracy_script.METRICS.values() for name in metrics}
    for expected in load_model_tests(MODEL_DIR)["accuracy"]["expected"].values():
        assert set(expected["coco-2017-val"]) == metric_names
    assert list(declared["datasets"]) == ["coco-2017-val"]
    assert model["reference_metrics"][0]["metrics"] == {"coco_mask_map": 0.455, "coco_box_map": 0.544}


@pytest.mark.parametrize(
    ("precision", "calibrates", "per_token"), [("w8a8", True, False), ("w8a16", False, True), ("w16a16", False, False)]
)
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    assert arguments.get("per_token_quant", False) is per_token
    if calibrates:
        recipe.HistogramMSEMethod.assert_called_once_with(2048)


@pytest.mark.parametrize("precision", ["w16a16", "w8a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    precision: str,
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_images", lambda dataset, count: np.zeros((count, 640, 640, 3), np.float32))
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "cut.onnx")

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "coco.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert data is recipe.convert_data_generator_to_iterable.return_value
    else:
        assert data is None
