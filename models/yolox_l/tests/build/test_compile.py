"""Check raw-head surgery, BGR pixel-space calibration, and the compiled interface."""

import io
import json
import tarfile

import numpy as np
import onnx
import pytest
from model_registry.errors import ValidationError
from model_registry.model import load_model


def source_graph(surgery):
    helper = onnx.helper
    nodes, weights = [], []
    for index, ((name, weight), shape) in enumerate(zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values())):
        stride = surgery.IMAGE_SIZE // shape[2]
        weights.append(
            onnx.numpy_helper.from_array(np.full((shape[1], 3, 1, 1), (index + 1) / 100, np.float32), weight)
        )
        nodes.append(helper.make_node("Conv", [surgery.INPUT, weight], ["raw_" + name], strides=[stride, stride]))
        if "logit" in name:
            nodes.append(helper.make_node("Sigmoid", ["raw_" + name], ["prob_" + name]))
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["predictions"],
            value=onnx.numpy_helper.from_array(np.zeros(surgery.SOURCE_OUTPUT_SHAPES["predictions"], np.float32)),
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info(surgery.INPUT, onnx.TensorProto.FLOAT, surgery.INPUT_SHAPE)],
        [
            helper.make_tensor_value_info(
                "predictions", onnx.TensorProto.FLOAT, surgery.SOURCE_OUTPUT_SHAPES["predictions"]
            )
        ],
        weights,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)


def test_surgery_preserves_raw_values_and_interleaves_heads(tmp_path, load_script):
    import onnxruntime as ort

    surgery = load_script("surgery")
    graph = source_graph(surgery)
    source = tmp_path / "source.onnx"
    onnx.save(graph, source)
    rewritten = surgery.prepare_model(source, tmp_path / "out")
    cut = onnx.load(rewritten)
    assert [value.name for value in cut.graph.output] == surgery.OUTPUTS
    assert surgery.OUTPUTS == [f"{role}_{i}" for i in range(3) for role in ("bbox", "objectness_logit", "class_logit")]
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 9
    del graph.graph.output[:]
    graph.graph.output.extend(
        onnx.helper.make_tensor_value_info("raw_" + name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.OUTPUT_SHAPES.items()
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    a = ort.InferenceSession(graph.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(rewritten), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = {surgery.INPUT: np.random.default_rng(443).uniform(0, 255, surgery.INPUT_SHAPE).astype(np.float32)}
    for expected, actual in zip(a.run(None, inputs), b.run(None, inputs)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_surgery_rejects_a_missing_prediction_head(tmp_path, load_script):
    surgery = load_script("surgery")
    graph = source_graph(surgery)
    graph.graph.initializer[0].name = "other.weight"
    graph.graph.node[0].input[1] = "other.weight"
    path = tmp_path / "model.onnx"
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="prediction Conv"):
        surgery.prepare_model(path, tmp_path / "out")


def test_preprocessing_keeps_bgr_pixel_values_and_centered_padding(load_script):
    import cv2

    calibration = load_script("calibration")
    size = calibration.IMAGE_SIZE
    image = np.zeros((size // 2, size, 3), np.uint8)
    image[..., 0] = 255
    _, data = cv2.imencode(".png", image)
    result = calibration.preprocess_image(data.tobytes())
    assert result.dtype == np.float32 and result.shape == (size, size, 3)
    np.testing.assert_array_equal(result[size // 2, size // 2], [255, 0, 0])
    assert (result[: size // 4] == 114).all() and (result[-size // 4 :] == 114).all()


def test_calibration_uses_only_cal2017_and_is_reproducible(tmp_path, load_script):
    calibration = load_script("calibration")
    path = tmp_path / "data.tgz"
    with tarfile.open(path, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"{split}/{index}.jpg")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))
    names = calibration.calibration_names(path, 4)
    assert names == calibration.calibration_names(path, 4)
    assert len(set(names)) == 4 and all(name.startswith("cal2017/") for name in names)
    with pytest.raises(ValueError, match="cannot select"):
        calibration.calibration_names(path, 7)


def test_compile_profiles_and_sources_match_the_definition(model_dir, load_script):
    recipe = load_script("compile")
    model = load_model(model_dir)
    assert model["architecture"]["id"] == "yolox"
    assert model["components"][0]["source"]["type"] == "conversion"
    assert model["license"]["spdx"] == "Apache-2.0"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w8a8"]
    assert recipe.component_parameters(recipe.MODEL_DIR, "w8a8", recipe.COMPONENT_ID)["precision"] == "w8a8"
    for profile in ("w16a16", "w8a16", "unknown"):
        with pytest.raises(ValidationError, match=f"unknown build profile: {profile}"):
            recipe.component_parameters(recipe.MODEL_DIR, profile, recipe.COMPONENT_ID)


def write_mpk(path, recipe, case="valid"):
    outputs = [{"size": size} for size in recipe.OUTPUT_BYTES]
    if case == "order":
        outputs[0], outputs[1] = outputs[1], outputs[0]
    manifest = {
        "input_nodes": [{"name": key, "size": size} for key, size in recipe.INPUT_BYTES.items()],
        "plugins": [
            {"sequence": 0, "processor": "MLA", "config_params": {}},
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "pass_through"}, "output_nodes": outputs},
        ],
    }
    if case == "host":
        manifest["plugins"].insert(1, {"sequence": 1, "processor": "A65", "config_params": {}})
    with tarfile.open(path, "w:gz") as stream:
        for name, data in [("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("case", ["valid", "order", "host"])
def test_mpk_audit_rejects_wrong_output_order_or_cpu_fallback(tmp_path, case, load_script):
    recipe = load_script("compile")
    path = tmp_path / "model_mpk.tar.gz"
    write_mpk(path, recipe, case)
    if case == "valid":
        recipe.audit_mpk(path)
    else:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(path)


def test_real_compiled_component_has_the_declared_interface(compiled, load_script):
    archives = list(compiled.rglob("*_mpk.tar.gz"))
    assert len(archives) == 1
    load_script("compile").audit_mpk(archives[0])


# The raw-head output contract is checked on the host even before board accuracy baselines exist.
def outputs_for(accuracy):
    result = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    for index, output in enumerate(result):
        if index % 3:
            output[:] = -20
    return result


def test_decode_uses_xywh_objectness_and_one_best_class(load_script):
    accuracy = load_script("accuracy")
    outputs = outputs_for(accuracy)
    outputs[3][0, 3, 5] = [0.25, 0.5, np.log(4), np.log(2)]
    outputs[4][0, 3, 5] = 0
    outputs[5][0, 3, 5, [2, 7]] = [np.log(3), 0]
    rows = accuracy.decode(outputs, 42, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, np.arange(80) + 1)
    assert rows.shape == (1, 7)
    np.testing.assert_allclose(rows[0], [42, 52, 40, 64, 32, 0.375, 3], rtol=1e-6, atol=1e-6)


def test_nms_suppresses_same_class_and_preserves_other_classes(load_script):
    accuracy = load_script("accuracy")
    outputs = outputs_for(accuracy)
    for column, label, logit in [(5, 0, 2), (6, 0, 1), (7, 1, 1.5)]:
        outputs[6][0, 5, column] = [0, 0, np.log(8), np.log(8)]
        outputs[7][0, 5, column] = 3
        outputs[8][0, 5, column, label] = logit
    rows = accuracy.decode(outputs, 1, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, np.arange(80) + 1)
    assert len(rows) == 2
    np.testing.assert_array_equal(rows[:, 6], [1, 2])


def test_low_objectness_rejects_an_otherwise_confident_class(load_script):
    accuracy = load_script("accuracy")
    outputs = outputs_for(accuracy)
    outputs[2][0, 2, 2, 0] = 20
    rows = accuracy.decode(outputs, 1, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, np.arange(80) + 1)
    assert rows.shape == (0, 7)


def test_coordinate_mapping_inverts_padding_and_clips(load_script):
    accuracy = load_script("accuracy")
    width, height = 500, 375
    gain, _, (left, top, _, _) = accuracy.letterbox_geometry(width, height)
    original = np.array([[10, 20, 110, 220], [-10, -10, 600, 400]], dtype=float)
    transformed = original * gain + [left, top, left, top]
    np.testing.assert_allclose(
        accuracy.image_boxes(transformed, width, height), [[10, 20, 110, 220], [0, 0, width, height]]
    )


@pytest.mark.parametrize("index", [0, 1, 2, 8])
def test_nonfinite_outputs_are_rejected(index, load_script):
    accuracy = load_script("accuracy")
    outputs = outputs_for(accuracy)
    outputs[index].flat[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.decode(outputs, 1, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, np.arange(80) + 1)


def test_wrong_head_layout_is_rejected(load_script):
    accuracy = load_script("accuracy")
    outputs = [output.transpose(0, 3, 1, 2) for output in outputs_for(accuracy)]
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.decode(outputs, 1, accuracy.IMAGE_SIZE, accuracy.IMAGE_SIZE, np.arange(80) + 1)
