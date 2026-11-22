"""Static input, quantization, projection equivalence and real compiled-package contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from model_registry.errors import ValidationError
from model_registry.model import load_model


def test_model_uses_a_single_uncalibrated_w8a16_profile(model_dir):
    model = load_model(model_dir)
    assert [profile["id"] for profile in model["build_profiles"]] == ["w8a16"]
    assert not model["calibrations"]
    assert "calibration" not in model["build_profiles"][0]["components"][0]
    assert model["architecture"]["id"] == "depth_anything_3"


def test_compile_reads_the_declared_build_parameters(load_script):
    recipe = load_script("compile")

    parameters = recipe.component_parameters(recipe.MODEL_DIR, "w8a16", recipe.COMPONENT_ID)

    assert parameters == {"precision": "w8a16", "input_shape": [1, 3, 280, 504]}


def test_compile_rejects_an_undeclared_build_profile(load_script):
    recipe = load_script("compile")

    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(recipe.MODEL_DIR, "w32a32", recipe.COMPONENT_ID)


@pytest.mark.parametrize(
    ("source", "invocation", "message"),
    [
        ("model.onnx", "model", "a build profile compiles in one invocation named after it"),
        ("weights.pth", "w8a16", "--model must be the verified ONNX source"),
    ],
)
def test_compile_main_rejects_an_invocation_the_runner_would_not_make(
    load_script, tmp_path, monkeypatch, capsys, source, invocation, message
):
    recipe = load_script("compile")
    (tmp_path / source).touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(tmp_path / source), "--output-dir", str(tmp_path / "out"),
         "--build-profile", "w8a16", "--compile-invocation", invocation],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_preprocessing_preserves_rgb_and_applies_imagenet_normalization(load_script):
    cv2 = pytest.importorskip("cv2")
    preprocessing = load_script("preprocess")
    blue = np.zeros((480, 640, 3), dtype=np.uint8)
    blue[:, :, 0] = 255  # OpenCV's BGR input.
    _, encoded = cv2.imencode(".png", blue)
    image = preprocessing.preprocess_image(encoded.tobytes())
    assert image.shape == (preprocessing.HEIGHT, preprocessing.WIDTH, 3)
    assert image.dtype == np.float32 and image.flags.c_contiguous
    np.testing.assert_allclose(image[0, 0], (np.array([0, 0, 1]) - preprocessing.MEAN) / preprocessing.STD)


def test_fp32_baseline_rejects_an_incomplete_original_test_split(load_script, tmp_path, monkeypatch):
    baseline = load_script("fp32_baseline_accuracy")
    monkeypatch.setattr(baseline.accuracy, "FRAMES", 2)
    (tmp_path / "test").mkdir()
    (tmp_path / "test/rgb_0001.png").touch()
    (tmp_path / "provenance.json").write_text(json.dumps({"test_indices_one_based": [1, 9]}))
    with pytest.raises(ValueError, match="official test IDs"):
        baseline.original_frames(tmp_path)
    (tmp_path / "test/rgb_0009.png").touch()
    assert [path.name for path in baseline.original_frames(tmp_path)] == ["rgb_0001.png", "rgb_0009.png"]


def test_fp32_baseline_scores_original_float_depth_without_millimeter_rounding(load_script, tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    baseline = load_script("fp32_baseline_accuracy")
    scoring = baseline.accuracy
    monkeypatch.setattr(scoring, "FRAMES", 1)
    rows, columns = np.indices((scoring.HEIGHT, scoring.WIDTH))
    depth = (1 + rows * 0.001 + columns * 0.0005).astype(np.float32)
    _, (width, height), (left, top) = scoring.cover_geometry(640, 480)
    ys, xs = np.indices((480, 640))
    u = np.clip(np.floor((xs + 0.5) * width / 640 - left).astype(int), 0, scoring.WIDTH - 1)
    v = np.clip(np.floor((ys + 0.5) * height / 480 - top).astype(int), 0, scoring.HEIGHT - 1)
    truth = (1 / (0.8 / depth[v, u] + 0.1)).astype(np.float32)
    (tmp_path / "test").mkdir()
    assert cv2.imwrite(str(tmp_path / "test/rgb_0001.png"), np.zeros((480, 640, 3), np.uint8))
    np.save(tmp_path / "test/depth_0001.npy", truth)
    (tmp_path / "provenance.json").write_text(json.dumps({"test_indices_one_based": [1]}))

    def run(names, inputs):
        assert names == ["depth", "depth_conf"]
        assert inputs["image"].shape == (1, 3, scoring.HEIGHT, scoring.WIDTH)
        return [depth[None, None], np.ones_like(depth)[None, None]]

    monkeypatch.setattr(baseline.ort, "InferenceSession", lambda *args, **kwargs: SimpleNamespace(run=run))
    output = tmp_path / "baseline.json"
    baseline.main(["--onnx", "model.onnx", "--original-nyu-test", str(tmp_path), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["samples"] == 1
    assert result["metrics"]["abs_rel"] < 1e-6
    assert result["metrics"]["delta1"] == 1


def test_w8a16_uses_sima_per_token_quantization_without_calibration(load_script, monkeypatch):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "QuantizationParams", lambda **kwargs: kwargs)
    monkeypatch.setattr(recipe, "SkipCalibration", lambda: "skip")
    monkeypatch.setattr(recipe, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(recipe, "quantization_scheme", lambda **kwargs: kwargs)
    parameters = recipe.quantization()
    assert parameters["calibration_method"] == "skip"
    assert parameters["activation_quantization_scheme"] == "bf16"
    assert parameters["weight_quantization_scheme"] == {"asymmetric": False, "per_channel": True, "bits": 8}
    assert parameters["requantization_mode"] is recipe.RequantizationMode.sima
    assert parameters["per_token_quant"] is True


def test_qkv_split_preserves_head_and_token_order(load_script, monkeypatch):
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    nodes = [
        helper.make_node("Reshape", ["packed", "shape"], ["five_dims"], name="attention/Reshape"),
        helper.make_node("Transpose", ["five_dims"], ["qkv"], name="attention/Transpose", perm=[2, 0, 3, 1, 4]),
        *[
            helper.make_node("Gather", ["qkv", f"index{i}"], [name], name=f"attention/{name}", axis=0)
            for i, name in enumerate(("q", "k", "v"))
        ],
    ]
    constants = [numpy_helper.from_array(np.array([1, 5, 3, 2, 4], np.int64), "shape")]
    constants.extend(numpy_helper.from_array(np.array(i, np.int64), f"index{i}") for i in range(3))
    graph = helper.make_graph(
        nodes,
        "qkv",
        [helper.make_tensor_value_info("packed", TensorProto.FLOAT, [1, 5, 24])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, 5, 4]) for name in ("q", "k", "v")],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    inputs = {"packed": np.arange(120, dtype=np.float32).reshape(1, 5, 24)}
    expected = ReferenceEvaluator(model).run(None, inputs)
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    prepared = surgery.split_attention_qkv(model)
    actual = ReferenceEvaluator(prepared).run(None, inputs)
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert all(node.op_type != "Gather" for node in prepared.graph.node)


def test_camera_token_replacement_preserves_all_patch_tokens(load_script):
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    shape = [1, 1, 5, 4]
    graph = helper.make_graph(
        [helper.make_node("ScatterND", ["tokens", "indices", "camera"], ["output"], name="camera_assignment")],
        "camera",
        [helper.make_tensor_value_info("tokens", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        [
            numpy_helper.from_array(np.zeros((1, 1, 1, 3), np.int64), "indices"),
            numpy_helper.from_array(np.full((1, 1, 1, 4), -7, np.float32), "camera"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    inputs = {"tokens": np.arange(20, dtype=np.float32).reshape(shape)}
    expected = ReferenceEvaluator(model).run(None, inputs)
    prepared = surgery.replace_camera_token(model)
    actual = ReferenceEvaluator(prepared).run(None, inputs)
    np.testing.assert_array_equal(actual, expected)
    assert [node.op_type for node in prepared.graph.node] == ["Slice", "Concat"]


def test_projection_split_preserves_depth_and_confidence(load_script, monkeypatch):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "HEIGHT", 3)
    monkeypatch.setattr(surgery, "WIDTH", 5)
    random = np.random.RandomState(123)
    constants = [
        numpy_helper.from_array(random.normal(0, 0.1, (2, 32, 1, 1)).astype(np.float32), "weight"),
        numpy_helper.from_array(np.array([0.2, -0.4], np.float32), "bias"),
        *[
            numpy_helper.from_array(np.array(value, np.int64), name)
            for name, value in (
                ("start", [0]),
                ("end", [-1]),
                ("channel", [3]),
                ("step", [1]),
                ("confidence_channel", -1),
                ("output_shape", [1, 1, 3, 5]),
            )
        ],
        numpy_helper.from_array(np.array(1, np.float32), "one"),
    ]
    nodes = [
        helper.make_node("Conv", ["image", "weight", "bias"], ["logits"], name="projection"),
        helper.make_node("Transpose", ["logits"], ["nhwc"], name="layout", perm=[0, 2, 3, 1]),
        helper.make_node("Slice", ["nhwc", "start", "end", "channel", "step"], ["depth_logit"], name="depth_select"),
        helper.make_node("Exp", ["depth_logit"], ["depth_value"], name="depth_exp"),
        helper.make_node("Squeeze", ["depth_value", "channel"], ["depth_3d"], name="depth_squeeze"),
        helper.make_node("Reshape", ["depth_3d", "output_shape"], ["depth"], name="depth_shape"),
        helper.make_node("Gather", ["nhwc", "confidence_channel"], ["confidence_logit"], name="conf_select", axis=3),
        helper.make_node("Exp", ["confidence_logit"], ["confidence_exp"], name="conf_exp"),
        helper.make_node("Add", ["confidence_exp", "one"], ["confidence"], name="conf_add"),
        helper.make_node("Reshape", ["confidence", "output_shape"], ["depth_conf"], name="conf_shape"),
    ]
    graph = helper.make_graph(
        nodes,
        "head",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 32, 3, 5])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 3, 5]) for name in ("depth", "depth_conf")],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    inputs = {"image": random.normal(size=(1, 32, 3, 5)).astype(np.float32)}
    expected = ReferenceEvaluator(model).run(None, inputs)
    prepared = surgery.split_output_projection(model)
    actual = ReferenceEvaluator(prepared).run(None, inputs)
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    assert [node.op_type for node in prepared.graph.node] == ["Conv", "Exp", "Conv", "Exp", "Add"]


def test_compiled_package_keeps_computation_on_one_mla_segment(compiled: Path, model_dir, build_profile, load_script):
    archive = compiled / "model" / f"{model_dir.name}_{build_profile}_mpk.tar.gz"
    load_script("compile").audit_mpk(archive)
