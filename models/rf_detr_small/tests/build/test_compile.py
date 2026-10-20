from __future__ import annotations

import builtins
import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "rf_detr_small"
SCRIPTS_DIR = MODEL_DIR / "scripts"


def test_preprocessing_produces_normalized_nchw_image(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    image = Image.new("RGB", (16, 8), color=(255, 128, 0))
    content = io.BytesIO()
    image.save(content, format="PNG")

    result = calibration.preprocess_image(content.getvalue())

    assert result.shape == (3, 512, 512)
    assert result.dtype == np.float32
    expected = (np.asarray([1.0, 128 / 255, 0.0], dtype=np.float32) - calibration.MEAN[:, 0, 0]) / calibration.STD[
        :, 0, 0
    ]
    np.testing.assert_allclose(result[:, 0, 0], expected)


def test_surgery_import_does_not_require_onnxruntime(monkeypatch, load_script: Callable[[str], ModuleType]) -> None:
    original_import = builtins.__import__

    def without_runtime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_runtime)
    assert callable(load_script("surgery").prepare_model)


def test_image_selection_is_seeded_and_disjoint(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    for index in range(8):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(tmp_path / f"{index}.jpg")

    calibration_names = calibration.select_images(tmp_path, seed=42, start=0, count=3)
    fidelity_names = calibration.select_images(tmp_path, seed=42, start=3, count=3)

    assert calibration.select_images(tmp_path, seed=42, start=0, count=6) == calibration_names + fidelity_names
    assert len(fidelity_names) == 3
    with pytest.raises(ValueError, match="8 images, but 9 are required"):
        calibration.select_images(tmp_path, seed=42, start=3, count=6)


def test_afe_layout_converts_only_rank_four_tensors(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    images = np.zeros((2, 3, 5, 7), dtype=np.float32)
    proposals = np.zeros((2, 300, 4), dtype=np.float32)

    converted_images = calibration.to_afe_layout(images)
    converted_proposals = calibration.to_afe_layout(proposals)

    assert converted_images.shape == (2, 5, 7, 3)
    assert converted_images.flags.c_contiguous
    assert converted_proposals.shape == proposals.shape
    assert converted_proposals.flags.c_contiguous


def test_coco_archive_selection_and_webp_decoding(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    image = io.BytesIO()
    Image.new("RGB", (2, 2), color=(64, 128, 192)).save(image, format="WEBP")
    names = [f"val2017/{index:012d}.webp" for index in range(8)]
    selections = []
    for index, members in enumerate((names, list(reversed(names)))):
        dataset = tmp_path / f"coco-{index}.tgz"
        with tarfile.open(dataset, "w:gz") as archive:
            for name in members:
                member = tarfile.TarInfo(name)
                member.size = len(image.getvalue())
                archive.addfile(member, io.BytesIO(image.getvalue()))
        selected = calibration.select_images(dataset, seed=42, start=0, count=3)
        selections.append(selected)
        assert calibration.load_images(dataset, selected).shape == (3, 3, 512, 512)
    assert selections[0] == selections[1]


def test_top_proposals_keep_equal_scores_in_order(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    scores = np.asarray([[0.5, 0.9, 0.9, 0.1]], dtype=np.float32)
    boxes = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    result = calibration.select_top_proposals(scores, boxes, count=3)

    np.testing.assert_array_equal(result, boxes[:, [1, 2, 0]])


def _value(name: str, shape: list[int], dtype: int = onnx.TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=()) -> onnx.ModelProto:
    graph = onnx.helper.make_graph(nodes, "fixture", inputs, outputs, list(initializers))
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def test_surgery_requires_a_folded_source(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    model = _model(
        [onnx.helper.make_node("Shape", ["x"], ["shape"], name="shape")],
        [_value("x", [2])],
        [_value("shape", [1], onnx.TensorProto.INT64)],
    )

    with pytest.raises(ValueError, match="fold it with onnxsim"):
        surgery._require_folded(model)


@pytest.mark.parametrize("selects", [False, True])
def test_surgery_bypasses_only_noop_masks(selects: bool, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    mask = np.zeros((1, 4), dtype=bool)
    mask[0, 0] = selects
    grid = np.stack(np.meshgrid(np.arange(1), np.arange(3), indexing="ij"), -1).astype(np.int64)
    model = _model(
        [
            onnx.helper.make_node("Where", ["mask", "fill", "x"], ["masked"], name="where"),
            onnx.helper.make_node("ScatterND", ["data", "indices", "u"], ["scattered"], name="scatter"),
            onnx.helper.make_node("Relu", ["masked"], ["y"], name="y"),
            onnx.helper.make_node("Relu", ["scattered"], ["z"], name="z"),
        ],
        [_value("x", [1, 4]), _value("u", [1, 3, 2])],
        [_value("y", [1, 4]), _value("z", [1, 3, 2])],
        [
            onnx.numpy_helper.from_array(mask, "mask"),
            onnx.numpy_helper.from_array(np.zeros((1, 4), dtype=np.float32), "fill"),
            onnx.numpy_helper.from_array(np.zeros((1, 3, 2), dtype=np.float32), "data"),
            onnx.numpy_helper.from_array(grid, "indices"),
        ],
    )
    expected = {"Where": 1, "ScatterND": 1}

    if selects:
        with pytest.raises(ValueError, match="selects elements"):
            surgery._remove_noop_masks(model, expected)
        return
    result = surgery._remove_noop_masks(model, expected)

    assert [(node.op_type, list(node.input)) for node in result.graph.node] == [("Relu", ["x"]), ("Relu", ["u"])]
    assert not result.graph.initializer


def test_surgery_bypasses_identities_and_unused_stacked_layers(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    model = _model(
        [
            onnx.helper.make_node("Unsqueeze", ["a", "axis"], ["stack_a"], name="stack_a"),
            onnx.helper.make_node("Unsqueeze", ["b", "axis"], ["stack_b"], name="stack_b"),
            onnx.helper.make_node("Concat", ["stack_a", "stack_b"], ["stacked"], name="stacked", axis=0),
            onnx.helper.make_node("Gather", ["stacked", "last"], ["selected"], name="selected", axis=0),
            onnx.helper.make_node("Cast", ["a"], ["cast"], name="cast", to=onnx.TensorProto.FLOAT),
            onnx.helper.make_node("Relu", ["selected"], ["y"], name="y"),
            onnx.helper.make_node("Relu", ["cast"], ["z"], name="z"),
        ],
        [_value("a", [1, 2]), _value("b", [1, 2])],
        [_value("y", [1, 2]), _value("z", [1, 2])],
        [
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "axis"),
            onnx.numpy_helper.from_array(np.array(-1, dtype=np.int64), "last"),
        ],
    )

    result = surgery._remove_identities(model, {"Gather": 1, "Cast": 1})

    assert [(node.op_type, list(node.input)) for node in result.graph.node] == [("Relu", ["b"]), ("Relu", ["a"])]


def _attention_block(factor: float) -> onnx.ModelProto:
    rng = np.random.RandomState(0)
    prefix = "/backbone/layer.0/attention"
    return _model(
        [
            onnx.helper.make_node("MatMul", ["x", "wq"], ["q_linear"], name=f"{prefix}/q/MatMul"),
            onnx.helper.make_node("Add", ["bq", "q_linear"], ["q_biased"], name=f"{prefix}/q/Add"),
            onnx.helper.make_node("Reshape", ["q_biased", "heads"], ["q_heads"], name=f"{prefix}/q/Reshape"),
            onnx.helper.make_node("Transpose", ["q_heads"], ["q"], name=f"{prefix}/q/Transpose", perm=[0, 2, 1, 3]),
            onnx.helper.make_node("Reshape", ["x", "heads"], ["k_heads"], name=f"{prefix}/k/Reshape"),
            onnx.helper.make_node("Transpose", ["k_heads"], ["k"], name=f"{prefix}/k/Transpose", perm=[0, 2, 3, 1]),
            onnx.helper.make_node("MatMul", ["q", "k"], ["scores"], name=f"{prefix}/MatMul"),
            onnx.helper.make_node("Mul", ["scores", "scale"], ["scaled"], name=f"{prefix}/Mul"),
            onnx.helper.make_node("Softmax", ["scaled"], ["y"], name=f"{prefix}/Softmax", axis=-1),
        ],
        [_value("x", [1, 4, 8])],
        [_value("y", [1, 2, 4, 4])],
        [
            onnx.numpy_helper.from_array(rng.standard_normal((8, 8)).astype(np.float32), "wq"),
            onnx.numpy_helper.from_array(rng.standard_normal(8).astype(np.float32), "bq"),
            onnx.numpy_helper.from_array(np.array([1, 4, 2, 4], dtype=np.int64), "heads"),
            onnx.numpy_helper.from_array(np.array(factor, dtype=np.float32), "scale"),
        ],
    )


def test_surgery_folds_the_backbone_score_scale_into_q(load_script: Callable[[str], ModuleType]) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    model = _attention_block(0.125)
    x = np.random.RandomState(1).standard_normal((1, 4, 8)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]

    result = surgery._fold_backbone_score_scale(model, 1)

    assert "Mul" not in {node.op_type for node in result.graph.node}
    np.testing.assert_array_equal(ReferenceEvaluator(result).run(None, {"x": x})[0], expected)


def test_surgery_folds_only_power_of_two_score_scales(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")

    with pytest.raises(ValueError, match="power-of-two scale"):
        surgery._fold_backbone_score_scale(_attention_block(0.1), 1)


def test_compile_recipe_uses_one_grouped_entrypoint_and_direct_afe_api() -> None:
    source = (SCRIPTS_DIR / "compile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--compile-invocation", required=True)' in source
    assert "export_model" not in source
    assert "prepare_model(source" in source
    assert "loaded.quantize(" in source
    assert "any_shape_on_mla=" not in source
    assert "automatic_layout_conversion=" not in source
    assert "flexible_batch_size=True" in source
    assert "quantized.compile(" in source
    assert "compile-outputs.json" in source
    assert '"derived_graphs"' in source
    assert '"correctness"' in source
    assert "shutil.copyfile" in source
    assert "requests" not in source
    assert "urllib" not in source
    assert 'get_context("spawn")' in source
    assert source.index('"RF-DETR backbone compilation"') < source.index('"RF-DETR decoder compilation and fidelity"')


def _metadata() -> dict:
    return yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))


def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    parameters = compile_module._component_parameters(_metadata(), "w8a16_w16a16")

    assert parameters == {
        "backbone": {"precision": "w8a16", "per_token": True},
        "decoder": {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 50},
    }
    assert compile_module._pipeline_description(parameters) == (
        "AFE W8A16 per-token backbone → CPU TopK/Gather → AFE W16A16 decoder (JAX execute)"
    )
    with pytest.raises(ValueError, match="no build profile 'w32a32'"):
        compile_module._component_parameters(_metadata(), "w32a32")
    with pytest.raises(ValueError, match="unsupported precision: int8"):
        compile_module._quantization_config({"precision": "int8"})
    with pytest.raises(ValueError, match="W8A8 component must declare calibration_bins"):
        compile_module._quantization_config({"precision": "w8a8"})


def test_the_decoder_calibrates_in_every_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    # AFE runs the decoder's query sine embedding as an INT8 lookup table even at W16A16.
    compile_module = load_script("compile")

    for profile in _metadata()["build_profiles"]:
        parameters = compile_module._component_parameters(_metadata(), profile["id"])
        assert compile_module._calibrates(parameters["decoder"]), profile["id"]


def test_only_calibrated_components_select_calibration_images(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    for index in range(8):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(tmp_path / f"{index}.jpg")
    calibrated = {
        "backbone": {"precision": "w8a16", "per_token": True},
        "decoder": {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 3},
    }
    uncalibrated = {"backbone": {"precision": "w8a16"}, "decoder": {"precision": "w16a16"}}
    selection = {"seed": 42, "offset": 3, "samples": 3}

    assert compile_module._calibration_names(uncalibrated, None, tmp_path, selection) == ()
    names = compile_module._calibration_names(calibrated, tmp_path, tmp_path, selection)
    assert names == compile_module.select_images(tmp_path, 42, 0, 3)
    with pytest.raises(ValueError, match="require --dataset"):
        compile_module._calibration_names(calibrated, None, tmp_path, selection)
    mismatched = {**calibrated, "backbone": {"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 4}}
    with pytest.raises(ValueError, match="must use the same images"):
        compile_module._calibration_names(mismatched, tmp_path, tmp_path, selection)
    with pytest.raises(ValueError, match="begin after the calibration images"):
        compile_module._calibration_names(calibrated, tmp_path, tmp_path, {**selection, "offset": 2})


def test_bf16_activation_components_quantize_without_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    loaded = MagicMock()
    monkeypatch.setattr(compile_module, "onnx_source", MagicMock())
    monkeypatch.setattr(compile_module, "load_model", MagicMock(return_value=loaded))

    compile_module._quantize_component(
        tmp_path / "graph.onnx", "backbone", {"precision": "w16a16"}, None, {"input": (1, 3, 8, 8)}
    )

    assert loaded.quantize.call_args.kwargs["calibration_data"] is None


@pytest.mark.parametrize(("precision", "fp32_intermediates"), [("w8a8", False), ("w8a16", True), ("w16a16", True)])
def test_bf16_activation_layer_norms_keep_fp32_intermediates(
    precision: str, fp32_intermediates: bool, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    quantized = MagicMock()

    with pytest.raises(RuntimeError, match="expected one backbone MPK"):
        compile_module._compile_component(quantized, tmp_path / "backbone", "backbone", precision)

    assert quantized.compile.call_args.kwargs["layer_norm_use_fp32_intermediates"] is fp32_intermediates


def test_fidelity_error_sums_match_the_metrics_over_all_images(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    rng = np.random.RandomState(0)
    expected = rng.standard_normal((2, 3, 4))
    actual = expected + rng.standard_normal((2, 3, 4)) * 0.1
    error = compile_module._OutputError("dets")

    for image in range(2):
        error.add(expected[image], actual[image])
    summary = error.summary()

    difference = actual - expected
    assert summary["name"] == "dets"
    np.testing.assert_allclose(
        summary["cosine_similarity"],
        np.dot(expected.ravel(), actual.ravel()) / (np.linalg.norm(expected) * np.linalg.norm(actual)),
    )
    np.testing.assert_allclose(
        summary["normalized_rmse"], np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(expected**2))
    )
    np.testing.assert_allclose(summary["mean_absolute_error"], np.mean(np.abs(difference)))
    np.testing.assert_allclose(summary["max_absolute_error"], np.max(np.abs(difference)))
    with pytest.raises(RuntimeError, match="dets shape mismatch"):
        error.add(expected[0], actual[0, :2])


def test_backbone_features_convert_to_the_onnx_boundary_layout(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    batch, channels, height, width = compile_module.BOUNDARY_SHAPES[compile_module.FEATURES]
    nchw = np.arange(batch * channels * height * width, dtype=np.float32).reshape(batch, channels, height, width)

    np.testing.assert_array_equal(compile_module._features_to_nchw(nchw.transpose(0, 2, 3, 1)), nchw)
    assert compile_module._features_to_nchw(nchw) is nchw
    with pytest.raises(RuntimeError, match="unexpected backbone feature shape"):
        compile_module._features_to_nchw(nchw[:, :1])


def test_surgery_simplifies_only_the_decoder_partition() -> None:
    source = (SCRIPTS_DIR / "surgery.py").read_text(encoding="utf-8")

    assert "decoder, valid = simplify(" in source
    assert "full, valid = simplify(" not in source


def test_model_declares_onnx_and_full_pipeline_correctness() -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    invocation = model["build_profiles"][0]["compile_invocations"][0]
    correctness = model["fidelity"][0]
    benchmarks = model["benchmark"]

    assert invocation["components"] == ["backbone", "decoder"]
    assert "python_packages" not in invocation
    assert model["components"][0]["source"]["type"] == "file"
    assert model["components"][0]["source"]["uri"].endswith("/model.onnx")
    assert model["calibrations"][0]["dataset"]["id"] == "coco-2017"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w8a8_w16a16", "w8a16_w16a16"]
    assert model["build_profiles"][0]["components"][0]["parameters"]["calibration_samples"] == 50
    assert model["build_profiles"][0]["components"][0]["parameters"]["calibration_bins"] == 2048
    assert correctness["scope"] == {"type": "package", "package": "default"}
    assert [(entry["id"], entry["scope"]) for entry in benchmarks] == [
        ("orchestration", {"type": "package", "package": "default"})
    ]
    assert benchmarks[0]["required_metrics"] == ["avg_power_watts", "energy_per_inference_joules"]


def test_an_afe_system_exit_fails_the_phase(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    monkeypatch.setattr(compile_module.logging, "basicConfig", lambda **_: None)

    def quantize() -> None:
        raise SystemExit()

    with pytest.raises(RuntimeError, match="quantize stopped in AFE"):
        compile_module._execute_phase(quantize, {})
