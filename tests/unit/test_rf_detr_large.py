from __future__ import annotations

import builtins
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "rf_detr_large"
SCRIPTS_DIR = MODEL_DIR / "scripts"


def load_module(name: str) -> ModuleType:
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rf_detr_large_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
    return module


def test_preprocessing_produces_normalized_nchw_image() -> None:
    calibration = load_module("calibration")
    image = Image.new("RGB", (16, 8), color=(255, 128, 0))
    content = io.BytesIO()
    image.save(content, format="PNG")

    result = calibration.preprocess_image(content.getvalue())

    assert result.shape == (3, 704, 704)
    assert result.dtype == np.float32
    expected = (np.asarray([1.0, 128 / 255, 0.0], dtype=np.float32) - calibration.MEAN[:, 0, 0]) / calibration.STD[
        :, 0, 0
    ]
    np.testing.assert_allclose(result[:, 0, 0], expected)


def test_surgery_import_does_not_require_onnxruntime(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_runtime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_runtime)
    assert callable(load_module("surgery").prepare_model)


def test_image_selection_is_seeded_and_disjoint(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    for index in range(8):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(tmp_path / f"{index}.jpg")

    first = calibration.select_images(tmp_path, calibration_count=3, fidelity_count=3, seed=42)
    second = calibration.select_images(tmp_path, calibration_count=3, fidelity_count=3, seed=42)

    assert first == second
    calibration_names, fidelity_names = first
    assert len(calibration_names) == 3
    assert len(fidelity_names) == 3
    assert set(calibration_names).isdisjoint(fidelity_names)


def test_afe_layout_converts_only_rank_four_tensors() -> None:
    calibration = load_module("calibration")
    images = np.zeros((2, 3, 5, 7), dtype=np.float32)
    proposals = np.zeros((2, 300, 4), dtype=np.float32)

    converted_images = calibration.to_afe_layout(images)
    converted_proposals = calibration.to_afe_layout(proposals)

    assert converted_images.shape == (2, 5, 7, 3)
    assert converted_images.flags.c_contiguous
    assert converted_proposals.shape == proposals.shape
    assert converted_proposals.flags.c_contiguous


def test_coco_archive_selection_and_webp_decoding(tmp_path: Path) -> None:
    calibration = load_module("calibration")
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
        selected = calibration.select_images(dataset, calibration_count=3, fidelity_count=3, seed=42)
        selections.append(selected)
        assert calibration.load_images(dataset, selected[0]).shape == (3, 3, 704, 704)
        assert set(selected[0]).isdisjoint(selected[1])
    assert selections[0] == selections[1]


def test_topk_gather_uses_stable_score_order() -> None:
    calibration = load_module("calibration")
    scores = np.asarray([[0.5, 0.9, 0.9, 0.1]], dtype=np.float32)
    boxes = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    result = calibration.stable_topk_gather(scores, boxes, count=3)

    np.testing.assert_array_equal(result, boxes[:, [1, 2, 0]])


def _value(name: str, shape: list[int], dtype: int = onnx.TensorProto.FLOAT) -> onnx.ValueInfoProto:
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=()) -> onnx.ModelProto:
    graph = onnx.helper.make_graph(nodes, "fixture", inputs, outputs, list(initializers))
    return onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])


def test_surgery_requires_a_folded_source() -> None:
    surgery = load_module("surgery")
    model = _model(
        [onnx.helper.make_node("Shape", ["x"], ["shape"], name="shape")],
        [_value("x", [2])],
        [_value("shape", [1], onnx.TensorProto.INT64)],
    )

    with pytest.raises(ValueError, match="fold it with onnxsim"):
        surgery._require_folded(model)


@pytest.mark.parametrize("selects", [False, True])
def test_surgery_bypasses_only_noop_masks(selects: bool) -> None:
    surgery = load_module("surgery")
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


def test_surgery_bypasses_identities_and_unused_stacked_layers() -> None:
    surgery = load_module("surgery")
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


def test_surgery_folds_the_backbone_score_scale_into_q() -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_module("surgery")
    model = _attention_block(0.125)
    x = np.random.RandomState(1).standard_normal((1, 4, 8)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]

    result = surgery._fold_backbone_score_scale(model, 1)

    assert "Mul" not in {node.op_type for node in result.graph.node}
    np.testing.assert_array_equal(ReferenceEvaluator(result).run(None, {"x": x})[0], expected)


def test_surgery_folds_only_power_of_two_score_scales() -> None:
    surgery = load_module("surgery")

    with pytest.raises(ValueError, match="power-of-two scale"):
        surgery._fold_backbone_score_scale(_attention_block(0.1), 1)


def test_compile_recipe_uses_one_grouped_entrypoint_and_direct_afe_api() -> None:
    source = (SCRIPTS_DIR / "compile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--compile-invocation", required=True)' in source
    assert 'parser.add_argument("--correctness-dataset", type=Path, required=True)' in source
    assert "export_model" not in source
    assert "prepare_model(source" in source
    assert "transformer_inputs(" in source
    assert "loaded.quantize(" in source
    assert "any_shape_on_mla=" not in source
    assert "automatic_layout_conversion=" not in source
    assert "with_per_token_quant(False)" in source
    assert "HistogramMSEMethod(calibration_bins)" in source
    assert "flexible_batch_size=True" in source
    assert 'layer_norm_use_fp32_intermediates=precision == "bf16"' in source
    assert "quantized.compile(" in source
    assert "compile-outputs.json" in source
    assert '"schema_version": 2' in source
    assert '"derived_graphs"' in source
    assert '"correctness"' in source
    assert "shutil.copyfile" in source
    assert "requests" not in source
    assert "urllib" not in source
    assert 'get_context("spawn")' in source
    assert source.index('"RF-DETR backbone compilation"') < source.index(
        '"RF-DETR transformer compilation and fidelity"'
    )


def test_surgery_simplifies_only_the_transformer_partition() -> None:
    source = (SCRIPTS_DIR / "surgery.py").read_text(encoding="utf-8")

    assert "transformer, valid = simplify(" in source
    assert "full, valid = simplify(" not in source


def test_model_declares_onnx_and_full_pipeline_correctness() -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    invocation = model["build_profiles"][0]["compile_invocations"][0]
    correctness = model["validation"][0]
    benchmarks = model["benchmark"]

    assert invocation["components"] == ["backbone", "transformer"]
    assert "python_packages" not in invocation
    assert model["components"][0]["source"]["type"] == "file"
    assert model["components"][0]["source"]["uri"].endswith("/model.onnx")
    assert model["calibrations"][0]["dataset"]["id"] == "coco-2017"
    assert model["build_profiles"][0]["components"][0]["parameters"]["calibration_samples"] == 50
    assert model["build_profiles"][0]["components"][1]["parameters"]["calibration_samples"] == 50
    assert model["build_profiles"][0]["components"][0]["parameters"]["calibration_bins"] == 2048
    assert model["build_profiles"][0]["components"][1]["parameters"]["calibration_bins"] == 2048
    assert correctness["scope"] == {"type": "package", "package": "default"}
    assert [entry["scope"]["component"] for entry in benchmarks] == ["backbone", "transformer"]
    assert all(entry["required_metrics"] == ["avg_power_watts", "energy_per_inference_joules"] for entry in benchmarks)


def _gelu_mlp() -> onnx.ModelProto:
    rng = np.random.RandomState(0)
    prefix = "/model/model/backbone/backbone/encoder/layer.0/mlp"
    constant = onnx.numpy_helper.from_array
    return _model(
        [
            onnx.helper.make_node("MatMul", ["x", "w1"], ["fc1"], name=f"{prefix}/fc1/MatMul"),
            onnx.helper.make_node("Add", ["b1", "fc1"], ["hidden"], name=f"{prefix}/fc1/Add"),
            onnx.helper.make_node("Div", ["hidden", "root_two"], ["scaled"], name=f"{prefix}/activation/Div"),
            onnx.helper.make_node("Erf", ["scaled"], ["erf"], name=f"{prefix}/activation/Erf"),
            onnx.helper.make_node("Add", ["erf", "one"], ["shift"], name=f"{prefix}/activation/Add"),
            onnx.helper.make_node("Mul", ["hidden", "shift"], ["gate"], name=f"{prefix}/activation/Mul"),
            onnx.helper.make_node("Mul", ["gate", "half"], ["gelu"], name=f"{prefix}/activation/Mul_1"),
            onnx.helper.make_node("MatMul", ["gelu", "w2"], ["y"], name=f"{prefix}/fc2/MatMul"),
        ],
        [_value("x", [1, 4, 8])],
        [_value("y", [1, 4, 8])],
        [
            constant(rng.standard_normal((8, 16)).astype(np.float32), "w1"),
            constant(rng.standard_normal(16).astype(np.float32), "b1"),
            constant(rng.standard_normal((16, 8)).astype(np.float32), "w2"),
            constant(np.array(np.sqrt(2), dtype=np.float32), "root_two"),
            constant(np.array(1, dtype=np.float32), "one"),
            constant(np.array(0.5, dtype=np.float32), "half"),
        ],
    )


def test_surgery_folds_gelu_constants_into_the_mlp_weights() -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_module("surgery")
    model = _gelu_mlp()
    x = np.random.RandomState(1).standard_normal((1, 4, 8)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]

    result = surgery._fold_gelu_constants(model, 1)

    assert [node.op_type for node in result.graph.node] == ["MatMul", "Add", "Erf", "Add", "Mul", "MatMul"]
    np.testing.assert_allclose(ReferenceEvaluator(result).run(None, {"x": x})[0], expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="expected 2"):
        surgery._fold_gelu_constants(_gelu_mlp(), 2)


def _query_sine_embedding(width: int = 8) -> onnx.ModelProto:
    prefix = "/model/model/decoder"
    constant = onnx.numpy_helper.from_array
    temperatures = 10000 ** (2 * (np.arange(width) // 2) / width)
    nodes = [
        onnx.helper.make_node("Split", ["coords", "ones"], [f"c{k}" for k in range(4)], name=f"{prefix}/Split", axis=-1)
    ]
    blocks = []
    # The exporter embeds (y, x, w, h); the merge must keep that column order.
    for block, k in enumerate((1, 0, 2, 3)):
        names = {
            part: f"{part}{block}"
            for part in ("scaled", "angles", "even", "odd", "sin", "cos", "us", "uc", "pair", "flat")
        }
        nodes += [
            onnx.helper.make_node("Mul", [f"c{k}", "scale"], [names["scaled"]], name=f"{prefix}/Mul_{block}"),
            onnx.helper.make_node(
                "Div", [names["scaled"], "temperatures"], [names["angles"]], name=f"{prefix}/Div_{block}"
            ),
            onnx.helper.make_node(
                "Slice",
                [names["angles"], "zero", "end", "last", "two"],
                [names["even"]],
                name=f"{prefix}/Slice_{block}a",
            ),
            onnx.helper.make_node(
                "Slice", [names["angles"], "one", "end", "last", "two"], [names["odd"]], name=f"{prefix}/Slice_{block}b"
            ),
            onnx.helper.make_node("Sin", [names["even"]], [names["sin"]], name=f"{prefix}/Sin_{block}"),
            onnx.helper.make_node("Cos", [names["odd"]], [names["cos"]], name=f"{prefix}/Cos_{block}"),
            onnx.helper.make_node(
                "Unsqueeze", [names["sin"], "axis"], [names["us"]], name=f"{prefix}/Unsqueeze_{block}a"
            ),
            onnx.helper.make_node(
                "Unsqueeze", [names["cos"], "axis"], [names["uc"]], name=f"{prefix}/Unsqueeze_{block}b"
            ),
            onnx.helper.make_node(
                "Concat", [names["us"], names["uc"]], [names["pair"]], name=f"{prefix}/Concat_{block}", axis=-1
            ),
            onnx.helper.make_node(
                "Reshape", [names["pair"], "flat_shape"], [names["flat"]], name=f"{prefix}/Reshape_{block}"
            ),
        ]
        blocks.append(names["flat"])
    nodes.append(onnx.helper.make_node("Concat", blocks, ["embedding"], name=f"{prefix}/Concat_9", axis=-1))
    return _model(
        nodes,
        [_value("coords", [1, 3, 4])],
        [_value("embedding", [1, 3, 4 * width])],
        [
            constant(np.ones(4, dtype=np.int64), "ones"),
            constant(np.array(2 * np.pi, dtype=np.float32), "scale"),
            constant(temperatures.astype(np.float32), "temperatures"),
            constant(np.array([0], dtype=np.int64), "zero"),
            constant(np.array([1], dtype=np.int64), "one"),
            constant(np.array([np.iinfo(np.int64).max], dtype=np.int64), "end"),
            constant(np.array([-1], dtype=np.int64), "last"),
            constant(np.array([2], dtype=np.int64), "two"),
            constant(np.array([-1], dtype=np.int64), "axis"),
            constant(np.array([1, 3, width], dtype=np.int64), "flat_shape"),
        ],
    )


def test_surgery_merges_the_query_sine_embedding_into_one_sin() -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_module("surgery")
    model = _query_sine_embedding()
    coords = np.random.RandomState(2).uniform(0, 1, (1, 3, 4)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"coords": coords})[0]

    result = surgery._merge_query_sine_embedding(model)

    assert [node.op_type for node in result.graph.node] == ["MatMul", "Add", "Sin"]
    np.testing.assert_allclose(ReferenceEvaluator(result).run(None, {"coords": coords})[0], expected, atol=1e-5)
