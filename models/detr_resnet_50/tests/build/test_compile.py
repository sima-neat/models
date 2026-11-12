from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import onnx.reference
import pytest
from PIL import Image


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), color).save(stream, format="PNG")
    return stream.getvalue()


def test_preprocessing_stretches_to_the_input_and_normalizes_with_imagenet_statistics(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")

    # A 4:3 image fills the whole square input, with no padding at any edge.
    result = calibration.preprocess_image(_png(500, 375, (255, 0, 0)))

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    red = (np.array([255, 0, 0]) / 255 - calibration.MEAN) / calibration.STD
    for row, column in ((0, 0), (0, 639), (639, 0), (639, 639), (320, 320)):
        np.testing.assert_allclose(result[row, column], red, rtol=1e-6)


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


def _model(nodes, inputs, outputs, initializers) -> onnx.ModelProto:
    helper = onnx.helper
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in inputs.items()],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in outputs.items()],
        [onnx.numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)


def _batch_norm_graph(scale_shape=(1, 4, 1, 1)) -> onnx.ModelProto:
    """A bias-free Conv followed by a frozen batch norm's Mul and Add, then a Relu."""
    rng = np.random.default_rng(0)
    helper = onnx.helper
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Mul", ["conv", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "shift"], ["shifted"]),
        helper.make_node("Relu", ["shifted"], ["y"]),
    ]
    initializers = {
        "w": rng.standard_normal((4, 3, 3, 3), dtype=np.float32),
        "scale": rng.uniform(0.5, 2.0, scale_shape).astype(np.float32),
        "shift": rng.standard_normal(scale_shape, dtype=np.float32),
    }
    return _model(nodes, {"x": [1, 3, 8, 8]}, {"y": [1, 4, 8, 8]}, initializers)


def test_batch_norms_fold_into_the_convolution_without_changing_outputs(
    load_script: Callable[[str], ModuleType],
) -> None:
    surgery = load_script("surgery")
    source = _batch_norm_graph()
    feeds = {"x": np.random.default_rng(1).standard_normal((1, 3, 8, 8), dtype=np.float32)}

    folded = surgery._fold_frozen_batch_norms(onnx.ModelProto.FromString(source.SerializeToString()), 1)

    assert [node.op_type for node in folded.graph.node] == ["Conv", "Relu"]
    np.testing.assert_allclose(_run(folded, feeds)[0], _run(source, feeds)[0], rtol=1e-5, atol=1e-5)


def test_batch_norm_folding_rejects_an_unexpected_count_or_shape(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    with pytest.raises(ValueError, match="expected 2 frozen batch norms, found 1"):
        surgery._fold_frozen_batch_norms(_batch_norm_graph(), 2)
    with pytest.raises(ValueError, match="per-channel batch norm"):
        surgery._fold_frozen_batch_norms(_batch_norm_graph(scale_shape=(1, 1, 1, 1)), 1)


def _attention_graph(queries: int, keys: int, mask: np.ndarray | None = None) -> onnx.ModelProto:
    """One exported DETR attention: per-head Q/K/V, scaled scores, an optional additive mask, softmax, projection."""
    heads, dim = 2, 4
    channels = heads * dim
    rng = np.random.default_rng(2)
    helper = onnx.helper
    initializers = {
        **{name: rng.standard_normal((channels, channels), dtype=np.float32) for name in ("wq", "wk", "wv", "wo")},
        "bo": rng.standard_normal(channels, dtype=np.float32),
        "scale": np.array(dim**-0.5, np.float32),
        "heads_q": np.array([1, queries, heads, dim], np.int64),
        "heads_k": np.array([1, keys, heads, dim], np.int64),
        "merged": np.array([1, queries, channels], np.int64),
    }
    nodes = [
        helper.make_node("MatMul", ["q_in", "wq"], ["q2"]),
        helper.make_node("Reshape", ["q2", "heads_q"], ["q3"]),
        helper.make_node("Transpose", ["q3"], ["q"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["kv_in", "wk"], ["k2"]),
        helper.make_node("Reshape", ["k2", "heads_k"], ["k3"]),
        helper.make_node("Transpose", ["k3"], ["k"], perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["kv_in", "wv"], ["v2"]),
        helper.make_node("Reshape", ["v2", "heads_k"], ["v3"]),
        helper.make_node("Transpose", ["v3"], ["v"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q", "k"], ["scores"]),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
    ]
    logits = "scaled"
    if mask is not None:
        initializers["mask"] = mask.astype(np.float32)
        nodes.append(helper.make_node("Add", ["scaled", "mask"], ["masked"]))
        logits = "masked"
    nodes += [
        helper.make_node("Softmax", [logits], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "v"], ["values"]),
        helper.make_node("Transpose", ["values"], ["merge"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["merge", "merged"], ["flat"]),
        helper.make_node("MatMul", ["flat", "wo"], ["projected"]),
        helper.make_node("Add", ["bo", "projected"], ["y"]),
    ]
    inputs = {"q_in": [1, queries, channels], "kv_in": [1, keys, channels]}
    return _model(nodes, inputs, {"y": [1, queries, channels]}, initializers)


@pytest.mark.parametrize(("queries", "keys"), [(5, 5), (3, 7)], ids=["self", "cross"])
def test_attention_rewrite_keeps_outputs_with_the_softmax_over_channels(
    load_script: Callable[[str], ModuleType], queries: int, keys: int
) -> None:
    surgery = load_script("surgery")
    source = _attention_graph(queries, keys)
    rng = np.random.default_rng(3)
    feeds = {
        "q_in": rng.standard_normal((1, queries, 8), dtype=np.float32),
        "kv_in": rng.standard_normal((1, keys, 8), dtype=np.float32),
    }

    rewritten = surgery._optimize_attention(onnx.ModelProto.FromString(source.SerializeToString()), 1)

    operations = [node.op_type for node in rewritten.graph.node]
    assert operations.count("Einsum") == 2 and "Conv" in operations and operations.count("MatMul") == 3
    (softmax,) = [node for node in rewritten.graph.node if node.op_type == "Softmax"]
    assert onnx.helper.get_attribute_value(softmax.attribute[0]) == 1
    np.testing.assert_allclose(_run(rewritten, feeds)[0], _run(source, feeds)[0], rtol=1e-4, atol=1e-4)


def test_only_an_all_zero_attention_mask_is_removed(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    removed = surgery._remove_zero_masks(_attention_graph(3, 7, np.zeros((1, 1, 3, 7))), 1)
    assert "mask" not in {node.input[1] for node in removed.graph.node if node.op_type == "Add"}

    excluding = np.zeros((1, 1, 3, 7))
    excluding[..., -1] = -np.inf
    with pytest.raises(ValueError, match="excludes positions"):
        surgery._remove_zero_masks(_attention_graph(3, 7, excluding), 1)


def test_surgery_requires_the_folded_interface(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    helper = onnx.helper
    outputs = {name: shape for name, shape in surgery.SOURCE_OUTPUT_SHAPES.items()}
    nodes = [
        helper.make_node("Constant", [], [name], value=onnx.numpy_helper.from_array(np.zeros(shape, np.float32)))
        for name, shape in outputs.items()
    ]
    source = tmp_path / "source.onnx"
    onnx.save(_model(nodes, {surgery.INPUT: surgery.INPUT_SHAPE}, outputs, {}), source)
    with pytest.raises(ValueError, match="fold it with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")

    onnx.save(_model(nodes, {surgery.INPUT: [1, 3, 800, 800]}, outputs, {}), source)
    with pytest.raises(ValueError, match="unexpected input interface"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    return {
        "input_nodes": [{"name": "input", "size": 1 * 3 * 640 * 640 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in compile_module.OUTPUT_BYTES],
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
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[1] = outputs[1], outputs[0]
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


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        ("w16a16", {"precision": "w16a16"}),
        (
            "w8a8_mixed",
            {
                "precision": "w8a8",
                "calibration_bins": 2048,
                "calibration_samples": 50,
                "bf16_regions": ["attention", "residual", "ffn", "heads"],
            },
        ),
    ],
)
def test_compile_reads_the_declared_build_parameters(
    load_script: Callable[[str], ModuleType], profile_id: str, expected: dict
) -> None:
    assert load_script("compile")._parameters(profile_id) == expected


def test_bf16_regions_belong_to_a_w8a8_profile_marked_mixed(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    assert recipe._bf16_regions("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["heads"]}) == ["heads"]
    assert recipe._bf16_regions("w16a16", {"precision": "w16a16"}) == []

    for profile_id, parameters, error in [
        ("w8a8", {"precision": "w8a8", "bf16_regions": ["heads"]}, "mixed token"),
        ("w8a8_mixed", {"precision": "w8a8"}, "mixed token"),
        ("w16a16_mixed", {"precision": "w16a16", "bf16_regions": ["heads"]}, "W8A8 build"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["heads", "heads"]}, "distinct regions"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["decoder"]}, "unknown BF16 regions"),
    ]:
        with pytest.raises(ValueError, match=error):
            recipe._bf16_regions(profile_id, parameters)


@pytest.mark.parametrize(("precision", "calibrates"), [("w8a8", True), ("w16a16", False)])
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration


def _heads_graph(extra_relu: bool = False) -> onnx.ModelProto:
    """The decoder's last LayerNorm feeding the class Linear and the three-layer box MLP, as the export names them.

    The box MLP ends in the Sigmoid the converted graph keeps, before surgery cuts it.
    """
    rng = np.random.default_rng(4)
    helper = onnx.helper
    box = ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add", "Sigmoid"]
    if extra_relu:
        box.insert(2, "Relu")
    nodes = [
        helper.make_node("LayerNormalization", ["h", "gamma", "beta"], ["normed"], name="norm", axis=-1),
        helper.make_node("MatMul", ["normed", "class_w"], ["class_mm"], name="class_mm"),
        helper.make_node("Add", ["class_mm", "class_b"], ["logits"], name="class_add"),
    ]
    initializers = {
        "gamma": np.ones(8, np.float32),
        "beta": np.zeros(8, np.float32),
        "class_w": rng.standard_normal((8, 5), dtype=np.float32),
        "class_b": rng.standard_normal(5, dtype=np.float32),
    }
    previous, layer = "normed", 0
    for index, kind in enumerate(box):
        name = f"box_{index}"
        if kind == "MatMul":
            layer += 1
            width = 4 if layer == 3 else 8
            initializers[f"box_w{layer}"] = rng.standard_normal((8, width), dtype=np.float32)
            initializers[f"box_b{layer}"] = rng.standard_normal(width, dtype=np.float32)
            inputs = [previous, f"box_w{layer}"]
        elif kind == "Add":
            inputs = [previous, f"box_b{layer}"]
        else:
            inputs = [previous]
        output = "pred_boxes" if index == len(box) - 1 else name
        nodes.append(helper.make_node(kind, inputs, [output], name=name))
        previous = output
    return _model(nodes, {"h": [1, 3, 8]}, {"logits": [1, 3, 5], "pred_boxes": [1, 3, 4]}, initializers)


def test_box_sigmoid_is_cut_so_the_package_returns_the_box_logits(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = _heads_graph()
    rewritten = surgery._cut_box_sigmoid(onnx.ModelProto.FromString(source.SerializeToString()))

    assert [value.name for value in rewritten.graph.output] == ["logits", "box_logits"]
    assert "Sigmoid" not in {node.op_type for node in rewritten.graph.node}
    hidden = {"h": np.random.default_rng(5).standard_normal((1, 3, 8), dtype=np.float32)}
    logits, boxes = onnx.reference.ReferenceEvaluator(source).run(None, hidden)
    cut_logits, box_logits = onnx.reference.ReferenceEvaluator(rewritten).run(None, hidden)
    np.testing.assert_array_equal(cut_logits, logits)
    np.testing.assert_allclose(1 / (1 + np.exp(-box_logits)), boxes, rtol=1e-6)


def test_box_sigmoid_cut_rejects_logits_read_elsewhere(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    model = _heads_graph()
    sigmoid = next(node for node in model.graph.node if node.op_type == "Sigmoid")
    model.graph.node.append(onnx.helper.make_node("Relu", [sigmoid.input[0]], ["side"], name="side"))
    model.graph.output.append(onnx.helper.make_tensor_value_info("side", onnx.TensorProto.FLOAT, [1, 3, 4]))
    with pytest.raises(ValueError, match="box logits read outside"):
        surgery._cut_box_sigmoid(model)


def test_heads_annotation_marks_each_head_operation_bf16_and_keeps_the_outputs(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    rewritten = surgery._cut_box_sigmoid(_heads_graph())
    graph = tmp_path / "rewritten.onnx"
    onnx.save(rewritten, graph)

    assert surgery.annotate_bf16_regions(graph, ["heads"]) == {"bfloat16": surgery.BF16_REGIONS["heads"], "int8": 0}

    annotated = onnx.load(graph)
    producers = {output: node for node in annotated.graph.node for output in node.output}
    marks = [node for node in annotated.graph.node if node.op_type == "AnnotatePrecision"]
    assert len(marks) == surgery.BF16_REGIONS["heads"]
    assert all(node.domain == "ai.sima" and node.attribute[0].s == b"bfloat16" for node in marks)
    assert {producers[node.input[0]].name for node in marks} == {
        node.name for node in rewritten.graph.node if node.op_type != "LayerNormalization"
    }
    assert {producers[name].op_type for name in ("logits", "box_logits")} == {"AnnotatePrecision"}
    assert ("ai.sima", 1) in {(opset.domain, opset.version) for opset in annotated.opset_import}


def test_heads_annotation_rejects_an_unexpected_head(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    graph = tmp_path / "rewritten.onnx"
    onnx.save(surgery._cut_box_sigmoid(_heads_graph(extra_relu=True)), graph)
    with pytest.raises(ValueError, match="expected 10 heads operations, found 11"):
        surgery.annotate_bf16_regions(graph, ["heads"])


def _layer_graph() -> onnx.ModelProto:
    """One DETR encoder layer and the heads, as the export names its operations.

    The layer adds a position embedding to the queries and keys, attends, adds the result to its input, normalizes,
    runs the feed-forward network, adds and normalizes again; the decoder's last LayerNorm then feeds the heads.
    """
    tokens, heads, dim = 3, 2, 4
    channels = heads * dim
    rng = np.random.default_rng(6)
    helper = onnx.helper
    initializers = {
        "position": rng.standard_normal((1, tokens, channels), dtype=np.float32),
        "scale": np.array(dim**-0.5, np.float32),
        "split_heads": np.array([1, tokens, heads, dim], np.int64),
        "merged": np.array([1, tokens, channels], np.int64),
        **{f"gamma{index}": np.ones(channels, np.float32) for index in (1, 2)},
        **{f"beta{index}": np.zeros(channels, np.float32) for index in (1, 2)},
        "fc1_w": rng.standard_normal((channels, 16), dtype=np.float32),
        "fc1_b": rng.standard_normal(16, dtype=np.float32),
        "fc2_w": rng.standard_normal((16, channels), dtype=np.float32),
        "fc2_b": rng.standard_normal(channels, dtype=np.float32),
    }
    nodes = [helper.make_node("Add", ["x", "position"], ["embedded"], name="self_attn/Add")]
    projections = (("q", "embedded", [0, 2, 1, 3]), ("k", "embedded", [0, 2, 3, 1]), ("v", "x", [0, 2, 1, 3]))
    for name, source, perm in projections:
        initializers[f"w{name}"] = rng.standard_normal((channels, channels), dtype=np.float32)
        initializers[f"b{name}"] = rng.standard_normal(channels, dtype=np.float32)
        nodes += [
            helper.make_node("MatMul", [source, f"w{name}"], [f"{name}_mm"], name=f"{name}_proj/MatMul"),
            helper.make_node("Add", [f"b{name}", f"{name}_mm"], [f"{name}_add"], name=f"{name}_proj/Add"),
            helper.make_node("Reshape", [f"{name}_add", "split_heads"], [f"{name}_heads"], name=f"{name}/Reshape"),
            helper.make_node("Transpose", [f"{name}_heads"], [name], name=f"{name}/Transpose", perm=perm),
        ]
    initializers["wo"] = rng.standard_normal((channels, channels), dtype=np.float32)
    initializers["bo"] = rng.standard_normal(channels, dtype=np.float32)
    nodes += [
        helper.make_node("MatMul", ["q", "k"], ["scores"], name="scores"),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"], name="scaled"),
        helper.make_node("Softmax", ["scaled"], ["probabilities"], name="self_attn/Softmax", axis=-1),
        helper.make_node("MatMul", ["probabilities", "v"], ["values"], name="values"),
        helper.make_node("Transpose", ["values"], ["merge"], name="merge", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["merge", "merged"], ["flat"], name="flat"),
        helper.make_node("MatMul", ["flat", "wo"], ["projected"], name="o_proj/MatMul"),
        helper.make_node("Add", ["bo", "projected"], ["attended"], name="o_proj/Add"),
        helper.make_node("Add", ["x", "attended"], ["residual1"], name="Add"),
        helper.make_node("LayerNormalization", ["residual1", "gamma1", "beta1"], ["n1"], name="norm1", axis=-1),
        helper.make_node("MatMul", ["n1", "fc1_w"], ["fc1_mm"], name="fc1/MatMul"),
        helper.make_node("Add", ["fc1_mm", "fc1_b"], ["fc1"], name="fc1/Add"),
        helper.make_node("Relu", ["fc1"], ["activated"], name="Relu"),
        helper.make_node("MatMul", ["activated", "fc2_w"], ["fc2_mm"], name="fc2/MatMul"),
        helper.make_node("Add", ["fc2_mm", "fc2_b"], ["fc2"], name="fc2/Add"),
        helper.make_node("Add", ["n1", "fc2"], ["residual2"], name="Add_1"),
        helper.make_node("LayerNormalization", ["residual2", "gamma2", "beta2"], ["h"], name="norm2", axis=-1),
    ]
    heads = _heads_graph()
    initializers.update({value.name: onnx.numpy_helper.to_array(value) for value in heads.graph.initializer})
    nodes += list(heads.graph.node)
    outputs = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in heads.graph.output
    }
    return _model(nodes, {"x": [1, tokens, channels]}, outputs, initializers)


# The fixture's region sizes: one attention block with its head splits and merge and a shared Q and K position
# embedding, two residual sums and three LayerNorms, one feed-forward network and the heads.
LAYER_REGIONS = {"attention": 19, "residual": 5, "ffn": 5, "heads": 10}


def _rewritten_layer(surgery: ModuleType, monkeypatch: pytest.MonkeyPatch) -> onnx.ModelProto:
    monkeypatch.setattr(surgery, "BF16_REGIONS", LAYER_REGIONS)
    return surgery._cut_box_sigmoid(surgery._optimize_attention(_layer_graph(), 1))


def test_attention_and_residual_marks_end_at_the_feed_forward_network(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_script("surgery")
    model = _rewritten_layer(surgery, monkeypatch)

    marks = surgery.precision_marks(model, ["attention", "residual", "heads"])

    assert {name for name, precision in marks.items() if precision == "int8"} == {"fc1/MatMul"}
    kinds = {node.name: node.op_type for node in model.graph.node}
    bf16 = {kinds[name] for name, precision in marks.items() if precision == "bfloat16"}
    assert {"Einsum", "Softmax", "Concat", "Conv", "LayerNormalization"} <= bf16
    assert {"self_attn/Add", "Add", "Add_1", "norm1", "norm2", "norm"} <= set(marks)


def test_whole_transformer_marks_every_computing_operation_bf16(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_script("surgery")
    model = _rewritten_layer(surgery, monkeypatch)

    marks = surgery.precision_marks(model, ["attention", "residual", "ffn", "heads"])

    computing = {node.name for node in model.graph.node if node.op_type not in surgery.LAYOUT_OPS}
    assert computing <= set(marks)
    assert set(marks.values()) == {"bfloat16"}


def test_precision_marks_reject_an_unknown_region_or_count(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_script("surgery")
    model = _rewritten_layer(surgery, monkeypatch)
    with pytest.raises(ValueError, match="unknown BF16 regions"):
        surgery.precision_marks(model, ["decoder"])
    monkeypatch.setitem(LAYER_REGIONS, "ffn", 4)
    with pytest.raises(ValueError, match="expected 4 ffn operations, found 5"):
        surgery.precision_marks(model, ["ffn"])


def test_bf16_check_requires_every_attention_softmax_and_head_split_and_merge(
    load_script: Callable[[str], ModuleType],
) -> None:
    from collections import Counter

    recipe = load_script("compile")
    recipe._require_bf16_regions(Counter(SoftmaxOp=17, SliceConcatOp=67, LayerNormOp=30), ["attention", "residual"])
    recipe._require_bf16_regions(Counter(MatMulOp=4), ["heads"])
    # The head splits and merges stayed INT8: AFE requantizes every attention's Q, K, V and values.
    with pytest.raises(RuntimeError, match="expected BF16 attention operations"):
        recipe._require_bf16_regions(Counter(SoftmaxOp=17), ["attention", "heads"])
    with pytest.raises(RuntimeError, match="expected BF16 attention operations"):
        recipe._require_bf16_regions(Counter(), ["heads"])


@pytest.mark.parametrize(
    ("profile_id", "parameters", "int8_layer_norms", "fp32_layer_norms"),
    [
        ("w16a16", {"precision": "w16a16"}, 0, True),
        (
            "w8a8_mixed",
            {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 2, "bf16_regions": ["residual"]},
            0,
            True,
        ),
        (
            "w8a8_mixed",
            {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 2, "bf16_regions": ["heads"]},
            30,
            False,
        ),
    ],
    ids=["w16a16", "bf16-layer-norms", "int8-layer-norms"],
)
def test_only_int8_builds_calibrate_and_fp32_layer_norms_need_every_layer_norm_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    profile_id: str,
    parameters: dict,
    int8_layer_norms: int,
    fp32_layer_norms: bool,
) -> None:
    from collections import Counter

    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_images", lambda dataset, count: np.zeros((count, 640, 640, 3), np.float32))
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "rewritten.onnx")
    annotated, checked = [], []
    monkeypatch.setattr(recipe, "annotate_bf16_regions", lambda graph, regions: annotated.append(regions) or {})
    bf16 = Counter(LayerNormOp=30 - int8_layer_norms, SoftmaxOp=17)
    monkeypatch.setattr(recipe, "_operation_counts", lambda quantized: (Counter(LayerNormOp=30, SoftmaxOp=17), bf16))
    monkeypatch.setattr(recipe, "_require_bf16_regions", lambda counts, regions: checked.append(regions))

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", profile_id, tmp_path / "coco.tgz")

    int8 = parameters["precision"] == "w8a8"
    quantize = recipe.load_model.return_value.quantize
    data = quantize.call_args.kwargs["calibration_data"]
    assert (data is recipe.convert_data_generator_to_iterable.return_value) is int8
    regions = parameters.get("bf16_regions")
    assert annotated == checked == ([regions] if regions else [])
    compile_options = quantize.return_value.compile.call_args.kwargs
    assert compile_options["layer_norm_use_fp32_intermediates"] is fp32_layer_norms
