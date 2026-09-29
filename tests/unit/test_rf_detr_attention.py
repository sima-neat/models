"""Semantic and rejection checks for the two recipe-level attention rewrites."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

PATH = Path(__file__).resolve().parents[2] / "models/rf_detr_small/attention.py"
SPEC = importlib.util.spec_from_file_location("rf_detr_attention", PATH)
attention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attention)


def window_model(windows=2, heads=3, tokens=5, dim=4):
    prefix = "/backbone/block/"
    nodes = []

    def node(kind, inputs, output, **attributes):
        nodes.append(helper.make_node(kind, inputs, [output], name=prefix + output, **attributes))

    node("Transpose", ["q"], "qt", perm=[0, 2, 1, 3])
    node("Transpose", ["k"], "kt", perm=[0, 2, 3, 1])
    node("Transpose", ["v"], "vt", perm=[0, 2, 1, 3])
    node("Shape", ["qt"], "q_shape")
    node("Slice", ["q_shape", "start", "end"], "head_dim")
    node("Cast", ["head_dim"], "scale", to=TensorProto.FLOAT)
    node("Mul", ["qt", "scale"], "qs")
    node("Mul", ["kt", "k_scale"], "ks")
    node("MatMul", ["qs", "ks"], "scores")
    node("Softmax", ["scores"], "probabilities", axis=-1)
    node("MatMul", ["probabilities", "vt"], "pv")
    node("Transpose", ["pv"], "output", perm=[0, 2, 1, 3])
    shape = [windows, tokens, heads, dim]
    initializers = [
        numpy_helper.from_array(value, name)
        for name, value in (
            ("start", np.array([-1], np.int64)),
            ("end", np.array([np.iinfo(np.int64).max], np.int64)),
            ("k_scale", np.array([0.25], np.float32)),
        )
    ]
    graph = helper.make_graph(
        nodes,
        "window",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("q", "k", "v")],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("windows,heads,tokens,dim", [(2, 3, 5, 4), (4, 6, 7, 8), (3, 2, 4, 5)])
def test_window_permutation_preserves_outputs_and_inputs(windows, heads, tokens, dim):
    model = window_model(windows, heads, tokens, dim)
    original = model.SerializeToString()
    optimized = attention.optimize_window_attention(model, expected_blocks=1)
    rng = np.random.default_rng(42)
    feed = {name: rng.normal(size=(windows, tokens, heads, dim)).astype(np.float32) for name in ("q", "k", "v")}
    (expected,) = ReferenceEvaluator(model).run(None, feed)
    (actual,) = ReferenceEvaluator(optimized).run(None, feed)
    np.testing.assert_array_equal(actual, expected)
    assert model.SerializeToString() == original
    assert list(model.graph.input) == list(optimized.graph.input)
    assert list(model.graph.output) == list(optimized.graph.output)
    assert list(model.graph.initializer) == list(optimized.graph.initializer)


def test_window_rewrite_leaves_global_attention_unchanged():
    model = window_model(windows=1)
    optimized = attention.optimize_window_attention(model, expected_blocks=0)
    assert list(model.graph.node) == list(optimized.graph.node)


@pytest.mark.parametrize("case", ["scores_output", "q_output", "shape_output", "pv_user", "axis", "scale", "mask"])
def test_window_rewrite_rejects_observable_or_incompatible_intermediates(case):
    model = window_model()
    by_output = {n.output[0]: n for n in model.graph.node}
    if case in ("scores_output", "q_output", "shape_output"):
        name = {"scores_output": "scores", "q_output": "qt", "shape_output": "q_shape"}[case]
        shape = {"scores": [2, 3, 5, 5], "qt": [2, 3, 5, 4], "q_shape": [4]}[name]
        dtype = TensorProto.INT64 if name == "q_shape" else TensorProto.FLOAT
        model.graph.output.append(helper.make_tensor_value_info(name, dtype, shape))
    elif case == "pv_user":
        model.graph.node.append(helper.make_node("Identity", ["pv"], ["extra"], name="extra"))
    elif case == "axis":
        by_output["probabilities"].attribute[0].i = 2
    elif case == "scale":
        by_output["qs"].input[1] = "channel_scale"
        model.graph.initializer.append(numpy_helper.from_array(np.ones(4, np.float32), "channel_scale"))
    elif case == "mask":
        index = list(model.graph.node).index(by_output["probabilities"])
        model.graph.node.insert(index, helper.make_node("Add", ["scores", "k_scale"], ["masked"], name="mask"))
        by_output["probabilities"].input[0] = "masked"
    original = model.SerializeToString()
    with pytest.raises(ValueError, match="Unsupported RF-DETR attention"):
        attention.optimize_window_attention(model, expected_blocks=1)
    assert model.SerializeToString() == original


def decoder_model(heads=3, tokens=5, dim=4):
    prefix = "/transformer/decoder/layers.0/self_attn/"
    nodes = [
        helper.make_node(kind, inputs, [name], name=prefix + name, **attributes)
        for kind, inputs, name, attributes in (
            ("MatMul", ["q", "k"], "scores", {}),
            ("Softmax", ["scores"], "probabilities", {"axis": -1}),
            ("MatMul", ["probabilities", "v"], "pv", {}),
            ("Transpose", ["pv"], "transposed", {"perm": [2, 0, 1, 3]}),
            ("Reshape", ["transposed", "merged_shape"], "merged", {}),
            ("Gemm", ["merged", "weight", "bias"], "output", {"transB": 1}),
        )
    ]
    rng = np.random.default_rng(7)
    channels = heads * dim + 2  # Exercise a nonsquare projection and nonzero bias.
    initializers = [
        numpy_helper.from_array(value, name)
        for name, value in (
            ("merged_shape", np.array([tokens, heads * dim], np.int64)),
            ("weight", rng.normal(size=(channels, heads * dim)).astype(np.float32)),
            ("bias", rng.normal(size=channels).astype(np.float32)),
        )
    ]
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in (
            ("q", [1, heads, tokens, dim]),
            ("k", [1, heads, dim, tokens]),
            ("v", [1, heads, tokens, dim]),
        )
    ]
    graph = helper.make_graph(
        nodes,
        "decoder",
        inputs,
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [tokens, channels])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("heads,tokens,dim", [(3, 5, 4), (2, 7, 3)])
def test_decoder_rewrite_preserves_head_order_projection_and_bias(heads, tokens, dim):
    model = decoder_model(heads, tokens, dim)
    optimized = attention.optimize_decoder_attention(model, expected_blocks=1)
    rng = np.random.default_rng(17)
    feed = {
        v.name: rng.normal(size=[d.dim_value for d in v.type.tensor_type.shape.dim]).astype(np.float32)
        for v in model.graph.input
    }
    (expected,) = ReferenceEvaluator(model).run(None, feed)
    (actual,) = ReferenceEvaluator(optimized).run(None, feed)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert list(model.graph.input) == list(optimized.graph.input)
    assert list(model.graph.output) == list(optimized.graph.output)


@pytest.mark.parametrize("case", ["public_scores", "shared_merge", "gemm_scale", "softmax_axis", "count"])
def test_decoder_rewrite_rejects_unsupported_patterns(case):
    model = decoder_model()
    if case == "public_scores":
        model.graph.output.append(helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3, 5, 5]))
    elif case == "shared_merge":
        model.graph.node.append(helper.make_node("Identity", ["merged"], ["extra"], name="extra"))
    elif case == "gemm_scale":
        model.graph.node[-1].attribute.append(helper.make_attribute("alpha", 2.0))
    elif case == "softmax_axis":
        model.graph.node[1].attribute[0].i = 2
    with pytest.raises(ValueError, match="Unsupported RF-DETR attention"):
        attention.optimize_decoder_attention(model, expected_blocks=2 if case == "count" else 1)


def test_decoder_rewrite_leaves_other_attention_alone():
    model = decoder_model()
    for node in model.graph.node:
        node.name = node.name.replace("self_attn", "cross_attn")
    optimized = attention.optimize_decoder_attention(model, expected_blocks=0)
    assert list(model.graph.node) == list(optimized.graph.node)
