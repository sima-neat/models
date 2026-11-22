"""Precision-profile, graph-rewrite and installed artifact contracts for Nano."""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from model_registry.model import load_model
from PIL import Image

DECODER = {"precision": "w16a16"}


@pytest.fixture
def compiler(load_script):
    return load_script("compile")


@pytest.fixture
def surgery(load_script):
    return load_script("surgery")


@pytest.mark.parametrize(
    ("profile", "backbone"),
    [
        ("w16a16", {"precision": "w16a16"}),
        ("w8a8_w16a16", {"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 20}),
        (
            "w8a8_mixed_w16a16",
            {
                "precision": "w8a8",
                "calibration_bins": 1024,
                "calibration_samples": 20,
                "bf16_regions": ["backbone", "nonconv", "outputs"],
            },
        ),
    ],
)
def test_compile_reads_each_declared_build_profile(compiler, model_dir, profile, backbone):
    assert compiler._component_parameters(load_model(model_dir), profile) == {"backbone": backbone, "decoder": DECODER}


def test_only_int8_backbones_calibrate(compiler, model_dir):
    # The W16A16 decoder keeps no INT8 lookup after surgery, so BF16 activations need no ranges.
    metadata = load_model(model_dir)
    for profile in metadata["build_profiles"]:
        parameters = compiler._component_parameters(metadata, profile["id"])
        assert not compiler.calibrates(parameters["decoder"]), profile["id"]
        assert compiler.calibrates(parameters["backbone"]) == (parameters["backbone"]["precision"] == "w8a8")


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "w8a16",
        "decoder_int8",
        "decoder_regions",
        "decoder_calibration",
        "half_calibration",
        "zero_bins",
        "bool_samples",
        "token",
    ],
)
def test_component_parameters_reject_undeclared_or_inconsistent_profiles(compiler, model_dir, case):
    metadata = copy.deepcopy(load_model(model_dir))
    profile = metadata["build_profiles"][0]
    backbone, decoder = (binding["parameters"] for binding in profile["components"])
    assert profile["id"] == "w16a16"
    if case == "unknown":
        profile["id"] = "w32a32"
    elif case == "w8a16":
        backbone["precision"] = "w8a16"
    elif case == "decoder_int8":
        decoder["precision"] = "w8a8"
    elif case == "decoder_regions":
        decoder["bf16_regions"] = ["outputs"]
    elif case == "decoder_calibration":
        decoder.update(calibration_bins=2048, calibration_samples=50)
    elif case == "half_calibration":
        backbone["calibration_bins"] = 2048
    elif case == "zero_bins":
        backbone.update(calibration_bins=0, calibration_samples=50)
    elif case == "bool_samples":
        backbone.update(calibration_bins=2048, calibration_samples=True)
    else:
        backbone.update(precision="w8a8", calibration_bins=1024, calibration_samples=50, bf16_regions=["outputs"])
    with pytest.raises(ValueError):
        compiler._component_parameters(metadata, "w16a16")


@pytest.mark.parametrize("regions", [[], "outputs", ["typo"], ["outputs", "outputs"], [False], [{}]])
def test_mixed_regions_are_explicit_and_validated(compiler, model_dir, regions):
    metadata = copy.deepcopy(load_model(model_dir))
    candidate = [p for p in metadata["build_profiles"] if p["id"] == "w8a8_mixed_w16a16"]
    candidate[0]["components"][0]["parameters"]["bf16_regions"] = regions
    with pytest.raises(ValueError, match="BF16 regions"):
        compiler._component_parameters(metadata, "w8a8_mixed_w16a16")


@pytest.mark.parametrize(
    ("parameters", "calibrates"),
    [
        ({"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 50}, True),
        ({"precision": "w16a16"}, False),
    ],
)
def test_quantization_config_states_calibration_per_precision(compiler, monkeypatch, parameters, calibrates):
    for name in ("QuantizationParams", "HistogramMSEMethod", "SkipCalibration", "bfloat16_scheme"):
        monkeypatch.setattr(compiler, name, MagicMock())
    monkeypatch.setattr(compiler, "quantization_scheme", lambda **kwargs: kwargs)
    monkeypatch.setattr(compiler, "BiasCorrectionType", SimpleNamespace(REGULAR="regular"))

    compiler.quantization_config(parameters)

    arguments = compiler.QuantizationParams.call_args.kwargs
    if calibrates:
        compiler.HistogramMSEMethod.assert_called_once_with(parameters["calibration_bins"])
        assert arguments["calibration_method"] is compiler.HistogramMSEMethod.return_value
    else:
        assert arguments["calibration_method"] is compiler.SkipCalibration.return_value
    assert "per_token_quant" not in arguments
    if parameters["precision"] == "w8a8":
        assert arguments["activation_quantization_scheme"] == {"asymmetric": True, "per_channel": False, "bits": 8}
        assert arguments["weight_quantization_scheme"] == {"asymmetric": False, "per_channel": True, "bits": 8}
        assert arguments["biascorr_type"] == "regular"
    else:
        assert arguments["activation_quantization_scheme"] is compiler.bfloat16_scheme.return_value
        assert arguments["weight_quantization_scheme"] is compiler.bfloat16_scheme.return_value
        assert "biascorr_type" not in arguments


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"precision": "int8"}, "Unsupported"),
        ({"precision": "w8a16"}, "Unsupported"),
        ({"precision": "w8a8"}, "W8A8 component must declare calibration_bins"),
    ],
)
def test_quantization_config_rejects_undeclared_settings(compiler, monkeypatch, parameters, message):
    monkeypatch.setattr(compiler, "SkipCalibration", MagicMock())
    monkeypatch.setattr(compiler, "quantization_scheme", lambda **kwargs: kwargs)
    with pytest.raises(ValueError, match=message):
        compiler.quantization_config(parameters)


def test_calibration_images_are_selected_only_for_a_calibrated_backbone(compiler, tmp_path):
    for index in range(8):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(tmp_path / f"{index}.jpg")
    calibrated = {
        "backbone": {"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 3},
        "decoder": DECODER,
    }
    uncalibrated = {"backbone": {"precision": "w16a16"}, "decoder": {"precision": "w16a16"}}
    selection = {"seed": 42, "offset": 3, "samples": 3}

    assert compiler.calibration_images(uncalibrated, None, tmp_path, selection) == ()
    names = compiler.calibration_images(calibrated, tmp_path, tmp_path, selection)
    assert names == compiler.select_images(tmp_path, 42, 0, 3)
    with pytest.raises(ValueError, match="require --dataset"):
        compiler.calibration_images(calibrated, None, tmp_path, selection)
    with pytest.raises(ValueError, match="disjoint"):
        compiler.calibration_images(calibrated, tmp_path, tmp_path, {**selection, "offset": 2})
    with pytest.raises(ValueError, match="same verified dataset"):
        compiler.calibration_images(calibrated, tmp_path, tmp_path / "other", selection)


@pytest.mark.parametrize("mixed", [False, True])
def test_quantization_enforces_applied_precision_only_for_the_mixed_profile(compiler, monkeypatch, mixed):
    quantized, checked, calls = object(), [], []

    def quantize(**kwargs):
        calls.append(kwargs)
        return quantized

    monkeypatch.setattr(compiler, "load_model", lambda source: SimpleNamespace(quantize=quantize))
    monkeypatch.setattr(compiler, "onnx_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(compiler, "DataGenerator", lambda data: data)
    monkeypatch.setattr(compiler, "InputName", str)
    monkeypatch.setattr(compiler, "convert_data_generator_to_iterable", lambda data: data)
    monkeypatch.setattr(compiler, "quantization_config", lambda parameters: "config")
    monkeypatch.setattr(compiler, "require_mixed_precision", checked.append)
    parameters = {"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 1}
    if mixed:
        parameters["bf16_regions"] = ["backbone", "nonconv", "outputs"]
    result = compiler.quantize_component(
        Path("unused.onnx"), "candidate", parameters, {"input": np.ones((1, 3, 2, 2), np.float32)}
    )
    assert result is quantized
    assert checked == ([quantized] if mixed else [])
    assert list(calls[0]["calibration_data"]) == ["input"]


def test_bf16_activation_components_quantize_without_calibration_data(compiler, monkeypatch):
    calls = []
    monkeypatch.setattr(compiler, "load_model", lambda source: SimpleNamespace(quantize=lambda **k: calls.append(k)))
    monkeypatch.setattr(compiler, "onnx_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(compiler, "quantization_config", lambda parameters: "config")
    compiler.quantize_component(Path("unused.onnx"), "backbone", {"precision": "w16a16"}, None)
    assert calls[0]["calibration_data"] is None


@pytest.mark.parametrize("case", ["mixed", "ignored", "all_bf16", "ingress_only", "no_convolutions"])
def test_mixed_policy_checks_applied_convolutions_not_annotations_or_ingress(compiler, monkeypatch, case):
    monkeypatch.setattr(compiler, "QuantizationPrecision", SimpleNamespace(BFLOAT_16="bf16", INT_8="int8"))
    operations = {
        "mixed": [("ConvAddActivationOp", "bf16"), ("ConvAddActivationOp", "int8")],
        "ignored": [("ConvAddActivationOp", "int8")],
        "all_bf16": [("ConvAddActivationOp", "bf16")],
        "ingress_only": [("ConvAddActivationOp", "bf16"), ("PlaceholderOp", "int8")],
        "no_convolutions": [("PlaceholderOp", "bf16"), ("PlaceholderOp", "int8")],
    }
    nodes = [
        SimpleNamespace(
            ir=SimpleNamespace(
                operation=type(operation, (), {})(),
                quant_config=SimpleNamespace(quantization_precision=SimpleNamespace(value=precision)),
            )
        )
        for operation, precision in operations[case]
    ]
    # AFE's recursive iterator also yields nested network containers, not only operations.
    nodes.append(SimpleNamespace(ir=SimpleNamespace()))
    model = SimpleNamespace(_net=SimpleNamespace(iter_nodes_recursive=lambda: iter(nodes)))
    if case == "mixed":
        compiler.require_mixed_precision(model)
    else:
        with pytest.raises(RuntimeError, match="Expected BF16 and INT8 convolutions"):
            compiler.require_mixed_precision(model)


@pytest.mark.parametrize("backbone_calibrates", [False, True])
def test_backbone_phase_calibrates_only_w8a8_and_saves_owned_evaluation_outputs(
    compiler, tmp_path, monkeypatch, backbone_calibrates
):
    image = np.zeros((1, 3, 640, 640), np.float32)
    tensors = [
        np.full((1, 2000, 128), 2, np.float32),
        np.full((1, 2000, 128), 3, np.float32),
        np.arange(2000, dtype=np.float32)[None],
        np.full((1, 2000, 4), 4, np.float32),
    ]
    chosen = object()
    calls = []
    monkeypatch.setattr(compiler, "load_images", lambda *args: image)
    monkeypatch.setattr(compiler, "quantize_component", lambda *args: calls.append(args) or chosen)

    def execute(model, pixels):
        assert model is chosen
        return tensors

    def compile_component(*args):
        # Simulate AFE recycling the execution storage after it has been saved.
        for tensor in tensors:
            tensor.fill(-99)

    monkeypatch.setattr(compiler, "execute_backbone", execute)
    monkeypatch.setattr(compiler, "compile_component", compile_component)
    if backbone_calibrates:
        parameters = {"precision": "w8a8", "calibration_bins": 1024, "calibration_samples": 1}
    else:
        parameters = {"precision": "w16a16"}
    names = ("cal",) if backbone_calibrates else ()
    compiler.backbone_phase("profile", parameters, tmp_path, names, ("eval",), tmp_path)
    assert calls[0][1] == "dfine_n_profile_backbone"
    assert calls[0][2] is parameters
    if backbone_calibrates:
        assert list(calls[0][3]) == ["input"] and calls[0][3]["input"] is image
    else:
        assert calls[0][3] is None
    assert not (tmp_path / "decoder-calibration.npz").exists()
    with np.load(tmp_path / "backbone-evaluation.npz", allow_pickle=False) as values:
        assert np.all(values["attention_memory"] == 2)
        assert np.all(values["query_content"] == 3)
        assert np.all(values["proposal_scores"] == np.arange(2000))
        assert np.all(values["reference_logits"] == 4)


def test_calibration_is_rgb_over_255_and_selection_is_disjoint(load_script, tmp_path):
    calibration = load_script("calibration")
    for i in range(8):
        Image.new("RGB", (4, 2), (255, 128, 0)).save(tmp_path / f"{i}.jpg")
    first = calibration.select_images(tmp_path, 42, 0, 3)
    next_ = calibration.select_images(tmp_path, 42, 3, 2)
    assert not set(first) & set(next_)
    assert first + next_ == calibration.select_images(tmp_path, 42, 0, 5)
    stream = io.BytesIO()
    Image.new("RGB", (4, 2), (255, 128, 0)).save(stream, format="PNG")
    pixels = calibration.preprocess_image(stream.getvalue())
    assert pixels.shape == (3, 640, 640) and pixels.dtype == np.float32
    np.testing.assert_array_equal(pixels[:, 0, 0], np.array([1, 128 / 255, 0], np.float32))
    assert calibration.MEAN.tolist() == [0.0, 0.0, 0.0] and calibration.STD.tolist() == [1.0, 1.0, 1.0]


def test_query_bridge_preserves_memory_and_stable_owned_rows(load_script):
    calibration = load_script("calibration")
    memory = np.ones((1, 2000, 128), np.float32)
    content = np.arange(2000 * 128, dtype=np.float32).reshape(1, 2000, 128)
    scores = np.zeros((1, 2000), np.float32)
    refs = np.arange(2000 * 4, dtype=np.float32).reshape(1, 2000, 4)
    forwarded, selected, references = calibration.decoder_inputs([memory, content, scores, refs])
    assert forwarded is memory
    np.testing.assert_array_equal(selected, content[:, :300])
    np.testing.assert_array_equal(references, refs[:, :300])
    assert not np.shares_memory(selected, content) and not np.shares_memory(references, refs)


def regional_graph(tmp_path):
    h = onnx.helper

    def value(name, dimensions=(1, 1, 2, 2), dtype=onnx.TensorProto.FLOAT):
        return h.make_tensor_value_info(name, dtype, dimensions)

    nodes = [
        h.make_node("Conv", ["input", "weight"], ["hidden"], name="/model/backbone/stem/Conv"),
        h.make_node("Conv", ["hidden", "weight"], ["projection"], name="/model/encoder/Conv"),
        h.make_node("Mul", ["projection", "projection"], ["feature"], name="activation"),
        h.make_node("Shape", ["hidden"], ["dimensions"], name="shape"),
        h.make_node("Reshape", ["feature", "dimensions"], ["view"], name="view"),
        h.make_node("Concat", ["view", "view"], ["output"], name="concat", axis=1),
    ]
    model = h.make_model(
        h.make_graph(
            nodes,
            "regional",
            [value("input")],
            [value("output", (1, 2, 2, 2))],
            [onnx.numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weight")],
            value_info=[value(name) for name in ("hidden", "projection", "feature", "view")]
            + [value("dimensions", (4,), onnx.TensorProto.INT64)],
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "regional.onnx"
    onnx.save(model, path)
    return path


def test_regional_policy_keeps_neck_int8_and_freezes_the_complete_float_domain(surgery, tmp_path):
    path = regional_graph(tmp_path)
    before = path.read_bytes()
    policy = surgery.region_precision_map(path, list(surgery.BF16_REGIONS))
    assert policy["graph_sha256"] == surgery.sha256_file(path)[0]
    assert policy["nodes"] == {
        "/model/backbone/stem/Conv": "bfloat16",
        "/model/encoder/Conv": "int8",
        "activation": "bfloat16",
        "view": "bfloat16",
        "concat": "bfloat16",
    }
    assert path.read_bytes() == before


def test_outputs_region_reaches_numerical_producers_without_following_shape_inputs(surgery, tmp_path):
    path = regional_graph(tmp_path)
    policy = surgery.region_precision_map(path, ["outputs"])["nodes"]
    assert policy["activation"] == policy["view"] == policy["concat"] == "bfloat16"
    assert policy["/model/backbone/stem/Conv"] == policy["/model/encoder/Conv"] == "int8"
    assert "shape" not in policy


def test_requested_regions_cannot_silently_stop_matching_after_a_graph_change(surgery, tmp_path):
    path = regional_graph(tmp_path)
    graph = onnx.load(path)
    graph.graph.node[0].name = "/model/renamed/stem/Conv"
    onnx.save(graph, path)
    with pytest.raises(ValueError, match="BF16 regions did not resolve.*backbone"):
        surgery.region_precision_map(path, list(surgery.BF16_REGIONS))


def test_regions_do_not_silently_turn_a_mixed_profile_into_all_bf16(surgery, tmp_path):
    graph, _ = precision_graph(tmp_path)
    with pytest.raises(ValueError, match="mixed arithmetic"):
        surgery.region_precision_map(graph, ["nonconv"])


def precision_graph(tmp_path):
    h = onnx.helper

    def value(name):
        return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])

    model = h.make_model(
        h.make_graph(
            [
                h.make_node("Relu", ["input"], ["hidden"], name="first"),
                h.make_node("Relu", ["hidden"], ["output"], name="last"),
            ],
            "precision",
            [value("input")],
            [value("output")],
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    model.graph.value_info.append(value("hidden"))
    onnx.helper.set_model_props(model, {"contract": "preserved"})
    graph = tmp_path / "backbone.onnx"
    onnx.save(model, graph)
    from model_registry.fetch import sha256_file

    return graph, {"graph_sha256": sha256_file(graph)[0], "nodes": {"first": "int8", "last": "bfloat16"}}


@pytest.mark.parametrize("case", ["hash", "unknown", "dtype", "not_mixed", "missing", "duplicate"])
def test_precision_maps_cannot_silently_drift_or_promote_everything(surgery, tmp_path, case):
    graph, precision_map = precision_graph(tmp_path)
    original = graph.read_bytes()
    if case == "hash":
        precision_map["graph_sha256"] = "0" * 64
    elif case == "unknown":
        precision_map["nodes"]["typo"] = "int8"
    elif case == "dtype":
        precision_map["nodes"]["first"] = "fp32"
    elif case == "not_mixed":
        precision_map["nodes"]["first"] = "bfloat16"
    elif case == "missing":
        del precision_map["nodes"]
    else:
        model = onnx.load(graph)
        model.graph.node[1].name = "first"
        onnx.save(model, graph)
        from model_registry.fetch import sha256_file

        precision_map["graph_sha256"] = sha256_file(graph)[0]
        original = graph.read_bytes()
    with pytest.raises(ValueError):
        surgery.annotate_backbone(graph, precision_map)
    assert graph.read_bytes() == original


def test_mixed_uses_bundled_exporter_without_changing_model_metadata(surgery, tmp_path, monkeypatch):
    graph, precision_map = precision_graph(tmp_path)
    seen = []

    def annotate_graph(model_graph, nodes):
        seen.append(nodes)
        # The bundled helper rebuilds GraphProto and drops its internal value_info.
        return onnx.helper.make_graph(model_graph.node, model_graph.name, model_graph.input, model_graph.output)

    monkeypatch.setitem(
        sys.modules, "neural_compressor.annotate_precision", SimpleNamespace(annotate_graph=annotate_graph)
    )
    surgery.annotate_backbone(graph, precision_map)
    assert seen == [precision_map["nodes"]]
    model = onnx.load(graph)
    assert dict((entry.key, entry.value) for entry in model.metadata_props) == {"contract": "preserved"}
    assert [v.name for v in model.graph.input] == ["input"]
    assert [v.name for v in model.graph.output] == ["output"]
    assert [v.name for v in model.graph.value_info] == ["hidden"]


def reference_graph():
    h, n = onnx.helper, onnx.numpy_helper
    constants = [
        n.from_array(np.array(value, np.float32), name) for name, value in [("zero", 0), ("one", 1), ("eps", 1e-5)]
    ]
    nodes = [
        h.make_node("Sigmoid", ["reference_logits"], ["sigmoid"]),
        h.make_node("Clip", ["sigmoid", "zero", "one"], ["bounded"]),
        h.make_node("Sub", ["one", "bounded"], ["complement"]),
        h.make_node("Clip", ["bounded", "eps"], ["numerator"]),
        h.make_node("Clip", ["complement", "eps"], ["denominator"]),
        h.make_node("Div", ["numerator", "denominator"], ["ratio"]),
        h.make_node("Log", ["ratio"], ["result"]),
    ]

    def value(name):
        return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 300, 4])

    graph = h.make_graph(nodes, "reference_chain", [value("reference_logits")], [value("result")], constants)
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8)


def test_reference_rewrite_is_narrow_and_retains_the_tensor_contract(surgery):
    original = reference_graph()
    candidate = surgery.bounded_reference_logits(original)
    assert [v.SerializeToString() for v in candidate.graph.input] == [
        v.SerializeToString() for v in original.graph.input
    ]
    assert [v.SerializeToString() for v in candidate.graph.output] == [
        v.SerializeToString() for v in original.graph.output
    ]
    assert len(candidate.graph.node) == 1 and candidate.graph.node[0].op_type == "Clip"
    assert any(n.op_type == "Log" for n in original.graph.node), "caller graph was mutated"


@pytest.mark.parametrize("case", ["epsilon", "upper", "complement", "unrelated", "two_logs"])
def test_reference_rewrite_rejects_other_log_semantics(surgery, case):
    graph = reference_graph()
    if case == "epsilon":
        graph.graph.initializer[2].CopyFrom(onnx.numpy_helper.from_array(np.array(1e-4, np.float32), "eps"))
    elif case == "upper":
        graph.graph.node[3].input.append("eps")
    elif case == "complement":
        graph.graph.node[2].input[0] = "zero"
    elif case == "unrelated":
        graph.graph.node[0].input[0] = "other_input"
    else:
        graph.graph.node.append(onnx.helper.make_node("Log", ["ratio"], ["unrelated_log"]))
    with pytest.raises(ValueError):
        surgery.bounded_reference_logits(graph)


def test_reference_rewrite_handles_saturation_but_is_not_claimed_bitwise_equal(surgery):
    ort = pytest.importorskip("onnxruntime")
    original = reference_graph()
    candidate = surgery.bounded_reference_logits(original)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for m in (original, candidate)
    ]
    values = np.resize(np.array([-30, -11.25, 0, 1, 11.2525, 30], np.float32), 1200).reshape(1, 300, 4)
    before, after = [s.run(None, {"reference_logits": values})[0] for s in sessions]
    assert np.isfinite(after).all() and not np.array_equal(before, after)
    assert np.max(np.abs(before - after)) < 0.011


def attention_graph(surgery, layers=None):
    """Three upstream-style deformable-attention samplings, with the heads folded into the GridSample batch."""
    h, n = onnx.helper, onnx.numpy_helper
    heads, channels, queries, levels, points = (
        surgery.HEADS,
        surgery.HEAD_CHANNELS,
        surgery.QUERY_COUNT,
        surgery.LEVELS,
        surgery.POINTS,
    )
    tokens, total = surgery.TOKENS, sum(points)
    inputs, outputs, nodes, initializers = [], [], [], []
    for layer in range(surgery.DECODER_LAYERS if layers is None else layers):
        p = f"layer{layer}/"
        inputs += [
            h.make_tensor_value_info(p + "value", onnx.TensorProto.FLOAT, [1, tokens, heads, channels]),
            h.make_tensor_value_info(p + "grid", onnx.TensorProto.FLOAT, [1, queries, heads, total, 2]),
            h.make_tensor_value_info(p + "weights", onnx.TensorProto.FLOAT, [1, queries, heads, total]),
        ]
        outputs.append(h.make_tensor_value_info(p + "output", onnx.TensorProto.FLOAT, [1, queries, heads * channels]))
        initializers += [
            n.from_array(np.array([heads, channels, tokens], np.int64), p + "flat_shape"),
            n.from_array(np.array([heads, queries, total, 2], np.int64), p + "grid_shape"),
            n.from_array(np.array(points, np.int64), p + "points"),
            n.from_array(np.array([heads, 1, queries, total], np.int64), p + "weight_shape"),
            n.from_array(np.array([-1], np.int64), p + "last"),
            n.from_array(np.array([1, heads * channels, queries], np.int64), p + "merged_shape"),
        ]
        nodes += [
            h.make_node("Transpose", [p + "value"], [p + "value_t"], name=p + "value_t", perm=[0, 2, 3, 1]),
            h.make_node("Reshape", [p + "value_t", p + "flat_shape"], [p + "flat"], name=p + "flat"),
            h.make_node("Transpose", [p + "grid"], [p + "grid_t"], name=p + "grid_t", perm=[0, 2, 1, 3, 4]),
            h.make_node("Reshape", [p + "grid_t", p + "grid_shape"], [p + "grids"], name=p + "grids"),
            h.make_node(
                "Split",
                [p + "grids", p + "points"],
                [p + f"grid{i}" for i in range(len(levels))],
                name=p + "split",
                axis=-2,
            ),
            h.make_node("Transpose", [p + "weights"], [p + "weights_t"], name=p + "weights_t", perm=[0, 2, 1, 3]),
            h.make_node("Reshape", [p + "weights_t", p + "weight_shape"], [p + "weight"], name=p + "weight"),
        ]
        begin = 0
        for level, (height, width) in enumerate(levels):
            q = p + f"level{level}/"
            initializers += [
                n.from_array(np.array([begin], np.int64), q + "start"),
                n.from_array(np.array([begin + height * width], np.int64), q + "end"),
                n.from_array(np.array([2], np.int64), q + "axis"),
                n.from_array(np.array([heads, channels, height, width], np.int64), q + "map_shape"),
            ]
            nodes += [
                h.make_node(
                    "Slice", [p + "flat", q + "start", q + "end", q + "axis"], [q + "tokens"], name=q + "tokens"
                ),
                h.make_node("Reshape", [q + "tokens", q + "map_shape"], [q + "map"], name=q + "map"),
                h.make_node(
                    "GridSample",
                    [q + "map", p + f"grid{level}"],
                    [q + "sampled"],
                    name=q + "sampled",
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=0,
                ),
            ]
            begin += height * width
        nodes += [
            h.make_node(
                "Concat",
                [p + f"level{i}/sampled" for i in range(len(levels))],
                [p + "sampled"],
                name=p + "sampled",
                axis=-1,
            ),
            h.make_node("Mul", [p + "sampled", p + "weight"], [p + "weighted"], name=p + "weighted"),
            h.make_node("ReduceSum", [p + "weighted", p + "last"], [p + "reduced"], name=p + "reduced", keepdims=0),
            h.make_node("Reshape", [p + "reduced", p + "merged_shape"], [p + "merged"], name=p + "merged"),
            h.make_node("Transpose", [p + "merged"], [p + "output"], name=p + "output", perm=[0, 2, 1]),
        ]
    model = h.make_model(
        h.make_graph(nodes, "attention", inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def random_feed(model, seed):
    rng = np.random.default_rng(seed)
    feed = {}
    for value in model.graph.input:
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        feed[value.name] = (
            (2 * rng.random(dims, dtype=np.float32) - 1) if "grid" in value.name else rng.random(dims, dtype=np.float32)
        )
    return feed


def test_per_head_attention_samples_each_head_alone_without_changing_the_result(surgery):
    ort = pytest.importorskip("onnxruntime")
    original = attention_graph(surgery)
    rewritten = surgery.per_head_attention(original)
    samples = [node for node in rewritten.graph.node if node.op_type == "GridSample"]
    assert len(samples) == surgery.DECODER_LAYERS * surgery.HEADS * len(surgery.LEVELS)
    shapes = {tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in rewritten.graph.value_info}
    assert not any(dims and dims[0] == surgery.HEADS for dims in shapes)
    assert len(original.graph.node) == len(attention_graph(surgery).graph.node), "caller graph was mutated"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for m in (original, rewritten)
    ]
    feed = random_feed(original, 7)
    before, after = (session.run(None, feed) for session in sessions)
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", ["layers", "points", "span", "interpolation", "weights"])
def test_per_head_attention_rejects_other_attention_layouts(surgery, case):
    if case == "layers":
        model = attention_graph(surgery, layers=2)
    else:
        model = attention_graph(surgery)
        node = {n.name: n for n in model.graph.node}
        init = {i.name: i for i in model.graph.initializer}
        if case == "points":
            init["layer0/points"].CopyFrom(onnx.numpy_helper.from_array(np.array([3, 9], np.int64), "layer0/points"))
        elif case == "span":
            init["layer0/level1/start"].CopyFrom(
                onnx.numpy_helper.from_array(np.array([0], np.int64), "layer0/level1/start")
            )
            init["layer0/level1/end"].CopyFrom(
                onnx.numpy_helper.from_array(np.array([400], np.int64), "layer0/level1/end")
            )
        elif case == "interpolation":
            (mode,) = [a for a in node["layer0/level0/sampled"].attribute if a.name == "mode"]
            mode.s = b"nearest"
        else:
            node["layer0/weights_t"].attribute[0].ints[:] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Unsupported D-FINE graph"):
        surgery.per_head_attention(model)


def integral_graph(surgery, layers=None, bins=None):
    """Three upstream-style Integral projections, with queries and sides folded into the softmax rows."""
    h, n = onnx.helper, onnx.numpy_helper
    queries, bins = surgery.QUERY_COUNT, bins or surgery.BINS
    inputs, outputs, nodes, initializers = [], [], [], []
    for layer in range(surgery.DECODER_LAYERS if layers is None else layers):
        p = f"integral{layer}/"
        inputs.append(h.make_tensor_value_info(p + "corners", onnx.TensorProto.FLOAT, [1, queries, 4 * bins]))
        outputs.append(h.make_tensor_value_info(p + "boxes", onnx.TensorProto.FLOAT, [1, queries, 4]))
        initializers += [
            n.from_array(np.linspace(-4, 4, bins, dtype=np.float32), p + "weights"),
            n.from_array(np.array([-1, bins], np.int64), p + "rows"),
            n.from_array(np.array([-1, 4], np.int64), p + "sides"),
            n.from_array(np.array([1, queries, 4], np.int64), p + "shape"),
        ]
        nodes += [
            h.make_node("Identity", [p + "weights"], [p + "project"], name=p + "project"),
            h.make_node("Reshape", [p + "corners", p + "rows"], [p + "flat"], name=p + "flat"),
            h.make_node("Softmax", [p + "flat"], [p + "probabilities"], name=p + "probabilities", axis=1),
            h.make_node("MatMul", [p + "probabilities", p + "project"], [p + "expected"], name=p + "expected"),
            h.make_node("Reshape", [p + "expected", p + "sides"], [p + "folded"], name=p + "folded"),
            h.make_node("Reshape", [p + "folded", p + "shape"], [p + "boxes"], name=p + "boxes"),
        ]
    model = h.make_model(
        h.make_graph(nodes, "integral", inputs, outputs, initializers),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def test_batch_preserving_integral_keeps_the_batch_axis_without_changing_the_result(surgery):
    ort = pytest.importorskip("onnxruntime")
    original = integral_graph(surgery)
    rewritten = surgery.batch_preserving_integral(original)
    softmaxes = [node for node in rewritten.graph.node if node.op_type == "Softmax"]
    values = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in rewritten.graph.value_info}
    assert [values[node.input[0]] for node in softmaxes] == [(1, 1200, surgery.BINS)] * surgery.DECODER_LAYERS
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = [
        ort.InferenceSession(m.SerializeToString(), options, providers=["CPUExecutionProvider"])
        for m in (original, rewritten)
    ]
    feed = random_feed(original, 11)
    before, after = (session.run(None, feed) for session in sessions)
    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", ["layers", "bins", "shared_softmax"])
def test_batch_preserving_integral_rejects_other_projections(surgery, case):
    if case == "layers":
        model = integral_graph(surgery, layers=2)
    elif case == "bins":
        model = integral_graph(surgery, bins=surgery.BINS - 1)
    else:
        model = integral_graph(surgery)
        model.graph.node.append(onnx.helper.make_node("Relu", ["integral0/probabilities"], ["extra"], name="extra"))
        model.graph.output.append(onnx.helper.make_tensor_value_info("extra", onnx.TensorProto.FLOAT, [1200, 33]))
    with pytest.raises(ValueError, match="Unsupported D-FINE graph"):
        surgery.batch_preserving_integral(model)


def test_sdk_system_exit_cannot_look_like_a_successful_phase(compiler):
    def sdk_failure():
        raise SystemExit()

    with pytest.raises(RuntimeError, match="failed"):
        compiler.execute_phase(sdk_failure, {})


@pytest.mark.parametrize("invocation", ["w16a16", "other"])
def test_registry_compile_cli_requires_the_profile_invocation(compiler, monkeypatch, tmp_path, invocation):
    calls = []
    model = tmp_path / "model.onnx"
    model.touch()
    monkeypatch.setattr(compiler, "compile_model", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(model),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            invocation,
            "--dataset",
            str(tmp_path),
            "--fidelity-dataset",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "compiled"),
        ],
    )
    if invocation == "other":
        with pytest.raises(SystemExit) as error:
            compiler.main()
        assert error.value.code == 2
        assert calls == []
    else:
        assert compiler.main() == 0
        assert calls == [(model, invocation, tmp_path, tmp_path, tmp_path / "compiled")]


def test_compiled_packages_have_native_placement_and_real_checksums(compiler, model_dir, compiled, build_profile):
    from model_registry.fetch import sha256_file

    receipt = json.loads((compiled / "compile-outputs.json").read_text())
    assert receipt["model_id"] == "dfine_n" and receipt["build_profile_id"] == build_profile
    assert build_profile in {profile["id"] for profile in load_model(model_dir)["build_profiles"]}
    assert [item["component_id"] for item in receipt["outputs"]] == ["backbone", "decoder"]
    for item in receipt["outputs"]:
        path = compiled / item["mpk_path"]
        assert sha256_file(path)[0] == item["mpk_sha256"]
        compiler.audit_mpk(path, item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text())
    assert fidelity["samples"] == 5 and fidelity["selection"]["offset"] >= 50
    assert [m["name"] for m in fidelity["metrics"]] == ["pred_boxes", "pred_logits"]
    assert len(receipt["derived_graphs"]) == 2
