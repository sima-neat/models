"""Precision-profile, graph-rewrite and installed artifact contracts for Large."""

from __future__ import annotations

import copy
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from model_registry.model import load_model
from PIL import Image


@pytest.fixture
def compiler(load_script):
    return load_script("compile")


@pytest.fixture
def surgery(load_script):
    return load_script("surgery")


def test_large_geometry_uses_three_levels_and_256_channels(surgery):
    assert surgery.geometry("dfine_l") == (256, (6400, 1600, 400))


@pytest.mark.parametrize("tokens,width", [(2000, 128), (8400, 128), (2000, 256)])
def test_large_proposal_selection_rejects_nano_dimensions(load_script, tokens, width):
    calibration = load_script("calibration")
    with pytest.raises(ValueError, match="Large proposal boundary"):
        calibration.select_top_queries(
            np.zeros((1, tokens), np.float32),
            np.zeros((1, tokens, width), np.float32),
            np.zeros((1, tokens, 4), np.float32),
        )


def test_only_registered_profiles_are_admitted_and_decoder_stays_w16a16(compiler, model_dir):
    metadata = load_model(model_dir)
    assert [p["id"] for p in metadata["build_profiles"]] == ["w8a16_w16a16", "w8a8_w16a16", "w16a16_w16a16"]
    decoder = {"precision": "w16a16", "calibration_bins": 2048, "calibration_samples": 50}
    for profile, precision, bins in [
        ("w8a16_w16a16", "w8a16", 2048),
        ("w8a8_w16a16", "w8a8", 1024),
        ("w16a16_w16a16", "w16a16", 2048),
    ]:
        candidate = compiler.component_parameters(metadata, profile)
        assert list(candidate) == ["backbone", "decoder"]
        assert candidate["backbone"] == {"precision": precision, "calibration_bins": bins, "calibration_samples": 50}
        assert candidate["decoder"] == decoder
    for profile in ("w8a8_mixed_w16a16", "unregistered_profile"):
        with pytest.raises(ValueError):
            compiler.component_parameters(metadata, profile)
    for key, value in [("precision", "w8a8"), ("calibration_bins", 0), ("calibration_samples", True)]:
        changed = copy.deepcopy(metadata)
        changed["build_profiles"][0]["components"][0]["parameters"][key] = value
        with pytest.raises(ValueError):
            compiler.component_parameters(changed, "w8a16_w16a16")


def test_both_quantization_schemes_are_bf16_without_unmeasured_option_changes(compiler, monkeypatch):
    calls = []

    class Config:
        def with_activation_quantization(self, value):
            calls.append(("activation", value))
            return self

        def with_weight_quantization(self, value):
            calls.append(("weight", value))
            return self

        def with_calibration(self, value):
            calls.append(("bins", value))
            return self

    monkeypatch.setattr(compiler, "default_quantization", Config())
    monkeypatch.setattr(compiler, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(compiler, "HistogramMSEMethod", lambda bins: bins)
    compiler.quantization_config({"precision": "w16a16", "calibration_bins": 2048})
    assert calls == [("activation", "bf16"), ("weight", "bf16"), ("bins", 2048)]


@pytest.mark.parametrize("profile", ["w8a16_w16a16", "w8a8_w16a16", "w8a8_mixed_w16a16"])
def test_candidate_profile_parameters_do_not_change_the_decoder(compiler, model_dir, profile):
    metadata = copy.deepcopy(load_model(model_dir))
    candidate = copy.deepcopy(metadata["build_profiles"][0])
    candidate["id"] = profile
    candidate["components"][0]["parameters"]["precision"] = compiler.PROFILES[profile]
    if "mixed" in profile:
        candidate["components"][0]["parameters"]["bf16_regions"] = ["backbone", "nonconv", "outputs"]
    metadata["build_profiles"] = [candidate]
    parameters = compiler.component_parameters(metadata, profile)
    assert parameters["decoder"] == candidate["components"][1]["parameters"]
    candidate["components"][1]["parameters"]["precision"] = "w8a8"
    with pytest.raises(ValueError, match="decoder"):
        compiler.component_parameters(metadata, profile)


@pytest.mark.parametrize("regions", [[], "outputs", ["typo"], ["outputs", "outputs"], [False], [{}]])
def test_mixed_regions_are_explicit_and_validated(compiler, model_dir, regions):
    metadata = copy.deepcopy(load_model(model_dir))
    candidate = copy.deepcopy(metadata["build_profiles"][0])
    candidate["id"] = "w8a8_mixed_w16a16"
    candidate["components"][0]["parameters"].update(precision="w8a8", bf16_regions=regions)
    metadata["build_profiles"] = [candidate]
    with pytest.raises(ValueError, match="BF16 regions"):
        compiler.component_parameters(metadata, candidate["id"])


def masked_reference_graph(tmp_path):
    h = onnx.helper

    def value(name, shape=(1, 2, 4), dtype=onnx.TensorProto.FLOAT):
        return h.make_tensor_value_info(name, dtype, shape)

    model = h.make_model(
        h.make_graph(
            [
                h.make_node("Conv", ["image", "weight"], ["feature"], name="convolution"),
                h.make_node("Where", ["valid", "anchors", "sentinel"], ["masked"], name="mask"),
                h.make_node("Add", ["masked", "offsets"], ["reference_logits"], name="references"),
            ],
            "references",
            [
                value("image", (1, 1, 1, 1)),
                value("valid", dtype=onnx.TensorProto.BOOL),
                value("anchors"),
                value("offsets"),
            ],
            [value("reference_logits")],
            [
                onnx.numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "weight"),
                onnx.numpy_helper.from_array(np.array(np.finfo(np.float32).max, np.float32), "sentinel"),
            ],
        ),
        opset_imports=[h.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    path = tmp_path / "references.onnx"
    onnx.save(model, path)
    return path


def test_reference_policy_preserves_math_and_keeps_convolutions_int8(surgery, tmp_path):
    path = masked_reference_graph(tmp_path)
    before = path.read_bytes()
    assert surgery.reference_precision_map(path) == {
        "graph_sha256": surgery.sha256_file(path)[0],
        "nodes": {"convolution": "int8", "mask": "bfloat16", "references": "bfloat16"},
    }
    assert path.read_bytes() == before


@pytest.mark.parametrize("case", ["sentinel", "addition", "mask"])
def test_reference_policy_rejects_an_unrecognized_boundary(surgery, tmp_path, case):
    path = masked_reference_graph(tmp_path)
    model = onnx.load(path)
    if case == "sentinel":
        model.graph.initializer[1].CopyFrom(onnx.numpy_helper.from_array(np.array(0, np.float32), "sentinel"))
    elif case == "addition":
        model.graph.node[-1].op_type = "Mul"
    else:
        model.graph.node[-1].input[0] = "anchors"
    onnx.checker.check_model(model)
    onnx.save(model, path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        surgery.reference_precision_map(path)
    assert path.read_bytes() == before


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


@pytest.mark.parametrize("precision", ["w8a16", "w8a8"])
def test_int8_policies_keep_a16_defaults_and_only_enable_measured_int8_correction(compiler, monkeypatch, precision):
    calls = []

    class Config:
        def __getattr__(self, name):
            def apply(value):
                calls.append((name, value))
                return self

            return apply

    monkeypatch.setattr(compiler, "default_quantization", Config())
    monkeypatch.setattr(compiler, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(compiler, "quantization_scheme", lambda **kwargs: kwargs)
    monkeypatch.setattr(compiler, "HistogramMSEMethod", lambda bins: bins)
    monkeypatch.setattr(compiler, "BiasCorrectionType", SimpleNamespace(REGULAR="regular"))
    compiler.quantization_config({"precision": precision, "calibration_bins": 1024})
    assert calls[1] == ("with_weight_quantization", {"asymmetric": False, "per_channel": True, "bits": 8})
    assert calls[2] == ("with_calibration", 1024)
    if precision == "w8a8":
        assert calls[0][1] == {"asymmetric": True, "per_channel": False, "bits": 8}
        assert calls[3:] == [("with_bias_correction", "regular"), ("with_per_token_quant", False)]
    else:
        assert calls[0] == ("with_activation_quantization", "bf16") and len(calls) == 3


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


@pytest.mark.parametrize("case", ["bf16", "mixed", "wrong_input_dtype", "wrong_output_dtype"])
def test_package_audit_requires_uniform_bf16_backbone_outputs(compiler, tmp_path, case):
    # This exercises the recipe's manifest audit, not executable/ELF admission.
    plugins = [{"name": "MLA_0", "processor": "MLA", "sequence": 1, "output_nodes": []}]
    inputs = []
    for index, shape in enumerate(compiler.OUTPUT_SHAPES["backbone"]):
        name = f"head_{index}"
        size = int(np.prod(shape)) * 4
        params = {"in_dtype": "bfloat16", "out_dtype": "float32"}
        kernel = "cast_transform"
        if index == 0:
            if case == "mixed":
                kernel = "dequantization_transform"
            elif case == "wrong_input_dtype":
                params["in_dtype"] = "int8"
            elif case == "wrong_output_dtype":
                params["out_dtype"] = "bfloat16"
        plugins.append(
            {
                "sequence": index + 2,
                "processor": "EV74",
                "output_nodes": [{"name": name, "size": size}],
                "config_params": {"kernel": kernel, "params": params},
            }
        )
        inputs.append({"name": name, "size": size})
    plugins.append(
        {
            "sequence": len(plugins) + 1,
            "processor": "EV74",
            "input_nodes": inputs,
            "output_nodes": [{"name": f"public_{i}", "size": v["size"]} for i, v in enumerate(inputs)],
            "config_params": {"kernel": "pass_through"},
        }
    )
    manifest = {
        "model_sdk_version": compiler.COMPILER_VERSION,
        "input_nodes": [{"name": "input", "size": 1 * 3 * 640 * 640 * 4}],
        "plugins": plugins,
    }
    archive = tmp_path / "audit-only.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, data in [("audit_mpk.json", json.dumps(manifest).encode()), ("audit.elf", b"not executed")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    if case == "bf16":
        assert compiler.audit_mpk(archive, "backbone") == manifest
    else:
        with pytest.raises(ValueError, match="uniform BF16"):
            compiler.audit_mpk(archive, "backbone")


@pytest.mark.parametrize("mixed", [False, True])
def test_prepare_promotes_output_paths_without_overwriting_reference_or_mixed_policy(
    compiler, monkeypatch, tmp_path, mixed
):
    graph, seen = tmp_path / "backbone.onnx", []
    monkeypatch.setattr(compiler, "prepare_model", lambda *args: (graph, tmp_path / "decoder.onnx"))
    outputs = [np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32)]
    monkeypatch.setattr(compiler, "session", lambda *args: SimpleNamespace(run=lambda *args: outputs))
    monkeypatch.setattr(compiler, "load_images", lambda *args: [np.zeros((3, 1, 1), np.float32)])
    monkeypatch.setattr(compiler, "decoder_feed", lambda *args: {})
    initial = {
        "graph_sha256": "source",
        "nodes": {"body": "bfloat16" if mixed else "int8", "mask": "bfloat16", "output": "int8"},
    }

    def regions(path, selected):
        assert path == graph
        if selected == ["outputs"]:
            return {"nodes": {"body": "int8", "mask": "int8", "output": "bfloat16"}}
        assert mixed and selected == ["backbone", "nonconv", "outputs"]
        return copy.deepcopy(initial)

    monkeypatch.setattr(compiler, "reference_precision_map", lambda path: copy.deepcopy(initial))
    monkeypatch.setattr(compiler, "region_precision_map", regions)
    monkeypatch.setattr(compiler, "annotate_backbone", lambda path, policy: seen.append(copy.deepcopy(policy)))
    parameters = {"precision": "w8a8"}
    if mixed:
        parameters["bf16_regions"] = ["backbone", "nonconv", "outputs"]
    compiler.prepare_phase(tmp_path, tmp_path, [], tmp_path, parameters)
    expected = copy.deepcopy(initial)
    expected["nodes"]["output"] = "bfloat16"
    assert seen == [expected]
    assert json.loads((tmp_path / "precision-map.json").read_text()) == expected


@pytest.mark.parametrize("precision", ["w16a16", "w8a16"])
def test_prepare_leaves_existing_bf16_activation_profiles_unannotated(compiler, monkeypatch, tmp_path, precision):
    monkeypatch.setattr(compiler, "prepare_model", lambda *args: (tmp_path, tmp_path))
    outputs = [np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32)]
    monkeypatch.setattr(compiler, "session", lambda *args: SimpleNamespace(run=lambda *args: outputs))
    monkeypatch.setattr(compiler, "load_images", lambda *args: [np.zeros((3, 1, 1), np.float32)])
    monkeypatch.setattr(compiler, "decoder_feed", lambda *args: {})

    def unexpected(*args):
        pytest.fail("BF16-activation profiles do not need output annotations")

    monkeypatch.setattr(compiler, "region_precision_map", unexpected)
    monkeypatch.setattr(compiler, "annotate_backbone", unexpected)
    compiler.prepare_phase(tmp_path, tmp_path, [], tmp_path, {"precision": precision})
    assert not (tmp_path / "precision-map.json").exists()


def test_decoder_calibration_uses_owned_outputs_from_the_selected_backbone(compiler, tmp_path, monkeypatch):
    image = np.zeros((1, 3, 640, 640), np.float32)
    tensors = [
        np.full((1, 8400, 256), 2, np.float32),
        np.full((1, 8400, 256), 3, np.float32),
        np.arange(8400, dtype=np.float32)[None],
        np.full((1, 8400, 4), 4, np.float32),
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
    parameters = {"precision": "w8a16", "calibration_bins": 2048, "calibration_samples": 1}
    compiler.backbone_phase("w8a16_w16a16", parameters, tmp_path, ("cal",), ("eval",), tmp_path)
    assert calls[0][1] == "dfine_l_w8a16_w16a16_backbone"
    assert calls[0][2] is parameters
    with np.load(tmp_path / "decoder-calibration.npz", allow_pickle=False) as values:
        assert np.all(values["attention_memory"] == 2)
        assert np.all(values["query_content"] == 3)
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


def test_query_bridge_preserves_memory_and_stable_owned_rows(load_script):
    calibration = load_script("calibration")
    memory = np.ones((1, 8400, 256), np.float32)
    content = np.arange(8400 * 256, dtype=np.float32).reshape(1, 8400, 256)
    scores = np.zeros((1, 8400), np.float32)
    refs = np.arange(8400 * 4, dtype=np.float32).reshape(1, 8400, 4)
    forwarded, selected, references = calibration.decoder_inputs([memory, content, scores, refs])
    assert forwarded is memory
    np.testing.assert_array_equal(selected, content[:, :300])
    np.testing.assert_array_equal(references, refs[:, :300])
    assert not np.shares_memory(selected, content) and not np.shares_memory(references, refs)


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


def test_reference_rewrite_is_narrow_and_retains_the_tensor_contract(load_script):
    surgery = load_script("surgery")
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
def test_reference_rewrite_rejects_other_log_semantics(load_script, case):
    surgery = load_script("surgery")
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


def test_reference_rewrite_handles_saturation_but_is_not_claimed_bitwise_equal(load_script):
    ort = pytest.importorskip("onnxruntime")
    surgery = load_script("surgery")
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


def test_sdk_system_exit_cannot_look_like_a_successful_phase(compiler):
    def sdk_failure():
        raise SystemExit()

    with pytest.raises(RuntimeError, match="failed"):
        compiler.execute_phase(sdk_failure, {})


@pytest.mark.parametrize("invocation", ["w16a16_w16a16", "other"])
def test_registry_compile_cli_requires_the_profile_invocation(compiler, monkeypatch, tmp_path, invocation):
    calls = []
    monkeypatch.setattr(compiler, "compile_model", lambda *args: calls.append(args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(tmp_path),
            "--build-profile",
            "w16a16_w16a16",
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
        assert calls == [(tmp_path, invocation, tmp_path, tmp_path, tmp_path / "compiled")]


def test_compiled_packages_have_native_placement_and_real_checksums(compiler, compiled, build_profile):
    from model_registry.fetch import sha256_file

    receipt = json.loads((compiled / "compile-outputs.json").read_text())
    assert receipt["model_id"] == "dfine_l" and receipt["build_profile_id"] == build_profile
    assert build_profile in compiler.PROFILES
    assert [item["component_id"] for item in receipt["outputs"]] == ["backbone", "decoder"]
    for item in receipt["outputs"]:
        path = compiled / item["mpk_path"]
        assert sha256_file(path)[0] == item["mpk_sha256"]
        compiler.audit_mpk(path, item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text())
    assert fidelity["samples"] == 10 and fidelity["selection"]["offset"] >= 50
    assert [m["name"] for m in fidelity["metrics"]] == ["pred_boxes", "pred_logits"]
    assert len(receipt["derived_graphs"]) == 2


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


@pytest.mark.parametrize("mixed", [False, True])
def test_quantization_enforces_applied_precision_only_for_the_mixed_profile(compiler, monkeypatch, mixed):
    quantized, checked = object(), []
    monkeypatch.setattr(compiler, "load_model", lambda source: SimpleNamespace(quantize=lambda **kwargs: quantized))
    monkeypatch.setattr(compiler, "onnx_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(compiler, "DataGenerator", lambda data: data)
    monkeypatch.setattr(compiler, "InputName", str)
    monkeypatch.setattr(compiler, "convert_data_generator_to_iterable", lambda data: data)
    monkeypatch.setattr(compiler, "quantization_config", lambda parameters: None)
    monkeypatch.setattr(compiler, "require_mixed_precision", checked.append)
    parameters = {"precision": "w8a8", "calibration_bins": 1024}
    if mixed:
        parameters["bf16_regions"] = ["backbone", "nonconv", "outputs"]
    result = compiler.quantize_component(
        Path("unused.onnx"), "candidate", parameters, {"input": np.ones((1, 3, 2, 2), np.float32)}
    )
    assert result is quantized
    assert checked == ([quantized] if mixed else [])


@pytest.mark.parametrize("case", ["valid", "missing", "distribution", "geometry", "interpolation"])
def test_large_admits_exact_six_layer_attention_geometry(surgery, monkeypatch, tmp_path, case):
    shapes = [(1, 32, 80, 80)] * 48 + [(1, 16, 40, 40)] * 96 + [(1, 32, 20, 20)] * 48
    if case == "missing":
        shapes.pop()
    elif case == "distribution":
        shapes[0] = (1, 32, 20, 20)
    elif case == "geometry":
        shapes[0] = (1, 32, 81, 80)
    nodes, values = [], []
    for i, shape in enumerate(shapes):
        name = f"source_{i}"
        values.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        mode = "nearest" if case == "interpolation" and i == 0 else "bilinear"
        nodes.append(onnx.helper.make_node("GridSample", [name, "grid"], [f"out_{i}"], mode=mode))
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, "attention-admission", [], [], value_info=values))
    monkeypatch.setattr(surgery.onnx, "load", lambda path: model)
    monkeypatch.setattr(surgery, "split_model", lambda model, model_id: (model, model, None))
    monkeypatch.setattr(surgery, "bounded_reference_logits", lambda model: model)
    if case == "valid":
        assert all(path.is_file() for path in surgery.prepare_model("unused.onnx", tmp_path / "graphs"))
    else:
        with pytest.raises(ValueError, match="attention samples|interpolation"):
            surgery.prepare_model("unused.onnx", tmp_path / "graphs")
