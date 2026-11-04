"""Validate the static D-FINE export and fail closed on incompatible source contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import pytest


@pytest.fixture
def converter():
    path = Path(__file__).resolve().parents[2] / "scripts" / "convert.py"
    spec = importlib.util.spec_from_file_location(f"{path.parents[1].name}_convert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
@pytest.mark.parametrize("variant", ["", "_discrete"])
def test_family_identity(converter, size, variant):
    assert converter.identity(f"dfine_{size}{variant}") == (size, "discrete" if variant else "default")


@pytest.mark.parametrize("name", ["dfine", "dfine_t", "dfine_s_other", "dfine_s_discrete_more", "other_n"])
def test_invalid_identity(converter, name):
    with pytest.raises(ValueError):
        converter.identity(name)


@pytest.mark.parametrize("case", ["valid", "permuted", "shape", "nan", "inf", "missing"])
def test_raw_outputs(converter, case):
    expected = [np.zeros(converter.INTERFACE[name], dtype=np.float32) for name in converter.OUTPUTS]
    expected[0][0, 0, 0] = 0.5
    actual = [x.copy() for x in expected]
    if case == "permuted":
        actual[0] = actual[0][:, ::-1]
    elif case == "shape":
        actual[1] = actual[1][:, :1]
    elif case in {"nan", "inf"}:
        expected[1][0, 0, 0] = actual[1][0, 0, 0] = float(case)
    elif case == "missing":
        actual.pop()
    else:
        actual[0] += 1e-5
        assert converter.compare_outputs(expected, actual)["pred_boxes"] < 1e-4
        return
    with pytest.raises((ValueError, AssertionError)):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "opset",
        "order",
        "dtype",
        "shape",
        "dynamic",
        "identity",
        "model_id",
        "box_format",
        "class_output",
        "missing_metadata",
    ],
)
def test_fixed_onnx_interface(tmp_path, converter, case):
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    if case == "order":
        values[1:] = values[:0:-1]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[0].type.tensor_type.shape.dim[0].dim_value = 2
    elif case == "dynamic":
        values[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [v.name],
            value=helper.make_tensor(
                v.name,
                onnx.TensorProto.FLOAT,
                converter.INTERFACE[v.name],
                np.zeros(converter.INTERFACE[v.name], dtype=np.float32).ravel(),
            ),
        )
        for v in values[1:]
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "fixture", values[:1], values[1:]),
        opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)],
    )
    _, sampling = converter.identity(converter.MODEL_DIR.name)
    metadata = {
        "model_id": converter.MODEL_DIR.name,
        "sampling": sampling,
        "box_format": "normalized_cxcywh",
        "class_output": "logits",
    }
    if case == "identity":
        metadata["sampling"] = "discrete" if sampling == "default" else "default"
    elif case == "model_id":
        metadata["model_id"] = "dfine_x" if converter.MODEL_DIR.name != "dfine_x" else "dfine_s"
    elif case == "box_format":
        metadata["box_format"] = "xyxy"
    elif case == "class_output":
        metadata["class_output"] = "probabilities"
    elif case == "missing_metadata":
        metadata.clear()
    onnx.helper.set_model_props(model, metadata)
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize(
    "case", ["valid", "model_type", "width", "queries", "classes", "sampling", "normalize", "learned_query"]
)
def test_source_config(tmp_path, converter, case):
    size, _ = converter.identity(converter.MODEL_DIR.name)
    config = dict(
        zip(
            (
                "encoder_hidden_dim",
                "d_model",
                "num_feature_levels",
                "decoder_layers",
                "feat_strides",
                "decoder_n_points",
            ),
            converter.SIZES[size],
            strict=True,
        )
    )
    config.update(
        model_type="d_fine",
        num_queries=300,
        max_num_bins=32,
        decoder_method="default",
        eval_size=None,
        learn_initial_query=False,
        encode_proj_layers=[converter.SIZES[size][2] - 1],
        id2label={str(i): f"class{i}" for i in range(80)},
    )
    processor = {
        "do_normalize": False,
        "do_pad": False,
        "do_rescale": True,
        "do_resize": True,
        "size": {"height": 640, "width": 640},
        "rescale_factor": 1 / 255,
    }
    if case == "model_type":
        config["model_type"] = "rt_detr"
    elif case == "width":
        config["encoder_hidden_dim"] += 1
    elif case == "queries":
        config["num_queries"] = 100
    elif case == "classes":
        config["id2label"].pop("79")
    elif case == "sampling":
        config["decoder_method"] = "discrete"
    elif case == "learned_query":
        config["learn_initial_query"] = True
    elif case == "normalize":
        processor["do_normalize"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    if case == "valid":
        converter.validate_config(tmp_path, converter.MODEL_DIR.name)
    else:
        with pytest.raises(ValueError):
            converter.validate_config(tmp_path, converter.MODEL_DIR.name)


def test_source_requires_complete_file_set(converter):
    source = {
        "type": "file_set",
        "entrypoint": "model.safetensors",
        "files": [{"path": name} for name in converter.SOURCE[2]],
    }
    converter.validate_source(source)
    source["files"].pop()
    with pytest.raises(ValueError):
        converter.validate_source(source)


def test_converted_interface(converter, converted):
    converter.validate_export(converted)


@pytest.mark.parametrize("case", ["same", "tie", "far", "different_set", "duplicate", "negative", "nan", "shape"])
def test_proposal_alignment(converter, case):
    scores = np.linspace(1, 0, 350, dtype=np.float32)[None]
    actual_scores = scores.copy()
    expected = np.arange(300, dtype=np.int64)[None]
    actual = expected.copy()
    if case in ("tie", "far"):
        actual[0, [10, 11]] = actual[0, [11, 10]]
        if case == "tie":
            scores[0, 11] = scores[0, 10]
            actual_scores[0, 11] = actual_scores[0, 10] - 1e-6
    elif case == "different_set":
        actual[0, -1] = 300
    elif case == "duplicate":
        actual[0, -1] = 0
    elif case == "negative":
        actual[0, -1] = -1
    elif case == "nan":
        actual_scores[0, 0] = float("nan")
    elif case == "shape":
        actual = actual[:, :1]
    if case in ("same", "tie"):
        order = converter.proposal_alignment(scores, actual_scores, expected, actual)
        np.testing.assert_array_equal(actual[:, order], expected)
    else:
        with pytest.raises((ValueError, AssertionError)):
            converter.proposal_alignment(scores, actual_scores, expected, actual)


@pytest.mark.parametrize("proposal_nodes", [0, 1, 2])
def test_selection_diagnostics_do_not_modify_published_graph(tmp_path, converter, proposal_nodes):
    helper = onnx.helper
    scores = helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, 350])
    nodes = []
    for index, count in enumerate([4] + [300] * proposal_nodes):
        nodes.extend(
            [
                helper.make_node(
                    "Constant",
                    [],
                    [f"k{index}"],
                    value=helper.make_tensor(f"k{index}", onnx.TensorProto.INT64, [1], [count]),
                ),
                helper.make_node("TopK", ["scores", f"k{index}"], [f"values{index}", f"indices{index}"]),
            ]
        )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "fixture",
            [scores],
            [helper.make_tensor_value_info("values0", onnx.TensorProto.FLOAT, [1, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    before = path.read_bytes()
    if proposal_nodes == 1:
        diagnostic, names = converter.selection_model(path)
        assert names == ["scores", "indices1"]
        assert len(onnx.load_from_string(diagnostic).graph.output) == 3
    else:
        with pytest.raises(ValueError, match="one 300-query"):
            converter.selection_model(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "channels,shapes,points",
    [
        (16, [(40, 40), (20, 20)], [6, 6]),
        (32, [(40, 40), (20, 20)], [6, 6]),
        (32, [(80, 80), (40, 40), (20, 20)], [3, 6, 3]),
    ],
)
def test_default_attention_preserves_source_and_only_chunks_c32_40(converter, monkeypatch, channels, shapes, points):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.d_fine.modeling_d_fine import multi_scale_deformable_attention_v2

    generator = torch.Generator().manual_seed(123)
    value = torch.rand((1, sum(h * w for h, w in shapes), 8, channels), generator=generator)
    locations = 2 * torch.rand((1, 7, 8, sum(points), 2), generator=generator) - 0.5
    weights = torch.rand((1, 7, 8, sum(points)), generator=generator).softmax(-1)
    expected = multi_scale_deformable_attention_v2(value, shapes, locations, weights, points, "default")
    calls = []
    original = torch.nn.functional.grid_sample

    def record(image, grid, **kwargs):
        calls.append(tuple(image.shape))
        assert kwargs == {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
        return original(image, grid, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", record)
    actual = converter.per_head_default_attention(value, shapes, locations, weights, points)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_calls = []
    for _head in range(8):
        for h, w in shapes:
            count = 2 if channels == 32 and (h, w) == (40, 40) else 1
            expected_calls.extend([(1, channels // count, h, w)] * count)
    assert calls == expected_calls


def test_attention_preparation_is_instance_local_and_retains_loaded_scales(converter):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DFineConfig
    from transformers.models.d_fine.modeling_d_fine import (
        DFineMultiscaleDeformableAttention,
        multi_scale_deformable_attention_v2,
    )

    default = DFineMultiscaleDeformableAttention(DFineConfig(decoder_method="default"))
    discrete = DFineMultiscaleDeformableAttention(DFineConfig(decoder_method="discrete"))
    untouched = DFineMultiscaleDeformableAttention(DFineConfig(decoder_method="default"))
    default.num_points_scale.fill_(0.12345)
    scales = default.num_points_scale.clone()
    assert converter.prepare_default_attention(torch.nn.ModuleList([default, discrete])) == 1
    assert default.ms_deformable_attn_core is converter.per_head_default_attention
    assert discrete.ms_deformable_attn_core is multi_scale_deformable_attention_v2
    assert untouched.ms_deformable_attn_core is multi_scale_deformable_attention_v2
    torch.testing.assert_close(default.num_points_scale, scales, atol=0, rtol=0)


def test_attention_preparation_rejects_unexpected_source_implementation(converter):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DFineConfig
    from transformers.models.d_fine.modeling_d_fine import DFineMultiscaleDeformableAttention

    module = DFineMultiscaleDeformableAttention(DFineConfig(decoder_method="default"))
    module.ms_deformable_attn_core = lambda *args: None
    with pytest.raises(ValueError, match="Unexpected D-FINE attention"):
        converter.prepare_default_attention(module)


@pytest.mark.parametrize("case", ["batch", "length", "points", "heads", "channels", "locations", "method"])
def test_default_attention_rejects_incompatible_geometry(converter, case):
    torch = pytest.importorskip("torch")
    value = torch.zeros(1, 5, 8, 16)
    locations = torch.zeros(1, 7, 8, 4, 2)
    weights = torch.zeros(1, 7, 8, 4)
    points, method = [2, 2], "default"
    if case == "batch":
        value = value.repeat(2, 1, 1, 1)
    elif case == "length":
        value = value[:, :4]
    elif case == "points":
        points = [2]
    elif case == "heads":
        value = value[:, :, :4]
    elif case == "channels":
        value = value[:, :, :, :12]
    elif case == "locations":
        locations = locations[..., :1]
    else:
        method = "discrete"
    with pytest.raises(ValueError):
        converter.per_head_default_attention(value, [(2, 2), (1, 1)], locations, weights, points, method)


@pytest.mark.parametrize("model_id,prepared_count", [("dfine_s", 2), ("dfine_m", 3), ("dfine_s_discrete", 1)])
def test_conversion_rejects_incomplete_or_unexpected_attention_preparation(
    converter, monkeypatch, tmp_path, model_id, prepared_count
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(encode_proj_layers=[], feat_strides=[])
            self.model = SimpleNamespace(encoder=SimpleNamespace(aifi=[]))

    monkeypatch.setattr(converter, "MODEL_DIR", tmp_path / model_id)
    monkeypatch.setattr(converter, "load_reference", lambda *_: (Detector(), {}))
    monkeypatch.setattr(converter, "prepare_default_attention", lambda _: prepared_count)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("Must reject before export"))
    with pytest.raises(ValueError, match="Prepared attention count"):
        converter.convert(tmp_path / "model.safetensors", tmp_path / "output")


@pytest.mark.parametrize("reg_scale", [4.0, 8.0])
@pytest.mark.parametrize("case", ["random", "uniform", "extreme", "batch2", "strided"])
def test_integral_preserves_source_projection(converter, reg_scale, case):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DFineConfig
    from transformers.models.d_fine.modeling_d_fine import DFineIntegral, weighting_function

    reference = DFineIntegral(DFineConfig(max_num_bins=32))
    prepared = DFineIntegral(DFineConfig(max_num_bins=32))
    assert converter.prepare_integral(prepared) == 1
    generator = torch.Generator().manual_seed(741)
    corners = torch.randn((1, 7, 132), generator=generator)
    if case == "uniform":
        corners.zero_()
    elif case == "extreme":
        corners = torch.where(corners >= 0, 1000.0, -1000.0)
    elif case == "batch2":
        corners = torch.randn((2, 7, 132), generator=generator)
    elif case == "strided":
        corners = torch.randn((1, 7, 264), generator=generator)[..., ::2]
    project = weighting_function(32, torch.tensor([0.5]), torch.tensor([reg_scale]))
    original_project = project.clone()
    torch.testing.assert_close(prepared(corners, project), reference(corners, project), rtol=0, atol=0)
    assert reference.forward.__func__ is DFineIntegral.forward
    torch.testing.assert_close(project, original_project, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["bins", "subclass", "forward", "already_prepared"])
def test_integral_preparation_rejects_unexpected_source(converter, case):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DFineConfig
    from transformers.models.d_fine.modeling_d_fine import DFineIntegral

    class OtherIntegral(DFineIntegral):
        pass

    cls = OtherIntegral if case == "subclass" else DFineIntegral
    module = cls(DFineConfig(max_num_bins=16 if case == "bins" else 32))
    if case == "forward":
        module.forward = lambda *args: None
    elif case == "already_prepared":
        assert converter.prepare_integral(module) == 1
    with pytest.raises(ValueError, match="Unexpected D-FINE Integral"):
        converter.prepare_integral(module)


@pytest.mark.parametrize("case", ["rank", "width", "project_rank", "project_size"])
def test_integral_rejects_incompatible_geometry(converter, case):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    corners, project = torch.zeros(1, 7, 132), torch.zeros(33)
    if case == "rank":
        corners = corners[0]
    elif case == "width":
        corners = corners[..., :-1]
    elif case == "project_rank":
        project = project[None]
    else:
        project = project[:-1]
    with pytest.raises(ValueError):
        converter.batch_preserving_integral(SimpleNamespace(max_num_bins=32), corners, project)


@pytest.mark.parametrize("prepared_count", [0, 2])
def test_conversion_rejects_unexpected_integral_count(converter, monkeypatch, tmp_path, prepared_count):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(encode_proj_layers=[], feat_strides=[])
            self.model = SimpleNamespace(encoder=SimpleNamespace(aifi=[]))

    monkeypatch.setattr(converter, "MODEL_DIR", tmp_path / "dfine_s")
    monkeypatch.setattr(converter, "load_reference", lambda *_: (Detector(), {}))
    monkeypatch.setattr(converter, "prepare_default_attention", lambda _: 3)
    monkeypatch.setattr(converter, "prepare_integral", lambda _: prepared_count)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("Must reject before export"))
    with pytest.raises(ValueError, match="Prepared Integral count"):
        converter.convert(tmp_path / "model.safetensors", tmp_path / "output")
