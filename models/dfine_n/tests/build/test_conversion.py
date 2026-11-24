"""Validate the static D-FINE Nano export and fail closed on incompatible source contracts."""

import json

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version

OPSET = onnx_opset_version()


@pytest.fixture
def converter(load_script):
    return load_script("convert")


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
    ["valid", "opset", "order", "dtype", "shape", "dynamic", "model_id", "box_format", "class_output", "no_metadata"],
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
        opset_imports=[helper.make_opsetid("", OPSET + 1 if case == "opset" else OPSET)],
    )
    metadata = dict(converter.METADATA)
    if case == "model_id":
        metadata["model_id"] = "dfine_x"
    elif case == "box_format":
        metadata["box_format"] = "xyxy"
    elif case == "class_output":
        metadata["class_output"] = "probabilities"
    elif case == "no_metadata":
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
    "case",
    ["valid", "model_type", "width", "levels", "queries", "classes", "sampling", "normalize", "learned_query"],
)
def test_source_config(tmp_path, converter, case):
    config = dict(converter.CONFIG, id2label={str(i): f"class{i}" for i in range(80)})
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
    elif case == "levels":
        config["feat_strides"] = [8, 16, 32]
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
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError):
            converter.validate_config(tmp_path)


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


def test_conversion_report_describes_the_upstream_export(converted):
    report = json.loads((converted.parent / "conversion.json").read_text(encoding="utf-8"))
    assert report["model_id"] == "dfine_n"
    assert report["effective_refinement"] == {"reg_scale": 4.0, "up": 0.5}
    assert report["static_position_embeddings"] == 1
    assert "preparation" not in report
    assert [list(check) for check in report["max_absolute_errors"]] == [["pred_boxes", "pred_logits"]] * 3


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
        opset_imports=[helper.make_opsetid("", OPSET)],
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
