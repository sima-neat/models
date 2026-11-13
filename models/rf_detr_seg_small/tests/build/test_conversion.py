"""Small checks for conversion failures that a synthetic smoke input can miss."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.mark.parametrize("case", ["valid", "permuted", "shape", "nan", "inf"])
def test_output_agreement(case, converter: ModuleType):
    expected = [np.zeros(converter.INTERFACE[name], dtype=np.float32) for name in converter.OUTPUTS]
    expected[0][0, 0, 0] = 0.5
    actual = [x.copy() for x in expected]
    if case == "permuted":
        actual[0] = actual[0][:, ::-1]
    elif case == "shape":
        actual[0] = actual[0][:, :1]
    elif case in {"nan", "inf"}:
        expected[0][0, 0, 0] = actual[0][0, 0, 0] = float(case)
    else:
        actual[0] += 1e-5
        converter.compare_outputs(expected, actual)
        return
    with pytest.raises((ValueError, AssertionError)):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
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
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [value.name],
            value=helper.make_tensor(
                value.name,
                onnx.TensorProto.FLOAT,
                converter.INTERFACE[value.name],
                np.zeros(converter.INTERFACE[value.name]).ravel(),
            ),
        )
        for value in values[1:]
    ]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType):
    model = load_metadata(ROOT / "models/rf_detr_seg_small")
    converter.validate_source(model["conversion"]["default"]["source"])
    pth = {"type": "file", "uri": "https://example.com/checkpoint.pth"}
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source(pth)


@pytest.mark.parametrize("model_type", ["rf_detr", "detr"])
def test_config_must_name_rf_detr(tmp_path, model_type, converter: ModuleType):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if model_type == "rf_detr":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="rf_detr"):
            converter.validate_config(tmp_path)


@pytest.mark.parametrize("case", ["same", "tie", "different"])
def test_selection_allows_only_swaps_of_tied_proposals(case, converter: ModuleType):
    scores = np.linspace(1, 0, converter.PROPOSAL_COUNT, dtype=np.float32)[None]
    actual, indices = scores.copy(), np.arange(converter.QUERY_COUNT)[None]
    if case == "tie":
        actual[0, 11] = actual[0, 10]
        indices[0, [10, 11]] = [11, 10]
    elif case == "different":
        # A proposal far below the selected ones.
        indices[0, 10] = converter.QUERY_COUNT + 100
    if case == "different":
        with pytest.raises(ValueError, match="different proposals"):
            converter.compare_selection(scores, actual, indices)
    else:
        assert converter.compare_selection(scores, actual, indices) is (case == "same")
