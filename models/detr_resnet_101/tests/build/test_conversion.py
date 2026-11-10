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
    model = load_metadata(ROOT / "models/detr_resnet_101")
    converter.validate_source(model["conversion"]["default"]["source"])
    pth = {"type": "file", "uri": "https://example.com/checkpoint.pth"}
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source(pth)


@pytest.mark.parametrize("field", ["valid", "model_type", "backbone", "dilation", "num_queries"])
def test_config_must_name_this_edition(tmp_path, field, converter: ModuleType):
    config = {"model_type": "detr", "backbone": converter.BACKBONE, "dilation": converter.DILATION, "num_queries": 100}
    other_edition = {
        "model_type": "rf_detr",
        "backbone": "resnet50" if converter.BACKBONE == "resnet101" else "resnet101",
        "dilation": not converter.DILATION,
        "num_queries": 300,
    }
    if field != "valid":
        config[field] = other_edition[field]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if field == "valid":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="config.json declares"):
            converter.validate_config(tmp_path)


def test_converted_graph_keeps_no_trigonometry_or_mask(converted: Path):
    """The sine position embedding and the all-ones pixel mask fold to constants at the fixed input size."""
    model = onnx.load(converted)
    operations = {node.op_type for node in model.graph.node}
    assert not operations & {"Sin", "Cos", "CumSum", "Constant", "Shape", "ConstantOfShape"}
