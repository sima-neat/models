"""Small checks for conversion failures that a synthetic smoke input can miss."""

import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rf_detr_convert", ROOT / "models/rf_detr_small/scripts/convert.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


@pytest.mark.parametrize("case", ["valid", "permuted", "shape", "nan", "inf"])
def test_output_agreement(case):
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
def test_fixed_onnx_interface(tmp_path, case):
    helper = onnx.helper
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
              for name, shape in converter.INTERFACE.items()]
    if case == "order":
        values[1:] = values[:0:-1]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[0].type.tensor_type.shape.dim[0].dim_value = 2
    nodes = [helper.make_node("Constant", [], [value.name],
                             value=helper.make_tensor(value.name, onnx.TensorProto.FLOAT,
                                                      converter.INTERFACE[value.name],
                                                      np.zeros(converter.INTERFACE[value.name]).ravel()))
             for value in values[1:]]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)
