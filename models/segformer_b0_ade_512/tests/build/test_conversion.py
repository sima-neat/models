"""Small checks for conversion failures that a synthetic smoke input can miss."""

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


def _logits(seed=0):
    return np.random.RandomState(seed).normal(0, 4, (1, 150, 128, 128)).astype(np.float32)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "drift"])
def test_logits_agreement(case, converter: ModuleType):
    expected = _logits()
    actual = expected + 1e-5
    error, match = ValueError, "Unexpected or non-finite"
    if case == "shape":
        actual = actual[:, :, :-1]
    elif case == "nan":
        actual[0, 0, 0, 0] = np.nan
    elif case == "drift":
        actual = _logits(1)
        error, match = AssertionError, "Not equal to tolerance"
    else:
        converter.compare_logits(expected, actual)
        return
    with pytest.raises(error, match=match):
        converter.compare_logits(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    if case == "order":
        values.reverse()
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[1].type.tensor_type.shape.dim[1].dim_value = 19
    shape = [dim.dim_value for dim in values[1].type.tensor_type.shape.dim]
    node = helper.make_node(
        "Constant",
        [],
        [values[1].name],
        value=helper.make_tensor(values[1].name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape)))),
    )
    graph = helper.make_graph([node], "fixture", values[:1], values[1:])
    opset = converter.OPSET - 1 if case == "opset" else converter.OPSET
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType):
    model = load_metadata(ROOT / "models/segformer_b0_ade_512")
    converter.validate_source(model["conversion"]["default"]["source"])
    source = dict(model["conversion"]["default"]["source"])
    source["files"] = [item for item in source["files"] if item["path"] != "model.safetensors"]
    with pytest.raises(ValueError, match="config.json, model.safetensors"):
        converter.validate_source(source)
