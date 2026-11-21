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


def _interface(converter, case):
    helper = onnx.helper
    inputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.INPUT_SHAPES[c].items()
    ]
    outputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.OUTPUT_SHAPES[c].items()
    ]
    if case == "dtype":
        inputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[0].type.tensor_type.shape.dim[2].dim_value = 1024
    elif case == "missing":
        outputs.pop()
    nodes = []
    for value in outputs:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor = helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape))))
        nodes.append(helper.make_node("Constant", [], [value.name], value=tensor))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    opset = converter.OPSET - 1 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "dtype", "shape", "missing"])
def test_onnx_interface(tmp_path, case, converter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "drift"])
def test_onnx_outputs_must_match_pytorch(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 10, (1, 768, 1, 64)).astype(np.float32)
    actual = expected + 1e-4
    error, match = ValueError, "Unexpected or non-finite"
    if case == "shape":
        actual = actual[:, :767]
    elif case == "nan":
        actual[0, 0, 0, 0] = np.nan
    elif case == "drift":
        actual = expected + 0.1
        error, match = AssertionError, "features"
    else:
        converter.compare(expected, actual, "features")
        return
    with pytest.raises(error, match=match):
        converter.compare(expected, actual, "features")


def test_source_must_be_the_pinned_safetensors(converter: ModuleType):
    model = load_metadata(ROOT / "models/vjepa2_1_vitb_16x256")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="model.safetensors"):
        converter.validate_source({"type": "file", "uri": "hf://models/mnmly/vjepa2.1-vit-base-384@main/model.pt"})
