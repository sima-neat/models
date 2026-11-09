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


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "changed"])
def test_complete_upstream_output_agreement(case, converter: ModuleType):
    expected = np.random.default_rng(352).normal(0, 1, (1, 84, 8400)).astype(np.float32)
    actual = expected.copy()
    if case == "valid":
        converter.compare_outputs(expected, actual)
    elif case == "shape":
        with pytest.raises(ValueError, match="predictions"):
            converter.compare_outputs(expected, actual[:, :, :-1])
    elif case == "nan":
        actual[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            converter.compare_outputs(expected, actual)
    else:
        actual[0, 0, 0] += 1
        with pytest.raises(AssertionError):
            converter.compare_outputs(expected, actual)


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
        values[1].type.tensor_type.shape.dim[1].dim_value = 100
    node = helper.make_node(
        "Constant",
        [],
        [values[1].name],
        value=helper.make_tensor(values[1].name, onnx.TensorProto.FLOAT, [1, 84, 8400], np.zeros(84 * 8400)),
    )
    graph = helper.make_graph([node], "fixture", values[:1], values[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13 if case == "opset" else 12)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType):
    model = load_metadata(ROOT / "models/yolov8l")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="yolov8l.pt"):
        converter.validate_source({"type": "file", "uri": "hf://models/Ultralytics/YOLOv8@main/yolov8l-seg.pt"})
