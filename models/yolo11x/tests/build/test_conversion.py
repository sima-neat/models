"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version
from model_registry.model import load_model as load_metadata

OPSET = onnx_opset_version()
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


class Reference:
    """Stands in for the torch tensors Ultralytics returns: compare_heads only calls numpy()."""

    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array


def _heads():
    levels = [80, 40, 20]
    boxes = [np.random.RandomState(level).normal(0, 4, (1, 64, level, level)).astype(np.float32) for level in levels]
    scores = [
        np.random.RandomState(level + 1).normal(0, 4, (1, 80, level, level)).astype(np.float32) for level in levels
    ]
    expected = {
        "boxes": Reference(np.concatenate([value.reshape(1, 64, -1) for value in boxes], axis=2)),
        "scores": Reference(np.concatenate([value.reshape(1, 80, -1) for value in scores], axis=2)),
    }
    return expected, [*boxes, *scores]


@pytest.mark.parametrize("case", ["valid", "level_order", "shape", "nan"])
def test_head_agreement(case, converter: ModuleType):
    expected, actual = _heads()
    error, match = ValueError, "Unexpected or non-finite"
    if case == "level_order":
        actual[3], actual[4] = actual[4], actual[3]
        error, match = AssertionError, "scores"
    elif case == "shape":
        actual[0] = actual[0][:, :, :-1]
    elif case == "nan":
        actual[5] = actual[5].copy()
        actual[5][0, 0, 0, 0] = np.nan
    else:
        actual = [value + 1e-5 for value in actual]
        converter.compare_heads(expected, actual)
        return
    with pytest.raises(error, match=match):
        converter.compare_heads(expected, actual)


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
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET + 1 if case == "opset" else OPSET)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType):
    model = load_metadata(ROOT / "models/yolo11x")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="yolo11x.pt"):
        converter.validate_source({"type": "file", "uri": "hf://models/Ultralytics/YOLO11@main/yolo11x-seg.pt"})
