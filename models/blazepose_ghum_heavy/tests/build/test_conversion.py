"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.mark.parametrize("case", ["valid", "missing", "shape", "nan", "world_landmarks"])
def test_output_agreement(case, converter: ModuleType):
    rng = np.random.RandomState(0)
    expected = {
        name: rng.uniform(-1, 1, shape).astype(np.float32) for name, shape in list(converter.INTERFACE.items())[1:]
    }
    actual = {name: value + 1e-6 for name, value in expected.items()}
    if case == "missing":
        del actual["Identity_4"]
    elif case == "shape":
        actual["Identity"] = actual["Identity"][:, :-1]
    elif case == "nan":
        actual["Identity_1"][0, 0] = np.nan
    elif case == "world_landmarks":
        # One millimetre off in one world coordinate.
        actual["Identity_4"][0, 0] += 1e-3
    else:
        assert max(converter.compare_outputs(expected, actual).values()) < 2e-6
        return
    with pytest.raises(ValueError):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    nodes = [
        helper.make_node("Constant", [], [value.name], value=onnx.numpy_helper.from_array(np.zeros(shape, np.float32)))
        for value, shape in zip(values[1:], list(converter.INTERFACE.values())[1:])
    ]
    if case == "order":
        values[1], values[2] = values[2], values[1]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[0].type.tensor_type.shape.dim[3].dim_value = 4
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13 if case == "opset" else 16)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def _numbered_export(path, scale, offset, unused=False):
    """A two-node export whose constants carry tf2onnx's run-dependent numbers, listed in reverse order of use."""
    helper = onnx.helper
    initializers = [
        onnx.numpy_helper.from_array(np.full((1, 3), 2, np.float32), scale),
        onnx.numpy_helper.from_array(np.full((1, 3), 1, np.float32), offset),
    ]
    if unused:
        initializers.append(onnx.numpy_helper.from_array(np.zeros(1, np.float32), "const_fold_opt__9"))
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", offset], ["shifted"]), helper.make_node("Mul", ["shifted", scale], ["y"])],
        "tf2onnx",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])],
        initializers,
        doc_string=f"converted from {path}",
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)]), path)


def test_repeated_conversions_name_constants_alike(tmp_path, converter: ModuleType):
    first, second = tmp_path / "first.onnx", tmp_path / "second.onnx"
    _numbered_export(first, "const_fold_opt__2385", "Relu6__131_min__737")
    _numbered_export(second, "const_fold_opt__2085", "Relu6__333_min__1372")

    converter.name_constants(first)
    converter.name_constants(second)

    assert first.read_bytes() == second.read_bytes()
    model = onnx.load(first)
    assert model.graph.doc_string == ""
    assert [(node.op_type, list(node.input)) for node in model.graph.node] == [
        ("Add", ["x", "constant_0"]), ("Mul", ["shifted", "constant_1"])
    ]
    assert [value.name for value in model.graph.initializer] == ["constant_0", "constant_1"]
    assert onnx.numpy_helper.to_array(model.graph.initializer[1]).tolist() == [[2, 2, 2]]
    _numbered_export(first, "a", "b", unused=True)
    with pytest.raises(ValueError, match="1 initializers are not node inputs"):
        converter.name_constants(first)


def test_source_must_be_the_pinned_tflite(model_dir: Path, converter: ModuleType):
    converter.validate_source(load_metadata(model_dir)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="pose_landmark_heavy.tflite"):
        converter.validate_source(
            {
                "type": "file",
                "uri": "hf://models/Pictorial/mediapipe@main/modules/pose_landmark/pose_landmark_full.tflite",
            }
        )
