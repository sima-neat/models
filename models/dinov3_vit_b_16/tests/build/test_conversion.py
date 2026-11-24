"""Checks of the conversion's source pin, the checkpoint it accepts, and the weights.pth and reference.onnx it
writes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.conversion import conversion_outputs
from model_registry.dependencies import onnx_opset_version
from model_registry.model import load_model as load_metadata

OPSET = onnx_opset_version()
MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_file_set(converter: ModuleType) -> None:
    converter.validate_source(load_metadata(MODEL_DIR)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="file set"):
        converter.validate_source({"type": "file_set", "entrypoint": "model.safetensors",
                                   "files": [{"path": "model.safetensors"}, {"path": "config.json"}]})


@pytest.mark.parametrize("case", ["valid", "architecture", "features"])
def test_config_must_name_the_converted_architecture(tmp_path: Path, converter: ModuleType, case: str) -> None:
    # timm's _qkvb variant of the same size has query, key and value biases, which surgery.py does not rebuild.
    architecture = converter.ARCHITECTURE + ("_qkvb" if case == "architecture" else "")
    features = converter.HIDDEN_SIZE * (2 if case == "features" else 1)
    (tmp_path / "config.json").write_text(json.dumps({"architecture": architecture, "num_features": features}))
    if case == "valid":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="config.json declares"):
            converter.validate_config(tmp_path)


def test_conversion_writes_the_weights_and_then_the_reference(converter: ModuleType) -> None:
    # compile.py receives the first output as --model and reads the second beside it.
    assert conversion_outputs(MODEL_DIR) == ["weights.pth", "reference.onnx"]


def test_the_interface_is_the_one_surgery_exports(
    converter: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")

    assert converter.INTERFACE == surgery.INTERFACE
    assert converter.ROTARY_EMBEDDING == surgery.ROTARY_EMBEDDING
    assert (converter.PATCHES, converter.HEAD_SIZE, converter.REGISTERS) == (
        surgery.PATCHES, surgery.HEAD_SIZE, surgery.REGISTERS)


@pytest.mark.parametrize("case", ["valid", "opset", "name", "dtype", "shape"])
def test_reference_interface(tmp_path: Path, converter: ModuleType, case: str) -> None:
    helper = onnx.helper
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
              for name, shape in converter.INTERFACE.items()]
    if case == "name":
        values[1].name = "pooler_output"
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[2].type.tensor_type.shape.dim[1].dim_value = converter.PATCHES + 1 + converter.REGISTERS
    nodes = []
    for value in values[1:]:
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        zeros = helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel())
        nodes.append(helper.make_node("Constant", [], [value.name], value=zeros))
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:])
    path = tmp_path / "reference.onnx"
    opset = OPSET + 1 if case == "opset" else OPSET
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "different"])
def test_reference_must_follow_timm(converter: ModuleType, case: str) -> None:
    expected = [np.zeros(converter.INTERFACE[name], np.float32) for name in converter.OUTPUTS]
    actual = [value + 5e-4 for value in expected]
    if case == "shape":
        actual[1] = actual[1][:, 1:]
    elif case == "nan":
        actual[0][0, 0] = np.nan
    elif case == "different":
        actual[1][0, 3, 7] = 0.01
    if case == "valid":
        converter.compare_outputs(expected, actual)
    else:
        with pytest.raises((ValueError, AssertionError)):
            converter.compare_outputs(expected, actual)


def test_converted_weights_are_timms_state_dict_with_the_rotary_embedding(
    converted: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The conversion stage's real output: the float32 tensors surgery.py rebuilds the encoder from, and beside it
    reference.onnx with the package interface."""
    torch = pytest.importorskip("torch")
    surgery, converter = load_script("surgery"), load_script("convert")
    weights = torch.load(converted, weights_only=True)

    assert {name: list(value.shape) for name, value in weights.items()} == surgery.weight_shapes()
    assert all(value.dtype == torch.float32 for value in weights.values())
    # Each row holds sines then cosines of the same phases.
    sines, cosines = weights[surgery.ROTARY_EMBEDDING].tensor_split(2, dim=1)
    torch.testing.assert_close(sines.square() + cosines.square(), torch.ones_like(sines))
    converter.validate_export(converted.with_name(conversion_outputs(MODEL_DIR)[1]))
