"""Small checks for conversion failures that a synthetic smoke input can miss."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "inf", "different"])
def test_output_agreement(case, converter: ModuleType):
    expected = np.zeros(converter.INTERFACE["logits"], dtype=np.float32)
    expected[0, 7] = 4.0
    actual = expected.copy()
    if case == "shape":
        actual = actual[:, :1]
    elif case in {"nan", "inf"}:
        expected[0, 0] = actual[0, 0] = float(case)
    elif case == "different":
        actual[0, 7] = 3.9
    else:
        actual += 1e-5
        converter.compare_outputs(expected, actual)
        return
    with pytest.raises((ValueError, AssertionError)):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "name", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    helper = onnx.helper
    pixels = helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, converter.INTERFACE["pixel_values"])
    logits = helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, converter.INTERFACE["logits"])
    if case == "name":
        logits.name = "output"
    elif case == "dtype":
        pixels.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        pixels.type.tensor_type.shape.dim[0].dim_value = 2
    shape = converter.INTERFACE["logits"]
    zeros = helper.make_tensor(logits.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel())
    node = helper.make_node("Constant", [], [logits.name], value=zeros)
    graph = helper.make_graph([node], "fixture", [pixels], [logits])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType):
    model = load_metadata(MODEL_DIR)
    converter.validate_source(model["conversion"]["default"]["source"])
    pth = {"type": "file", "uri": "https://example.com/checkpoint.pth"}
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source(pth)


@pytest.mark.parametrize(
    ("architecture", "classes"), [("hrnet_w32", 1000), ("hrnet_w18", 1000), ("hrnet_w32", 11821)]
)
def test_config_must_name_the_architecture_and_imagenet_classes(tmp_path, architecture, classes, converter: ModuleType):
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture": architecture, "num_classes": classes}), encoding="utf-8"
    )
    if (architecture, classes) == ("hrnet_w32", 1000):
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="expected"):
            converter.validate_config(tmp_path)


def test_converted_onnx_keeps_the_interface(converted: Path, converter: ModuleType):
    converter.validate_export(converted)
