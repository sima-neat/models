"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

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
    expected = np.full(converter.INTERFACE["predicted_depth"], 500.0, dtype=np.float32)
    expected[0, 7, 7] = 900.0
    actual = expected.copy()
    if case == "shape":
        actual = actual[:, :1]
    elif case in {"nan", "inf"}:
        expected[0, 0, 0] = actual[0, 0, 0] = float(case)
    elif case == "different":
        actual[0, 7, 7] = 899.0
    else:
        actual += 0.05
        converter.compare_outputs(expected, actual)
        return
    with pytest.raises((ValueError, AssertionError)):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "name", "dtype", "shape", "shape_operation"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    helper = onnx.helper
    pixels = helper.make_tensor_value_info("pixel_values", onnx.TensorProto.FLOAT, converter.INTERFACE["pixel_values"])
    depth = helper.make_tensor_value_info(
        "predicted_depth", onnx.TensorProto.FLOAT, converter.INTERFACE["predicted_depth"]
    )
    if case == "name":
        depth.name = "output"
    elif case == "dtype":
        pixels.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        pixels.type.tensor_type.shape.dim[0].dim_value = 2
    shape = converter.INTERFACE["predicted_depth"]
    zeros = helper.make_tensor(depth.name, onnx.TensorProto.FLOAT, shape, np.zeros(shape).ravel())
    nodes = [helper.make_node("Constant", [], [depth.name], value=zeros)]
    if case == "shape_operation":
        # A padding size left to compute at run time from the input's shape.
        nodes.append(helper.make_node("Shape", ["pixel_values"], ["pixel_shape"]))
    graph = helper.make_graph(nodes, "fixture", [pixels], [depth])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_is_the_authors_release_checkpoint(converter: ModuleType):
    source = load_metadata(MODEL_DIR)["conversion"]["default"]["source"]
    uri = urlsplit(source["uri"])
    assert (uri.scheme, uri.netloc) == ("https", "github.com")
    assert uri.path.split("/")[1:] == ["isl-org", "MiDaS", "releases", "download", "v2_1", converter.CHECKPOINT]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="checkpoint"):
        converter.validate_source({"type": "file", "uri": "https://example.com/other.pt"})


def test_converted_onnx_keeps_the_interface(converted: Path, converter: ModuleType):
    converter.validate_export(converted)
