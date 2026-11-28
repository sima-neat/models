"""Verify the upstream output contract and conversion's numerical guard."""

from urllib.parse import urlsplit

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version
from model_registry.model import load_model

OPSET = onnx_opset_version()


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "changed"])
def test_complete_upstream_predictions_agree(case, load_script):
    converter = load_script("convert")
    expected = np.random.default_rng(443).normal(size=(1, 12, 85)).astype(np.float32)
    actual = expected.copy()
    if case == "shape":
        with pytest.raises(ValueError):
            converter.compare_outputs(expected, actual[:, :-1])
    elif case == "nan":
        actual[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            converter.compare_outputs(expected, actual)
    elif case == "changed":
        actual[0, 0, 0] += 1
        with pytest.raises(AssertionError):
            converter.compare_outputs(expected, actual)
    else:
        converter.compare_outputs(expected, actual)


def test_source_is_the_original_megvii_checkpoint(model_dir, load_script):
    converter = load_script("convert")
    model = load_model(model_dir)
    source = model["conversion"]["default"]["source"]
    uri = urlsplit(source["uri"])
    assert (uri.scheme, uri.netloc) == ("https", "github.com")
    assert uri.path.split("/")[1:] == [
        "Megvii-BaseDetection",
        "YOLOX",
        "releases",
        "download",
        "0.1.1rc0",
        converter.CHECKPOINT,
    ]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="checkpoint"):
        converter.validate_source({"type": "file", "uri": "https://example.com/other.pth"})


@pytest.mark.parametrize("case", ["valid", "opset", "shape", "dtype"])
def test_converted_interface_is_fixed(tmp_path, case, load_script):
    converter = load_script("convert")
    helper = onnx.helper
    values = [helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in converter.INTERFACE.items()]
    if case == "shape":
        values[1].type.tensor_type.shape.dim[2].dim_value = 84
    if case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["predictions"],
                value=onnx.numpy_helper.from_array(np.zeros(converter.INTERFACE["predictions"], np.float32)),
            )
        ],
        "fixture",
        values[:1],
        values[1:],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET + 1 if case == "opset" else OPSET)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_exported_artifact_keeps_the_upstream_forward(converted, load_script):
    load_script("convert").validate_export(converted)
    # The conversion includes probabilities and decoded boxes; only surgery removes this tail.
    graph = onnx.load(converted)
    assert any(node.op_type == "Sigmoid" for node in graph.graph.node)
    assert any(node.op_type == "Exp" for node in graph.graph.node)
