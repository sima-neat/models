"""Verify the upstream source, the parameter mapping, the output contract and conversion's numerical guard."""

from urllib.parse import urlsplit

import numpy as np
import onnx
import pytest
from model_registry.model import load_model


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "changed"])
def test_embeddings_must_agree_with_pytorch(case, load_script):
    converter = load_script("convert")
    expected = np.random.default_rng(662).normal(size=(1, converter.FEATURES)).astype(np.float32)
    actual = expected.copy()
    if case == "shape":
        with pytest.raises(ValueError):
            converter.compare_outputs(expected, actual[:, :-1])
    elif case == "nan":
        actual[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            converter.compare_outputs(expected, actual)
    elif case == "changed":
        actual[0, :8] += 1
        with pytest.raises(AssertionError):
            converter.compare_outputs(expected, actual)
    else:
        converter.compare_outputs(expected, actual)


def test_source_is_the_fast_reid_release_asset(model_dir, load_script):
    converter = load_script("convert")
    source = load_model(model_dir)["conversion"]["default"]["source"]
    uri = urlsplit(source["uri"])
    assert (uri.scheme, uri.netloc) == ("https", "github.com")
    assert uri.path.split("/")[1:] == ["JDAI-CV", "fast-reid", "releases", "download", "v0.1.1", converter.CHECKPOINT]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="checkpoint"):
        converter.validate_source({"type": "file", "uri": "https://example.com/other.pth"})


def test_upstream_parameters_map_onto_the_rebuilt_network(load_script):
    converter = load_script("convert")
    state = {
        "pixel_mean": np.zeros((1, 3, 1, 1)),
        "pixel_std": np.ones((1, 3, 1, 1)),
        "backbone.conv1.weight": 1,
        "backbone.layer4.2.bn3.running_var": 2,
        "heads.bnneck.weight": 3,
        "heads.classifier.weight": 4,
    }

    mapped = converter.upstream_state(state)

    assert mapped["backbone.0.weight"] == 1
    assert mapped["backbone.7.2.bn3.running_var"] == 2
    assert mapped["bnneck.weight"] == 3
    assert "heads.classifier.weight" not in mapped and mapped.keys() >= {"pixel_mean", "pixel_std"}
    with pytest.raises(ValueError, match="unexpected checkpoint parameter"):
        converter.upstream_state({"heads.extra": 0})


@pytest.mark.parametrize("case", ["valid", "opset", "shape", "dtype"])
def test_converted_interface_is_fixed(tmp_path, case, load_script):
    converter = load_script("convert")
    helper = onnx.helper
    values = [helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in converter.INTERFACE.items()]
    if case == "shape":
        values[1].type.tensor_type.shape.dim[1].dim_value = 512
    if case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    constant = onnx.numpy_helper.from_array(np.zeros(converter.INTERFACE["embedding"], np.float32))
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["embedding"], value=constant)], "fixture", values[:1], values[1:]
    )
    opset = converter.OPSET + 1 if case == "opset" else converter.OPSET
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_exported_artifact_keeps_the_upstream_forward(converted, load_script):
    converter = load_script("convert")
    converter.validate_export(converted)
    graph = onnx.load(converted).graph
    # The network normalizes pixels itself, then pools and applies the batch-norm neck.
    operations = [node.op_type for node in graph.node]
    assert operations[:2] == ["Sub", "Div"]
    assert operations.count("Conv") == 53 and "GlobalAveragePool" in operations
