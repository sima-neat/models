"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]
# The rebuilt network's parameters, which the checkpoint's 1,838 tensors fill when it loads strictly.
PARAMETERS = 9_676_350


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
        values[1].type.tensor_type.shape.dim[1].dim_value = 21
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
    model = load_metadata(ROOT / "models/fcn_hrnet_w18")
    source = model["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match=converter.CHECKPOINT):
        converter.validate_source({**source, "uri": source["uri"].replace("ade20k", "voc12aug")})


def test_network_has_the_checkpoint_layout(converter: ModuleType):
    torch = pytest.importorskip("torch")
    network = converter.build_network(converter.WIDTHS, converter.CLASS_COUNT)
    state = network.state_dict()

    assert sum(parameter.numel() for parameter in network.parameters()) == PARAMETERS
    # MMSegmentation's key names, from the stem through the last fuse layer to the classifier.
    assert state["backbone.conv1.weight"].shape == (64, 3, 3, 3)
    # Branch 0 reaches branch 3 through three strided convolutions; the last one widens 18 channels to 144.
    assert state["backbone.stage4.2.fuse_layers.3.0.2.0.weight"].shape == (144, 18, 3, 3)
    assert state["decode_head.conv_seg.weight"].shape == (150, sum(converter.WIDTHS), 1, 1)
    with torch.inference_mode():
        logits = network.eval()(torch.zeros(converter.INTERFACE["image"]))
    assert list(logits.shape) == converter.INTERFACE["logits"]


def test_converted_graph_has_only_convolutions_and_bilinear_resizes(converted: Path):
    graph = onnx.load(converted).graph
    operations = Counter(node.op_type for node in graph.node)
    resize_modes = {
        (attribute.name, attribute.s.decode())
        for node in graph.node
        if node.op_type == "Resize"
        for attribute in node.attribute
        if attribute.name in {"mode", "coordinate_transformation_mode"}
    }

    assert set(operations) == {"Conv", "Relu", "Add", "Resize", "Concat"}
    # 31 fuse paths resize twice, by 2**(j - i) and then to the target size, the head resizes all four branches, and
    # the logits are resized to the input.
    assert operations["Resize"] == 67
    assert resize_modes == {("mode", "linear"), ("coordinate_transformation_mode", "half_pixel")}
