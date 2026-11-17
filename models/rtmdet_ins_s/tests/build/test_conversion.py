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


def _outputs(converter: ModuleType) -> list[np.ndarray]:
    rng = np.random.RandomState(0)
    return [rng.normal(0, 10, converter.INTERFACE[name]).astype(np.float32) for name in converter.OUTPUTS]


@pytest.mark.parametrize(
    ("case", "passes"),
    [("same", True), ("box_rounding", True), ("box_shift", False), ("score_shift", False), ("nan", False),
     ("shape", False)],
)
def test_output_agreement_keeps_a_sub_pixel_box_bound(case: str, passes: bool, converter: ModuleType) -> None:
    expected = _outputs(converter)
    actual = [value.copy() for value in expected]
    box = converter.OUTPUTS.index("bbox_pred_0")
    if case == "box_rounding":
        actual[box][0, 0, 0, 0] += 0.03
    elif case == "box_shift":
        actual[box][0, 0, 0, 0] += 0.2
    elif case == "score_shift":
        expected[0][0, 0, 0, 0] = 0.0
        actual[0][0, 0, 0, 0] = 0.01
    elif case == "nan":
        actual[0][0, 0, 0, 0] = expected[0][0, 0, 0, 0] = np.nan
    elif case == "shape":
        actual[0] = actual[0][:, :1]
    if passes:
        converter.compare_outputs(expected, actual)
    else:
        with pytest.raises((ValueError, AssertionError)):
            converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    if case == "order":
        values[1:] = values[:0:-1]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[0].type.tensor_type.shape.dim[2].dim_value = 320
    nodes = [
        helper.make_node(
            "ConstantOfShape",
            [f"{value.name}_shape"],
            [value.name],
            value=helper.make_tensor("zero", onnx.TensorProto.FLOAT, [1], [0.0]),
        )
        for value in values[1:]
    ]
    shapes = [
        onnx.numpy_helper.from_array(np.asarray(converter.INTERFACE[value.name], np.int64), f"{value.name}_shape")
        for value in values[1:]
    ]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:], shapes)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/rtmdet_ins_s")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="rtmdet-ins_s_8xb32-300e_coco"):
        converter.validate_source({"type": "file", "uri": "https://example.com/mmdetection/v3.0/x-12345678.pth"})


def test_the_stub_serves_every_compiled_op_module(converter: ModuleType) -> None:
    stub = converter._CompiledOpsStub()
    assert stub.find_spec("mmcv.ops", None) is not None
    assert stub.find_spec("mmcv.ops.roi_align", None) is not None
    assert stub.find_spec("mmcv.cnn", None) is None
    module = stub.create_module(stub.find_spec("mmcv.ops.nms", None))
    assert callable(module.batched_nms) and module.__path__ == []
