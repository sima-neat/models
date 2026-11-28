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


def _interface(converter, case):
    helper = onnx.helper
    inputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.INPUT_SHAPES[c].items()
    ]
    outputs = [
        helper.make_tensor_value_info(converter.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in converter.COMPONENTS
        for n, s in converter.OUTPUT_SHAPES[c].items()
    ]
    if case == "dtype":
        inputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[0].type.tensor_type.shape.dim[3].dim_value = 512
    elif case == "missing":
        outputs.pop()
    nodes = []
    for value in outputs:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor = helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape))))
        nodes.append(helper.make_node("Constant", [], [value.name], value=tensor))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    opset = converter.OPSET - 1 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "dtype", "shape", "missing"])
def test_side_by_side_onnx_interface(tmp_path, case, converter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "drift"])
def test_onnx_outputs_must_match_pytorch(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 4, (1, 81, 1, 100)).astype(np.float32)
    actual = expected + 1e-5
    error, match = ValueError, "Unexpected or non-finite"
    if case == "shape":
        actual = actual[:, :80]
    elif case == "nan":
        actual[0, 0, 0, 0] = np.nan
    elif case == "drift":
        actual = expected + 0.01
        error, match = AssertionError, "classes"
    else:
        converter.compare(expected, actual, "classes")
        return
    with pytest.raises(error, match=match):
        converter.compare(expected, actual, "classes")


@pytest.mark.parametrize("case", ["valid", "flipped_queries", "sign_flips", "drift"])
def test_mask_logits_tolerate_only_rare_mask_flips(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 8, (1, 100, 96, 96)).astype(np.float32)
    actual = expected + 1e-5
    if case == "flipped_queries":
        # Mask decisions at float ties moved a few queries' logits, away from 0.
        actual[0, :8] += 0.2 * np.sign(actual[0, :8])
    elif case == "sign_flips":
        actual[0, :20] = -actual[0, :20]
    elif case == "drift":
        actual = expected * 1.01
    if case in ("valid", "flipped_queries"):
        converter.compare_masks(expected, actual, "masks")
    else:
        with pytest.raises(ValueError, match="agreement"):
            converter.compare_masks(expected, actual, "masks")


@pytest.mark.parametrize("case", ["valid", "one_query", "wrong_classes"])
def test_class_logits_must_pick_the_same_classes(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 4, (1, 100, 81)).astype(np.float32)
    actual = expected + 1e-5
    if case == "one_query":
        actual[0, 3] = actual[0, 3, ::-1]
    elif case == "wrong_classes":
        actual = actual[:, :, ::-1].copy()
    if case in ("valid", "one_query"):
        converter.compare_decoder(expected, actual, "classes")
    else:
        with pytest.raises(ValueError, match="agreement"):
            converter.compare_decoder(expected, actual, "classes")


def test_source_must_be_the_pinned_file_set(converter: ModuleType):
    model = load_metadata(ROOT / "models/mask2former_swin_t_coco")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="config.json, model.safetensors"):
        converter.validate_source(
            {
                "type": "file_set",
                "entrypoint": "config.json",
                "files": [{"path": "config.json"}, {"path": "pytorch_model.bin"}],
            }
        )


def test_tiled_sampling_equals_one_grid_sample(load_script: Callable[[str], ModuleType]):
    torch = pytest.importorskip("torch")
    blocks = load_script("mask2former_4d")
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(1, 8, 48, 48, generator=generator)
    # Points over the map and a little beyond it, and points on the tiles' shared pixels 23 and 24.
    x = torch.rand(1, 500, 4, 1, generator=generator) * 2.1 - 1.05
    y = torch.rand(1, 500, 4, 1, generator=generator) * 2.1 - 1.05
    x[0, :8, :, 0] = (torch.tensor([22.9, 23.0, 23.5, 24.0])[None] * 2 + 1) / 48 - 1

    tiled = blocks.sample(value, x, y)
    whole = torch.nn.functional.grid_sample(value, torch.cat([x, y], 3), mode="bilinear", align_corners=False)

    assert tiled.shape == whole.shape
    torch.testing.assert_close(tiled, whole, rtol=1e-5, atol=1e-5)
