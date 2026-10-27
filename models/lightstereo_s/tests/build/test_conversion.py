from __future__ import annotations

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


def _export(converter: ModuleType, case: str) -> onnx.ModelProto:
    """Two inputs summed into the disparity through a 1x1 Conv, or one fault validate_export must catch."""
    helper = onnx.helper
    shapes = dict(converter.INTERFACE)
    if case == "shape":
        shapes["right"] = [1, 3, 544, 1024]
    inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shapes[name]) for name in converter.INPUTS]
    if case == "order":
        inputs.reverse()
    output = helper.make_tensor_value_info("disparity", onnx.TensorProto.FLOAT, shapes["disparity"])
    weight = onnx.numpy_helper.from_array(np.ones((1, 3, 1, 1), np.float32), "weight")
    nodes = [
        helper.make_node("Add", ["left", "right"], ["sum"]),
        helper.make_node("Conv", ["sum", "weight"], ["disparity"]),
    ]
    if case == "shape_arithmetic":
        nodes.append(helper.make_node("Shape", ["sum"], ["unused"]))
    graph = helper.make_graph(nodes, "fixture", inputs, [output], [weight])
    opset = 13 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "shape", "shape_arithmetic"])
def test_export_must_keep_the_stereo_interface_without_index_arithmetic(
    tmp_path: Path, case: str, converter: ModuleType
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_export(converter, case), path)

    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_interface_is_two_padded_scene_flow_views_and_one_full_resolution_disparity(converter: ModuleType) -> None:
    # OpenStereo's Scene Flow evaluation pads to 544x960; the disparity comes back at the same resolution.
    assert converter.INTERFACE == {
        "left": [1, 3, 544, 960],
        "right": [1, 3, 544, 960],
        "disparity": [1, 1, 544, 960],
    }
    # MAX_DISP 192 at quarter resolution: 48 cost-volume bins.
    assert converter.BINS == 48


def test_disparity_comparison_rejects_shape_nan_and_excess_error(converter: ModuleType) -> None:
    expected = np.random.RandomState(0).uniform(0, 192, (1, 1, 544, 960)).astype(np.float32)

    assert converter.compare_disparity(expected, expected + 0.004) == pytest.approx(0.004, abs=1e-5)
    with pytest.raises(ValueError, match="differs from PyTorch by 0.02"):
        converter.compare_disparity(expected, expected + 0.02)
    with pytest.raises(ValueError, match="non-finite"):
        converter.compare_disparity(expected, np.full_like(expected, np.nan))
    with pytest.raises(ValueError, match="Unexpected"):
        converter.compare_disparity(expected, expected[:, :, :-1])


def test_synthetic_pair_shifts_the_right_view_by_a_seeded_disparity(converter: ModuleType) -> None:
    left, right = converter.synthetic_pair(123)

    shifts = [d for d in range(converter.MAX_DISP) if np.array_equal(np.roll(left, -d, axis=3), right)]
    assert left.shape == right.shape == (1, 3, 544, 960)
    # Exactly one shift in 1..191 maps the left view onto the right, and the same seed gives the same pair.
    assert len(shifts) == 1 and 1 <= shifts[0] < 192
    assert np.array_equal(converter.synthetic_pair(123)[1], right)


def test_source_must_be_the_pinned_scene_flow_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/lightstereo_s")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="LightStereo-S-SceneFlow.ckpt"):
        converter.validate_source(
            {
                "type": "file",
                "uri": "hf://models/XiandaGuo/OpenStereo@main/checkpoint/LightStereo/LightStereo-S-KITTI.ckpt",
            }
        )


def test_converted_onnx_keeps_the_interface_and_mla_operators(converted: Path, converter: ModuleType) -> None:
    converter.validate_export(converted)
