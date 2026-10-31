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
    """The side-by-side frame sliced into its halves and summed into the disparity through a 1x1 Conv, with the
    padded rows removed, or one fault validate_export must catch."""
    helper = onnx.helper
    shapes = dict(converter.INTERFACE)
    if case == "shape":
        shapes["stereo"] = [1, 3, 544, 1920]
    inputs = [helper.make_tensor_value_info("stereo", onnx.TensorProto.FLOAT, shapes["stereo"])]
    if case == "two_inputs":
        inputs.append(helper.make_tensor_value_info("right", onnx.TensorProto.FLOAT, [1, 3, 540, 960]))
    output = helper.make_tensor_value_info("disparity", onnx.TensorProto.FLOAT, shapes["disparity"])
    constants = {
        "weight": np.ones((1, 3, 1, 1), np.float32),
        "width": np.array([3]),
        "zero": np.array([0]),
        "half": np.array([960]),
        "end": np.array([1920]),
    }
    if case == "constants":
        constants["table"] = np.zeros(1000, np.float32)
    initializers = [onnx.numpy_helper.from_array(value, name) for name, value in constants.items()]
    nodes = [
        helper.make_node("Slice", ["stereo", "zero", "half", "width"], ["left"]),
        helper.make_node("Slice", ["stereo", "half", "end", "width"], ["right_view"]),
        helper.make_node("Add", ["left", "right_view"], ["sum"]),
        helper.make_node("Conv", ["sum", "weight"], ["disparity"]),
    ]
    if case == "edge_pad":
        nodes.insert(0, helper.make_node("Pad", ["stereo", "pads"], ["padded"], mode="edge"))
        initializers.append(onnx.numpy_helper.from_array(np.array([0, 0, 2, 0, 0, 0, 2, 0]), "pads"))
    if case == "grid_sample":
        nodes.append(helper.make_node("GridSample", ["sum", "grid"], ["sampled"]))
        initializers.append(onnx.numpy_helper.from_array(np.zeros((1, 1, 1, 2), np.float32), "grid"))
    graph = helper.make_graph(nodes, "fixture", inputs, [output], initializers)
    opset = 13 if case == "opset" else converter.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("valid", None),
        ("opset", "requires opset"),
        ("two_inputs", "Expected inputs and outputs"),
        ("shape", "Unexpected dtype or shape"),
        ("edge_pad", r"still holds \['Pad'\]"),
        ("grid_sample", r"still holds \['GridSample'\]"),
        ("constants", "constant values"),
    ],
)
def test_export_must_keep_the_stereo_interface_and_mla_operators(
    tmp_path: Path, case: str, error: str | None, converter: ModuleType
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_export(converter, case), path)

    if error is None:
        converter.validate_export(path, max_constants=1000)
    else:
        with pytest.raises(ValueError, match=error):
            converter.validate_export(path, max_constants=1000)


def test_interface_is_the_native_scene_flow_pair_and_the_left_views_disparity(converter: ModuleType) -> None:
    # Scene Flow views are 540x960; the graph pads them to 544 rows itself, so Neat's resize leaves a pair as it is.
    assert converter.INTERFACE == {"stereo": [1, 3, 540, 1920], "disparity": [1, 1, 540, 960]}
    assert converter.PADDED_HEIGHT % 32 == 0 and converter.PADDED_HEIGHT - converter.HEIGHT == 2 * converter.PAD
    # Upstream's realtime evaluation: 7 iterations over a 4-level correlation pyramid of radius 4.
    assert (converter.ITERATIONS, converter.LEVELS, converter.RADIUS) == (7, 4, 4)


def test_normalize_joins_the_views_and_applies_the_imagenet_mean_and_std(converter: ModuleType) -> None:
    left = np.zeros((540, 960, 3), np.float32)
    right = np.full((540, 960, 3), 255, np.float32)

    frame = converter.normalize(left, right)

    assert frame.shape == (1, 3, 540, 1920) and frame.dtype == np.float32
    np.testing.assert_allclose(frame[0, :, 0, 0], -converter.MEAN / converter.STD, rtol=1e-6)
    np.testing.assert_allclose(frame[0, :, 0, 960], (1 - converter.MEAN) / converter.STD, rtol=1e-6)


def test_disparity_comparison_rejects_shape_nan_and_excess_error(converter: ModuleType) -> None:
    expected = np.random.RandomState(0).uniform(0, 192, (1, 1, 540, 960)).astype(np.float32)

    assert converter.compare_disparity(expected, expected + 0.004) == pytest.approx(0.004, abs=1e-4)
    with pytest.raises(ValueError, match="differs from PyTorch by 0.02"):
        converter.compare_disparity(expected, expected + 0.02)
    with pytest.raises(ValueError, match="non-finite"):
        converter.compare_disparity(expected, np.full_like(expected, np.nan))
    with pytest.raises(ValueError, match="Unexpected"):
        converter.compare_disparity(expected, expected[:, :, :-1])


def test_synthetic_pair_shifts_the_right_view_by_a_disparity_growing_down_the_image(converter: ModuleType) -> None:
    pytest.importorskip("torch")
    left, right = converter.stereo_pair(123)

    assert left.shape == right.shape == (540, 960, 3)
    # Row r of the right view samples the left view's texture d = 8 + 40 r / 540 columns further right.
    for row, disparity in ((0, 8), (270, 28)):
        np.testing.assert_allclose(right[row, :100], left[row, disparity : disparity + 100], atol=1e-3)
    assert np.array_equal(converter.stereo_pair(123)[1], right)


def test_source_must_be_the_pinned_realtime_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/raft_stereo_realtime")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="raftstereo-realtime.pth"):
        converter.validate_source(
            {
                "type": "file",
                "uri": "hf://models/shriarul5273/RAFT-Stereo@97526bde05a11452d25cce032ccb1ad030bb775e/raftstereo-sceneflow.pth",
            }
        )


def test_converted_onnx_keeps_the_interface_and_mla_operators(converted: Path, converter: ModuleType) -> None:
    # The network's 9.9 M weights, the lookup's offsets and the folded batch norms' small constants.
    converter.validate_export(converted, max_constants=12_000_000)
