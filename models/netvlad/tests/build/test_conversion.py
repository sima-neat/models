"""Checks of the NetVLAD export's interface, its checkpoint guard, and the reference it is compared with."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_reference_aggregates_soft_assigned_residuals_per_cluster(converter: ModuleType) -> None:
    # Two clusters over three-channel features with an identity WPCA, so the descriptor is the normalized VLAD.
    centroids = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assignment = np.zeros((2, 3, 1, 1))
    identity, no_bias = np.eye(6), np.zeros(6)
    # One cell holding 3 * e_0 normalizes to e_0; equal logits split it evenly, leaving residual e_0 - c_k per cluster.
    one_cell = np.asarray([3.0, 0.0, 0.0]).reshape(3, 1, 1)
    expected = np.asarray([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]]) / np.sqrt(2) / np.sqrt(2)

    descriptor = converter.reference_descriptor(one_cell, centroids, assignment, identity, no_bias)

    np.testing.assert_allclose(descriptor, expected.ravel())
    # A strong assignment to cluster 0 moves the cell's weight there; cluster 1's residual shrinks to zero.
    assignment[0, 0] = 100.0
    descriptor = converter.reference_descriptor(one_cell, centroids, assignment, identity, no_bias)
    np.testing.assert_allclose(descriptor, np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2), atol=1e-12)


def test_descriptor_comparison_rejects_shape_nan_and_drift(converter: ModuleType) -> None:
    expected = converter._l2_normalize(np.random.RandomState(1).normal(0, 1, converter.DESCRIPTOR_DIM), 0)
    actual = expected[None].astype(np.float32)

    assert converter.compare_descriptors(expected, actual) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unexpected or non-finite"):
        converter.compare_descriptors(expected, actual[:, :512])
    with pytest.raises(ValueError, match="Unexpected or non-finite"):
        converter.compare_descriptors(expected, np.where(np.arange(actual.size) == 7, np.nan, actual))
    drifted = actual.copy()
    drifted[0, 0] += 1e-3
    with pytest.raises(AssertionError, match="descriptor"):
        converter.compare_descriptors(expected, drifted)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape", "unfolded"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, converter: ModuleType) -> None:
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
        values[1].type.tensor_type.shape.dim[1].dim_value = 512
    zeros = onnx.numpy_helper.from_array(np.zeros((1, 4096), np.float32), "zeros")
    if case == "unfolded":
        node = helper.make_node("Constant", [], [values[1].name], value=zeros)
    else:
        node = helper.make_node("Identity", ["zeros"], [values[1].name])
    graph = helper.make_graph([node], "fixture", values[:1], values[1:], [zeros])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_checkpoint_must_hold_the_vladv1_pooling_and_4096_wpca(converter: ModuleType) -> None:
    # validate_state reads only shapes, so stand-ins avoid allocating the 4096x32768 WPCA.
    state = {name: SimpleNamespace(shape=shape) for name, shape in converter.STATE_SHAPES.items()}
    state["encoder.0.weight"] = SimpleNamespace(shape=(64, 3, 3, 3))
    converter.validate_state(state)

    state["pool.conv.bias"] = SimpleNamespace(shape=(64,))
    with pytest.raises(ValueError, match="pool.conv.bias"):
        converter.validate_state(state)
    del state["pool.conv.bias"]
    state["WPCA.0.weight"] = SimpleNamespace(shape=(512, 32768, 1, 1))
    with pytest.raises(ValueError, match="Unexpected NetVLAD checkpoint tensors"):
        converter.validate_state(state)


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType, model_dir: Path) -> None:
    converter.validate_source(load_metadata(model_dir)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="pitts_WPCA4096.pth.tar"):
        converter.validate_source(
            {"type": "file", "uri": "hf://models/TobiasRobotics/Patch-NetVLAD@main/mapillary_WPCA4096.pth.tar"}
        )


def test_converted_onnx_maps_one_image_to_one_descriptor(converted: Path, converter: ModuleType) -> None:
    converter.validate_export(converted)
