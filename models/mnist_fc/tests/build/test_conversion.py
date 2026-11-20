"""Checks of convert.py's 4D export of the MLP, and of the converted ONNX when the conversion stage supplies it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def convert(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def _weights(convert: ModuleType) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    shapes = {
        "input_layer": (convert.HIDDEN_SIZE, convert.INPUT_SIZE),
        "mid_layer": (convert.HIDDEN_SIZE, convert.HIDDEN_SIZE),
        "output_layer": (convert.CLASSES, convert.HIDDEN_SIZE),
    }
    weights = {}
    for name, shape in shapes.items():
        weights[f"{name}.weight"] = rng.normal(0, 0.05, shape).astype(np.float32)
        weights[f"{name}.bias"] = rng.normal(0, 0.05, shape[0]).astype(np.float32)
    return weights


def test_convolutions_equal_the_normalized_dense_layers(convert: ModuleType) -> None:
    weights = _weights(convert)
    images = np.random.default_rng(2).random((4, 28, 28), dtype=np.float32)
    layers = convert.convolutions(weights)
    assert [weight.shape for weight, _ in layers] == [(256, 1, 28, 28), (256, 256, 1, 1), (10, 256, 1, 1)]

    # A 28x28 convolution over a 28x28 image is a dot product over its pixels; a 1x1 convolution is a matrix product.
    hidden = np.einsum("nhw,ochw->no", images, layers[0][0]) + layers[0][1]
    for index, (weight, bias) in enumerate(layers[1:], start=1):
        hidden = np.maximum(hidden, 0) @ weight[:, :, 0, 0].T + bias
    np.testing.assert_allclose(hidden, convert.reference_logits(weights, images), rtol=1e-4, atol=1e-4)


def test_graph_is_three_convolutions_with_relus_and_a_4d_interface(
    convert: ModuleType, tmp_path: Path
) -> None:
    import onnx

    model = convert.build_graph(_weights(convert))
    assert [node.op_type for node in model.graph.node] == ["Conv", "Relu", "Conv", "Relu", "Conv"]
    onnx.save(model, str(tmp_path / "model.onnx"))
    convert.validate_export(tmp_path / "model.onnx")


def test_source_and_config_must_be_the_pinned_mlp(convert: ModuleType) -> None:
    files = [{"path": "config.json"}, {"path": "model.safetensors"}]
    convert.validate_source({"type": "file_set", "entrypoint": "config.json", "files": files})
    with pytest.raises(ValueError, match="must pin exactly"):
        convert.validate_source({"type": "file_set", "entrypoint": "config.json", "files": files[1:]})
    convert.check_config({"architectures": ["MLP"], "input_size": 784, "hidden_size": 256, "output_size": 10})
    with pytest.raises(ValueError, match="unexpected mnist-mlp config"):
        convert.check_config({"architectures": ["MLP"], "input_size": 784, "hidden_size": 512, "output_size": 10})


def test_converted_model_has_the_declared_interface(converted: Path, convert: ModuleType) -> None:
    convert.validate_export(converted)
