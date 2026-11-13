"""Checks of the conversion's source pin, its checkpoint guard, and the weights.pth it publishes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from model_registry.model import load_model as load_metadata


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


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


def test_converted_weights_are_the_float32_netvlad_tensors(converted: Path, converter: ModuleType) -> None:
    """The conversion stage's real output: VGG16's encoder and the NetVLAD and WPCA tensors, all float32."""
    torch = pytest.importorskip("torch")
    weights = torch.load(converted, weights_only=True)
    converter.validate_state(weights)
    assert all(value.dtype == torch.float32 for value in weights.values())
