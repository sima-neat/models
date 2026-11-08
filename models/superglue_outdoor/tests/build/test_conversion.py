"""Checks of the conversion's source pin and the weights.pth it publishes."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/superglue_outdoor")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="model.safetensors"):
        converter.validate_source({"uri": "hf://models/magic-leap-community/superglue_indoor@main/model.bin"})


def test_converted_weights_are_superglue_tensors(converted: Path) -> None:
    """The conversion stage's real output: a plain tensor state dict holding SuperGlue's matcher weights."""
    torch = pytest.importorskip("torch")
    weights = torch.load(converted, weights_only=True)
    assert {"bin_score", "final_projection.final_proj.weight"} <= set(weights)
    assert all(value.dtype == torch.float32 for value in weights.values())
