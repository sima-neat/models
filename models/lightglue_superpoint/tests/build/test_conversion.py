"""Checks of the conversion's source pin, the architecture it accepts, and the weights.pth it publishes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_file_set(converter: ModuleType) -> None:
    converter.validate_source(load_metadata(MODEL_DIR)["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="file set"):
        converter.validate_source({"type": "file_set", "entrypoint": "model.safetensors",
                                   "files": [{"path": "model.safetensors"}]})


def test_config_must_declare_the_architecture_surgery_builds(
    tmp_path: Path, converter: ModuleType, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    (tmp_path / "config.json").write_text(json.dumps({**converter.ARCHITECTURE, "depth_confidence": 0.95}))

    converter.validate_config(tmp_path)
    # surgery.py writes the same architecture out, so it needs only the weights.
    assert {key: surgery.CONFIG[key] for key in converter.ARCHITECTURE if key != "model_type"} == {
        key: value for key, value in converter.ARCHITECTURE.items() if key != "model_type"}
    (tmp_path / "config.json").write_text(json.dumps({**converter.ARCHITECTURE, "num_attention_heads": 8}))
    with pytest.raises(ValueError, match="config.json declares"):
        converter.validate_config(tmp_path)


def test_converted_weights_are_lightglues_matcher_without_superpoint(converted: Path) -> None:
    """The conversion stage's real output: a plain float32 state dict of the matcher, leaving out SuperPoint."""
    torch = pytest.importorskip("torch")
    weights = torch.load(converted, weights_only=True)

    assert {"positional_encoder.projector.weight", "match_assignment_layers.8.matchability.weight"} <= set(weights)
    assert not [name for name in weights if name.startswith("keypoint_detector.")]
    assert all(value.dtype == torch.float32 for value in weights.values())
