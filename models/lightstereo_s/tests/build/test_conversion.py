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


def test_source_must_be_the_pinned_scene_flow_checkpoint(converter: ModuleType) -> None:
    model = load_metadata(ROOT / "models/lightstereo_s")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="LightStereo-S-SceneFlow.ckpt"):
        converter.validate_source(
            {"uri": "hf://models/XiandaGuo/OpenStereo@main/checkpoint/LightStereo/LightStereo-S-KITTI.ckpt"}
        )


def test_converted_weights_are_openstereo_tensors(converted: Path) -> None:
    """The conversion stage's real output: OpenStereo's state dict, under the names surgery.py loads strictly."""
    torch = pytest.importorskip("torch")
    weights = torch.load(converted, weights_only=True)
    # Upstream's backbone slices timm's MobileNetV2 blocks, so its fourth block keeps the index 3 in its names.
    assert {"backbone.conv_stem.weight", "backbone.block3.3.0.conv_pw.weight", "refine_3.block.0.weight"} <= set(
        weights
    )
    # BatchNorm's num_batches_tracked counters are int64; every weight is float32.
    weights = {name: value for name, value in weights.items() if not name.endswith("num_batches_tracked")}
    assert all(value.dtype == torch.float32 for value in weights.values())
