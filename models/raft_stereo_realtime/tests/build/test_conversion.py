"""Checks of the conversion's source pin and the weights.pth it publishes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


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


def test_unwrap_drops_the_data_parallel_prefix_from_every_tensor(converter: ModuleType) -> None:
    assert converter.unwrap({"module.cnet.conv1.weight": 1, "module.conv2.1.bias": 2}) == {
        "cnet.conv1.weight": 1,
        "conv2.1.bias": 2,
    }
    # A checkpoint saved without the wrapper, or an empty one, is not the pinned realtime checkpoint.
    with pytest.raises(ValueError, match="under 'module.'"):
        converter.unwrap({"module.cnet.conv1.weight": 1, "conv2.1.bias": 2})
    with pytest.raises(ValueError, match="under 'module.'"):
        converter.unwrap({})


def test_converted_weights_are_the_realtime_networks_tensors(converted: Path) -> None:
    """The conversion stage's real output: the checkpoint's state dict under the names surgery.py loads strictly."""
    torch = pytest.importorskip("torch")
    weights = torch.load(converted, weights_only=True)
    assert not any(name.startswith("module.") for name in weights)
    # The shared backbone and its context heads, the two GRU levels the realtime model updates, the convex
    # upsampling mask and the correlation features' instance-normalized block.
    assert {
        "cnet.conv1.weight",
        "cnet.outputs08.0.0.conv1.weight",
        "cnet.outputs16.1.1.weight",
        "update_block.gru08.convz.weight",
        "update_block.gru16.convq.bias",
        "update_block.mask.2.weight",
        "context_zqr_convs.1.weight",
        "conv2.0.conv1.weight",
        "conv2.1.weight",
    } <= set(weights)
    # BatchNorm's num_batches_tracked counters are int64; every weight is float32.
    tensors = {name: value for name, value in weights.items() if not name.endswith("num_batches_tracked")}
    assert all(value.dtype == torch.float32 for value in tensors.values())
    # 9.87 M weights, without the counters.
    assert sum(value.numel() for value in tensors.values()) == 9_874_976
