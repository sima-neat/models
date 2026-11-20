"""Conversion pins AIMET's FP32 checkpoint and exports its uncollapsed network; surgery.py does the collapse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import onnx
import pytest
from model_registry.model import load_model


def test_source_is_the_matching_aimet_checkpoint(model_dir: Path, load_script: Callable[[str], ModuleType]) -> None:
    converter = load_script("convert")
    source = load_model(model_dir)["conversion"]["default"]["source"]
    converter.validate_source(source)
    assert source["uri"].endswith(f"/{converter.CHECKPOINT}")
    with pytest.raises(ValueError, match="checkpoint"):
        converter.validate_source({**source, "uri": source["uri"].replace("m7", "m99").replace("m11", "m99")})


def test_sample_inputs_are_tiles_in_the_unit_range(load_script: Callable[[str], ModuleType]) -> None:
    converter = load_script("convert")
    samples = converter.sample_inputs()
    assert len(samples) == len(converter.SEEDS)
    for sample in samples:
        assert list(sample.shape) == converter.INPUT_SHAPE
        assert 0 <= sample.min() and sample.max() <= 1


def test_network_loads_the_checkpoint_layout(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("torch")
    sesr = load_script("sesr")
    names = list(sesr.build_network().state_dict())
    assert names[:2] == ["anchor.net.weight", "anchor.net.bias"]
    assert f"residual_block.{sesr.NUM_BLOCKS - 1}.conv_squeeze.bias" in names
    assert f"residual_block.{sesr.NUM_BLOCKS}.conv_squeeze.bias" not in names


def test_conversion_exports_the_uncollapsed_tile_network(
    converted: Path, load_script: Callable[[str], ModuleType]
) -> None:
    converter = load_script("convert")
    converter.validate_export(converted)
    model = onnx.load(converted)
    sesr = load_script("sesr")
    # One anchor, then an expand and a squeeze for the first block, every residual block and the last block.
    assert sum(node.op_type == "Conv" for node in model.graph.node) == 1 + 2 * (sesr.NUM_BLOCKS + 2)
    assert [node.op_type for node in model.graph.node][-1] == "DepthToSpace"
