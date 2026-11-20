"""Checks of the CPU conditioning in dit.py and of convert.py's source and export checks."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture
def dit(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("dit")


def test_timestep_features_are_diffusers_flipped_sinusoids(dit: ModuleType) -> None:
    features = dit.timestep_features(500.0)
    assert features.shape == (dit.GUIDANCE_BATCH, 256, 1, 1)
    # diffusers' Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1): cos then sin of t * 10000^(-i/127).
    frequency = math.exp(-math.log(10000) * 5 / 127)
    assert features[0, 5, 0, 0] == pytest.approx(math.cos(500 * frequency), abs=1e-5)
    assert features[1, 128 + 5, 0, 0] == pytest.approx(math.sin(500 * frequency), abs=1e-5)


def test_class_one_hot_puts_the_class_then_the_null_class(dit: ModuleType) -> None:
    one_hot = dit.class_one_hot(207)
    assert one_hot.shape == (2, 1001, 1, 1) and one_hot.sum() == 2
    assert one_hot[0, 207, 0, 0] == 1 and one_hot[1, 1000, 0, 0] == 1
    with pytest.raises(ValueError):
        dit.class_one_hot(1000)


def test_guidance_moves_from_the_null_prediction_toward_the_class(dit: ModuleType) -> None:
    prediction = np.zeros((2, 8, 32, 32), np.float32)
    prediction[0, :4], prediction[1, :4], prediction[:, 4:] = 1.0, 0.5, 9.0
    np.testing.assert_allclose(dit.guided_noise(prediction, 4.0), np.full((1, 4, 32, 32), 2.5))


def test_transformer_parts_cover_the_blocks_once(dit: ModuleType) -> None:
    blocks = [index for part in range(dit.TRANSFORMER_PARTS) for index in dit.part_blocks(part)]
    assert blocks == list(range(dit.BLOCKS))
    # Each part takes the conditioning; the first the latent, the others the previous part's hidden grid.
    assert list(dit.INPUT_SHAPES["transformer_0"]) == ["latent", "timestep_features", "class_one_hot"]
    for previous, part in zip(dit.TRANSFORMER, dit.TRANSFORMER[1:]):
        assert dit.INPUT_SHAPES[part]["previous_hidden"] == dit.OUTPUT_SHAPES[previous]["hidden"] == (1, 1152, 16, 16)


def test_source_must_be_the_pinned_pipeline_files(load_script: Callable[[str], ModuleType]) -> None:
    convert = load_script("convert")
    source = {"type": "file_set", "entrypoint": "model_index.json", "files": [{"path": p} for p in convert.FILES]}
    convert.validate_source(source)
    with pytest.raises(ValueError):
        convert.validate_source({**source, "files": source["files"][:-1]})


def test_converted_model_has_the_declared_interface(converted: Path, load_script: Callable[[str], ModuleType]) -> None:
    load_script("convert").validate_export(converted)
