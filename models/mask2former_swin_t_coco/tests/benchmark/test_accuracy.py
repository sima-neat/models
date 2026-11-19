from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_output_layout_reads_both_layouts(accuracy_script: ModuleType, layout: str) -> None:
    rng = np.random.default_rng(0)
    classes = rng.normal(size=(81, 100)).astype(np.float32)
    masks = rng.normal(size=(100, 96, 96)).astype(np.float32)
    if layout == "nchw":
        board = classes[None, :, None, :], masks[None]
    else:
        board = classes.T[None, None], masks.transpose(1, 2, 0)[None]

    read_classes, read_masks = accuracy_script.output_layout(*board)

    np.testing.assert_array_equal(read_classes, classes.T)
    np.testing.assert_array_equal(read_masks, masks)


def test_board_decoding_matches_the_pipeline(accuracy_script: ModuleType, load_script: Callable[[str], ModuleType]):
    pipeline = load_script("pipeline")
    rng = np.random.default_rng(1)
    classes = rng.normal(0, 3, (100, 81)).astype(np.float32)
    masks = rng.normal(0, 4, (100, 96, 96)).astype(np.float32)

    board = accuracy_script.instances(classes, masks, (300, 400))
    reference = pipeline.decode(classes.T[None, :, None, :], masks[None], (300, 400))

    assert [label for label, _, _ in board] == [label for label, _, _ in reference]
    np.testing.assert_allclose([s for _, s, _ in board], [s for _, s, _ in reference], rtol=1e-3, atol=1e-4)
    agreement = np.mean([np.mean(a == b) for (_, _, a), (_, _, b) in zip(board, reference)])
    assert agreement > 0.995


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
