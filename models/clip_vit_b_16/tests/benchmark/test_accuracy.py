from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.array.copy() if copy else self.array


def test_mpk_arguments_name_exactly_both_encoders(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    mpks = script.parse_mpks(["image_encoder=/data/image_mpk.tar.gz", "text_encoder=/data/text_mpk.tar.gz"])
    assert mpks == {"image_encoder": Path("/data/image_mpk.tar.gz"), "text_encoder": Path("/data/text_mpk.tar.gz")}
    with pytest.raises(SystemExit, match="exactly image_encoder=<path> and text_encoder=<path>"):
        script.parse_mpks(["model=/data/clip_mpk.tar.gz"])


def test_class_logits_are_told_apart_from_the_embedding_by_size(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    logits = np.arange(1000, dtype=np.float32)

    assert script.class_logits([_Tensor(np.zeros((1, 512), np.float32)), _Tensor(logits[None])]) is not None
    np.testing.assert_array_equal(script.class_logits([_Tensor(logits[None]), _Tensor(np.zeros((1, 512)))]), logits)
    with pytest.raises(ValueError, match="output sizes"):
        script.class_logits([_Tensor(logits[None])])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
