from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/model_mpk.tar.gz"]) == Path("/data/model_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/model_mpk.tar.gz"])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_knn_scores_each_image_by_its_nearest_other_images(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("accuracy")
    monkeypatch.setattr(script, "NEIGHBOURS", 2)
    monkeypatch.setattr(script, "BLOCK", 3)
    # Three images near each of two directions, and one image of class 1 whose nearest images are of class 0.
    embeddings = np.asarray(
        [[1, 0], [1, 0.02], [1, -0.02], [0, 1], [0.02, 1], [-0.02, 1], [1, 0.3]], dtype=np.float32
    ) * np.asarray([[1], [5], [0.2], [1], [3], [1], [2]], dtype=np.float32)
    classes = np.asarray([0, 0, 0, 1, 1, 1, 1])

    # Vector lengths do not matter, and the misplaced image is the only wrong one.
    assert script.knn_top1(embeddings, classes) == pytest.approx(6 / 7)


def test_knn_never_counts_an_image_as_its_own_neighbour(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script("accuracy")
    monkeypatch.setattr(script, "NEIGHBOURS", 1)
    # Each image's only other neighbours belong to the other class.
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)

    assert script.knn_top1(embeddings, np.asarray([0, 1, 0])) == 0.0
    with pytest.raises(ValueError, match="more than 1 images"):
        script.knn_top1(embeddings[:1], np.asarray([0]))


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
