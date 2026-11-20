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

    assert script.parse_mpks(["model=/data/bot_reid_r50_mpk.tar.gz"]) == Path("/data/bot_reid_r50_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/bot_reid_r50_mpk.tar.gz"])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


def test_evaluation_follows_the_market_1501_protocol(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    # Query 0 is person 1 on camera 0; query 1 is person 2, absent from the gallery, and is skipped.
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_ids = np.array([[1, 0], [2, 1]])
    # The nearest gallery image is person 1 on the query's own camera, which the protocol ignores; next come a
    # different person, then person 1 on another camera.
    gallery = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
    gallery_ids = np.array([[1, 0], [3, 2], [1, 1], [4, 3]])

    metrics = script.evaluate(query, gallery, query_ids, gallery_ids)

    assert metrics["cmc_rank1"] == 0.0 and metrics["cmc_rank5"] == 1.0 and metrics["cmc_rank10"] == 1.0
    # One correct match, found second: average precision 1/2.
    assert metrics["reid_map"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no query identity"):
        script.evaluate(query[1:], gallery, query_ids[1:], gallery_ids)


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
