from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def test_mpk_argument_names_exactly_the_model_component(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")

    assert script.parse_mpks(["model=/data/resnet_50_mpk.tar.gz"]) == Path("/data/resnet_50_mpk.tar.gz")
    with pytest.raises(SystemExit, match="exactly model=<path>"):
        script.parse_mpks(["backbone=/data/resnet_50_mpk.tar.gz"])


def test_prefetched_images_keep_their_order(load_script: Callable[[str], ModuleType]) -> None:
    script = load_script("accuracy")
    items = list(range(script.PREFETCH * 3 + 1))

    assert list(script.prefetched(lambda item: item * 2, items)) == [item * 2 for item in items]


# Lowest accepted value of each required metric per build profile: the build's latest result in the
# develop catalog at b64f221, less 0.01, rounded down.
MINIMUM_ACCURACY = {
    "modalix_int8": {"top1_accuracy": 0.7664},
}


def test_accuracy_meets_its_minimum(accuracy: dict[str, Any], build_profile: str) -> None:
    assert build_profile in MINIMUM_ACCURACY, f"no accuracy minimum for {build_profile}"
    for name, minimum in MINIMUM_ACCURACY[build_profile].items():
        value = accuracy["metrics"][name]["value"]
        assert value >= minimum, f"{name} {value:.4f} is below its minimum {minimum}"
