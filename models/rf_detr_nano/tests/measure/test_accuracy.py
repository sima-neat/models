from __future__ import annotations

from typing import Any

# Lowest accepted value of each required metric per build profile: the build's latest result in the
# develop catalog at b64f221, less 0.01, rounded down.
MINIMUM_ACCURACY = {
    "modalix_int8_bf16": {"coco_map": 0.4086},
}


def test_accuracy_meets_its_minimum(accuracy: dict[str, Any], build_profile: str) -> None:
    assert build_profile in MINIMUM_ACCURACY, f"no accuracy minimum for {build_profile}"
    for name, minimum in MINIMUM_ACCURACY[build_profile].items():
        value = accuracy["metrics"][name]["value"]
        assert value >= minimum, f"{name} {value:.4f} is below its minimum {minimum}"
