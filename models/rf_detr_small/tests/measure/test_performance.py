from __future__ import annotations

from typing import Any

# Lowest accepted throughput per build profile and component: 90% of the build's latest result in the
# develop catalog at b64f221, rounded down.
MINIMUM_THROUGHPUT_FPS = {
    ("modalix_int8_bf16", "backbone"): 127.9,
    ("modalix_int8_bf16", "transformer"): 478.6,
}


def test_throughput_meets_its_minimum(performance: dict[str, Any], build_profile: str) -> None:
    key = (build_profile, performance["model"]["component_id"])
    assert key in MINIMUM_THROUGHPUT_FPS, f"no throughput minimum for {key}"
    fps = performance["metrics"]["throughput_fps"]["value"]
    assert fps >= MINIMUM_THROUGHPUT_FPS[key], f"{fps:.1f} fps is below its minimum {MINIMUM_THROUGHPUT_FPS[key]}"
