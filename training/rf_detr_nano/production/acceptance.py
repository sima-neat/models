"""Absolute AP budgets supplement, never weaken, normal registry regressions."""

from __future__ import annotations

import math

FP32_MAP = 0.48407727873279055
MAX_FP32_DROP_POINTS = 1.0
MAX_SOURCE_COMPILED_DROP_POINTS = 0.1


def accuracy_shortfalls(
    report: dict, *, source_map: float, expected_checkpoint_sha256: str, expected_reference_sha256: str
) -> list[str]:
    if not math.isfinite(source_map) or not 0 <= source_map <= 1:
        raise ValueError("source_map must be a measured COCO AP ratio, not AP points")
    dataset = report.get("datasets", {}).get("coco-2017-val", {})
    details = dataset.get("details", {})
    value = dataset.get("metrics", {}).get("coco_map")
    failures = []
    if (
        dataset.get("evaluated_samples") != 5000
        or dataset.get("available_samples") != 5000
        or details.get("official_full5000") is not True
    ):
        failures.append("accuracy requires all 5,000 official validation images")
    if details.get("checkpoint_sha256") != expected_checkpoint_sha256:
        failures.append("accuracy checkpoint does not match the source score")
    if details.get("reference_sha256") != expected_reference_sha256:
        failures.append("accuracy reference does not match the source score")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        failures.append("missing or invalid compiled COCO AP ratio")
        return failures
    # Budgets are points; registry metrics are ratios. They do not accumulate.
    if (FP32_MAP - value) * 100 > MAX_FP32_DROP_POINTS + 1e-12:
        failures.append("compiled AP is more than 1.0 point below original FP32")
    if (FP32_MAP - source_map) * 100 > MAX_FP32_DROP_POINTS + 1e-12:
        failures.append("source QAT AP itself is more than 1.0 point below original FP32")
    if (source_map - value) * 100 > MAX_SOURCE_COMPILED_DROP_POINTS + 1e-12:
        failures.append("compiled AP is more than 0.1 point below the same QAT source")
    return failures
