"""Published metrics: the scores a source model's authors published, which the browser shows beside measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError

ACCURACY_METRICS_RELATIVE_PATH = Path("taxonomies/accuracy-metrics.yaml")


def load_accuracy_metrics(path: str | Path) -> dict[str, dict[str, Any]]:
    """The accuracy metric vocabulary: each metric name with its display name and unit."""
    metrics_path = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{metrics_path}: could not load accuracy metrics: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"{metrics_path}: accuracy metrics must be a non-empty mapping")
    for name, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"display_name", "unit"} or entry["unit"] != "ratio":
            raise ValidationError(f"{metrics_path}: accuracy metric {name!r} needs a display_name and unit: ratio")
    return value


def validate_published_metrics(entries: list[dict[str, Any]], vocabulary: dict[str, dict[str, Any]]) -> None:
    """Check that each published score names a known metric, in its unit, at most once per dataset."""
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"published_metrics[{index}]"
        dataset_id = entry["dataset"]["id"]
        for name, value in entry["metrics"].items():
            if name not in vocabulary:
                raise ValidationError(
                    f"{label}: {name!r} is not an accuracy metric in {ACCURACY_METRICS_RELATIVE_PATH}"
                )
            if not 0 <= value <= 1:
                raise ValidationError(f"{label}: {name} must be a ratio from 0 to 1; found {value}")
            if (dataset_id, name) in seen:
                raise ValidationError(f"{label}: {name} on {dataset_id} is already published by an earlier entry")
            seen.add((dataset_id, name))


def published_metrics_by_model(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each model's published scores, keyed by model ID, for the model browser."""
    return {
        model_id: model["published_metrics"]
        for model_id, model in sorted(models.items())
        if model.get("published_metrics")
    }
