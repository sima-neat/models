from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .errors import ValidationError


def load_performance_config(
    path: str | Path,
    schema_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    schema_file = Path(schema_path).expanduser().resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{config_path}: could not load performance configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{config_path}: performance configuration must be a mapping")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors
        )
        raise ValidationError(f"{config_path}: {details}")
    metric_names = {metric if isinstance(metric, str) else metric["name"] for metric in value["metrics"]}
    missing = set(value.get("required_metrics", ())) - metric_names
    if missing:
        raise ValidationError(
            f"{config_path}: required_metrics are not declared in metrics: {', '.join(sorted(missing))}"
        )
    return value
