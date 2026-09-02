from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .errors import ValidationError

DEFAULT_BENCHMARK = {
    "samples": 1000,
    "trials": 3,
    "include_plugin_latency": False,
}


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
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValidationError(f"{config_path}: {details}")
    return value


def benchmark_settings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    configured = config.get("benchmark")
    if configured is None:
        benchmark = dict(DEFAULT_BENCHMARK)
    else:
        benchmark = {
            "samples": configured["samples"],
            "trials": configured.get("trials", DEFAULT_BENCHMARK["trials"]),
            "include_plugin_latency": configured.get("include_plugin_latency", False),
            "mode": configured["mode"],
            "warmup_samples": configured["warmup_samples"],
        }
        if "original_width" in configured:
            benchmark.update(
                {
                    "original_width": configured["original_width"],
                    "original_height": configured["original_height"],
                }
            )
    return benchmark, dict(config.get("model_options", {}))
