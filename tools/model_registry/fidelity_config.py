from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .errors import ValidationError


def load_recipe_fidelity_config(path: str | Path, schema_path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    schema_file = Path(schema_path).expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{config_path}: could not load recipe fidelity configuration: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ValidationError(f"{config_path}: {details}")
    return config


def validate_recipe_fidelity_result(
    config: dict[str, Any],
    report: dict[str, Any],
    path: str | Path,
) -> None:
    report_path = Path(path)
    expected = {
        "component_id": config["result_component"],
        "status": config["status"],
        "dataset": {
            key: config["dataset"][key]
            for key in ("id", "variant", "sha256")
        },
        "selection": config["selection"],
        "thresholds": config["thresholds"],
        "samples": config["selection"]["samples"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValidationError(f"{report_path}: numerical fidelity {field} does not match its configuration")

    metrics = {metric["name"]: metric for metric in report["metrics"]}
    if set(metrics) != set(config["outputs"]):
        raise ValidationError(f"{report_path}: numerical fidelity metrics do not match configured outputs")
    minimum_cosine = config["thresholds"]["minimum_cosine_similarity"]
    maximum_nrmse = config["thresholds"]["maximum_normalized_rmse"]
    failures = [
        name
        for name, metric in metrics.items()
        if metric["cosine_similarity"] < minimum_cosine
        or metric["normalized_rmse"] > maximum_nrmse
    ]
    if failures:
        raise ValidationError(
            f"{report_path}: numerical fidelity thresholds failed for: {', '.join(sorted(failures))}"
        )


def find_recipe_fidelity_config(
    model_dir: str | Path,
    model: dict[str, Any],
    *,
    build_profile_id: str,
    component_id: str,
) -> tuple[Path, dict[str, Any]] | None:
    directory = Path(model_dir).expanduser().resolve()
    packages = {package["id"]: package for package in model["packages"]}
    matches = []
    for case in model["tests"]["correctness"]:
        if build_profile_id not in case["build_profiles"]:
            continue
        scope = case["scope"]
        applies = scope == {"type": "component", "component": component_id}
        if scope["type"] == "package":
            applies = component_id in packages[scope["package"]]["components"]
        if not applies:
            continue
        path = (directory / case["config"]).resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise ValidationError(f"{path}: could not inspect correctness YAML: {exc}") from exc
        if isinstance(raw, dict) and raw.get("engine") == "model_recipe":
            matches.append(path)
    if not matches:
        return None
    if len(matches) != 1:
        raise ValidationError(
            f"{directory}: expected exactly one model-recipe correctness test for "
            f"{build_profile_id}/{component_id}"
        )
    schema = directory.parents[1] / "schemas" / "recipe-fidelity-v1.schema.json"
    return matches[0], load_recipe_fidelity_config(matches[0], schema)
