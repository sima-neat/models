"""A model's benchmark declarations: what scripts/performance.py and scripts/accuracy.py must report, and for
which build profiles, plus the reference scores its authors published.

datasets/<id>.yaml says which accuracy metrics every model must report on that dataset, by the model's task, so
model.yaml names only the datasets accuracy evaluates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from model_report import METRICS, accuracy_value_error

from .errors import ValidationError

BENCHMARK_STAGES = ("performance", "accuracy")
SCRIPTS = {"performance": "scripts/performance.py", "accuracy": "scripts/accuracy.py"}
ACCURACY_REQUIREMENTS = "scripts/requirements-accuracy.txt"
DATASET_REQUIREMENTS_DIR = "datasets"
DATASET_REQUIREMENTS_SCHEMA = "schemas/dataset-requirements.schema.json"


def benchmarked_profiles(model: dict[str, Any], stage: str) -> list[str]:
    """The build profiles the stage benchmarks, in model.yaml order."""
    declaration = model.get(stage)
    if declaration is None:
        return []
    return declaration.get("build_profiles", [profile["id"] for profile in model["build_profiles"]])


def resolve_accuracy_requirements(model: dict[str, Any], repository: Path, directory: Path) -> None:
    """Give each accuracy dataset the metrics its datasets/<id>.yaml requires for any of the model's tasks."""
    for name, dataset in model.get("accuracy", {}).get("datasets", {}).items():
        relative = f"{DATASET_REQUIREMENTS_DIR}/{dataset['id']}.yaml"
        if not (repository / relative).is_file():
            raise ValidationError(f"{directory}: accuracy dataset {name!r} needs {relative}")
        by_task = _dataset_requirements(repository, relative)
        required = [metric for task in model["tasks"] for metric in by_task.get(task, [])]
        if not required:
            raise ValidationError(
                f"{directory}: {relative} requires no metrics for the tasks {', '.join(model['tasks'])}"
            )
        dataset["required_metrics"] = list(dict.fromkeys(required))


def _dataset_requirements(repository: Path, relative: str) -> dict[str, list[str]]:
    path = repository / relative
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads((repository / DATASET_REQUIREMENTS_SCHEMA).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: could not load the dataset requirements: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda error: list(error.absolute_path))
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or relative
        raise ValidationError(f"{path}: {location}: {errors[0].message}")
    unknown = sorted(
        {name for names in document["required_metrics"].values() for name in names
         if METRICS.get(name, {}).get("stage") != "accuracy"}
    )
    if unknown:
        raise ValidationError(f"{path}: requires metrics model_report does not define: {', '.join(unknown)}")
    return document["required_metrics"]


def validate_benchmark_declarations(model: dict[str, Any], directory: Path) -> None:
    profiles = [profile["id"] for profile in model["build_profiles"]]
    for stage in BENCHMARK_STAGES:
        declaration = model.get(stage)
        if declaration is None:
            continue
        if not (directory / SCRIPTS[stage]).is_file():
            raise ValidationError(f"{directory}: {stage} needs {SCRIPTS[stage]}")
        unknown_profiles = set(declaration.get("build_profiles", ())) - set(profiles)
        if unknown_profiles:
            names = ", ".join(sorted(unknown_profiles))
            raise ValidationError(f"{directory}: {stage} names unknown build profiles: {names}")
    _validate_reference_metrics(model.get("reference_metrics", []), directory)


def _validate_reference_metrics(entries: list[dict[str, Any]], directory: Path) -> None:
    """Each published score names an accuracy metric, in the metric's unit, at most once per dataset."""
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"{directory}: reference_metrics[{index}]"
        for name, value in entry["metrics"].items():
            if METRICS.get(name, {}).get("stage") != "accuracy":
                raise ValidationError(f"{label}: {name!r} is not an accuracy metric")
            if error := accuracy_value_error(name, value):
                raise ValidationError(f"{label}: {error}")
            if (entry["dataset"]["id"], name) in seen:
                raise ValidationError(f"{label}: {name} on {entry['dataset']['id']} is already published")
            seen.add((entry["dataset"]["id"], name))


def reference_metrics_by_model(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each model's published scores, keyed by model ID, for the model browser."""
    return {
        model_id: model["reference_metrics"]
        for model_id, model in sorted(models.items())
        if model.get("reference_metrics")
    }
