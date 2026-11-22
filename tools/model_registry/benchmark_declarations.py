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
# An optional host script that measures a reference_metrics entry with origin fp32_baseline; no stage runs it.
FP32_BASELINE_SCRIPT = "scripts/fp32_baseline_accuracy.py"
ACCURACY_REQUIREMENTS = "scripts/requirements-accuracy.txt"
PERFORMANCE_REQUIREMENTS = "scripts/requirements-performance.txt"
DATASET_REQUIREMENTS_DIR = "datasets"
DATASET_REQUIREMENTS_SCHEMA = "schemas/dataset-requirements.schema.json"


def benchmarked_profiles(model: dict[str, Any], stage: str) -> list[str]:
    """The build profiles the stage benchmarks, in model.yaml order."""
    declaration = model.get(stage)
    if declaration is None:
        return []
    return declaration.get("build_profiles", [profile["id"] for profile in model["build_profiles"]])


def benchmark_runner(model: dict[str, Any]) -> str:
    """Which Python measures the model on a DevKit: "mole" for a deployment that generates text, else "pyneat".

    A speech model is packaged as a deployment folder too, and PyNeat's speech server runs it.
    """
    deployment = model["components"][0]["interface"]["format"] == "huggingface"
    return "mole" if deployment and model["compositions"][0].get("entrypoint") == "generate" else "pyneat"


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


def dataset_names(repository: str | Path) -> dict[str, str]:
    """The name the model browser shows for each dataset, by its Dataset Registry ID."""
    root = Path(repository)
    return {
        path.stem: _dataset_document(root, f"{DATASET_REQUIREMENTS_DIR}/{path.name}")["name"]
        for path in sorted((root / DATASET_REQUIREMENTS_DIR).glob("*.yaml"))
    }


def _dataset_requirements(repository: Path, relative: str) -> dict[str, list[str]]:
    return _dataset_document(repository, relative)["required_metrics"]


def _dataset_document(repository: Path, relative: str) -> dict[str, Any]:
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
    return document


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
    datasets = {dataset["id"] for dataset in model.get("accuracy", {}).get("datasets", {}).values()}
    _validate_reference_metrics(model.get("reference_metrics", []), datasets, directory)


def _validate_reference_metrics(entries: list[dict[str, Any]], datasets: set[str], directory: Path) -> None:
    """Each reference score names an accuracy metric, in the metric's unit, at most once per origin and dataset.

    An FP32 baseline is measured on one of the model's accuracy datasets by the model's baseline script.
    """
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"{directory}: reference_metrics[{index}]"
        origin, dataset = entry.get("origin", "published"), entry["dataset"]["id"]
        if origin == "fp32_baseline":
            if dataset not in datasets:
                raise ValidationError(f"{label}: the FP32 baseline's dataset {dataset} is not an accuracy dataset")
            if not (directory / entry["script"]).is_file():
                raise ValidationError(f"{label}: needs {entry['script']}")
        for name, value in entry["metrics"].items():
            if METRICS.get(name, {}).get("stage") != "accuracy":
                raise ValidationError(f"{label}: {name!r} is not an accuracy metric")
            if error := accuracy_value_error(name, value):
                raise ValidationError(f"{label}: {error}")
            if (origin, dataset, name) in seen:
                raise ValidationError(f"{label}: {name} on {dataset} already has a {origin} score")
            seen.add((origin, dataset, name))


def reference_metrics_by_model(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each model's reference scores, keyed by model ID, for the model browser."""
    return {
        model_id: model["reference_metrics"]
        for model_id, model in sorted(models.items())
        if model.get("reference_metrics")
    }
