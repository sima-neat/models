"""Accuracy result documents and their registry publication.

A result holds any task's named metrics, measured by a model's ``scripts/accuracy.py`` on a DevKit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .accuracy_config import accuracy_components, find_devkit_accuracy_config
from .errors import ValidationError
from .fetch import sha256_file
from .model import ACCURACY_SCRIPT, load_and_validate_model

JsonObject = dict[str, Any]
ACCURACY_RESULT_FILE = "accuracy-result.json"
ACCURACY_RESULT_SCHEMA = "schemas/accuracy-result.schema.json"
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def load_accuracy_result(path: str | Path, schema_path: str | Path) -> JsonObject:
    """Read an accuracy result and check it against its schema."""
    result_path = Path(path).expanduser().resolve()
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
        schema = json.loads(Path(schema_path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read accuracy result or schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "result"
        raise ValidationError(f"{result_path}: accuracy result {location}: {errors[0].message}")
    dataset = report["dataset"]
    if dataset["evaluated_samples"] > dataset["available_samples"]:
        raise ValidationError(f"{result_path}: accuracy evaluated_samples exceeds available_samples")
    return report


def compiled_artifacts(
    compiled_dir: str | Path, model_id: str, build_profile_id: str, component_ids: list[str]
) -> dict[str, str]:
    """SHA-256 of each component's compiled MPK, from the build's compile-outputs.json, checked against the file."""
    manifest_path = Path(compiled_dir).expanduser().resolve() / model_id / build_profile_id / "compile-outputs.json"
    try:
        outputs = json.loads(manifest_path.read_text(encoding="utf-8"))["outputs"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValidationError(f"could not read compile outputs {manifest_path}: {exc}") from exc
    artifacts = {}
    for component_id in component_ids:
        matches = [item for item in outputs if isinstance(item, dict) and item.get("component_id") == component_id]
        if len(matches) != 1:
            raise ValidationError(f"{manifest_path}: accuracy needs exactly one compiled {component_id} MPK")
        try:
            digest, _ = sha256_file(manifest_path.parent / matches[0]["mpk_path"])
        except (OSError, KeyError, TypeError) as exc:
            raise ValidationError(f"{manifest_path}: could not read the compiled {component_id} MPK: {exc}") from exc
        if digest != matches[0].get("mpk_sha256"):
            raise ValidationError(f"{manifest_path}: {component_id} MPK SHA-256 does not match")
        artifacts[component_id] = digest
    return artifacts


def build_accuracy_result(
    *,
    repository: str | Path,
    model_id: str,
    build_profile_id: str,
    script_output: str | Path,
    compiled_dir: str | Path,
    git_sha: str,
) -> JsonObject:
    """Wrap scripts/accuracy.py output with the model, accuracy entry, dataset and MPKs it measured."""
    root = Path(repository).expanduser().resolve()
    model_dir = root / "models" / model_id
    model = load_and_validate_model(model_dir, root / "schemas" / "model.schema.json")
    entry = find_devkit_accuracy_config(model, build_profile_id)
    if entry is None:
        raise ValidationError(f"{model_id} declares no devkit_script validation for {build_profile_id}")
    try:
        output = json.loads(Path(script_output).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read accuracy script output: {exc}") from exc
    metrics = output.get("metrics") if isinstance(output, dict) else None
    if not isinstance(metrics, dict):
        raise ValidationError("accuracy script output must contain a metrics object")
    undeclared = set(metrics) - set(entry["metrics"])
    if undeclared:
        raise ValidationError(f"accuracy script measured undeclared metrics: {', '.join(sorted(undeclared))}")
    missing = set(entry.get("required_metrics", ())) - set(metrics)
    if missing:
        raise ValidationError(f"accuracy script did not measure required metrics: {', '.join(sorted(missing))}")

    if not GIT_SHA.fullmatch(git_sha):
        raise ValidationError("git SHA must contain exactly 40 lowercase hexadecimal characters")
    artifacts = compiled_artifacts(compiled_dir, model_id, build_profile_id, accuracy_components(model, entry))
    scope = entry["scope"]
    dataset = output.get("dataset", {})
    return {
        "schema_version": 2,
        "model": {
            "id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": entry["result_component"],
            "package_id": scope["package"] if scope["type"] == "package" else None,
            "artifacts": artifacts,
        },
        "validation": {"id": entry["id"], "engine": entry["engine"], "status": entry["status"]},
        "dataset": {
            "id": entry["dataset"]["id"],
            "variant": entry["dataset"]["variant"],
            "split": dataset.get("split"),
            "sha256": entry["dataset"]["sha256"],
            "evaluated_samples": dataset.get("evaluated_samples"),
            "available_samples": dataset.get("available_samples"),
        },
        "metrics": _with_published_references(metrics, entry),
        "execution": {
            "core_version": output.get("core_version"),
            "script_sha256": hashlib.sha256((model_dir / ACCURACY_SCRIPT).read_bytes()).hexdigest(),
            "models_commit": git_sha,
        },
        "details": output.get("details", {}),
    }


def _with_published_references(metrics: JsonObject, entry: JsonObject) -> JsonObject:
    """Attach the entry's published score to each metric it names, as that metric's reference."""
    published = entry.get("reference", {}).get("metrics", {})
    return {
        name: {**metric, "reference": published[name]} if name in published and isinstance(metric, dict) else metric
        for name, metric in metrics.items()
    }


def check_accuracy_result(
    report: JsonObject, *, model: JsonObject, build_profile_id: str, artifacts: dict[str, str], models_commit: str
) -> None:
    """Check that a result measured the declared entry with these exact compiled MPKs."""
    entry = find_devkit_accuracy_config(model, build_profile_id)
    if entry is None:
        raise ValidationError(f"{model['id']} declares no devkit_script validation for {build_profile_id}")
    expected = {
        "model": model["id"],
        "build profile": build_profile_id,
        "result component": entry["result_component"],
        "validation": entry["id"],
        "dataset": entry["dataset"]["sha256"],
        "models commit": models_commit,
    }
    actual = {
        "model": report["model"]["id"],
        "build profile": report["model"]["build_profile_id"],
        "result component": report["model"]["component_id"],
        "validation": report["validation"]["id"],
        "dataset": report["dataset"]["sha256"],
        "models commit": report["execution"]["models_commit"],
    }
    for field, value in expected.items():
        if actual[field] != value:
            raise ValidationError(f"accuracy result {field} {actual[field]!r} does not match {value!r}")
    components = accuracy_components(model, entry)
    if report["model"]["artifacts"] != {component_id: artifacts.get(component_id) for component_id in components}:
        raise ValidationError("accuracy result did not measure the compiled MPKs")
    missing = set(entry.get("required_metrics", ())) - set(report["metrics"])
    if missing:
        raise ValidationError(f"required accuracy metrics were not measured: {', '.join(sorted(missing))}")


def accuracy_measurements(report: JsonObject, fingerprint: str) -> list[JsonObject]:
    dataset = report["dataset"]
    dimensions = {
        "model_id": report["model"]["id"],
        "build_profile_id": report["model"]["build_profile_id"],
        "component_id": report["model"]["component_id"],
        "dataset_id": dataset["id"],
        "dataset_variant": dataset["variant"],
        "dataset_split": dataset["split"],
        "validation_id": report["validation"]["id"],
        "validation_status": report["validation"]["status"],
    }
    return [
        {
            "measurement_key": f"{fingerprint}-{name}",
            "name": f"model.accuracy.{name}",
            "unit": metric["unit"],
            "value": metric["value"],
            "direction": metric["direction"],
            "sample_count": dataset["evaluated_samples"],
            "aggregation": "raw",
            "dimensions": dimensions,
        }
        for name, metric in sorted(report["metrics"].items())
    ]


def _artifact_metadata(report: JsonObject) -> JsonObject:
    dataset = report["dataset"]
    return {
        "model_id": report["model"]["id"],
        "build_profile_id": report["model"]["build_profile_id"],
        "component_id": report["model"]["component_id"],
        "dataset_id": dataset["id"],
        "dataset_variant": dataset["variant"],
        "dataset_split": dataset["split"],
        "evaluated_samples": dataset["evaluated_samples"],
        "available_samples": dataset["available_samples"],
        "validation_id": report["validation"]["id"],
        "validation_status": report["validation"]["status"],
    }


def publish_accuracy_result(
    client: Any,
    store: Any,
    *,
    bucket: str,
    run_id: str,
    path: Path,
    report: JsonObject,
    sha256: str,
    fingerprint: str,
    object_key: str,
    stable_key: str,
) -> JsonObject:
    """Attach an accuracy result to a mutable registry run as an artifact, a test result and measurements."""
    stored = store.ensure_native_sha256(path, object_key, sha256, media_type="application/json")
    artifact = client.request(
        "PUT",
        f"/v1/runs/{run_id}/artifacts/accuracy-result",
        {
            "artifact_type": "accuracy-result",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "application/json",
            "retention_class": "branch",
            "metadata": _artifact_metadata(report),
        },
        f"{stable_key}-accuracy-artifact",
    )
    test_result: JsonObject = {"status": "passed", "version": str(report["schema_version"]), "summary": report}
    if artifact.get("artifact_id"):
        test_result["report_artifact_id"] = artifact["artifact_id"]
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/test-results/accuracy/{report['validation']['id']}",
        test_result,
        f"{stable_key}-accuracy-test-result",
    )
    client.request(
        "POST",
        f"/v1/runs/{run_id}/measurements/batch",
        {"measurements": accuracy_measurements(report, fingerprint)},
        f"{stable_key}-accuracy-measurements",
    )
    return {
        "s3_key": stored.key,
        "result_fingerprint": fingerprint,
        "evaluated_samples": report["dataset"]["evaluated_samples"],
    }
