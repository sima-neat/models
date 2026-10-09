"""Accuracy result documents and their registry publication.

Schema 1 is the compile-time image classification report from ``model_registry evaluate``.
Schema 2 holds any task's named metrics, measured by a model's ``scripts/accuracy.py`` on a DevKit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .accuracy_config import find_devkit_accuracy_config, validation_components
from .errors import ValidationError
from .model import ACCURACY_SCRIPT, load_and_validate_model

JsonObject = dict[str, Any]
ACCURACY_RESULT_FILE = "accuracy-result-v2.json"
ACCURACY_RESULT_SCHEMA = "schemas/accuracy-result-v2.schema.json"


def load_accuracy_result(path: str | Path, schema_path: str | Path) -> JsonObject:
    """Read a schema-2 accuracy result and check it against its schema."""
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


def build_accuracy_result(
    *, repository: str | Path, model_id: str, build_profile_id: str, script_output: str | Path, result_root: str | Path
) -> JsonObject:
    """Wrap scripts/accuracy.py output with the model, validation entry, dataset and MPKs it measured."""
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

    manifests = {}
    for component_id in validation_components(model, entry):
        manifest_path = Path(result_root) / component_id / "performance-test.json"
        try:
            manifests[component_id] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"accuracy needs the benchmarked {component_id} MPK: {exc}") from exc
    commits = {manifest["provenance"]["models_commit"] for manifest in manifests.values()}
    if len(commits) != 1:
        raise ValidationError("accuracy components were benchmarked from different source revisions")
    scope = entry["scope"]
    dataset = output.get("dataset", {})
    return {
        "schema_version": 2,
        "model": {
            "id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": entry["result_component"],
            "package_id": scope["package"] if scope["type"] == "package" else None,
            "artifacts": {component_id: manifest["model"]["sha256"] for component_id, manifest in manifests.items()},
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
        "metrics": metrics,
        "execution": {
            "core_version": output.get("core_version"),
            "script_sha256": hashlib.sha256((model_dir / ACCURACY_SCRIPT).read_bytes()).hexdigest(),
            "models_commit": commits.pop(),
        },
        "details": output.get("details", {}),
    }


def check_accuracy_result(
    report: JsonObject, *, model: JsonObject, build_profile_id: str, artifacts: dict[str, str], models_commit: str
) -> None:
    """Check that a schema-2 result measured the declared entry with these exact benchmarked MPKs."""
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
    components = validation_components(model, entry)
    if report["model"]["artifacts"] != {component_id: artifacts.get(component_id) for component_id in components}:
        raise ValidationError("accuracy result did not measure the benchmarked MPKs")
    missing = set(entry.get("required_metrics", ())) - set(report["metrics"])
    if missing:
        raise ValidationError(f"required accuracy metrics were not measured: {', '.join(sorted(missing))}")


def accuracy_test_name(report: JsonObject) -> str:
    return report["report_scope"] if report["schema_version"] == 1 else report["validation"]["id"]


def accuracy_measurements(report: JsonObject, fingerprint: str) -> list[JsonObject]:
    dataset = report["dataset"]
    dimensions = {
        "model_id": report["model"]["id"],
        "build_profile_id": report["model"]["build_profile_id"],
        "component_id": report["model"]["component_id"],
        "dataset_id": dataset["id"],
        "dataset_variant": dataset["variant"],
        "dataset_split": dataset["split"],
    }
    if report["schema_version"] == 1:
        return _classification_measurements(report, fingerprint, dimensions)
    validation = report["validation"]
    dimensions.update({"validation_id": validation["id"], "validation_status": validation["status"]})
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


def _classification_measurements(report: JsonObject, fingerprint: str, dimensions: JsonObject) -> list[JsonObject]:
    metric_fields = (
        ("candidate", "top1_accuracy", "model.accuracy.candidate.top1", "higher"),
        ("candidate", "top5_accuracy", "model.accuracy.candidate.top5", "higher"),
        ("reference", "top1_accuracy", "model.accuracy.reference.top1", "higher"),
        ("reference", "top5_accuracy", "model.accuracy.reference.top5", "higher"),
        ("comparison", "top1_agreement", "model.accuracy.top1_agreement", "higher"),
        ("comparison", "top1_delta", "model.accuracy.top1_delta", "higher"),
        ("comparison", "top5_delta", "model.accuracy.top5_delta", "higher"),
    )
    dimensions = {
        **dimensions,
        "evaluation_profile": report["evaluation"]["profile"],
        "report_scope": report["report_scope"],
        "evaluation_status": report["evaluation"]["status"],
    }
    return [
        {
            "measurement_key": f"{fingerprint}-{group}-{field}",
            "name": name,
            "unit": "ratio",
            "value": report["metrics"][group][field],
            "direction": direction,
            "sample_count": report["dataset"]["evaluated_samples"],
            "aggregation": "mean",
            "dimensions": dimensions,
        }
        for group, field, name, direction in metric_fields
    ]


def _artifact_metadata(report: JsonObject) -> JsonObject:
    dataset = report["dataset"]
    metadata = {
        "model_id": report["model"]["id"],
        "build_profile_id": report["model"]["build_profile_id"],
        "component_id": report["model"]["component_id"],
        "dataset_id": dataset["id"],
        "dataset_variant": dataset["variant"],
        "dataset_split": dataset["split"],
        "evaluated_samples": dataset["evaluated_samples"],
        "available_samples": dataset["available_samples"],
    }
    if report["schema_version"] == 1:
        metadata.update(
            {
                "evaluation_profile": report["evaluation"]["profile"],
                "evaluation_status": report["evaluation"]["status"],
                "report_scope": report["report_scope"],
                "candidate_top1_accuracy": report["metrics"]["candidate"]["top1_accuracy"],
                "candidate_top5_accuracy": report["metrics"]["candidate"]["top5_accuracy"],
            }
        )
    else:
        validation = report["validation"]
        metadata.update({"validation_id": validation["id"], "validation_status": validation["status"]})
    return metadata


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
        f"/v1/runs/{run_id}/test-results/accuracy/{accuracy_test_name(report)}",
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
