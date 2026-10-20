from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker

from .accuracy_config import accuracy_components, find_devkit_accuracy_config
from .accuracy_result import (
    ACCURACY_RESULT_SCHEMA,
    check_accuracy_result,
    compiled_artifacts,
    load_accuracy_result,
    publish_accuracy_result,
)
from .benchmark_metrics import measurement_name, registry_metrics
from .dependencies import resolve_benchmark_runtime
from .errors import ValidationError
from .fetch import sha256_file
from .model import benchmark_entry, load_and_validate_model
from .model_tests import require_test_outcome
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256
from .stage_keys import (
    ACCURACY_FINGERPRINT_VERSION,
    ACCURACY_RUN_TYPE,
    PERFORMANCE_FINGERPRINT_VERSION,
    PERFORMANCE_RESULT_SCHEMA,
    PERFORMANCE_RUN_TYPE,
    accuracy_key,
    performance_key,
)

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]

PERFORMANCE_RESULT_FILE = "performance-result.json"


def benchmark_identity_document(result: JsonObject) -> JsonObject:
    model = result["model"]
    document = {
        "schema_version": result["schema_version"],
        "benchmark_type": result["benchmark_type"],
        "model": {
            "id": model["id"],
            "build_profile_id": model["build_profile_id"],
            "component_id": model["component_id"],
            "sha256": model["sha256"],
        },
        "compatibility": {
            "platform_version": result["compatibility"]["expected_platform_version"],
            "neat_version": result["compatibility"]["expected_neat_version"],
        },
        "configuration": result["configuration"],
        "environment_fingerprint": result["environment"]["fingerprint"],
        "tool": result["tool"],
    }
    if "components" in result:
        document["model"]["package_id"] = model["package_id"]
        document["components"] = [
            {"component_id": item["component_id"], "sha256": item["sha256"]} for item in result["components"]
        ]
    return document


def load_benchmark_result(result_path: str | Path, schema_path: str | Path | None = None) -> JsonObject:
    path = Path(result_path).expanduser().resolve()
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        schema = (
            Path(schema_path).expanduser().resolve()
            if schema_path is not None
            else Path(__file__).resolve().parents[2] / PERFORMANCE_RESULT_SCHEMA
        )
        schema_value = json.loads(schema.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read benchmark result or schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema_value, format_checker=FormatChecker()).iter_errors(result),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "result"
        raise ValidationError(f"benchmark result {location}: {first.message}")
    if result["identity"]["fingerprint"] is None:
        raise ValidationError("benchmark result does not have a publishable identity fingerprint")
    expected = canonical_sha256(benchmark_identity_document(result))
    if result["identity"]["fingerprint"] != expected:
        raise ValidationError("benchmark result identity fingerprint does not match its canonical inputs")
    return result


def validate_benchmark_set(
    *,
    repository: str | Path,
    result_root: str | Path,
    model_id: str,
    build_profile_id: str,
    component_ids: list[str],
    git_sha: str,
) -> JsonObject:
    root = Path(repository).expanduser().resolve()
    results_root = Path(result_root).expanduser().resolve()
    if len(component_ids) != len(set(component_ids)) or not component_ids:
        raise ValidationError("benchmark component IDs must be non-empty and unique")
    commit_sha = _require_git_sha(git_sha)
    actual_paths = {path.resolve() for path in results_root.rglob(PERFORMANCE_RESULT_FILE)}
    expected_paths = {(results_root / component / PERFORMANCE_RESULT_FILE).resolve() for component in component_ids}
    if actual_paths != expected_paths:
        raise ValidationError("benchmark result tree does not match the declared component set")

    validated = []
    for path in sorted(expected_paths):
        component_id = path.parent.name
        result = load_benchmark_result(path, root / PERFORMANCE_RESULT_SCHEMA)
        model = result["model"]
        expected = (model_id, build_profile_id, component_id)
        actual = (model["id"], model["build_profile_id"], _host_component(result))
        if actual != expected:
            raise ValidationError(f"{path}: benchmark identity does not match {expected}")
        provenance = result["provenance"]
        if provenance.get("models_commit") != commit_sha:
            raise ValidationError(f"{path}: benchmark models_commit does not match the source revision")
        _require_compile_run_id(result, None)
        definition = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
        entry = benchmark_entry(definition, component_id, build_profile_id)
        if entry is None:
            raise ValidationError(f"{path}: no benchmark entry applies to {expected}")
        measured = registry_metrics(result)
        for metric_name in entry.get("required_metrics", ()):
            if metric_name not in measured:
                raise ValidationError(f"{path}: required benchmark metric {metric_name!r} was not measured")
        validated.append(str(path.resolve()))
    return {"count": len(validated), "results": validated}


def _subject(result: JsonObject) -> JsonObject:
    """What a result measured: a component, or a package's components running as one pipeline."""
    model = result["model"]
    return {"package_id": model["package_id"]} if "components" in result else {"component_id": model["component_id"]}


def _host_component(result: JsonObject) -> str:
    """The component whose build the result's run is under: its own, or its package's last."""
    return result["components"][-1]["component_id"] if "components" in result else result["model"]["component_id"]


def _builds(result: JsonObject, compile_run_id: str) -> dict[str, JsonObject]:
    """Each component build the result measured, by component ID, with its compile run and MPK digest."""
    if "components" in result:
        return {
            item["component_id"]: {"run_id": item["compile_run_id"], "artifact_sha256": item["sha256"]}
            for item in result["components"]
        }
    return {result["model"]["component_id"]: {"run_id": compile_run_id, "artifact_sha256": result["model"]["sha256"]}}


def _require_compile_run_id(result: JsonObject, declared: str | None) -> str:
    provenance_run_id = result["provenance"].get("compile_run_id")
    run_id = declared or provenance_run_id
    if not isinstance(run_id, str) or not run_id:
        raise ValidationError("compile run ID is required as --compile-run-id or provenance.compile_run_id")
    try:
        normalized = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise ValidationError("compile run ID must be a UUID") from exc
    if declared and provenance_run_id and normalized != str(provenance_run_id):
        raise ValidationError("--compile-run-id does not match provenance.compile_run_id")
    return normalized


def _require_git_sha(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    return normalized


def _measurements(result: JsonObject, fingerprint: str) -> list[JsonObject]:
    dimensions = {
        "model_id": result["model"]["id"],
        "build_profile_id": result["model"]["build_profile_id"],
        **_subject(result),
        "benchmark_type": result["benchmark_type"],
    }
    return [
        {
            "measurement_key": f"{fingerprint}-{field}",
            "name": measurement_name(field),
            "unit": metric["unit"],
            "value": metric["value"],
            "direction": metric["direction"],
            "sample_count": metric["sample_count"],
            "aggregation": metric["aggregation"],
            "dimensions": dict(dimensions),
        }
        for field, metric in registry_metrics(result).items()
    ]


def publish_benchmark_result(
    *,
    result_path: str | Path,
    log_path: str | Path | None = None,
    api_url: str,
    aws_region: str,
    bucket: str,
    environment: str,
    branch: str,
    git_sha: str,
    compile_run_id: str | None = None,
    kms_key_id: str | None = None,
    github_run_url: str | None = None,
    schema_path: str | Path | None = None,
    repository: str | Path,
    test_status: str,
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    """Publish a DevKit performance result as a ``model_benchmark`` run under its compile run: a component's
    under its own, a package's under its last component's.

    The run fingerprint is the performance key, read from ``repository`` at the measured revision.
    A passed result's test result carries ``test_status``, the outcome of the model's performance tests on it;
    the run still completes when they fail, so the stage is not measured again.
    """
    test_status = require_test_outcome(test_status)
    root = Path(repository).expanduser().resolve()
    path = Path(result_path).expanduser().resolve()
    log = Path(log_path).expanduser().resolve() if log_path is not None else None
    result = load_benchmark_result(path, schema_path)
    parent_run_id = _require_compile_run_id(result, compile_run_id)
    commit_sha = _require_git_sha(git_sha)
    if not branch.strip():
        raise ValidationError("branch must not be empty")

    fingerprint = result["identity"]["fingerprint"]
    model_id = result["model"]["id"]
    build_profile_id = result["model"]["build_profile_id"]
    subject = _subject(result)
    builds = _builds(result, parent_run_id)
    measurements = _measurements(result, fingerprint)
    execution_digest = canonical_sha256(result)
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    stage_key = performance_key(root, model, build_profile_id, _host_component(result), builds)
    stable_digest = hashlib.sha256(
        f"{parent_run_id}:{stage_key}:{fingerprint}:{execution_digest}".encode()
    ).hexdigest()
    stable_key = f"models-benchmark-{stable_digest}"
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)

    create_body: JsonObject = {
        # This is the registry run-envelope schema version, not the benchmark
        # result artifact's independently versioned schema.
        "schema_version": 1,
        "run_type": PERFORMANCE_RUN_TYPE,
        "parent_run_id": parent_run_id,
        "commit_sha": commit_sha,
        "git_ref": f"refs/heads/{branch}",
        "run_fingerprint": stage_key,
        "environment": {
            "fingerprint": result["environment"]["fingerprint"],
            "fingerprint_version": PERFORMANCE_FINGERPRINT_VERSION,
            "toolchain_name": "sima-neat",
            "toolchain_version": result["compatibility"]["actual_neat_version"],
            "target_platform": result["compatibility"]["actual_platform_version"],
            "target_device": result["environment"]["board_model"],
            "operating_system": result["environment"]["kernel_release"],
            "properties": {
                "vulcan_environment": environment,
                "machine": result["environment"]["machine"],
                "cpu_count": result["environment"]["cpu_count"],
                "cpu_governor": result["environment"]["cpu_governor"],
            },
        },
        "metadata": {
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            **subject,
            "benchmark_type": result["benchmark_type"],
            "compile_run_id": parent_run_id,
            "benchmark_execution_digest": execution_digest,
            "performance_key": stage_key,
        },
    }
    if github_run_url:
        create_body.update({"ci_provider": "github-actions", "ci_run_url": github_run_url})
    created = client.request("POST", "/v1/runs", create_body, f"{stable_key}-create")
    run_id_value = created.get("run_id")
    try:
        run_id = str(uuid.UUID(str(run_id_value)))
    except ValueError as exc:
        raise ValidationError("Vulcan did not return a valid benchmark run ID") from exc

    # A package's run records every build its pipeline ran, as an accuracy run does.
    for component_id, build in builds.items():
        name = f"compiled-model-{component_id.replace('_', '-')}" if "package_id" in subject else "compiled-model"
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/{name}",
            {
                "input_type": "model-pack",
                "source_reference": f"vulcan://runs/{build['run_id']}/artifacts/mpk",
                "sha256": build["artifact_sha256"],
                "media_type": "application/gzip",
                "metadata": {
                    "model_id": model_id,
                    "build_profile_id": build_profile_id,
                    "component_id": component_id,
                    "compile_run_id": build["run_id"],
                },
            },
            f"{stable_key}-{name}",
        )
    parameters = benchmark_identity_document(result)
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/parameters",
        {
            "parameters": parameters,
            "canonical_sha256": canonical_sha256(parameters),
            "canonicalization_version": 1,
        },
        f"{stable_key}-parameters",
    )

    result_sha, _ = sha256_file(path)
    result_prefix = (
        f"models/{quote(branch, safe='')}/{commit_sha[:12]}/{model_id}/{build_profile_id}/"
        f"{result['model'].get('package_id', result['model']['component_id'])}/"
        f"benchmarks/{fingerprint}/{execution_digest}"
    )
    object_key = f"{result_prefix}/{PERFORMANCE_RESULT_FILE}"
    stored = store.ensure_native_sha256(path, object_key, result_sha, media_type="application/json")
    artifact = client.request(
        "PUT",
        f"/v1/runs/{run_id}/artifacts/performance-result",
        {
            "artifact_type": "benchmark-result",
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": "application/json",
            "retention_class": "branch",
            "metadata": {
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                **subject,
                "result_fingerprint": fingerprint,
                "benchmark_execution_digest": execution_digest,
            },
        },
        f"{stable_key}-artifact",
    )
    test_body: JsonObject = {
        "status": test_status if result["status"] == "passed" else "error",
        "version": str(result["schema_version"]),
        "summary": result,
    }
    if artifact.get("artifact_id"):
        test_body["report_artifact_id"] = artifact["artifact_id"]
    log_stored = None
    if log is not None:
        log_sha, _ = sha256_file(log)
        log_stored = store.ensure_native_sha256(log, f"{result_prefix}/benchmark.log", log_sha, media_type="text/plain")
        log_artifact = client.request(
            "PUT",
            f"/v1/runs/{run_id}/artifacts/benchmark-log",
            {
                "artifact_type": "log",
                "s3_bucket": bucket,
                "s3_key": log_stored.key,
                "s3_version_id": log_stored.version_id,
                "sha256": log_stored.sha256,
                "size_bytes": log_stored.size_bytes,
                "media_type": "text/plain",
                "retention_class": "branch",
                "metadata": {"model_id": model_id, "build_profile_id": build_profile_id, **subject},
            },
            f"{stable_key}-log-artifact",
        )
        if log_artifact.get("artifact_id"):
            test_body["log_artifact_id"] = log_artifact["artifact_id"]
    if result.get("failure_classification"):
        test_body["failure_classification"] = result["failure_classification"]
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/test-results/performance/model-synthetic",
        test_body,
        f"{stable_key}-test-result",
    )
    if measurements:
        client.request(
            "POST",
            f"/v1/runs/{run_id}/measurements/batch",
            {"measurements": measurements},
            f"{stable_key}-measurements",
        )
    client.request(
        "POST",
        f"/v1/runs/{run_id}/finalize",
        {"status": "completed" if result["status"] == "passed" else "failed"},
        f"{stable_key}-finalize",
    )
    return {
        "benchmark_run_id": run_id,
        "compile_run_id": parent_run_id,
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        **subject,
        "result_fingerprint": fingerprint,
        "performance_key": stage_key,
        "execution_digest": execution_digest,
        "s3_key": stored.key,
        "log_s3_key": log_stored.key if log_stored is not None else None,
    }


def _compile_run_id(compiled_dir: Path, model_id: str, build_profile_id: str, component_id: str) -> str:
    path = compiled_dir / model_id / build_profile_id / component_id / "registry-publication.json"
    try:
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
        run_id = str(uuid.UUID(str(runs[0]["run_id"])))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValidationError(f"accuracy needs the registered {component_id} compile run: {path}: {exc}") from exc
    if len(runs) != 1 or runs[0].get("component_id") != component_id:
        raise ValidationError(f"{path}: registry publication does not belong to {component_id}")
    return run_id


def publish_accuracy_run(
    *,
    repository: str | Path,
    compiled_dir: str | Path,
    result_path: str | Path,
    log_path: str | Path | None = None,
    api_url: str,
    aws_region: str,
    bucket: str,
    environment: str,
    branch: str,
    git_sha: str,
    test_status: str,
    kms_key_id: str | None = None,
    github_run_url: str | None = None,
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    """Publish a build profile's DevKit accuracy result as a ``model_accuracy`` run under its result component.

    The run's inputs are the compiled MPKs the accuracy script ran; its fingerprint is the accuracy key. Its
    test result carries ``test_status``, the outcome of the model's accuracy tests on the result; the run
    still completes when they fail, so the stage is not measured again.
    """
    test_status = require_test_outcome(test_status)
    root = Path(repository).expanduser().resolve()
    compiled = Path(compiled_dir).expanduser().resolve()
    path = Path(result_path).expanduser().resolve()
    commit_sha = _require_git_sha(git_sha)
    if not branch.strip():
        raise ValidationError("branch must not be empty")
    report = load_accuracy_result(path, root / ACCURACY_RESULT_SCHEMA)
    model_id = report["model"]["id"]
    build_profile_id = report["model"]["build_profile_id"]
    component_id = report["model"]["component_id"]
    model = load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")
    entry = find_devkit_accuracy_config(model, build_profile_id)
    if entry is None:
        raise ValidationError(f"{path}: {model_id} declares no DevKit accuracy for {build_profile_id}")
    components = accuracy_components(model, entry)
    artifacts = compiled_artifacts(compiled, model_id, build_profile_id, components)
    check_accuracy_result(
        report, model=model, build_profile_id=build_profile_id, artifacts=artifacts, models_commit=commit_sha
    )
    builds = {
        component: {
            "run_id": _compile_run_id(compiled, model_id, build_profile_id, component),
            "artifact_sha256": artifacts[component],
        }
        for component in components
    }
    stage_key = accuracy_key(root, model, build_profile_id, builds)
    parent_run_id = builds[component_id]["run_id"]
    report_digest = canonical_sha256(report)
    stable_digest = hashlib.sha256(f"{parent_run_id}:{stage_key}:{report_digest}".encode()).hexdigest()
    stable_key = f"models-accuracy-{stable_digest}"
    runtime = resolve_benchmark_runtime(root)
    core_version = report["execution"]["core_version"]
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)

    create_body: JsonObject = {
        "schema_version": 1,
        "run_type": ACCURACY_RUN_TYPE,
        "parent_run_id": parent_run_id,
        "commit_sha": commit_sha,
        "git_ref": f"refs/heads/{branch}",
        "run_fingerprint": stage_key,
        "environment": {
            "fingerprint": canonical_sha256({"runtime": runtime, "core_version": core_version}),
            "fingerprint_version": ACCURACY_FINGERPRINT_VERSION,
            "toolchain_name": "sima-neat",
            "toolchain_version": core_version,
            "target_platform": runtime["devkit"]["platform_version"],
            "properties": {"vulcan_environment": environment},
        },
        "metadata": {
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": component_id,
            "accuracy_id": entry["id"],
            "compile_run_id": parent_run_id,
            "accuracy_key": stage_key,
        },
    }
    if github_run_url:
        create_body.update({"ci_provider": "github-actions", "ci_run_url": github_run_url})
    created = client.request("POST", "/v1/runs", create_body, f"{stable_key}-create")
    try:
        run_id = str(uuid.UUID(str(created.get("run_id"))))
    except ValueError as exc:
        raise ValidationError("Vulcan did not return a valid accuracy run ID") from exc

    for component, build in builds.items():
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/compiled-model-{component.replace('_', '-')}",
            {
                "input_type": "model-pack",
                "source_reference": f"vulcan://runs/{build['run_id']}/artifacts/mpk",
                "sha256": build["artifact_sha256"],
                "media_type": "application/gzip",
                "metadata": {
                    "model_id": model_id,
                    "build_profile_id": build_profile_id,
                    "component_id": component,
                    "compile_run_id": build["run_id"],
                },
            },
            f"{stable_key}-input-{component}",
        )
    # Vulcan completes a run only once its canonical parameters are stored.
    parameters = {"accuracy": entry, "builds": builds, "runtime": runtime, "core_version": core_version}
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/parameters",
        {
            "parameters": parameters,
            "canonical_sha256": canonical_sha256(parameters),
            "canonicalization_version": 1,
        },
        f"{stable_key}-parameters",
    )
    prefix = (
        f"models/{quote(branch, safe='')}/{commit_sha[:12]}/{model_id}/{build_profile_id}/{component_id}/"
        f"accuracy/{stage_key}/{report_digest}"
    )
    publication = publish_accuracy_result(
        client,
        store,
        bucket=bucket,
        run_id=run_id,
        path=path,
        report=report,
        sha256=sha256_file(path)[0],
        fingerprint=stage_key,
        object_key=f"{prefix}/{path.name}",
        stable_key=stable_key,
        test_status=test_status,
    )
    if log_path is not None:
        log = Path(log_path).expanduser().resolve()
        log_sha, _ = sha256_file(log)
        log_stored = store.ensure_native_sha256(log, f"{prefix}/accuracy.log", log_sha, media_type="text/plain")
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/artifacts/accuracy-log",
            {
                "artifact_type": "log",
                "s3_bucket": bucket,
                "s3_key": log_stored.key,
                "s3_version_id": log_stored.version_id,
                "sha256": log_stored.sha256,
                "size_bytes": log_stored.size_bytes,
                "media_type": "text/plain",
                "retention_class": "branch",
                "metadata": {"model_id": model_id, "build_profile_id": build_profile_id, "component_id": component_id},
            },
            f"{stable_key}-log-artifact",
        )
    client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
    return {
        "accuracy_run_id": run_id,
        "compile_run_id": parent_run_id,
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "component_id": component_id,
        "accuracy_key": stage_key,
        **publication,
    }
