from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker

from .benchmark_manifest import load_component_performance_config
from .errors import ValidationError
from .fetch import sha256_file
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]

REGISTRY_AGGREGATION_ALIASES = {
    "first": "raw",
    "last": "raw",
    "maximum": "max",
    "minimum": "min",
    "single": "raw",
}
REGISTRY_AGGREGATIONS = {
    "raw",
    "min",
    "max",
    "mean",
    "median",
    "p50",
    "p90",
    "p95",
    "p99",
    "sum",
}
LEGACY_THROUGHPUT_DEFINITION = ("model.throughput", "fps")
CORE_THROUGHPUT_DEFINITION = ("model.inference_rate", "inferences/s")


def benchmark_identity_document(result: JsonObject) -> JsonObject:
    model = result["model"]
    if result["schema_version"] <= 3:
        model_identity = {
            "id": model["id"],
            "variant_id": model["variant_id"],
            "sha256": model["sha256"],
        }
    else:
        model_identity = {
            "id": model["id"],
            "build_profile_id": model["build_profile_id"],
            "component_id": model["component_id"],
            "sha256": model["sha256"],
        }
    return {
        "schema_version": result["schema_version"],
        "benchmark_type": result["benchmark_type"],
        "model": model_identity,
        "compatibility": {
            "platform_version": result["compatibility"]["expected_platform_version"],
            "neat_version": result["compatibility"]["expected_neat_version"],
        },
        "configuration": result["configuration"],
        "environment_fingerprint": result["environment"]["fingerprint"],
        "tool": result["tool"],
    }


def load_benchmark_result(result_path: str | Path, schema_path: str | Path | None = None) -> JsonObject:
    path = Path(result_path).expanduser().resolve()
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        version = result.get("schema_version") if isinstance(result, dict) else None
        if version not in {1, 2, 3, 4, 5}:
            raise ValidationError(f"unsupported benchmark result schema version: {version!r}")
        schema = (
            Path(schema_path).expanduser().resolve()
            if schema_path is not None
            else Path(__file__).resolve().parents[2] / "schemas" / f"performance-result-v{version}.schema.json"
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
    actual_paths = {path.resolve() for path in results_root.rglob("performance-result-v*.json")}
    expected_paths = set()
    for component_id in component_ids:
        matches = list((results_root / component_id).glob("performance-result-v[45].json"))
        if len(matches) != 1:
            raise ValidationError("benchmark result tree must contain exactly one result per component")
        expected_paths.add(matches[0].resolve())
    if actual_paths != expected_paths:
        raise ValidationError("benchmark result tree does not match the declared component set")

    validated = []
    for path in sorted(expected_paths):
        component_id = path.parent.name
        result = load_benchmark_result(path, root / "schemas" / f"{path.stem}.schema.json")
        model = result["model"]
        expected = (model_id, build_profile_id, component_id)
        actual = (model["id"], model["build_profile_id"], model["component_id"])
        if actual != expected:
            raise ValidationError(f"{path}: benchmark identity does not match {expected}")
        provenance = result["provenance"]
        if provenance.get("models_commit") != commit_sha:
            raise ValidationError(f"{path}: benchmark models_commit does not match the source revision")
        _require_compile_run_id(result, None)
        config = load_component_performance_config(root, model_id, component_id, build_profile_id)
        for metric_name in config.get("required_metrics", ()):
            metric = result["metrics"].get(metric_name)
            if not isinstance(metric, dict) or metric.get("status") != "measured" or metric.get("value") is None:
                raise ValidationError(f"{path}: required benchmark metric {metric_name!r} was not measured")
        validated.append(str(path.resolve()))
    return {"count": len(validated), "results": validated}


def _build_identity(result: JsonObject) -> tuple[str, str]:
    """Map legacy variant identities onto the v2 build coordinates."""
    model = result["model"]
    if result["schema_version"] <= 3:
        return model["variant_id"], "model"
    return model["build_profile_id"], model["component_id"]


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
    build_profile_id, component_id = _build_identity(result)
    definitions = {
        "latency_avg_ms": ("model.latency.avg", "lower"),
        "latency_p50_ms": ("model.latency.p50", "lower"),
        "latency_p95_ms": ("model.latency.p95", "lower"),
        "throughput_fps": ("model.throughput", "higher"),
        "avg_power_watts": ("board.power.avg", "lower"),
        "energy_joules": ("board.energy", "lower"),
        "energy_per_inference_joules": ("model.energy_per_inference", "lower"),
        "model_load_ms": ("model.initialization.load", "lower"),
        "time_to_first_inference_ms": ("model.initialization.time_to_first_inference", "lower"),
        "mla_memory_peak_delta_mib": ("model.mla_memory.peak_delta", "lower"),
    }
    dimensions = {
        "model_id": result["model"]["id"],
        "build_profile_id": build_profile_id,
        "component_id": component_id,
        "benchmark_type": result["benchmark_type"],
    }
    values = []
    for field, (name, direction) in definitions.items():
        metric = result["metrics"].get(field)
        if not isinstance(metric, dict) or metric.get("status") != "measured" or metric.get("value") is None:
            continue
        if field == "throughput_fps":
            name, expected_unit = (
                CORE_THROUGHPUT_DEFINITION if result["schema_version"] == 5 else LEGACY_THROUGHPUT_DEFINITION
            )
            if metric["unit"] != expected_unit:
                raise ValidationError(
                    f"benchmark result schema v{result['schema_version']} requires {field} unit {expected_unit!r}"
                )
        result_aggregation = metric["aggregation"]
        aggregation = REGISTRY_AGGREGATION_ALIASES.get(result_aggregation, result_aggregation)
        if aggregation not in REGISTRY_AGGREGATIONS:
            raise ValidationError(
                f"benchmark metric {field} uses aggregation {result_aggregation!r}, "
                "which cannot be represented by the Vulcan registry"
            )
        metric_dimensions = dict(dimensions)
        if aggregation != result_aggregation:
            metric_dimensions["result_aggregation"] = result_aggregation
        values.append(
            {
                "measurement_key": f"{fingerprint}-{field}",
                "name": name,
                "unit": metric["unit"],
                "value": metric["value"],
                "direction": direction,
                "sample_count": metric["sample_count"],
                "aggregation": aggregation,
                "dimensions": metric_dimensions,
            }
        )
    return values


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
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    path = Path(result_path).expanduser().resolve()
    log = Path(log_path).expanduser().resolve() if log_path is not None else None
    result = load_benchmark_result(path, schema_path)
    parent_run_id = _require_compile_run_id(result, compile_run_id)
    commit_sha = _require_git_sha(git_sha)
    if not branch.strip():
        raise ValidationError("branch must not be empty")

    fingerprint = result["identity"]["fingerprint"]
    model_id = result["model"]["id"]
    build_profile_id, component_id = _build_identity(result)
    measurements = _measurements(result, fingerprint)
    execution_digest = canonical_sha256(result)
    stable_digest = hashlib.sha256(
        f"{parent_run_id}:{fingerprint}:{execution_digest}".encode()
    ).hexdigest()
    stable_key = f"models-benchmark-{stable_digest}"
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)

    create_body: JsonObject = {
        # This is the registry run-envelope schema version, not the benchmark
        # result artifact's independently versioned schema.
        "schema_version": 1,
        "run_type": "model_benchmark",
        "parent_run_id": parent_run_id,
        "commit_sha": commit_sha,
        "git_ref": f"refs/heads/{branch}",
        "run_fingerprint": fingerprint,
        "environment": {
            "fingerprint": result["environment"]["fingerprint"],
            "fingerprint_version": result["environment"]["fingerprint_version"],
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
            "component_id": component_id,
            "benchmark_type": result["benchmark_type"],
            "compile_run_id": parent_run_id,
            "benchmark_execution_digest": execution_digest,
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

    client.request(
        "PUT",
        f"/v1/runs/{run_id}/inputs/compiled-model",
        {
            "input_type": "model-pack",
            "source_reference": result["model"]["uri"] or f"vulcan://runs/{parent_run_id}/artifacts/mpk",
            "sha256": result["model"]["sha256"],
            "media_type": "application/gzip",
            "metadata": {
                "model_id": model_id,
                "build_profile_id": build_profile_id,
                "component_id": component_id,
                "compile_run_id": parent_run_id,
            },
        },
        f"{stable_key}-input",
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
        f"models/{quote(branch, safe='')}/{commit_sha[:12]}/{model_id}/{build_profile_id}/{component_id}/"
        f"benchmarks/{fingerprint}/{execution_digest}"
    )
    object_key = f"{result_prefix}/performance-result-v{result['schema_version']}.json"
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
                "component_id": component_id,
                "result_fingerprint": fingerprint,
                "benchmark_execution_digest": execution_digest,
            },
        },
        f"{stable_key}-artifact",
    )
    test_body: JsonObject = {
        "status": "passed" if result["status"] == "passed" else "error",
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
                "metadata": {
                    "model_id": model_id,
                    "build_profile_id": build_profile_id,
                    "component_id": component_id,
                },
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
        "component_id": component_id,
        "result_fingerprint": fingerprint,
        "execution_digest": execution_digest,
        "s3_key": stored.key,
        "log_s3_key": log_stored.key if log_stored is not None else None,
    }
