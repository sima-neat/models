from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker

from .errors import ValidationError
from .fetch import sha256_file
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]


def benchmark_identity_document(result: JsonObject) -> JsonObject:
    return {
        "schema_version": result["schema_version"],
        "benchmark_type": result["benchmark_type"],
        "model": {
            "id": result["model"]["id"],
            "variant_id": result["model"]["variant_id"],
            "sha256": result["model"]["sha256"],
        },
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
        if version not in {1, 2}:
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
    definitions = {
        "latency_avg_ms": ("model.latency.avg", "lower"),
        "latency_p50_ms": ("model.latency.p50", "lower"),
        "latency_p95_ms": ("model.latency.p95", "lower"),
        "throughput_fps": ("model.throughput", "higher"),
        "avg_power_watts": ("board.power.avg", "lower"),
        "energy_joules": ("board.energy", "lower"),
        "energy_per_inference_joules": ("model.energy_per_inference", "lower"),
    }
    dimensions = {
        "model_id": result["model"]["id"],
        "variant_id": result["model"]["variant_id"],
        "benchmark_type": result["benchmark_type"],
    }
    values = []
    for field, (name, direction) in definitions.items():
        metric = result["metrics"][field]
        if metric["status"] != "measured" or metric["value"] is None:
            continue
        values.append(
            {
                "measurement_key": f"{fingerprint}-{field}",
                "name": name,
                "unit": metric["unit"],
                "value": metric["value"],
                "direction": direction,
                "sample_count": metric["sample_count"],
                "aggregation": metric["aggregation"],
                "dimensions": dimensions,
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
    variant_id = result["model"]["variant_id"]
    execution_digest = canonical_sha256(result)
    stable_digest = hashlib.sha256(
        f"{parent_run_id}:{fingerprint}:{execution_digest}".encode()
    ).hexdigest()
    stable_key = f"models-benchmark-{stable_digest}"
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)

    create_body: JsonObject = {
        "schema_version": result["schema_version"],
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
            "variant_id": variant_id,
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
            "metadata": {"model_id": model_id, "variant_id": variant_id, "compile_run_id": parent_run_id},
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
        f"models/{quote(branch, safe='')}/{commit_sha[:12]}/{model_id}/{variant_id}/"
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
                "variant_id": variant_id,
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
                "metadata": {"model_id": model_id, "variant_id": variant_id},
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
    measurements = _measurements(result, fingerprint)
    if measurements:
        client.request(
            "POST",
            f"/v1/runs/{run_id}/measurements:batch",
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
        "variant_id": variant_id,
        "result_fingerprint": fingerprint,
        "execution_digest": execution_digest,
        "s3_key": stored.key,
        "log_s3_key": log_stored.key if log_stored is not None else None,
    }
