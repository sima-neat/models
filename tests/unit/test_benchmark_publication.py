from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from model_registry.benchmark_publication import load_benchmark_result, publish_benchmark_result
from model_registry.errors import ValidationError
from model_registry.registry import S3Object, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
COMPILE_RUN_ID = "00000000-0000-4000-8000-000000000001"
BENCHMARK_RUN_ID = "00000000-0000-4000-8000-000000000002"


class FakeStore:
    def __init__(self) -> None:
        self.objects: list[tuple[Path, str, str, str]] = []

    def ensure_native_sha256(
        self, path: Path, key: str, sha256: str, *, media_type: str = "application/gzip"
    ) -> S3Object:
        self.objects.append((path, key, sha256, media_type))
        return S3Object(key, "version-1", sha256, path.stat().st_size)


def load_benchmarker() -> ModuleType:
    path = ROOT / "tools" / "runtime" / "benchmarker.py"
    spec = importlib.util.spec_from_file_location("publication_benchmarker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_result(tmp_path: Path) -> Path:
    benchmarker = load_benchmarker()
    manifest = {
        "schema_version": 1,
        "model": {
            "id": "resnet_50",
            "variant_id": "modalix_bf16",
            "path": str(tmp_path / "model.tar.gz"),
            "sha256": "a" * 64,
        },
        "compatibility": {"platform_version": "2.1.3", "neat_version": "0.4.0"},
        "benchmark": {"samples": 100, "trials": 3},
        "provenance": {"compile_run_id": COMPILE_RUN_ID},
    }
    preflight = {
        "status": "available",
        "reason": None,
        "runtime_user": "sima",
        "required_group": "i2c",
        "devices": [{"path": "/dev/i2c-3", "readable": True, "writable": True}],
    }
    result = benchmarker._result_base(
        manifest,
        Path(manifest["model"]["path"]),
        "2.1.3",
        preflight=preflight,
        actual_neat="0.4.0",
    )
    result["status"] = "passed"
    result["metrics"]["latency_avg_ms"] = benchmarker.metric(
        [{"latency_avg_ms": 3.5}, {"latency_avg_ms": 3.7}], "latency_avg_ms", "ms", "unavailable"
    )
    result["metrics"]["throughput_fps"] = benchmarker.metric(
        [{"throughput_fps": 450.0}, {"throughput_fps": 470.0}], "throughput_fps", "fps", "unavailable"
    )
    result["metrics"]["model_load_ms"] = benchmarker._single_measurement(
        125.0, "ms", details={"phase": "pyneat_model_construction"}
    )
    result["metrics"]["time_to_first_inference_ms"] = benchmarker._single_measurement(
        340.0,
        "ms",
        details={"phase": "before_model_construction_to_first_synchronous_result"},
    )
    result["metrics"]["mla_memory_peak_delta_mib"] = {
        "status": "measured",
        "reason": None,
        "unit": "MiB",
        "aggregation": "maximum",
        "value": 128.0,
        "minimum": 0.0,
        "maximum": 128.0,
        "standard_deviation": 32.0,
        "sample_count": 8,
    }
    result["environment"]["observations"]["end"] = benchmarker.runtime_observation()
    path = tmp_path / "performance-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_publishes_linked_benchmark_run_idempotently(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        calls.append((method, endpoint, body, key))
        if endpoint == "/v1/runs":
            return {"run_id": BENCHMARK_RUN_ID}
        if endpoint.endswith("/artifacts/performance-result"):
            return {"artifact_id": "00000000-0000-4000-8000-000000000003"}
        return {"ok": True}

    store = FakeStore()
    published = publish_benchmark_result(
        result_path=path,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        environment="staging",
        branch="feature/benchmark",
        git_sha="a" * 40,
        request=request,
        artifact_store=store,
    )

    assert published["compile_run_id"] == COMPILE_RUN_ID
    assert published["benchmark_run_id"] == BENCHMARK_RUN_ID
    assert published["model_id"] == "resnet_50"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert published["execution_digest"] == canonical_sha256(result)
    assert [endpoint for _, endpoint, _, _ in calls] == [
        "/v1/runs",
        f"/v1/runs/{BENCHMARK_RUN_ID}/inputs/compiled-model",
        f"/v1/runs/{BENCHMARK_RUN_ID}/parameters",
        f"/v1/runs/{BENCHMARK_RUN_ID}/artifacts/performance-result",
        f"/v1/runs/{BENCHMARK_RUN_ID}/test-results/performance/model-synthetic",
        f"/v1/runs/{BENCHMARK_RUN_ID}/measurements/batch",
        f"/v1/runs/{BENCHMARK_RUN_ID}/finalize",
    ]
    create = calls[0][2]
    assert create is not None
    assert create["schema_version"] == 1
    assert create["run_type"] == "model_benchmark"
    assert create["parent_run_id"] == COMPILE_RUN_ID
    assert create["run_fingerprint"] == published["result_fingerprint"]
    test_result = calls[4][2]
    assert test_result is not None and test_result["status"] == "passed"
    assert test_result["report_artifact_id"] == "00000000-0000-4000-8000-000000000003"
    measurements = calls[5][2]
    assert measurements is not None
    assert {item["name"] for item in measurements["measurements"]} == {
        "model.latency.avg",
        "model.throughput",
        "model.initialization.load",
        "model.initialization.time_to_first_inference",
        "model.mla_memory.peak_delta",
    }
    mla_memory = next(
        item for item in measurements["measurements"] if item["name"] == "model.mla_memory.peak_delta"
    )
    assert mla_memory["unit"] == "MiB"
    assert mla_memory["aggregation"] == "max"
    assert mla_memory["dimensions"]["result_aggregation"] == "maximum"
    assert mla_memory["direction"] == "lower"
    model_load = next(
        item for item in measurements["measurements"] if item["name"] == "model.initialization.load"
    )
    assert model_load["aggregation"] == "raw"
    assert model_load["dimensions"]["result_aggregation"] == "single"
    assert calls[-1][2] == {"status": "completed"}
    assert store.objects[0][3] == "application/json"
    assert store.objects[0][2] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.objects[0][1].endswith(f"/{published['execution_digest']}/performance-result-v3.json")


def test_publishes_log_beside_schema_versioned_result(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    log = tmp_path / "benchmark.log"
    log.write_text("benchmark complete\n", encoding="utf-8")

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        if endpoint == "/v1/runs":
            return {"run_id": BENCHMARK_RUN_ID}
        if endpoint.endswith("/artifacts/benchmark-log"):
            return {"artifact_id": "00000000-0000-4000-8000-000000000004"}
        return {"ok": True}

    store = FakeStore()
    published = publish_benchmark_result(
        result_path=path,
        log_path=log,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        environment="staging",
        branch="feature/benchmark",
        git_sha="a" * 40,
        request=request,
        artifact_store=store,
    )

    assert published["log_s3_key"].endswith("/benchmark.log")
    assert store.objects[0][1].rsplit("/", 1)[0] == store.objects[1][1].rsplit("/", 1)[0]
    assert store.objects[1][3] == "text/plain"


def test_idempotency_keys_are_stable_per_report_and_distinct_per_execution(tmp_path: Path) -> None:
    path = valid_result(tmp_path)

    def publish() -> tuple[dict, list[str], FakeStore]:
        keys: list[str] = []

        def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
            keys.append(key)
            if endpoint == "/v1/runs":
                return {"run_id": BENCHMARK_RUN_ID}
            return {"ok": True}

        store = FakeStore()
        published = publish_benchmark_result(
            result_path=path,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/benchmark",
            git_sha="a" * 40,
            request=request,
            artifact_store=store,
        )
        return published, keys, store

    first, first_keys, first_store = publish()
    retry, retry_keys, retry_store = publish()
    assert retry["execution_digest"] == first["execution_digest"]
    assert retry_keys == first_keys
    assert retry_store.objects[0][1] == first_store.objects[0][1]

    later_result = json.loads(path.read_text(encoding="utf-8"))
    later_result["timestamp_utc"] = "2026-08-26T12:00:00+00:00"
    path.write_text(json.dumps(later_result), encoding="utf-8")
    later, later_keys, later_store = publish()

    assert later["result_fingerprint"] == first["result_fingerprint"]
    assert later["execution_digest"] != first["execution_digest"]
    assert later_keys[0] != first_keys[0]
    assert later_store.objects[0][1] != first_store.objects[0][1]


def test_rejects_tampered_benchmark_identity(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["configuration"]["samples"] = 101
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValidationError, match="identity fingerprint"):
        load_benchmark_result(path)


def test_rejects_metric_aggregation_the_registry_cannot_represent(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["latency_avg_ms"]["aggregation"] = "harmonic_mean"
    path.write_text(json.dumps(result), encoding="utf-8")
    calls = []

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        calls.append((method, endpoint, body, key))
        return {"run_id": BENCHMARK_RUN_ID}

    with pytest.raises(ValidationError, match="cannot be represented by the Vulcan registry"):
        publish_benchmark_result(
            result_path=path,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/benchmark",
            git_sha="a" * 40,
            request=request,
            artifact_store=FakeStore(),
        )
    assert calls == []
