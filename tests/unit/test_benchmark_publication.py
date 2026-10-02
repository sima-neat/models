from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from model_registry.benchmark_publication import (
    benchmark_identity_document,
    load_benchmark_result,
    publish_benchmark_result,
    validate_benchmark_set,
)
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


def single_metric(value: float, unit: str) -> dict:
    return {
        "status": "measured",
        "reason": None,
        "unit": unit,
        "aggregation": "single",
        "value": value,
        "minimum": value,
        "maximum": value,
        "standard_deviation": 0.0,
        "sample_count": 1,
    }


def valid_result(tmp_path: Path) -> Path:
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result-v4.json").read_text())
    result["environment"].update(
        fingerprint_version=1, kernel_release="test", machine="aarch64", cpu_count=8, cpu_governor="performance"
    )
    result["provenance"]["compile_run_id"] = COMPILE_RUN_ID
    for metric in result["metrics"].values():
        metric.update(
            status="unavailable",
            reason="test_fixture",
            value=None,
            minimum=None,
            maximum=None,
            standard_deviation=None,
            sample_count=0,
        )
    result["metrics"].update(
        {
            "latency_avg_ms": single_metric(3.6, "ms"),
            "throughput_fps": single_metric(460.0, "fps"),
            "model_load_ms": single_metric(125.0, "ms"),
            "time_to_first_inference_ms": single_metric(340.0, "ms"),
            "mla_memory_peak_delta_mib": {**single_metric(128.0, "MiB"), "aggregation": "maximum"},
        }
    )
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    path = tmp_path / "performance-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def rf_detr_result(tmp_path: Path, component_id: str, *, include_power: bool = True) -> Path:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["model"].update(
        {
            "id": "rf_detr_small",
            "build_profile_id": "modalix_int8_bf16",
            "component_id": component_id,
        }
    )
    result["provenance"]["models_commit"] = "a" * 40
    if include_power:
        result["metrics"]["avg_power_watts"] = single_metric(8.5, "W")
        result["metrics"]["energy_per_inference_joules"] = single_metric(0.03, "J")
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    destination = tmp_path.parent / component_id / "performance-result-v4.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result), encoding="utf-8")
    return destination


def test_validates_complete_rf_detr_benchmark_set_and_required_power(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for component_id in ("backbone", "transformer"):
        component_tmp = tmp_path / f"tmp-{component_id}"
        component_tmp.mkdir()
        generated = rf_detr_result(component_tmp, component_id)
        destination = results / component_id / generated.name
        destination.parent.mkdir(parents=True)
        generated.replace(destination)

    validated = validate_benchmark_set(
        repository=ROOT,
        result_root=results,
        model_id="rf_detr_small",
        build_profile_id="modalix_int8_bf16",
        component_ids=["backbone", "transformer"],
        git_sha="a" * 40,
    )

    assert validated["count"] == 2


def test_rejects_rf_detr_benchmark_without_required_power(tmp_path: Path) -> None:
    component_tmp = tmp_path / "tmp-backbone"
    component_tmp.mkdir()
    generated = rf_detr_result(component_tmp, "backbone", include_power=False)
    results = tmp_path / "results"
    destination = results / "backbone" / generated.name
    destination.parent.mkdir(parents=True)
    generated.replace(destination)

    with pytest.raises(ValidationError, match="avg_power_watts.*was not measured"):
        validate_benchmark_set(
            repository=ROOT,
            result_root=results,
            model_id="rf_detr_small",
            build_profile_id="modalix_int8_bf16",
            component_ids=["backbone"],
            git_sha="a" * 40,
        )


@pytest.mark.parametrize("version", [2, 3])
def test_loads_historical_variant_identity_results(version: int) -> None:
    path = ROOT / "artifact-browser" / "fixtures" / f"performance-result-v{version}.json"

    result = load_benchmark_result(path)

    assert result["model"]["variant_id"] == "modalix_bf16"


def test_republishes_historical_variant_identity_result(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["schema_version"] = 3
    result["identity"]["canonicalization_version"] = 3
    result["model"]["variant_id"] = result["model"].pop("build_profile_id")
    result["model"].pop("component_id")
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    path.write_text(json.dumps(result), encoding="utf-8")
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        calls.append((method, endpoint, body, key))
        if endpoint == "/v1/runs":
            return {"run_id": BENCHMARK_RUN_ID}
        return {"ok": True}

    published = publish_benchmark_result(
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

    create = calls[0][2]
    assert create is not None
    assert create["metadata"]["build_profile_id"] == "modalix_bf16"
    assert create["metadata"]["component_id"] == "model"
    assert published["build_profile_id"] == "modalix_bf16"
    assert published["component_id"] == "model"


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
    assert published["component_id"] == "model"
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
    mla_memory = next(item for item in measurements["measurements"] if item["name"] == "model.mla_memory.peak_delta")
    assert mla_memory["unit"] == "MiB"
    assert mla_memory["aggregation"] == "max"
    assert mla_memory["dimensions"]["result_aggregation"] == "maximum"
    assert mla_memory["direction"] == "lower"
    model_load = next(item for item in measurements["measurements"] if item["name"] == "model.initialization.load")
    assert model_load["aggregation"] == "raw"
    assert model_load["dimensions"]["result_aggregation"] == "single"
    assert calls[-1][2] == {"status": "completed"}
    assert store.objects[0][3] == "application/json"
    assert store.objects[0][2] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.objects[0][1].endswith(f"/{published['execution_digest']}/performance-result-v4.json")


def test_publishes_core_throughput_as_logical_inference_rate(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["throughput_fps"]["unit"] = "inferences/s"
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    path.write_text(json.dumps(result), encoding="utf-8")
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        calls.append((method, endpoint, body, key))
        if endpoint == "/v1/runs":
            return {"run_id": BENCHMARK_RUN_ID}
        return {"ok": True}

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

    body = next(body for _, endpoint, body, _ in calls if endpoint.endswith("/measurements/batch"))
    assert body is not None
    throughput = next(item for item in body["measurements"] if item["measurement_key"].endswith("-throughput_fps"))
    assert throughput["name"] == "model.inference_rate"
    assert throughput["unit"] == "inferences/s"


def test_rejects_unknown_throughput_unit(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["throughput_fps"]["unit"] = "items/s"
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValidationError, match="unsupported throughput unit 'items/s'"):
        publish_benchmark_result(
            result_path=path,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/benchmark",
            git_sha="a" * 40,
            request=lambda *_: {},
            artifact_store=FakeStore(),
        )


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
