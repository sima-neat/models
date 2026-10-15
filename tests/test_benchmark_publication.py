from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from model_registry.benchmark_publication import (
    _measurements,
    benchmark_identity_document,
    load_benchmark_result,
    publish_accuracy_run,
    publish_benchmark_result,
    validate_benchmark_set,
)
from model_registry.dependencies import resolve_benchmark_runtime
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model
from model_registry.registry import S3Object, canonical_sha256
from model_registry.stage_keys import (
    ACCURACY_FINGERPRINT_VERSION,
    PERFORMANCE_FINGERPRINT_VERSION,
    accuracy_key,
    performance_key,
)

ROOT = Path(__file__).resolve().parents[1]
COMPILE_RUN_ID = "00000000-0000-4000-8000-000000000001"
BENCHMARK_RUN_ID = "00000000-0000-4000-8000-000000000002"
ENVIRONMENT_FINGERPRINT = "e" * 64
ACCURACY_RUN_ID = "00000000-0000-4000-8000-000000000006"
ACCURACY_ARTIFACT_ID = "00000000-0000-4000-8000-000000000007"
TEST_STATUS = "passed"
RF_DETR_COMPILE_RUN_IDS = {
    "backbone": "00000000-0000-4000-8000-000000000011",
    "transformer": "00000000-0000-4000-8000-000000000012",
}


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
    """A Core benchmark result, built from the browser's fixture."""
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result.json").read_text())
    result["tool"] = {"name": "pyneat.Model.benchmark", "version": "3.0.0"}
    result["model"]["build_profile_id"] = "modalix_bf16"
    result["core_report"] = {
        "core_version": "3.0.0",
        "latency_ms": 3.6,
        "fps": 460.0,
        "avg_power_watts": 8.5,
        "energy_joules": 1.0,
    }
    # Core benchmark results carry a SHA-256 environment fingerprint at version 2, the performance stage's version.
    result["environment"].update(
        fingerprint=ENVIRONMENT_FINGERPRINT,
        fingerprint_version=PERFORMANCE_FINGERPRINT_VERSION,
        kernel_release="test",
        machine="aarch64",
        cpu_count=8,
        cpu_governor="performance",
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
            "throughput_fps": single_metric(460.0, "inferences/s"),
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
    destination = tmp_path.parent / component_id / "performance-result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result), encoding="utf-8")
    return destination


def rf_detr_result_set(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    for component_id in ("backbone", "transformer"):
        component_tmp = tmp_path / f"tmp-{component_id}"
        component_tmp.mkdir()
        generated = rf_detr_result(component_tmp, component_id)
        destination = results / component_id / generated.name
        destination.parent.mkdir(parents=True)
        generated.replace(destination)
    return results


def test_validates_complete_rf_detr_benchmark_set_and_required_power(tmp_path: Path) -> None:
    validated = validate_benchmark_set(
        repository=ROOT,
        result_root=rf_detr_result_set(tmp_path),
        model_id="rf_detr_small",
        build_profile_id="modalix_int8_bf16",
        component_ids=["backbone", "transformer"],
        git_sha="a" * 40,
    )

    assert validated["count"] == 2
    assert "accuracy" not in validated


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


@pytest.mark.parametrize("version", [2, 3, 4])
def test_rejects_results_older_than_the_core_benchmark(tmp_path: Path, version: int) -> None:
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result.json").read_text())
    result["schema_version"] = version
    path = tmp_path / "performance-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValidationError, match="^benchmark result schema_version: 5 was expected"):
        load_benchmark_result(path)


def test_result_schema_requires_the_performance_fingerprint_version() -> None:
    schema = json.loads((ROOT / "schemas" / "performance-result.schema.json").read_text(encoding="utf-8"))
    environment = schema["properties"]["environment"]
    assert environment["properties"]["fingerprint_version"] == {"const": PERFORMANCE_FINGERPRINT_VERSION}
    assert {"fingerprint", "fingerprint_version"} <= set(environment["required"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fingerprint_version", 1, "environment.fingerprint_version: 2 was expected"),
        ("fingerprint", "E" * 64, "environment.fingerprint: 'E{64}' does not match"),
        ("fingerprint", None, "environment: 'fingerprint' is a required property"),
    ],
)
def test_rejects_result_environment_identity_the_stage_cannot_match(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        del result["environment"][field]
    else:
        result["environment"][field] = value
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_benchmark_result(path)


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
        repository=ROOT,
        test_status=TEST_STATUS,
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
    model = load_and_validate_model(ROOT / "models" / "resnet_50", ROOT / "schemas" / "model.schema.json")
    expected_key = performance_key(
        ROOT, model, "modalix_bf16", "model", compile_run_id=COMPILE_RUN_ID, artifact_sha256=result["model"]["sha256"]
    )
    assert create["run_fingerprint"] == expected_key
    assert create["run_fingerprint"] != published["result_fingerprint"]
    assert re.fullmatch(r"[0-9a-f]{64}", create["run_fingerprint"])
    assert create["metadata"]["performance_key"] == expected_key
    assert published["performance_key"] == expected_key
    assert create["environment"]["fingerprint"] == ENVIRONMENT_FINGERPRINT
    assert create["environment"]["fingerprint_version"] == PERFORMANCE_FINGERPRINT_VERSION == 2
    test_result = calls[4][2]
    assert test_result is not None and test_result["status"] == "passed"
    assert test_result["report_artifact_id"] == "00000000-0000-4000-8000-000000000003"
    measurements = calls[5][2]
    assert measurements is not None
    assert {item["name"] for item in measurements["measurements"]} == {
        "model.latency.avg",
        "model.inference_rate",
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
    assert store.objects[0][1].endswith(f"/{published['execution_digest']}/performance-result.json")


@pytest.mark.parametrize(
    ("result_status", "test_status", "published_status", "finalized_status"),
    [
        ("passed", "failed", "failed", "completed"),
        ("passed", "skipped", "skipped", "completed"),
        ("error", "failed", "error", "failed"),
    ],
)
def test_benchmark_test_result_carries_the_model_test_outcome(
    tmp_path: Path, result_status: str, test_status: str, published_status: str, finalized_status: str
) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["status"] = result_status
    if result_status == "error":
        result["error"] = {"type": "RuntimeError", "message": "benchmark failed"}
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
        repository=ROOT,
        test_status=test_status,
        request=request,
        artifact_store=FakeStore(),
    )

    test_result = next(body for _, endpoint, body, _ in calls if "/test-results/" in endpoint)
    assert test_result is not None and test_result["status"] == published_status
    # A failed model test still completes the run, so the measured stage is not measured again.
    assert calls[-1][2] == {"status": finalized_status}


def test_rejects_a_test_outcome_pytest_does_not_report(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="test outcome must be one of passed, failed, skipped, not 'error'"):
        publish_benchmark_result(
            result_path=valid_result(tmp_path),
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/benchmark",
            git_sha="a" * 40,
            repository=ROOT,
            test_status="error",
            request=lambda *_: {},
            artifact_store=FakeStore(),
        )


def test_publishes_core_throughput_as_logical_inference_rate(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
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
        repository=ROOT,
        test_status=TEST_STATUS,
        request=request,
        artifact_store=FakeStore(),
    )

    body = next(body for _, endpoint, body, _ in calls if endpoint.endswith("/measurements/batch"))
    assert body is not None
    throughput = next(item for item in body["measurements"] if item["measurement_key"].endswith("-throughput_fps"))
    assert throughput["name"] == "model.inference_rate"
    assert throughput["unit"] == "inferences/s"


def test_rejects_throughput_unit_that_is_not_the_inference_rate(tmp_path: Path) -> None:
    path = valid_result(tmp_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"]["throughput_fps"]["unit"] = "fps"
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValidationError, match="requires throughput_fps unit 'inferences/s'"):
        publish_benchmark_result(
            result_path=path,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/benchmark",
            git_sha="a" * 40,
            repository=ROOT,
            test_status=TEST_STATUS,
            request=lambda *_: {},
            artifact_store=FakeStore(),
        )


def test_publishes_log_beside_result(tmp_path: Path) -> None:
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
        repository=ROOT,
        test_status=TEST_STATUS,
        request=request,
        artifact_store=store,
    )

    assert published["log_s3_key"].endswith("/benchmark.log")
    assert store.objects[0][1].rsplit("/", 1)[0] == store.objects[1][1].rsplit("/", 1)[0]
    assert store.objects[1][3] == "text/plain"


def test_idempotency_keys_are_stable_per_report_and_distinct_per_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            repository=ROOT,
            test_status=TEST_STATUS,
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

    # A new stage key creates a new run even when the result itself is unchanged.
    monkeypatch.setattr("model_registry.benchmark_publication.performance_key", lambda *args, **kwargs: "f" * 64)
    _, rekeyed_keys, _ = publish()
    assert rekeyed_keys[0] != later_keys[0]


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
            repository=ROOT,
            test_status=TEST_STATUS,
            request=request,
            artifact_store=FakeStore(),
        )
    assert calls == []


def rf_detr_compilation(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Write a registered RF-DETR Small compilation: MPKs, compile-outputs.json and each compile run receipt."""
    compiled = tmp_path / "compiled"
    build = compiled / "rf_detr_small" / "modalix_int8_bf16"
    outputs = []
    digests = {}
    for component_id, run_id in RF_DETR_COMPILE_RUN_IDS.items():
        archive = build / component_id / f"rf_detr_small_{component_id}_mpk.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(f"{component_id} mpk".encode())
        digests[component_id] = hashlib.sha256(archive.read_bytes()).hexdigest()
        outputs.append(
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archive.name}",
                "mpk_sha256": digests[component_id],
            }
        )
        receipt = {
            "runs": [
                {
                    "run_id": run_id,
                    "model_id": "rf_detr_small",
                    "build_profile_id": "modalix_int8_bf16",
                    "component_id": component_id,
                }
            ]
        }
        (build / component_id / "registry-publication.json").write_text(json.dumps(receipt), encoding="utf-8")
    (build / "compile-outputs.json").write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    return compiled, digests


def rf_detr_accuracy(tmp_path: Path, artifacts: dict[str, str]) -> Path:
    report = {
        "schema_version": 2,
        "model": {
            "id": "rf_detr_small",
            "build_profile_id": "modalix_int8_bf16",
            "component_id": "transformer",
            "package_id": "default",
            "artifacts": artifacts,
        },
        "validation": {"id": "coco_accuracy", "engine": "devkit_script", "status": "ready"},
        "dataset": {
            "id": "coco-2017",
            "variant": "val",
            "split": "val2017",
            "sha256": "a40de1caf42b47c5a8976eb97efe3c83f3a41337c4d16d318eb15d65c293f2da",
            "evaluated_samples": 5000,
            "available_samples": 5000,
        },
        "metrics": {
            "coco_map": {"value": 0.4694, "unit": "ratio", "direction": "higher"},
            "coco_ap50": {"value": 0.6537, "unit": "ratio", "direction": "higher"},
        },
        "execution": {"core_version": "3.0.0", "script_sha256": "c" * 64, "models_commit": "a" * 40},
    }
    path = tmp_path / "accuracy-result.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def publish_rf_detr_accuracy(
    tmp_path: Path,
    *,
    artifacts: dict[str, str] | None = None,
    log_path: Path | None = None,
    test_status: str = TEST_STATUS,
) -> tuple[dict, list[tuple[str, str, dict | None, str]], FakeStore]:
    compiled, digests = rf_detr_compilation(tmp_path)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, endpoint: str, body: dict | None, key: str) -> dict:
        calls.append((method, endpoint, body, key))
        if endpoint == "/v1/runs":
            return {"run_id": ACCURACY_RUN_ID}
        if endpoint.endswith("/artifacts/accuracy-result"):
            return {"artifact_id": ACCURACY_ARTIFACT_ID}
        return {"ok": True}

    store = FakeStore()
    published = publish_accuracy_run(
        repository=ROOT,
        test_status=test_status,
        compiled_dir=compiled,
        result_path=rf_detr_accuracy(tmp_path, artifacts if artifacts is not None else digests),
        log_path=log_path,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        environment="staging",
        branch="feature/accuracy",
        git_sha="a" * 40,
        request=request,
        artifact_store=store,
    )
    return published, calls, store


def test_publishes_accuracy_as_its_own_run_under_the_result_component_compile_run(tmp_path: Path) -> None:
    published, calls, store = publish_rf_detr_accuracy(tmp_path)

    run = f"/v1/runs/{ACCURACY_RUN_ID}"
    assert [endpoint for _, endpoint, _, _ in calls] == [
        "/v1/runs",
        f"{run}/inputs/compiled-model-backbone",
        f"{run}/inputs/compiled-model-transformer",
        f"{run}/parameters",
        f"{run}/artifacts/accuracy-result",
        f"{run}/test-results/accuracy/coco_accuracy",
        f"{run}/measurements/batch",
        f"{run}/finalize",
    ]
    _, digests = rf_detr_compilation(tmp_path / "expected")
    model = load_and_validate_model(ROOT / "models" / "rf_detr_small", ROOT / "schemas" / "model.schema.json")
    builds = {
        component_id: {"run_id": RF_DETR_COMPILE_RUN_IDS[component_id], "artifact_sha256": digests[component_id]}
        for component_id in ("backbone", "transformer")
    }
    expected_key = accuracy_key(ROOT, model, "modalix_int8_bf16", builds)

    create = calls[0][2]
    assert create is not None
    assert create["run_type"] == "model_accuracy"
    assert create["parent_run_id"] == RF_DETR_COMPILE_RUN_IDS["transformer"]
    assert create["run_fingerprint"] == expected_key
    assert re.fullmatch(r"[0-9a-f]{64}", create["run_fingerprint"])
    assert create["metadata"]["accuracy_key"] == expected_key
    assert create["metadata"]["accuracy_id"] == "coco_accuracy"
    environment = create["environment"]
    assert environment["fingerprint_version"] == ACCURACY_FINGERPRINT_VERSION == 1
    assert environment["fingerprint"] == canonical_sha256(
        {"runtime": resolve_benchmark_runtime(ROOT), "core_version": "3.0.0"}
    )
    assert re.fullmatch(r"[0-9a-f]{64}", environment["fingerprint"])

    for component_id, (_, _, body, _) in zip(("backbone", "transformer"), calls[1:3]):
        assert body is not None
        assert body["input_type"] == "model-pack"
        assert body["source_reference"] == f"vulcan://runs/{RF_DETR_COMPILE_RUN_IDS[component_id]}/artifacts/mpk"
        assert body["sha256"] == digests[component_id]
        assert body["metadata"]["compile_run_id"] == RF_DETR_COMPILE_RUN_IDS[component_id]

    # Vulcan refuses to complete a run without its canonical parameters.
    parameters = calls[3][2]
    assert parameters is not None
    assert parameters["parameters"]["accuracy"]["id"] == "coco_accuracy"
    assert parameters["parameters"]["builds"] == builds
    assert parameters["parameters"]["core_version"] == "3.0.0"
    assert parameters["canonical_sha256"] == canonical_sha256(parameters["parameters"])

    test_result = calls[5][2]
    assert test_result is not None
    assert test_result["status"] == "passed"
    assert test_result["report_artifact_id"] == ACCURACY_ARTIFACT_ID
    assert calls[-1][2] == {"status": "completed"}
    assert published["accuracy_run_id"] == ACCURACY_RUN_ID
    assert published["compile_run_id"] == RF_DETR_COMPILE_RUN_IDS["transformer"]
    assert published["accuracy_key"] == expected_key
    assert published["evaluated_samples"] == 5000
    assert store.objects[0][3] == "application/json"


@pytest.mark.parametrize("test_status", ["failed", "skipped"])
def test_accuracy_test_result_carries_the_model_test_outcome(tmp_path: Path, test_status: str) -> None:
    _, calls, _ = publish_rf_detr_accuracy(tmp_path, test_status=test_status)

    test_result = next(body for _, endpoint, body, _ in calls if "/test-results/" in endpoint)
    assert test_result is not None and test_result["status"] == test_status
    assert calls[-1][2] == {"status": "completed"}


def test_accuracy_run_idempotency_keys_are_stable_across_retries(tmp_path: Path) -> None:
    _, first, _ = publish_rf_detr_accuracy(tmp_path / "first")
    _, retry, _ = publish_rf_detr_accuracy(tmp_path / "retry")

    keys = [key for _, _, _, key in first]
    assert keys == [key for _, _, _, key in retry]
    assert len(set(keys)) == len(keys)
    assert all(key.startswith("models-accuracy-") for key in keys)


def test_publishes_accuracy_log_before_finalizing(tmp_path: Path) -> None:
    log = tmp_path / "accuracy.log"
    log.write_text("accuracy complete\n", encoding="utf-8")

    _, calls, store = publish_rf_detr_accuracy(tmp_path, log_path=log)

    endpoints = [endpoint for _, endpoint, _, _ in calls]
    assert endpoints[-2:] == [
        f"/v1/runs/{ACCURACY_RUN_ID}/artifacts/accuracy-log",
        f"/v1/runs/{ACCURACY_RUN_ID}/finalize",
    ]
    assert store.objects[1][1].endswith("/accuracy.log")
    assert store.objects[1][3] == "text/plain"


def test_rejects_accuracy_measured_with_other_mpks(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="did not measure the compiled MPKs"):
        publish_rf_detr_accuracy(tmp_path, artifacts={"backbone": "d" * 64, "transformer": "d" * 64})


def test_rejects_accuracy_without_the_result_component_compile_run(tmp_path: Path) -> None:
    compiled, digests = rf_detr_compilation(tmp_path)
    (compiled / "rf_detr_small" / "modalix_int8_bf16" / "transformer" / "registry-publication.json").unlink()

    with pytest.raises(ValidationError, match="needs the registered transformer compile run"):
        publish_accuracy_run(
            repository=ROOT,
            test_status=TEST_STATUS,
            compiled_dir=compiled,
            result_path=rf_detr_accuracy(tmp_path, digests),
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            environment="staging",
            branch="feature/accuracy",
            git_sha="a" * 40,
            request=lambda *_: pytest.fail("no registry request is expected"),
            artifact_store=FakeStore(),
        )


def test_board_telemetry_metrics_are_published_as_measurements():
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result.json").read_text())
    measurements = {item["name"]: item for item in _measurements(result, "f" * 64)}

    assert {
        "board.power.peak", "board.mla_power.avg", "board.mla_power.peak", "board.die_temperature.avg",
        "board.die_temperature.peak", "board.som_temperature.avg", "board.som_temperature.peak",
        "board.cpu_usage.avg", "board.cpu_usage.peak",
        "model.mla_memory.avg", "model.mla_memory.peak", "board.cma_memory.avg", "board.cma_memory.peak",
        "board.linux_memory.avg", "board.linux_memory.peak",
    } <= set(measurements)
    assert measurements["board.die_temperature.peak"]["unit"] == "°C"
    assert measurements["board.die_temperature.peak"]["aggregation"] == "max"
    assert measurements["board.cpu_usage.avg"]["aggregation"] == "mean"
    assert measurements["board.power.peak"]["sample_count"] == 15


def test_latency_percentiles_and_busiest_core_are_published():
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result.json").read_text())
    measured = {"status": "measured", "reason": None, "aggregation": "raw", "minimum": None, "maximum": None,
                "standard_deviation": None, "sample_count": 1}
    result["metrics"]["latency_p99_ms"] = {**measured, "unit": "ms", "value": 3.1}
    result["metrics"]["avg_cpu_core_usage_pct"] = {**measured, "unit": "%", "value": 61.0, "aggregation": "mean"}
    result["metrics"]["peak_cpu_core_usage_pct"] = {**measured, "unit": "%", "value": 98.5, "aggregation": "max"}
    measurements = {item["name"]: item for item in _measurements(result, "f" * 64)}

    assert measurements["model.latency.p99"]["value"] == 3.1
    assert measurements["board.cpu_core_usage.avg"]["value"] == 61.0
    assert measurements["board.cpu_core_usage.peak"]["value"] == 98.5
