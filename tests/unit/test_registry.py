from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.registry import (
    RegistryPublicationError,
    S3Object,
    SigV4RegistryClient,
    canonical_sha256,
    publish_compilations,
    publish_taxonomy,
)

ROOT = Path(__file__).resolve().parents[2]


class FakeStore:
    def __init__(self) -> None:
        self.objects: list[tuple[Path, str, str, str]] = []

    def ensure_native_sha256(
        self,
        path: Path,
        key: str,
        sha256: str,
        *,
        media_type: str = "application/gzip",
    ) -> S3Object:
        self.objects.append((path, key, sha256, media_type))
        return S3Object(key=key, version_id="version-1", sha256=sha256, size_bytes=path.stat().st_size)


def _packaged_result(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    output = dist / "resnet_50" / "modalix_bf16"
    output.mkdir(parents=True)
    payload = tmp_path / "payload.elf"
    metadata = tmp_path / "fixture_mpk.json"
    payload.write_bytes(b"elf")
    metadata.write_text("{}\n", encoding="utf-8")
    archive = output / "resnet_50_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(payload, arcname="payload.elf")
        stream.add(metadata, arcname="fixture_mpk.json")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = {
        "archive": archive.name,
        "built_at": "2026-08-13T00:00:00Z",
        "schema_version": 1,
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
        "status": "success",
        "provenance": {
            "model_id": "resnet_50",
            "variant_id": "modalix_bf16",
            "target": "modalix",
            "compiler_version": "2.1.3",
            "compiler_label": "model-compiler-2.1.3-sha-fixture",
            "source_sha256": "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50",
            "git_sha": "a" * 40,
            "environment": "staging",
            "build_fingerprint": "b" * 64,
        },
    }
    (output / "compile-result.json").write_text(json.dumps(result), encoding="utf-8")
    return dist


def _add_accuracy_result(dist: Path) -> Path:
    path = dist / "resnet_50" / "modalix_bf16" / "accuracy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_scope": "sampled_accuracy_smoke",
                "model": {
                    "id": "resnet_50",
                    "variant_id": "modalix_bf16",
                    "source_sha256": "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50",
                },
                "dataset": {
                    "id": "imagenet-2012",
                    "variant": "mini",
                    "split": "validation",
                    "sha256": "d" * 64,
                    "evaluated_samples": 500,
                    "available_samples": 5000,
                },
                "evaluation": {"profile": "regular", "status": "provisional"},
                "metrics": {
                    "candidate": {"top1_accuracy": 0.762, "top5_accuracy": 0.934},
                    "reference": {"top1_accuracy": 0.798, "top5_accuracy": 0.95},
                    "comparison": {"top1_agreement": 0.886, "top1_delta": -0.036, "top5_delta": -0.016},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_canonical_sha256_is_stable() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_registry_errors_identify_the_failed_endpoint() -> None:
    def reject(method: str, path: str, body: dict | None, key: str) -> dict:
        raise RegistryPublicationError("artifact registry HTTP 403 unknown_error: request rejected")

    client = SigV4RegistryClient("https://registry.example.invalid", "us-west-2", request=reject)
    with pytest.raises(RegistryPublicationError, match=r"POST /v1/runs: artifact registry HTTP 403"):
        client.request("POST", "/v1/runs", {}, "create")


def test_publish_records_one_run_per_compiled_variant(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        return {"run_id": "00000000-0000-4000-8000-000000000001"} if path == "/v1/runs" else {"ok": True}

    store = FakeStore()
    result = publish_compilations(
        repository=ROOT,
        dist_dir=dist,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        kms_key_id="alias/artifacts",
        environment="staging",
        branch="feature/models",
        git_sha="a" * 40,
        github_run_id="1234",
        github_run_attempt="1",
        github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
        request=request,
        artifact_store=store,
    )

    assert result["count"] == 1
    assert [path for _, path, _, _ in calls] == [
        "/v1/repositories/sima-neat/models/taxonomies/2",
        "/v1/runs",
        "/v1/runs/00000000-0000-4000-8000-000000000001/inputs/source-model",
        "/v1/runs/00000000-0000-4000-8000-000000000001/parameters",
        "/v1/runs/00000000-0000-4000-8000-000000000001/artifacts/mpk",
        "/v1/runs/00000000-0000-4000-8000-000000000001/test-results/compile/mpk-validation",
        "/v1/runs/00000000-0000-4000-8000-000000000001/finalize",
    ]
    taxonomy = calls[0][2]
    assert taxonomy is not None
    assert taxonomy["taxonomy"]["tasks"]["image_classification"]["display_name"] == "Image Classification"
    assert canonical_sha256(taxonomy["taxonomy"]) == taxonomy["canonical_sha256"]
    create = calls[1][2]
    assert create is not None
    assert create["run_type"] == "model_compile"
    assert create["metadata"] == {
        "model_id": "resnet_50",
        "variant_id": "modalix_bf16",
        "artifact_kind": "mpk",
        "github_run_id": "1234",
        "license": {
            "spdx": "Apache-2.0",
            "redistribution": "allowed",
            "attribution": "Derived from microsoft/resnet-50 and distributed under Apache-2.0.",
        },
    }
    assert create["model_categories"] == {
            "taxonomy_version": 2,
        "model_type": "vision",
        "tasks": ["image_classification"],
        "modalities": {"inputs": ["image"], "outputs": ["class_scores"]},
        "architecture": {"id": "resnet_50", "paradigm": "cnn", "variant": "imagenet_1k"},
        "tags": [],
    }
    source_input = calls[2][2]
    assert source_input is not None
    assert source_input["source_reference"].startswith("hf://models/onnx-community/resnet-50-ONNX@")
    assert source_input["metadata"] == {
        "model_id": "resnet_50",
        "model_card": "https://huggingface.co/onnx-community/resnet-50-ONNX",
    }
    parameters = calls[3][2]["parameters"]
    assert parameters["compiler_version"] == "2.1.3"
    assert parameters["target"] == "modalix"
    artifact = calls[4][2]
    assert artifact["s3_version_id"] == "version-1"
    assert artifact["s3_key"].startswith("models/feature%2Fmodels/aaaaaaaaaaaa/")
    assert store.objects[0][1] == artifact["s3_key"]


def test_publish_attaches_accuracy_artifact_test_result_and_measurements(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    accuracy_path = _add_accuracy_result(dist)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        if path == "/v1/runs":
            return {"run_id": "00000000-0000-4000-8000-000000000001"}
        if path.endswith("/artifacts/accuracy-result"):
            return {"artifact_id": "00000000-0000-4000-8000-000000000002"}
        return {"ok": True}

    store = FakeStore()
    result = publish_compilations(
        repository=ROOT,
        dist_dir=dist,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        kms_key_id=None,
        environment="staging",
        branch="develop",
        git_sha="a" * 40,
        github_run_id="1234",
        github_run_attempt="1",
        github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
        request=request,
        artifact_store=store,
    )

    run_id = "00000000-0000-4000-8000-000000000001"
    endpoints = [path for _, path, _, _ in calls]
    assert f"/v1/runs/{run_id}/artifacts/accuracy-result" in endpoints
    assert f"/v1/runs/{run_id}/test-results/accuracy/sampled_accuracy_smoke" in endpoints
    assert f"/v1/runs/{run_id}/measurements/batch" in endpoints
    assert endpoints.index(f"/v1/runs/{run_id}/measurements/batch") < endpoints.index(f"/v1/runs/{run_id}/finalize")
    artifact_call = next(body for _, path, body, _ in calls if path.endswith("/artifacts/accuracy-result"))
    assert artifact_call is not None
    assert artifact_call["metadata"]["evaluated_samples"] == 500
    assert artifact_call["metadata"]["candidate_top1_accuracy"] == 0.762
    test_call = next(body for _, path, body, _ in calls if "/test-results/accuracy/" in path)
    assert test_call is not None
    assert test_call["report_artifact_id"] == "00000000-0000-4000-8000-000000000002"
    measurement_call = next(body for _, path, body, _ in calls if path.endswith("/measurements/batch"))
    assert measurement_call is not None
    assert {item["name"] for item in measurement_call["measurements"]} == {
        "model.accuracy.candidate.top1",
        "model.accuracy.candidate.top5",
        "model.accuracy.reference.top1",
        "model.accuracy.reference.top5",
        "model.accuracy.top1_agreement",
        "model.accuracy.top1_delta",
        "model.accuracy.top5_delta",
    }
    assert all(item["sample_count"] == 500 for item in measurement_call["measurements"])
    assert result["runs"][0]["accuracy"]["evaluated_samples"] == 500
    assert store.objects[1][0] == accuracy_path
    assert store.objects[1][3] == "application/json"


def test_publish_does_not_record_inherited_catalog_entries(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    calls: list[tuple[str, str, dict | None, str]] = []

    result = publish_compilations(
        repository=ROOT,
        dist_dir=dist,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="artifacts",
        kms_key_id=None,
        environment="staging",
        branch="feature/docs",
        git_sha="a" * 40,
        github_run_id="1234",
        github_run_attempt="1",
        github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
        request=lambda *args: calls.append(args) or {},
        artifact_store=FakeStore(),
    )

    assert result["count"] == 0
    assert result["runs"] == []
    assert result["taxonomy"]["version"] == 2
    assert [call[1] for call in calls] == ["/v1/repositories/sima-neat/models/taxonomies/2"]


def test_publish_retry_reuses_idempotency_keys_across_run_attempts(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)

    def publish(attempt: str) -> list[tuple[str, str, dict | None, str]]:
        calls: list[tuple[str, str, dict | None, str]] = []

        def request(method: str, path: str, body: dict | None, key: str) -> dict:
            calls.append((method, path, body, key))
            return {"run_id": "00000000-0000-4000-8000-000000000001"} if path == "/v1/runs" else {"ok": True}

        publish_compilations(
            repository=ROOT,
            dist_dir=dist,
            api_url="https://registry.example.invalid",
            aws_region="us-west-2",
            bucket="artifacts",
            kms_key_id=None,
            environment="staging",
            branch="develop",
            git_sha="a" * 40,
            github_run_id="1234",
            github_run_attempt=attempt,
            github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
            request=request,
            artifact_store=FakeStore(),
        )
        return calls

    assert publish("1") == publish("2")


def test_publish_taxonomy_is_independent_of_model_compilation() -> None:
    calls: list[tuple[str, str, dict | None, str]] = []

    result = publish_taxonomy(
        repository=ROOT,
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        request=lambda *args: calls.append(args) or {"stored": True},
    )

    assert result["version"] == 2
    assert result["registry"] == {"stored": True}
    assert len(calls) == 1
    assert calls[0][0:2] == ("PUT", "/v1/repositories/sima-neat/models/taxonomies/2")
