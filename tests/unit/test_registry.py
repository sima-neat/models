from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.errors import ValidationError
from model_registry.registry import (
    RegistryPublicationError,
    S3Object,
    SigV4RegistryClient,
    _component_source_media_type,
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
    output = dist / "resnet_50" / "modalix_bf16" / "model"
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
            "build_profile_id": "modalix_bf16",
            "component_id": "model",
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
    (output.parent / "compile-outputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "resnet_50",
                "build_profile_id": "modalix_bf16",
                "compile_invocation_id": "model",
                "outputs": [
                    {
                        "component_id": "model",
                        "mpk_path": f"model/{archive.name}",
                        "mpk_sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dist


def _add_accuracy_result(dist: Path) -> Path:
    path = dist / "resnet_50" / "modalix_bf16" / "model" / "accuracy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_scope": "sampled_accuracy_smoke",
                "model": {
                    "id": "resnet_50",
                    "build_profile_id": "modalix_bf16",
                    "component_id": "model",
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


def _add_fidelity_result(dist: Path) -> Path:
    path = dist / "resnet_50" / "modalix_bf16" / "model" / "fidelity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "resnet_50",
                "build_profile_id": "modalix_bf16",
                "component_id": "model",
                "scope": "component",
                "status": "provisional",
                "dataset": {"id": "imagenet-2012", "variant": "mini", "sha256": "d" * 64},
                "selection": {
                    "method": "seeded_random_without_replacement",
                    "seed": 42,
                    "offset": 0,
                    "samples": 4,
                },
                "samples": 4,
                "metrics": [
                    {
                        "name": "logits",
                        "cosine_similarity": 0.999,
                        "normalized_rmse": 0.01,
                        "mean_absolute_error": 0.02,
                        "max_absolute_error": 0.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_canonical_sha256_is_stable() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"type": "file"}, "application/onnx"),
        ({"type": "file_set"}, "application/vnd.sima.source-file-set+json"),
        ({"type": "file", "media_type": "application/custom"}, "application/custom"),
    ],
)
def test_component_source_media_type_matches_source_shape(source: dict, expected: str) -> None:
    assert _component_source_media_type(source) == expected


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
        "/v1/repositories/sima-neat/models/taxonomies/1",
        "/v1/runs",
        "/v1/runs/00000000-0000-4000-8000-000000000001/inputs/source-component",
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
        "build_profile_id": "modalix_bf16",
        "component_id": "model",
        "artifact_kind": "mpk",
        "github_run_id": "1234",
        "license": {
            "spdx": "Apache-2.0",
            "redistribution": "allowed",
            "attribution": "Derived from microsoft/resnet-50 and distributed under Apache-2.0.",
        },
    }
    assert create["model_categories"] == {
        "taxonomy_version": 1,
        "model_type": "vision",
        "tasks": ["image_classification"],
        "modalities": {"inputs": ["image"], "outputs": ["class_scores"]},
        "architecture": {"id": "resnet_50", "paradigm": "cnn", "variant": "imagenet_1k"},
        "tags": [],
    }
    source_input = calls[2][2]
    assert source_input is not None
    assert source_input["input_type"] == "onnx"
    assert source_input["media_type"] == "application/onnx"
    assert source_input["source_reference"].startswith("hf://models/onnx-community/resnet-50-ONNX@")
    assert source_input["metadata"] == {
        "model_id": "resnet_50",
        "component_id": "model",
        "source_type": "file",
        "model_card": "https://huggingface.co/onnx-community/resnet-50-ONNX",
    }
    parameters = calls[3][2]["parameters"]
    assert parameters["compiler_version"] == "2.1.3"
    assert parameters["compile_invocation_id"] == "model"
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


def test_publish_attaches_numerical_fidelity_result(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    fidelity_path = _add_fidelity_result(dist)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        if path == "/v1/runs":
            return {"run_id": "00000000-0000-4000-8000-000000000001"}
        if path.endswith("/artifacts/numerical-fidelity"):
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
    assert f"/v1/runs/{run_id}/artifacts/numerical-fidelity" in endpoints
    assert f"/v1/runs/{run_id}/test-results/correctness/numerical-fidelity" in endpoints
    measurements = next(
        body["measurements"] for _, path, body, _ in calls if path.endswith("/measurements/batch") and body is not None
    )
    assert {measurement["name"] for measurement in measurements} == {
        "model.fidelity.cosine_similarity",
        "model.fidelity.normalized_rmse",
        "model.fidelity.mean_absolute_error",
        "model.fidelity.max_absolute_error",
    }
    maximum = next(
        measurement for measurement in measurements if measurement["name"] == "model.fidelity.max_absolute_error"
    )
    assert maximum["aggregation"] == "max"
    assert result["runs"][0]["fidelity"]["samples"] == 4
    assert store.objects[1][0] == fidelity_path
    assert store.objects[1][3] == "application/json"


def test_publish_preflights_all_evidence_before_external_writes(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    fidelity_path = _add_fidelity_result(dist)
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    fidelity["unexpected"] = True
    fidelity_path.write_text(json.dumps(fidelity), encoding="utf-8")
    calls = []
    store = FakeStore()

    with pytest.raises(ValidationError, match="Additional properties"):
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
            github_run_attempt="1",
            github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
            request=lambda *args: calls.append(args),
            artifact_store=store,
        )

    assert calls == []
    assert store.objects == []


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
    assert result["taxonomy"]["version"] == 1
    assert [call[1] for call in calls] == ["/v1/repositories/sima-neat/models/taxonomies/1"]


def test_publish_rejects_archive_not_bound_to_output_manifest(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    result_path = dist / "resnet_50" / "modalix_bf16" / "model" / "compile-result.json"
    result = json.loads(result_path.read_text())
    stale = result_path.parent / "stale.tar.gz"
    stale.write_bytes((result_path.parent / result["archive"]).read_bytes())
    result["archive"] = stale.name
    result_path.write_text(json.dumps(result))
    calls = []
    store = FakeStore()

    with pytest.raises(ValidationError, match="archive does not match the compile output manifest"):
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
            github_run_attempt="1",
            github_run_url="https://github.com/sima-neat/models/actions/runs/1234",
            request=lambda *args: calls.append(args),
            artifact_store=store,
        )

    assert calls == []
    assert store.objects == []


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

    assert result["version"] == 1
    assert result["registry"] == {"stored": True}
    assert len(calls) == 1
    assert calls[0][0:2] == ("PUT", "/v1/repositories/sima-neat/models/taxonomies/1")
