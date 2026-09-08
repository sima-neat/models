from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.catalog import build_matrix, component_build_fingerprint, discover_catalog
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.registry import (
    RegistryPublicationError,
    S3Object,
    SigV4RegistryClient,
    _compile_timing,
    _component_source_media_type,
    build_catalog_snapshot,
    canonical_sha256,
    publish_catalog,
    publish_compilations,
    stage_catalog_build,
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
            "build_fingerprint": component_build_fingerprint(ROOT, "resnet_50", "model", "modalix_bf16", "2.1.3"),
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


@pytest.mark.parametrize("with_timing", [False, True])
def test_publish_records_one_run_per_compiled_variant(tmp_path: Path, monkeypatch, with_timing: bool) -> None:
    dist = _packaged_result(tmp_path)
    timing = {
        "scope": "compile_recipe",
        "started_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T01:29:52+00:00",
    }
    if with_timing:
        (dist / "resnet_50/modalix_bf16/compile-timing.json").write_text(json.dumps(timing))
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
        "/v1/runs",
        "/v1/runs/00000000-0000-4000-8000-000000000001/inputs/source-component",
        "/v1/runs/00000000-0000-4000-8000-000000000001/parameters",
        "/v1/runs/00000000-0000-4000-8000-000000000001/artifacts/mpk",
        "/v1/runs/00000000-0000-4000-8000-000000000001/test-results/compile/mpk-validation",
        "/v1/runs/00000000-0000-4000-8000-000000000001/finalize",
    ]
    create = calls[0][2]
    assert create is not None
    assert create["run_type"] == "model_compile"
    if with_timing:
        assert create["metadata"].pop("compile_timing") == timing
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
    assert "model_categories" not in create
    source_input = calls[1][2]
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
    parameters = calls[2][2]["parameters"]
    assert parameters["compiler_version"] == "2.1.3"
    assert parameters["compile_invocation_id"] == "model"
    assert parameters["target"] == "modalix"
    artifact = calls[3][2]
    assert result["runs"][0]["artifact_sha256"] == artifact["artifact_sha256"]
    assert artifact["s3_version_id"] == "version-1"
    assert artifact["s3_key"].startswith("models/feature%2Fmodels/aaaaaaaaaaaa/")
    assert store.objects[0][1] == artifact["s3_key"]

    receipt_path = dist / "resnet_50" / "modalix_bf16" / "model" / "registry-publication.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    catalog = discover_catalog(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["version"]
    selected = build_matrix(
        catalog,
        ["resnet_50"],
        ["modalix_bf16"],
        repository=ROOT,
        compiler_version=compiler_version,
    )
    monkeypatch.setattr("model_registry.registry.build_matrix", lambda *args, **kwargs: selected)
    snapshot = build_catalog_snapshot(
        repository=ROOT,
        branch="feature/models",
        git_sha="a" * 40,
        compiled_dir=dist,
    )
    resnet = next(model for model in snapshot["models"] if model["id"] == "resnet_50")
    assert resnet["builds"][0]["artifact_sha256"] == artifact["artifact_sha256"]


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
    assert calls == []


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


def _matching_catalog() -> dict:
    catalog = discover_catalog(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["version"]
    first = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="a" * 40,
        compiled_dir=None,
        previous={
            "models": [
                {
                    "id": model_id,
                    "builds": [
                        {
                            "build_profile_id": entry["build_profile_id"],
                            "component_id": component_id,
                            "compile_fingerprint": entry["build_fingerprint"],
                            "artifact_sha256": "e" * 64,
                            "run_id": f"00000000-0000-4000-8000-{index:012d}",
                        }
                        for entry in build_matrix(
                            catalog,
                            [model_id],
                            repository=ROOT,
                            compiler_version=compiler_version,
                        )["include"]
                        for component_id in entry["component_ids"]
                    ],
                }
                for index, model_id in enumerate(catalog, 1)
            ]
        },
    )

    return first


def test_catalog_snapshot_reuses_unchanged_compile_runs() -> None:
    first = _matching_catalog()
    assert first["catalog_revision"] == "a" * 40
    assert len(first["taxonomy_sha256"]) == 64
    assert "version" not in first["taxonomy"]
    assert all("taxonomy_version" not in model["categories"] for model in first["models"])


def test_publish_catalog_reads_current_then_publishes_complete_snapshot(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None, str]] = []
    read_urls: list[str] = []

    previous = {"models": []}
    monkeypatch.setattr(
        "model_registry.registry.build_catalog_snapshot",
        lambda **kwargs: {
            "schema_version": 1,
            "repository": "sima-neat/models",
            "git_ref": "refs/heads/develop",
            "catalog_revision": "a" * 40,
            "taxonomy_sha256": "b" * 64,
            "taxonomy": {},
            "models": [],
        },
    )

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        return {"stored": True}

    def read_request(url: str) -> dict:
        read_urls.append(url)
        return {"catalog": {"document": previous}}

    result = publish_catalog(
        repository=ROOT,
        api_url="https://registry.example.invalid",
        read_api_url="https://models-registry.example.invalid",
        aws_region="us-west-2",
        branch="develop",
        git_sha="a" * 40,
        request=request,
        read_request=read_request,
    )

    assert result["registry"] == {"stored": True}
    assert read_urls == [
        "https://models-registry.example.invalid/v1/repositories/sima-neat/models/"
        "catalogs/current?git_ref=refs%2Fheads%2Fdevelop"
    ]
    assert calls[0][0:2] == ("PUT", "/v1/repositories/sima-neat/models/catalogs/" + "a" * 40)


def test_stage_catalog_build_restores_and_verifies_durable_mpk(tmp_path: Path) -> None:
    dist = _packaged_result(tmp_path)
    source_archive = next(dist.glob("resnet_50/modalix_bf16/model/*_mpk.tar.gz"))
    payload = source_archive.read_bytes()
    run_id = "00000000-0000-4000-8000-000000000001"
    catalog = {
        "catalog_revision": "a" * 40,
        "models": [
            {
                "id": "resnet_50",
                "builds": [
                    {
                        "build_profile_id": "modalix_bf16",
                        "compile_invocation_id": "model",
                        "component_id": "model",
                        "run_id": run_id,
                        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ],
    }

    def read(url: str) -> dict:
        if "/catalogs/current?" in url:
            return {"catalog": {"document": catalog}}
        if url.endswith("/artifacts/mpk"):
            return {
                "artifact": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "download_url": "https://download.example.invalid/mpk",
                }
            }
        if url.endswith(run_id):
            return {"run": {"toolchain_version": "2.1.3"}}
        raise AssertionError(url)

    result = stage_catalog_build(
        repository=ROOT,
        output_dir=tmp_path / "staged",
        api_url="https://registry.example.invalid",
        branch="develop",
        catalog_revision="a" * 40,
        model_id="resnet_50",
        build_profile_id="modalix_bf16",
        compile_invocation_id="model",
        read_request=read,
        download_request=lambda url: payload,
    )

    assert result["count"] == 1
    profile = tmp_path / "staged" / "resnet_50" / "modalix_bf16"
    assert (profile / "registry-staged.json").is_file()
    assert (profile / "model" / "registry-publication.json").is_file()


@pytest.mark.parametrize("with_new_build", [False, True])
def test_new_branch_inherits_base_catalog_and_prefers_new_receipts(tmp_path: Path, with_new_build: bool) -> None:
    base = _matching_catalog()
    expected = json.loads(json.dumps(base["models"]))
    compiled_dir = None
    if with_new_build:
        model = base["models"][0]
        build = model["builds"][0]
        # This target invocation changed: the base's build is no longer reusable.
        target_fingerprint = build["compile_fingerprint"]
        build["compile_fingerprint"] = "0" * 64
        compiled_dir = tmp_path / "compiled"
        receipt_dir = compiled_dir / model["id"] / build["build_profile_id"] / build["component_id"]
        receipt_dir.mkdir(parents=True)
        receipt = {
            **build,
            "model_id": model["id"],
            "compile_fingerprint": target_fingerprint,
            "run_id": "00000000-0000-4000-8000-999999999999",
            "artifact_sha256": "f" * 64,
        }
        (receipt_dir / "registry-publication.json").write_text(json.dumps({"runs": [receipt]}))
        expected[0]["builds"][0]["run_id"] = receipt["run_id"]
        expected[0]["builds"][0]["artifact_sha256"] = receipt["artifact_sha256"]
    reads = []
    writes = []

    def read(url: str) -> dict:
        reads.append(url)
        if url.endswith("refs%2Fheads%2Ffeature%2Fnew-model"):
            raise RegistryPublicationError("HTTP 404 catalog_not_found: Model catalog was not found")
        assert url.endswith("refs%2Fheads%2Fdevelop")
        return {"catalog": {"document": base}}

    result = publish_catalog(
        repository=ROOT,
        api_url="https://write.invalid",
        read_api_url="https://read.invalid",
        aws_region="us-west-2",
        branch="feature/new-model",
        base_branch="develop",
        git_sha="b" * 40,
        compiled_dir=compiled_dir,
        read_request=read,
        request=lambda *args: writes.append(args) or {"stored": True},
    )
    assert len(reads) == 2
    assert result["catalog"]["git_ref"] == "refs/heads/feature/new-model"
    assert result["catalog"]["catalog_revision"] == "b" * 40
    assert len(writes) == 1
    assert writes[0][0] == "PUT"
    assert writes[0][2] == result["catalog"]
    for actual_model, expected_model in zip(result["catalog"]["models"], expected, strict=True):
        assert actual_model["id"] == expected_model["id"]
        for actual, prior in zip(actual_model["builds"], expected_model["builds"], strict=True):
            for field in ("run_id", "compile_fingerprint", "artifact_sha256"):
                assert actual[field] == prior[field]


def test_existing_branch_catalog_takes_precedence_over_base() -> None:
    current = _matching_catalog()
    reads = []

    def read(url: str) -> dict:
        reads.append(url)
        assert url.endswith("refs%2Fheads%2Ffeature%2Fexisting")
        return {"catalog": {"document": current}}

    result = publish_catalog(
        repository=ROOT,
        api_url="https://write.invalid",
        read_api_url="https://read.invalid",
        aws_region="us-west-2",
        branch="feature/existing",
        base_branch="develop",
        git_sha="b" * 40,
        read_request=read,
        request=lambda *args: {"stored": True},
    )
    assert len(reads) == 1
    assert result["catalog"]["models"] == current["models"]


@pytest.mark.parametrize("failure", ["missing", "fingerprint", "checksum"])
def test_unusable_base_catalog_blocks_publication(failure: str) -> None:
    base = _matching_catalog()
    if failure == "fingerprint":
        base["models"][0]["builds"][0]["compile_fingerprint"] = "0" * 64
    elif failure == "checksum":
        base["models"][0]["builds"][0].pop("artifact_sha256")
    writes = []

    def read(url: str) -> dict:
        if failure == "missing" or url.endswith("refs%2Fheads%2Ffeature%2Fnew-model"):
            raise RegistryPublicationError("HTTP 404 catalog_not_found: missing")
        return {"catalog": {"document": base}}

    with pytest.raises(ValidationError, match="no immutable"):
        publish_catalog(
            repository=ROOT,
            api_url="https://write.invalid",
            read_api_url="https://read.invalid",
            aws_region="us-west-2",
            branch="feature/new-model",
            base_branch="develop",
            git_sha="b" * 40,
            read_request=read,
            request=lambda *args: writes.append(args),
        )
    assert writes == []


@pytest.mark.parametrize("base_branch", [None, "develop"])
def test_catalog_inheritance_does_not_retry_same_branch(base_branch: str | None) -> None:
    reads = []

    def read(url: str) -> dict:
        reads.append(url)
        raise RegistryPublicationError("HTTP 404 catalog_not_found: missing")

    with pytest.raises(ValidationError, match="no immutable"):
        publish_catalog(
            repository=ROOT,
            api_url="https://write.invalid",
            read_api_url="https://read.invalid",
            aws_region="us-west-2",
            branch="develop",
            base_branch=base_branch,
            git_sha="b" * 40,
            read_request=read,
        )
    assert len(reads) == 1


@pytest.mark.parametrize("failing_branch", ["feature/new-model", "develop"])
@pytest.mark.parametrize("error", ["HTTP 503 unavailable", "HTTP 404 repository_not_found"])
def test_catalog_inheritance_propagates_registry_errors(failing_branch: str, error: str) -> None:
    reads = []
    writes = []

    def read(url: str) -> dict:
        reads.append(url)
        if url.endswith(failing_branch.replace("/", "%2F")):
            raise RegistryPublicationError(error)
        raise RegistryPublicationError("HTTP 404 catalog_not_found: missing")

    with pytest.raises(RegistryPublicationError, match=error):
        publish_catalog(
            repository=ROOT,
            api_url="https://write.invalid",
            read_api_url="https://read.invalid",
            aws_region="us-west-2",
            branch="feature/new-model",
            base_branch="develop",
            git_sha="b" * 40,
            read_request=read,
            request=lambda *args: writes.append(args),
        )
    assert len(reads) == (1 if failing_branch == "feature/new-model" else 2)
    assert writes == []


def test_incremental_catalog_initializes_without_compilation_receipts() -> None:
    snapshot = build_catalog_snapshot(repository=ROOT, branch="feature/progress", git_sha="a" * 40, incremental=True)
    assert snapshot["schema_version"] == 2
    assert snapshot["pending_builds"]
    assert all(not model["builds"] for model in snapshot["models"])


def _write_catalog_receipt(directory: Path, model: dict, build: dict) -> None:
    component_dir = directory / model["id"] / build["build_profile_id"] / build["component_id"]
    component_dir.mkdir(parents=True, exist_ok=True)
    (component_dir / "registry-publication.json").write_text(
        json.dumps(
            {
                "runs": [{**build, "model_id": model["id"]}],
            }
        )
    )


def test_incremental_publication_retries_without_losing_another_group(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("model_registry.registry.time.sleep", lambda _: None)
    ready = _matching_catalog()
    first_model = ready["models"][0]
    first_build = first_model["builds"][0]
    second_model = ready["models"][1]
    second_build = second_model["builds"][0]
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    _write_catalog_receipt(first_dir, first_model, first_build)
    for component_build in second_model["builds"]:
        if (component_build["build_profile_id"], component_build["compile_invocation_id"]) == (
            second_build["build_profile_id"],
            second_build["compile_invocation_id"],
        ):
            _write_catalog_receipt(second_dir, second_model, component_build)
    current = build_catalog_snapshot(repository=ROOT, branch="feature/progress", git_sha="a" * 40, incremental=True)
    attempts = []

    def read(url):
        return {"catalog": {"document": current, "canonical_sha256": canonical_sha256(current)}}

    def write(method, path, body, key):
        nonlocal current
        attempts.append(body)
        assert method == "POST" and path.endswith("/snapshots")
        if len(attempts) == 1:
            current = build_catalog_snapshot(
                repository=ROOT,
                branch="feature/progress",
                git_sha="a" * 40,
                incremental=True,
                previous=current,
                compiled_dir=second_dir,
            )
            raise RegistryPublicationError("HTTP 409 catalog_snapshot_conflict")
        assert body["previous_snapshot"] == canonical_sha256(current)
        current = body["catalog"]
        return {"stored": True}

    result = publish_catalog(
        repository=ROOT,
        api_url="https://write.invalid",
        read_api_url="https://read.invalid",
        aws_region="us-west-2",
        branch="feature/progress",
        git_sha="a" * 40,
        incremental=True,
        compiled_dir=first_dir,
        request=write,
        read_request=read,
    )
    assert len(attempts) == 2
    runs = {b["run_id"] for m in result["catalog"]["models"] for b in m["builds"]}
    assert {first_build["run_id"], second_build["run_id"]} <= runs
    assert result["catalog"]["pending_builds"]


def test_late_incremental_publication_cannot_replace_new_revision() -> None:
    current = _matching_catalog()
    with pytest.raises(RegistryPublicationError, match="revision is not current"):
        publish_catalog(
            repository=ROOT,
            api_url="https://write.invalid",
            read_api_url="https://read.invalid",
            aws_region="us-west-2",
            branch="develop",
            git_sha="b" * 40,
            incremental=True,
            read_request=lambda _: {"catalog": {"document": current, "canonical_sha256": canonical_sha256(current)}},
            request=lambda *args: pytest.fail("must not publish stale revision"),
        )


def test_incremental_catalog_keeps_previous_build_while_replacement_is_pending() -> None:
    previous = _matching_catalog()
    old = previous["models"][0]["builds"][0]
    old["compile_fingerprint"] = "0" * 64
    snapshot = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="b" * 40,
        incremental=True,
        previous=previous,
    )
    assert snapshot["models"][0]["builds"][0] == old
    assert any(item["model_id"] == previous["models"][0]["id"] for item in snapshot["pending_builds"])



@pytest.mark.parametrize("timing", [
    {"scope": "workflow", "started_at": "2026-08-13T00:00:00Z", "finished_at": "2026-08-13T01:00:00Z"},
    {"scope": "compile_recipe", "started_at": "2026-08-13T02:00:00Z", "finished_at": "2026-08-13T01:00:00Z"},
    {"scope": "compile_recipe", "started_at": "2026-08-13T00:00:00", "finished_at": "2026-08-13T01:00:00"},
    {"scope": "compile_recipe", "started_at": "invalid", "finished_at": "2026-08-13T01:00:00Z"},
    {},
    None,
])
def test_compile_timing_rejects_invalid_intervals(tmp_path: Path, timing) -> None:
    path = tmp_path / "compile-timing.json"
    path.write_text(json.dumps(timing))
    with pytest.raises(ValidationError, match="invalid compile timing"):
        _compile_timing(path)


def test_compile_timing_supports_legacy_outputs_and_utc_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "compile-timing.json"
    assert _compile_timing(path) is None
    path.write_text(json.dumps({
        "scope": "compile_recipe", "started_at": "2026-08-13T00:00:00Z", "finished_at": "2026-08-13T01:29:52Z",
    }))
    assert _compile_timing(path) == {
        "scope": "compile_recipe", "started_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T01:29:52+00:00",
    }
