from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.compile_matrix import build_matrix
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.fingerprints import benchmark_fingerprint, compile_fingerprint
from model_registry.model import load_model, load_models
from model_registry.package import assemble_package
from model_registry.registry import (
    RegistryPublicationError,
    S3Object,
    SigV4RegistryClient,
    _compile_timing,
    _component_source_media_type,
    _package_media_type,
    build_catalog_snapshot,
    canonical_sha256,
    publish_catalog,
    publish_compilations,
    publish_model_documentation,
    publish_packages,
    stage_catalog_build,
)

ROOT = Path(__file__).resolve().parents[1]
COMPILER = resolve_model_compiler(ROOT)
COMPILER_VERSION = COMPILER["version"]
COMPILER_IDENTITY = COMPILER["identity"]


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


def test_publish_model_documentation_creates_finalized_versioned_runs() -> None:
    calls: list[tuple[str, str, dict | None, str]] = []
    store = FakeStore()
    next_run = 0

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        nonlocal next_run
        calls.append((method, path, body, key))
        if method == "POST" and path == "/v1/runs":
            next_run += 1
            return {"run_id": f"00000000-0000-4000-8000-{next_run:012d}"}
        return {"stored": True}

    receipts = publish_model_documentation(
        repository=ROOT,
        api_url="https://writer.example.invalid",
        aws_region="us-west-2",
        bucket="model-artifacts",
        kms_key_id="alias/models",
        environment="staging",
        branch="feat/model cards",
        git_sha="a" * 40,
        github_run_url="https://github.example.invalid/actions/runs/123",
        request=request,
        artifact_store=store,
    )

    models = load_models(ROOT)
    assert set(receipts) == set(models)
    assert len(store.objects) == len(receipts) * 2
    assert len(calls) == len(receipts) * 5
    for model_id, receipt in receipts.items():
        assert receipt["format"] == "markdown"
        assert receipt["artifact_name"] == "readme"
        assert receipt["source_path"] == f"models/{model_id}/README.md"
        assert receipt["source_commit"] == "a" * 40
        assert receipt["hero"]["artifact_name"] == "hero"
        assert receipt["hero"]["media_type"] == "image/png"
        assert receipt["hero"]["source_path"] == f"models/{model_id}/hero.png"
        assert 50 <= len(receipt["overview_markdown"].split()) <= 120
    assert all("feat%2Fmodel%20cards" in key for _, key, _, _ in store.objects)
    artifact_calls = [call for call in calls if call[0:2] != ("POST", "/v1/runs") and "/artifacts/" in call[1]]
    parameter_calls = [call for call in calls if call[0] == "PUT" and call[1].endswith("/parameters")]
    assert all(call[2]["parameters"]["overview_markdown"] in {
        receipt["overview_markdown"] for receipt in receipts.values()
    } for call in parameter_calls)
    assert all(call[2]["parameters"]["hero"]["markdown_path"] == "hero.png" for call in parameter_calls)
    assert {call[2]["artifact_type"] for call in artifact_calls} == {"model-readme", "model-readme-hero"}
    assert {call[2]["media_type"] for call in artifact_calls} == {"text/markdown; charset=utf-8", "image/png"}
    assert all(call[2]["retention_class"] == "branch" for call in artifact_calls)


def test_publish_model_documentation_keys_bind_create_request_identity() -> None:
    def create_keys(environment: str, github_run_url: str) -> list[str]:
        calls: list[tuple[str, str, dict | None, str]] = []

        def request(method: str, path: str, body: dict | None, key: str) -> dict:
            calls.append((method, path, body, key))
            return {"run_id": "00000000-0000-4000-8000-000000000001"} if path == "/v1/runs" else {"ok": True}

        publish_model_documentation(
            repository=ROOT,
            api_url="https://writer.example.invalid",
            aws_region="us-west-2",
            bucket="model-artifacts",
            kms_key_id=None,
            environment=environment,
            branch="develop",
            git_sha="a" * 40,
            github_run_url=github_run_url,
            request=request,
            artifact_store=FakeStore(),
        )
        return [key for method, path, _, key in calls if method == "POST" and path == "/v1/runs"]

    first = create_keys("staging", "https://github.example.invalid/actions/runs/123")
    assert first == create_keys("staging", "https://github.example.invalid/actions/runs/123")
    assert first != create_keys("staging", "https://github.example.invalid/actions/runs/456")
    assert first != create_keys("production", "https://github.example.invalid/actions/runs/123")


COMPILE_TIMING = {
    "scope": "compile_recipe",
    "started_at": "2026-08-13T00:00:00+00:00",
    "finished_at": "2026-08-13T01:29:52+00:00",
}


def _packaged_result(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    output = dist / "resnet_50" / "w16a16" / "model"
    output.mkdir(parents=True)
    payload = tmp_path / "payload.elf"
    metadata = tmp_path / "fixture_mpk.json"
    payload.write_bytes(b"elf")
    metadata.write_text("{}\n", encoding="utf-8")
    archive = output / "resnet_50_w16a16_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(payload, arcname="payload.elf")
        stream.add(metadata, arcname="fixture_mpk.json")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = {
        "archive": archive.name,
        "built_at": "2026-08-13T00:00:00Z",
        "sha256": digest,
        "size_bytes": archive.stat().st_size,
        "status": "success",
        "provenance": {
            "model_id": "resnet_50",
            "build_profile_id": "w16a16",
            "component_id": "model",
            "target": "modalix",
            "compiler_version": COMPILER_VERSION,
            "compiler_label": COMPILER["label"],
            "source_sha256": "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50",
            "git_sha": "a" * 40,
            "environment": "staging",
            "compile_fingerprint": compile_fingerprint(ROOT, "resnet_50", "w16a16", "w16a16", COMPILER_IDENTITY),
        },
    }
    (output / "compile-result.json").write_text(json.dumps(result), encoding="utf-8")
    (output.parent / "compile-outputs.json").write_text(
        json.dumps(
            {
                "model_id": "resnet_50",
                "build_profile_id": "w16a16",
                "compile_invocation_id": "w16a16",
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
    (output.parent / "compile-timing.json").write_text(json.dumps(COMPILE_TIMING), encoding="utf-8")
    # resnet_50 declares numerical fidelity, so its package publication requires the profile's report.
    _add_fidelity_result(dist)
    return dist


def _add_fidelity_result(dist: Path) -> Path:
    config = load_model(ROOT / "models" / "resnet_50")["numerical_fidelity"]
    path = dist / "resnet_50" / "w16a16" / "fidelity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "resnet_50",
                "build_profile_id": "w16a16",
                "status": config["status"],
                "dataset": {key: config["dataset"][key] for key in ("id", "variant", "sha256")},
                "selection": config["selection"],
                "samples": config["selection"]["samples"],
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


def _assembled_package(tmp_path: Path) -> tuple[Path, dict]:
    compiled = _packaged_result(tmp_path)
    component_dir = compiled / "resnet_50/w16a16/model"
    (component_dir / "registry-publication.json").write_text(
        json.dumps({"runs": [{
            "run_id": "component-run", "model_id": "resnet_50",
            "build_profile_id": "w16a16", "component_id": "model",
        }]}),
        encoding="utf-8",
    )
    packages = tmp_path / "packages"
    assembled = assemble_package(
        repository=ROOT, compiled_dir=compiled, output_dir=packages,
        model_id="resnet_50", build_profile_id="w16a16", package_id="default",
    )
    return packages, assembled


def _publish_packages(packages: Path, request, store: FakeStore) -> dict:
    return publish_packages(
        repository=ROOT, package_dir=packages, api_url="https://registry.example.invalid", aws_region="us-west-2",
        bucket="artifacts", kms_key_id=None, environment="staging", branch="feature/models", git_sha="a" * 40,
        github_run_id="1234", github_run_attempt="1", github_run_url="https://example.invalid/run",
        request=request, artifact_store=store,
    )


def test_publish_full_package_creates_a_package_run_and_uploads_the_mpk(tmp_path: Path) -> None:
    packages, assembled = _assembled_package(tmp_path)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        return {"run_id": "package-run"} if path == "/v1/runs" else {"ok": True}

    store = FakeStore()
    result = _publish_packages(packages, request, store)

    assert result["count"] == 1
    assert result["packages"][0]["run_id"] == "package-run"
    assert result["packages"][0]["artifact_sha256"] == assembled["sha256"]
    assert [path for _, path, _, _ in calls] == [
        "/v1/runs", "/v1/runs/package-run/inputs/component-model", "/v1/runs/package-run/parameters",
        "/v1/runs/package-run/artifacts/package",
        "/v1/runs/package-run/artifacts/numerical-fidelity",
        "/v1/runs/package-run/test-results/correctness/numerical-fidelity",
        "/v1/runs/package-run/measurements/batch",
        "/v1/runs/package-run/finalize",
    ]
    assert calls[0][2]["run_type"] == "model_package"
    assert calls[3][2]["artifact_type"] == "model-package"
    assert store.objects[0][0].name == assembled["archive"]
    assert store.objects[0][3] == "application/gzip"
    assert calls[3][2]["media_type"] == "application/gzip"


def test_publish_package_attaches_the_profile_numerical_fidelity(tmp_path: Path) -> None:
    packages, _ = _assembled_package(tmp_path)
    calls: list[tuple[str, str, dict | None, str]] = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        if path == "/v1/runs":
            return {"run_id": "package-run"}
        if path.endswith("/artifacts/numerical-fidelity"):
            return {"artifact_id": "00000000-0000-4000-8000-000000000002"}
        return {"ok": True}

    store = FakeStore()
    _publish_packages(packages, request, store)

    bodies = {path: body for _, path, body, _ in calls}
    artifact = bodies["/v1/runs/package-run/artifacts/numerical-fidelity"]
    assert artifact["metadata"] == {"model_id": "resnet_50", "build_profile_id": "w16a16", "samples": 50}
    test_result = bodies["/v1/runs/package-run/test-results/correctness/numerical-fidelity"]
    assert test_result["report_artifact_id"] == "00000000-0000-4000-8000-000000000002"
    measurements = bodies["/v1/runs/package-run/measurements/batch"]["measurements"]
    assert {measurement["name"] for measurement in measurements} == {
        "model.fidelity.cosine_similarity",
        "model.fidelity.normalized_rmse",
        "model.fidelity.mean_absolute_error",
        "model.fidelity.max_absolute_error",
    }
    assert all(measurement["dimensions"] == {
        "model_id": "resnet_50", "build_profile_id": "w16a16", "output": "logits",
    } for measurement in measurements)
    maximum = next(
        measurement for measurement in measurements if measurement["name"] == "model.fidelity.max_absolute_error"
    )
    assert maximum["aggregation"] == "max"
    assert store.objects[1][0] == packages / "resnet_50/w16a16/packages/default/fidelity.json"
    assert store.objects[1][3] == "application/json"


@pytest.mark.parametrize(("change", "message"), [
    ("extra field", "Additional properties"),
    ("other output", "metrics must cover exactly"),
    ("other build profile", "build_profile_id must be 'w16a16'"),
    ("missing", "the package requires fidelity.json"),
])
def test_publish_package_rejects_fidelity_that_does_not_match_the_declaration(
    tmp_path: Path, change: str, message: str
) -> None:
    packages, _ = _assembled_package(tmp_path)
    fidelity_path = packages / "resnet_50/w16a16/packages/default/fidelity.json"
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if change == "extra field":
        fidelity["unexpected"] = True
    elif change == "other output":
        fidelity["metrics"][0]["name"] = "scores"
    elif change == "other build profile":
        fidelity["build_profile_id"] = "w8a8"
    fidelity_path.write_text(json.dumps(fidelity), encoding="utf-8")
    if change == "missing":
        fidelity_path.unlink()

    calls = []

    def request(method, path, *args):
        calls.append((method, path))
        return {"run_id": "package-run"}

    with pytest.raises(ValidationError, match=message):
        _publish_packages(packages, request, FakeStore())
    assert calls == []


@pytest.mark.parametrize(
    ("archive", "expected"),
    [
        ("resnet_50_w16a16_mpk.tar.gz", "application/gzip"),
        ("rf_detr_nano_w8a8_w16a16_default.zip", "application/zip"),
        ("resnet_50_w16a16.tar.gz", None),
    ],
)
def test_package_media_type_accepts_only_mpk_and_zip_packages(archive: str, expected: str | None) -> None:
    assert _package_media_type(archive) == expected


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


def test_publish_records_one_run_per_compiled_variant(tmp_path: Path, monkeypatch) -> None:
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
    assert create["metadata"].pop("compile_timing") == COMPILE_TIMING
    assert create["metadata"] == {
        "model_id": "resnet_50",
        "build_profile_id": "w16a16",
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
    assert parameters["compiler_version"] == COMPILER_VERSION
    assert parameters["compile_invocation_id"] == "w16a16"
    assert parameters["target"] == "modalix"
    artifact = calls[3][2]
    assert result["runs"][0]["artifact_sha256"] == artifact["artifact_sha256"]
    assert artifact["s3_version_id"] == "version-1"
    assert artifact["s3_key"].startswith("models/feature%2Fmodels/aaaaaaaaaaaa/")
    assert store.objects[0][1] == artifact["s3_key"]

    receipt_path = dist / "resnet_50" / "w16a16" / "model" / "registry-publication.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    models = load_models(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["identity"]
    selected = build_matrix(
        models,
        ["resnet_50"],
        ["w16a16"],
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
    # The fresh build carries its profile's whole-model performance fingerprint; w16a16 has no accuracy.
    assert resnet["benchmark_fingerprints"] == {"w16a16": {"performance": benchmark_fingerprint(
        ROOT, models["resnet_50"], "w16a16", {"model": resnet["builds"][0]}, "performance",
    )}}
    package_dir = tmp_path / "packages" / "resnet_50/w16a16/packages/default"
    package_dir.mkdir(parents=True)
    package_dir.joinpath("registry-publication.json").write_text(json.dumps({"packages": [{
        "model_id": "resnet_50", "build_profile_id": "w16a16", "package_id": "default",
        "run_id": "package-run", "artifact_sha256": "b" * 64, "size_bytes": 123,
        "manifest_sha256": "c" * 64, "package_fingerprint": "c" * 64,
        "component_run_ids": {"model": result["runs"][0]["run_id"]},
    }]}), encoding="utf-8")
    snapshot = build_catalog_snapshot(
        repository=ROOT, branch="feature/models", git_sha="a" * 40,
        compiled_dir=dist, package_dir=tmp_path / "packages",
    )
    resnet = next(model for model in snapshot["models"] if model["id"] == "resnet_50")
    assert resnet["packages"] == [{
        "package_id": "default", "build_profile_id": "w16a16", "run_id": "package-run",
        "artifact_sha256": "b" * 64, "size_bytes": 123, "package_fingerprint": "c" * 64,
        "component_run_ids": {"model": result["runs"][0]["run_id"]},
    }]


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


@pytest.mark.parametrize(("rebuilt", "message"), [
    (False, "compiled MPK must be named resnet_50_w16a16_mpk.tar.gz, found resnet_50_mpk.tar.gz"),
    (True, "archive does not match the compile output manifest"),
])
def test_publish_rejects_archive_not_named_or_bound_to_output_manifest(
    tmp_path: Path, rebuilt: bool, message: str
) -> None:
    dist = _packaged_result(tmp_path)
    result_path = dist / "resnet_50" / "w16a16" / "model" / "compile-result.json"
    result = json.loads(result_path.read_text())
    if rebuilt:
        # The result names the right file but records a checksum the compile output manifest never saw.
        result["sha256"] = "0" * 64
    else:
        outputs_path = result_path.parent.parent / "compile-outputs.json"
        outputs = json.loads(outputs_path.read_text())
        renamed = (result_path.parent / result["archive"]).rename(result_path.parent / "resnet_50_mpk.tar.gz")
        outputs["outputs"][0]["mpk_path"] = f"model/{renamed.name}"
        outputs_path.write_text(json.dumps(outputs))
        result["archive"] = renamed.name
    result_path.write_text(json.dumps(result))
    calls = []
    store = FakeStore()

    with pytest.raises(ValidationError, match=message):
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


def test_catalog_snapshot_reuses_unchanged_compile_runs() -> None:
    models = load_models(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["identity"]
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
                            "compile_fingerprint": entry["compile_fingerprint"],
                            "artifact_sha256": "e" * 64,
                            "run_id": f"00000000-0000-4000-8000-{index:012d}",
                        }
                        for entry in build_matrix(
                            models,
                            [model_id],
                            repository=ROOT,
                            compiler_version=compiler_version,
                        )["include"]
                        for component_id in entry["component_ids"]
                    ],
                }
                for index, model_id in enumerate(models, 1)
            ]
        },
    )

    assert first["catalog_revision"] == "a" * 40
    assert len(first["taxonomy_sha256"]) == 64
    assert "version" not in first["taxonomy"]
    assert all("taxonomy_version" not in model["categories"] for model in first["models"])


def test_catalog_records_benchmark_fingerprints_bound_to_their_compile_runs() -> None:
    models = load_models(ROOT)
    model = models["resnet_50"]
    matrix = build_matrix(
        models, sorted(models), repository=ROOT, compiler_version=COMPILER_IDENTITY,
    )["include"]

    def snapshot(run_prefix: str) -> dict:
        previous = {"models": [{"id": model_id, "builds": [
            {
                "build_profile_id": entry["build_profile_id"],
                "component_id": component_id,
                "compile_fingerprint": entry["compile_fingerprint"],
                "artifact_sha256": "e" * 64,
                "run_id": f"{run_prefix}-{index:012d}",
            }
            for index, entry in enumerate(matrix, 1)
            if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in models]}
        document = build_catalog_snapshot(repository=ROOT, branch="develop", git_sha="a" * 40, previous=previous)
        return next(item for item in document["models"] if item["id"] == "resnet_50")

    resnet = snapshot("00000000-0000-4000-8000")
    builds = {build["build_profile_id"]: {"model": build} for build in resnet["builds"]}
    assert all(set(build["model"]) == {
        "build_profile_id", "compile_invocation_id", "component_id", "compile_fingerprint", "artifact_sha256", "run_id",
    } for build in builds.values())
    # Each profile records one fingerprint per stage it benchmarks; only w8a8 declares accuracy.
    assert resnet["benchmark_fingerprints"] == {
        "w16a16": {"performance": benchmark_fingerprint(ROOT, model, "w16a16", builds["w16a16"], "performance")},
        "w8a8": {
            "performance": benchmark_fingerprint(ROOT, model, "w8a8", builds["w8a8"], "performance"),
            "accuracy": benchmark_fingerprint(ROOT, model, "w8a8", builds["w8a8"], "accuracy"),
        },
    }
    w8a8 = resnet["benchmark_fingerprints"]["w8a8"]
    assert w8a8["performance"] != w8a8["accuracy"]

    # A new compile run of identical inputs still needs its own benchmarks.
    rebuilt = snapshot("00000000-0000-4000-9000")
    assert [build["compile_fingerprint"] for build in rebuilt["builds"]] == [
        build["compile_fingerprint"] for build in resnet["builds"]
    ]
    for profile_id, fingerprints in resnet["benchmark_fingerprints"].items():
        for stage, fingerprint in fingerprints.items():
            assert rebuilt["benchmark_fingerprints"][profile_id][stage] != fingerprint


def test_incremental_catalog_keeps_prior_builds_for_missing_changed_invocations(monkeypatch) -> None:
    models = load_models(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["identity"]
    invocation = build_matrix(
        models, ["resnet_50"], ["w16a16"], repository=ROOT, compiler_version=compiler_version,
    )["include"][0]
    changed = dict(invocation, compile_fingerprint="f" * 64)
    monkeypatch.setattr("model_registry.registry.build_matrix", lambda *args, **kwargs: {"include": [changed]})
    prior_build = {
        "build_profile_id": "w16a16",
        "compile_invocation_id": invocation["compile_invocation_id"],
        "component_id": "model",
        "compile_fingerprint": "e" * 64,
        "artifact_sha256": "d" * 64,
        "run_id": "00000000-0000-4000-8000-000000000001",
    }
    prior_benchmark_fingerprints = {"w16a16": {"performance": "a" * 64}}

    snapshot = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="a" * 40,
        previous={"models": [
            {"id": "resnet_50", "builds": [prior_build], "benchmark_fingerprints": prior_benchmark_fingerprints},
        ]},
        incremental=True,
    )

    resnet = next(model for model in snapshot["models"] if model["id"] == "resnet_50")
    assert resnet["builds"] == [prior_build]
    # The prior build's benchmark fingerprints stay visible with it while the replacement is pending.
    assert resnet["benchmark_fingerprints"] == prior_benchmark_fingerprints
    assert snapshot["pending_builds"] == [{
        "model_id": "resnet_50",
        "build_profile_id": "w16a16",
        "compile_invocation_id": invocation["compile_invocation_id"],
        "component_ids": ["model"],
        "compile_fingerprint": "f" * 64,
    }]


def test_selected_same_fingerprint_rebuild_replaces_package_after_component_publication(
    monkeypatch, tmp_path: Path,
) -> None:
    models = load_models(ROOT)
    compiler_version = resolve_model_compiler(ROOT)["version"]
    invocation = build_matrix(
        models, ["resnet_50"], ["w16a16"], repository=ROOT, compiler_version=compiler_version,
    )["include"][0]
    monkeypatch.setattr("model_registry.registry.build_matrix", lambda *args, **kwargs: {"include": [invocation]})
    old_run_id = "00000000-0000-4000-8000-000000000001"
    new_run_id = "00000000-0000-4000-8000-000000000002"
    prior_build = {
        "build_profile_id": "w16a16",
        "compile_invocation_id": invocation["compile_invocation_id"],
        "component_id": "model",
        "compile_fingerprint": invocation["compile_fingerprint"],
        "artifact_sha256": "a" * 64,
        "run_id": old_run_id,
    }
    prior_package = {
        "package_id": "default",
        "build_profile_id": "w16a16",
        "run_id": "00000000-0000-4000-8000-000000000003",
        "artifact_sha256": "b" * 64,
        "size_bytes": 100,
        "package_fingerprint": "c" * 64,
        "component_run_ids": {"model": old_run_id},
    }
    selected_matrix = tmp_path / "selected-matrix.json"
    selected_matrix.write_text(json.dumps({"include": [invocation]}), encoding="utf-8")

    initial = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="d" * 40,
        selected_matrix=selected_matrix,
        previous={"models": [{"id": "resnet_50", "builds": [prior_build], "packages": [prior_package]}]},
        incremental=True,
    )

    resnet = next(model for model in initial["models"] if model["id"] == "resnet_50")
    assert resnet["builds"][0]["run_id"] == old_run_id
    assert resnet["builds"][0]["compile_fingerprint"] == invocation["compile_fingerprint"]
    assert resnet["packages"] == []
    assert initial["pending_builds"] == []

    compiled_dir = tmp_path / "compiled"
    component_dir = compiled_dir / "resnet_50" / "w16a16" / "model"
    component_dir.mkdir(parents=True)
    component_dir.joinpath("registry-publication.json").write_text(json.dumps({"runs": [{
        "model_id": "resnet_50",
        "build_profile_id": "w16a16",
        "component_id": "model",
        "run_id": new_run_id,
        "artifact_sha256": "d" * 64,
        "compile_fingerprint": invocation["compile_fingerprint"],
    }]}), encoding="utf-8")
    component_published = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="d" * 40,
        compiled_dir=compiled_dir,
        previous=initial,
        incremental=True,
    )
    resnet = next(model for model in component_published["models"] if model["id"] == "resnet_50")
    assert resnet["builds"][0]["run_id"] == new_run_id
    assert resnet["packages"] == []

    package_dir = tmp_path / "packages" / "resnet_50" / "w16a16" / "packages" / "default"
    package_dir.mkdir(parents=True)
    package_dir.joinpath("registry-publication.json").write_text(json.dumps({"packages": [{
        "model_id": "resnet_50",
        "build_profile_id": "w16a16",
        "package_id": "default",
        "run_id": "00000000-0000-4000-8000-000000000004",
        "artifact_sha256": "e" * 64,
        "size_bytes": 200,
        "manifest_sha256": "f" * 64,
        "package_fingerprint": "f" * 64,
        "component_run_ids": {"model": new_run_id},
    }]}), encoding="utf-8")
    complete = build_catalog_snapshot(
        repository=ROOT,
        branch="develop",
        git_sha="d" * 40,
        compiled_dir=compiled_dir,
        package_dir=tmp_path / "packages",
        previous=component_published,
        incremental=True,
    )
    resnet = next(model for model in complete["models"] if model["id"] == "resnet_50")
    assert resnet["packages"][0]["component_run_ids"] == {"model": new_run_id}


def test_publish_catalog_reads_current_then_publishes_complete_snapshot(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None, str]] = []
    read_urls: list[str] = []

    previous = {"models": []}
    target_catalog = {"canonical_sha256": "c" * 64, "catalog_revision": "a" * 40, "document": previous}
    monkeypatch.setattr(
        "model_registry.registry.build_catalog_snapshot",
        lambda **kwargs: {
            "schema_version": 2,
            "repository": "sima-neat/models",
            "git_ref": "refs/heads/develop",
            "catalog_revision": "a" * 40,
            "taxonomy_sha256": "b" * 64,
            "taxonomy": {},
            "models": [], "pending_builds": [],
        },
    )

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body, key))
        return {"stored": True}

    def read_request(url: str) -> dict:
        read_urls.append(url)
        return {"catalog": target_catalog}

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
    assert calls[0][0:2] == ("POST", "/v1/repositories/sima-neat/models/catalogs/" + "a" * 40 + "/snapshots")
    assert calls[0][2] == {"initialize": False, "previous_snapshot": "c" * 64, "catalog": result["catalog"]}
    assert calls[0][3] == (
        "models-catalog-develop-" + "a" * 40 + "-" + "c" * 16 + "-" + canonical_sha256(calls[0][2])[:16]
    )


def test_publish_catalog_keys_bind_dispatch_specific_documentation_receipts(monkeypatch) -> None:
    calls: list[tuple[str, str, dict | None, str]] = []
    target_catalog = {"canonical_sha256": "c" * 64, "catalog_revision": "b" * 40, "document": {"models": []}}
    monkeypatch.setattr(
        "model_registry.registry.publish_model_documentation",
        lambda **kwargs: {"fixture": {"run_id": kwargs["github_run_url"].rsplit("/", 1)[-1]}},
    )
    monkeypatch.setattr(
        "model_registry.registry.build_catalog_snapshot",
        lambda **kwargs: {"documentation": kwargs["documentation_receipts"]},
    )

    def publish(github_run_url: str) -> None:
        publish_catalog(
            repository=ROOT,
            api_url="https://registry.example.invalid",
            read_api_url="https://models-registry.example.invalid",
            aws_region="us-west-2",
            branch="develop",
            git_sha="a" * 40,
            bucket="artifacts",
            environment="staging",
            github_run_url=github_run_url,
            request=lambda *args: calls.append(args) or {"stored": True},
            read_request=lambda _: {"catalog": target_catalog},
        )

    publish("https://github.example.invalid/actions/runs/123")
    publish("https://github.example.invalid/actions/runs/456")

    assert calls[0][2] != calls[1][2]
    assert calls[0][3] != calls[1][3]


def test_publish_catalog_reads_current_catalog_after_documentation(monkeypatch) -> None:
    events: list[str] = []
    target_catalog = {"canonical_sha256": "c" * 64, "catalog_revision": "b" * 40, "document": {"models": []}}
    monkeypatch.setattr(
        "model_registry.registry.publish_model_documentation",
        lambda **kwargs: events.append("documentation") or {},
    )
    monkeypatch.setattr(
        "model_registry.registry.build_catalog_snapshot",
        lambda **kwargs: events.append("build") or {"models": []},
    )

    publish_catalog(
        repository=ROOT,
        api_url="https://registry.example.invalid",
        read_api_url="https://models-registry.example.invalid",
        aws_region="us-west-2",
        branch="develop",
        git_sha="a" * 40,
        bucket="artifacts",
        environment="staging",
        request=lambda method, *args: events.append(method) or {"stored": True},
        read_request=lambda _: events.append("read current") or {"catalog": target_catalog},
    )

    assert events == ["read current", "documentation", "read current", "build", "POST"]


def test_publish_catalog_reuses_documentation_receipts_of_a_cataloged_revision(monkeypatch) -> None:
    receipt = {"format": "markdown", "run_id": "first-run", "artifact_name": "readme"}
    target_catalog = {
        "canonical_sha256": "c" * 64,
        "catalog_revision": "a" * 40,
        "document": {"models": [{"id": "resnet_50", "documentation": receipt, "builds": []}]},
    }
    snapshots: list[dict] = []

    def publish_documentation(**kwargs):
        raise AssertionError("a cataloged revision must not publish new documentation runs")

    monkeypatch.setattr("model_registry.registry.publish_model_documentation", publish_documentation)
    monkeypatch.setattr(
        "model_registry.registry.build_catalog_snapshot",
        lambda **kwargs: snapshots.append(kwargs["documentation_receipts"]) or {"models": []},
    )

    publish_catalog(
        repository=ROOT,
        api_url="https://registry.example.invalid",
        read_api_url="https://models-registry.example.invalid",
        aws_region="us-west-2",
        branch="develop",
        git_sha="A" * 40,
        bucket="artifacts",
        environment="staging",
        github_run_url="https://github.example.invalid/actions/runs/456",
        request=lambda *args: {"stored": True},
        read_request=lambda _: {"catalog": target_catalog},
    )

    assert snapshots == [{"resnet_50": receipt}]


def test_publish_catalog_publishes_documentation_when_a_cataloged_model_has_none(monkeypatch) -> None:
    target_catalog = {
        "canonical_sha256": "c" * 64,
        "catalog_revision": "a" * 40,
        "document": {"models": [{"id": "resnet_50", "builds": []}]},
    }
    published: list[str] = []
    monkeypatch.setattr(
        "model_registry.registry.publish_model_documentation",
        lambda **kwargs: published.append(kwargs["git_sha"]) or {"resnet_50": {"run_id": "new-run"}},
    )
    monkeypatch.setattr("model_registry.registry.build_catalog_snapshot", lambda **kwargs: {"models": []})

    publish_catalog(
        repository=ROOT,
        api_url="https://registry.example.invalid",
        read_api_url="https://models-registry.example.invalid",
        aws_region="us-west-2",
        branch="develop",
        git_sha="a" * 40,
        bucket="artifacts",
        environment="staging",
        request=lambda *args: {"stored": True},
        read_request=lambda _: {"catalog": target_catalog},
    )

    assert published == ["a" * 40]


def _stage_catalog_build(tmp_path: Path, packages: list[dict]) -> Path:
    dist = _packaged_result(tmp_path)
    source_archive = next(dist.glob("resnet_50/w16a16/model/*_mpk.tar.gz"))
    payload = source_archive.read_bytes()
    run_id = "00000000-0000-4000-8000-000000000001"
    catalog = {
        "catalog_revision": "a" * 40,
        "models": [{
            "id": "resnet_50",
            "builds": [{
                "build_profile_id": "w16a16",
                "compile_invocation_id": "w16a16",
                "component_id": "model",
                "run_id": run_id,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            }],
            "packages": packages,
        }],
    }

    def read(url: str) -> dict:
        if "/catalogs/current?" in url:
            return {"catalog": {"document": catalog}}
        if url.endswith("/artifacts/mpk"):
            return {"artifact": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "download_url": "https://download.example.invalid/mpk",
            }}
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
        build_profile_id="w16a16",
        compile_invocation_id="w16a16",
        read_request=read,
        download_request=lambda url: payload,
    )
    assert result["count"] == 1
    return tmp_path / "staged" / "resnet_50" / "w16a16"


def test_stage_catalog_build_restores_and_verifies_durable_mpk(tmp_path: Path) -> None:
    profile = _stage_catalog_build(tmp_path, [])

    assert (profile / "model" / "registry-publication.json").is_file()
    # Restored builds carry the manifest a fresh compile writes, so later stages read both alike.
    staged = next((profile / "model").glob("*_mpk.tar.gz"))
    assert json.loads((profile / "compile-outputs.json").read_text(encoding="utf-8")) == {
        "model_id": "resnet_50",
        "build_profile_id": "w16a16",
        "compile_invocation_id": "w16a16",
        "outputs": [{
            "component_id": "model",
            "mpk_path": f"model/{staged.name}",
            "mpk_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
        }],
    }


def test_stage_catalog_build_writes_the_receipt_of_the_profile_default_package(tmp_path: Path) -> None:
    package = {
        "package_id": "default", "build_profile_id": "w16a16", "run_id": "00000000-0000-4000-8000-000000000002",
        "artifact_sha256": "b" * 64, "size_bytes": 123, "package_fingerprint": "c" * 64,
        "component_run_ids": {"model": "00000000-0000-4000-8000-000000000001"},
    }
    other_profile = {**package, "build_profile_id": "w8a8", "run_id": "00000000-0000-4000-8000-000000000003"}

    profile = _stage_catalog_build(tmp_path, [other_profile, package])

    # The benchmarks read the package run from the same receipt publish-package writes.
    receipt = profile / "packages" / "default" / "registry-publication.json"
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"packages": [{"model_id": "resnet_50", **package}]}


def test_stage_catalog_build_writes_no_package_receipt_without_a_cataloged_package(tmp_path: Path) -> None:
    other_profile = {
        "package_id": "default", "build_profile_id": "w8a8", "run_id": "00000000-0000-4000-8000-000000000003",
    }

    profile = _stage_catalog_build(tmp_path, [other_profile])

    assert not (profile / "packages").exists()


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


def test_compile_timing_is_required_and_normalized_to_utc(tmp_path: Path) -> None:
    path = tmp_path / "compile-timing.json"
    with pytest.raises(ValidationError, match="invalid compile timing"):
        _compile_timing(path)
    path.write_text(json.dumps({
        "scope": "compile_recipe", "started_at": "2026-08-13T00:00:00Z", "finished_at": "2026-08-13T01:29:52Z",
    }))
    assert _compile_timing(path) == {
        "scope": "compile_recipe", "started_at": "2026-08-13T00:00:00+00:00",
        "finished_at": "2026-08-13T01:29:52+00:00",
    }


def test_publish_preserves_runner_audit_without_changing_compile_identity(tmp_path):
    dist = _packaged_result(tmp_path)
    path = dist / "resnet_50/w16a16/model/compile-result.json"
    original = json.loads(path.read_text())
    runner = {
        "requested_runner_class": "default", "runner_class": "default",
        "requested_runner_market": "on-demand", "runner_market": "on-demand",
        "runner_actual_market": "on-demand", "runner_name": "runner-test",
        "runner_instance_id": "i-test", "runner_instance_type": "c8gd.4xlarge",
        "runner_approval_id": "sima-neat/models:resnet_50", "runner_approval_spec_revision": "a" * 64,
    }

    def publish():
        calls = []

        def request(method, endpoint, body, key):
            calls.append((endpoint, body, key))
            return {"run_id": "compile-run"} if endpoint == "/v1/runs" else {"ok": True}

        result = publish_compilations(
            repository=ROOT, dist_dir=dist, api_url="https://registry.example.invalid", aws_region="us-west-2",
            bucket="artifacts", kms_key_id=None, environment="staging", branch="feature/models", git_sha="a" * 40,
            github_run_id="1234", github_run_attempt="1", github_run_url="https://example.invalid/run",
            request=request, artifact_store=FakeStore(),
        )
        parameters = next(body["parameters"] for endpoint, body, _ in calls if endpoint.endswith("/parameters"))
        return calls[0], parameters, result

    before, legacy_parameters, legacy_result = publish()
    original["provenance"].update(runner)
    path.write_text(json.dumps(original))
    after, parameters, result = publish()
    assert {field: parameters[field] for field in runner} == runner
    assert all(field not in legacy_parameters for field in runner)
    assert before == after  # Includes run fingerprint and idempotency key.
    assert result["runs"][0]["artifact_sha256"] == legacy_result["runs"][0]["artifact_sha256"]
    assert parameters["build_fingerprint"] == legacy_parameters["build_fingerprint"]


def test_package_run_retains_provenance_when_artifact_is_only_an_mpk(tmp_path):
    packages, assembled = _assembled_package(tmp_path)
    path = next(packages.rglob("package-result.json"))
    result = json.loads(path.read_text())
    result["manifest"]["provenance"].update({
        "runner_market": "on-demand", "runner_instance_id": "i-test",
        "runner_actual_market": "on-demand", "runner_approval_spec_revision": "b" * 64,
    })
    path.write_text(json.dumps(result))
    calls = []

    def request(method, endpoint, body, key):
        calls.append((endpoint, body))
        return {"run_id": "package-run"} if endpoint == "/v1/runs" else {"ok": True}

    published = _publish_packages(packages, request, FakeStore())
    parameters = next(body["parameters"] for endpoint, body in calls if endpoint.endswith("/parameters"))
    assert parameters["provenance"] == result["manifest"]["provenance"]
    assert published["packages"][0]["artifact_sha256"] == assembled["sha256"]
