from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from model_registry.registry import S3Object, canonical_sha256, publish_compilations

ROOT = Path(__file__).resolve().parents[2]


class FakeStore:
    def __init__(self) -> None:
        self.objects: list[tuple[Path, str, str]] = []

    def ensure_native_sha256(self, path: Path, key: str, sha256: str) -> S3Object:
        self.objects.append((path, key, sha256))
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


def test_canonical_sha256_is_stable() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


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
        "/v1/runs",
        "/v1/runs/00000000-0000-4000-8000-000000000001/inputs/source-model",
        "/v1/runs/00000000-0000-4000-8000-000000000001/parameters",
        "/v1/runs/00000000-0000-4000-8000-000000000001/artifacts/mpk",
        "/v1/runs/00000000-0000-4000-8000-000000000001/test-results/compile/mpk-validation",
        "/v1/runs/00000000-0000-4000-8000-000000000001/finalize",
    ]
    create = calls[0][2]
    assert create is not None
    assert create["run_type"] == "model_compile"
    assert create["metadata"] == {
        "model_id": "resnet_50",
        "variant_id": "modalix_bf16",
        "artifact_kind": "mpk",
        "github_run_id": "1234",
    }
    parameters = calls[2][2]["parameters"]
    assert parameters["compiler_version"] == "2.1.3"
    assert parameters["target"] == "modalix"
    artifact = calls[3][2]
    assert artifact["s3_version_id"] == "version-1"
    assert artifact["s3_key"].startswith("models/feature%2Fmodels/aaaaaaaaaaaa/")
    assert store.objects[0][1] == artifact["s3_key"]


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

    assert result == {"count": 0, "runs": []}
    assert calls == []


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
