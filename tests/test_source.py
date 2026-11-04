from __future__ import annotations

import re
from pathlib import Path

import pytest
from model_registry.errors import SourcePolicyError
from model_registry.source import SourceSpec, redact_url

DIGEST = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"
ROOT = Path(__file__).resolve().parents[1]
GITHUB_RELEASE_DOWNLOAD = re.compile(r"github\.com/(?:repos/)?[\w.-]+/[\w.-]+/releases/(?:download|assets)/")


def test_hugging_face_uri_resolves_full_commit() -> None:
    source = SourceSpec(
        uri=f"hf://models/owner/repository@{REVISION}/nested/model.onnx",
        sha256=DIGEST,
        size_bytes=10,
    )
    assert source.resolve_url() == (f"https://huggingface.co/owner/repository/resolve/{REVISION}/nested/model.onnx")
    assert source.filename == "model.onnx"


@pytest.mark.parametrize("repo_type", ["datasets", "spaces"])
def test_hugging_face_non_model_repo_types(repo_type: str) -> None:
    source = SourceSpec(
        uri=f"hf://{repo_type}/owner/repository@{REVISION}/data/file.bin",
        sha256=DIGEST,
        size_bytes=10,
    )
    assert source.resolve_url().startswith(f"https://huggingface.co/{repo_type}/owner/repository/resolve/{REVISION}/")


@pytest.mark.parametrize(
    "uri",
    [
        "hf://models/owner/repository@main/model.onnx",
        "http://huggingface.co/owner/repository/model.onnx",
        "https://user:secret@huggingface.co/owner/repository/model.onnx",
        "https://huggingface.co/owner/repository/resolve/main/model.onnx",
        f"https://huggingface.co/owner/repository/resolve/{REVISION}/model.onnx?token=secret",
        f"https://huggingface.co:444/owner/repository/resolve/{REVISION}/model.onnx",
        "https://unapproved.example/model.onnx",
        f"hf://models/owner/repository@{REVISION}/../model.onnx",
    ],
)
def test_mutable_or_unapproved_sources_fail(uri: str) -> None:
    source = SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10)
    with pytest.raises(SourcePolicyError):
        source.resolve_url()


def test_redaction_removes_credentials_query_and_fragment() -> None:
    value = redact_url("https://user:secret@example.test/model.onnx?token=top-secret#fragment")
    assert value == "https://example.test/model.onnx"
    assert "secret" not in value


@pytest.mark.parametrize(
    "uri",
    [
        "https://artifacts.stg.neat.sima.ai/models-onnx/rf_detr_small/latest/model.onnx",
        f"https://artifacts.stg.neat.sima.ai/other/{DIGEST}/123-1/model.onnx",
    ],
)
def test_approved_hosts_reject_sources_outside_bounded_paths(uri: str) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()


@pytest.mark.parametrize(
    "uri",
    [
        "https://api.github.com/repos/sima-neat/models/releases/assets/508126877",
        "https://github.com/sima-neat/models/releases/download/source-v1/model.onnx",
        "https://github.com/sima-neat/models/raw/main/model.onnx",
    ],
)
def test_github_release_assets_are_not_sources(uri: str) -> None:
    with pytest.raises(SourcePolicyError, match="host is not approved"):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()


def test_models_and_shared_tooling_download_nothing_from_github_releases() -> None:
    # Sources are pinned in model.yaml and datasets come from the Dataset Registry, so no file fetches a release.
    offenders = [
        path.relative_to(ROOT).as_posix()
        for top in ("models", "scripts", "tools")
        for path in sorted((ROOT / top).rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and GITHUB_RELEASE_DOWNLOAD.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def test_vulcan_onnx_source() -> None:
    uri = f"https://artifacts.stg.neat.sima.ai/models-onnx/rf_detr_small/{DIGEST}/123-1/model.onnx"
    assert SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url() == uri


@pytest.mark.parametrize("path", ["detector.onnx", "temp/model_1.onnx", "project/export-v2.1/segmenter.onnx"])
def test_production_qat_onnx_source(path: str) -> None:
    uri = f"https://artifacts.neat.sima.ai/models/qat/{path}"
    source = SourceSpec.from_mapping({"uri": uri, "sha256": DIGEST, "size_bytes": 10})
    assert source.resolve_url() == uri
    assert source.filename == path.rsplit("/", 1)[-1]


@pytest.mark.parametrize(
    "uri",
    [
        "http://artifacts.neat.sima.ai/models/qat/model.onnx",
        "https://user:secret@artifacts.neat.sima.ai/models/qat/model.onnx",
        "https://artifacts.neat.sima.ai:444/models/qat/model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/model.onnx?token=secret",
        "https://artifacts.neat.sima.ai/models/qat/model.onnx?versionId=version",
        "https://artifacts.neat.sima.ai/models/qat/model.onnx#fragment",
        "https://artifacts.neat.sima.ai.evil.test/models/qat/model.onnx",
        "https://artifacts.neat.sima.ai/models/model.onnx",
        "https://artifacts.neat.sima.ai/models/qat-other/model.onnx",
        f"https://artifacts.neat.sima.ai/models-onnx/detector/{DIGEST}/123-1/model.onnx",
        "https://artifacts.stg.neat.sima.ai/models/qat/model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/./model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/../model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/%2e%2e/model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/%252e%252e/model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/temp%2f..%2fmodel.onnx",
        "https://artifacts.neat.sima.ai/models/qat/temp\\model.onnx",
        "https://artifacts.neat.sima.ai/models/qat//model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/temp//model.onnx",
        "https://artifacts.neat.sima.ai/models/qat/temp/",
        "https://artifacts.neat.sima.ai/models/qat/temp/model.zip",
        "https://artifacts.neat.sima.ai/models/qat/temp/model.onnx.tar.gz",
    ],
)
def test_production_qat_rejects_unsafe_or_unbounded_sources(uri: str) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()
