from __future__ import annotations

import pytest
from model_registry.errors import SourcePolicyError
from model_registry.source import SourceSpec, redact_url

DIGEST = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"


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
        "https://api.github.com/repos/sima-neat/models/releases/assets/508126877",
        "https://github.com/sima-neat/models/releases/download/source-v1/model.onnx",
        "https://github.com/owner/repository/releases/download/calibration-v1/data.npz",
        "https://api.github.com/repos/owner/repository/releases/assets/123456",
    ],
)
def test_github_release_assets_are_immutable_sources(uri: str) -> None:
    assert SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url() == uri


def test_google_cloud_storage_object_is_an_approved_checksum_verified_source() -> None:
    uri = "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"
    assert SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url() == uri


def test_python_package_artifact_is_an_approved_checksum_verified_source() -> None:
    uri = (
        "https://files.pythonhosted.org/packages/57/d4/"
        "082da42f39a29acadd47f487b4186c0d6203eb8ce4d145baa72a12103b52/"
        "rfdetr-1.9.4-py3-none-any.whl"
    )
    assert SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url() == uri


@pytest.mark.parametrize(
    "uri",
    [
        "https://storage.googleapis.com/unrelated/model.pth",
        "https://storage.googleapis.com/rfdetr/checkpoint.pth",
        "https://files.pythonhosted.org/packages/rfdetr-1.9.4-py3-none-any.whl",
        "https://files.pythonhosted.org/unrelated/package.whl",
    ],
)
def test_approved_hosts_reject_sources_outside_bounded_paths(uri: str) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()


@pytest.mark.parametrize(
    "uri",
    [
        "https://api.github.com/repos/sima-neat/models/releases/latest",
        "https://github.com/sima-neat/models/raw/main/model.onnx",
        "https://github.com/owner/repository/raw/main/data.npz",
        "https://github.com/owner/repository/releases/latest/download/data.npz",
        "https://api.github.com/repos/owner/repository/releases/latest",
    ],
)
def test_non_asset_github_urls_fail(uri: str) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()
