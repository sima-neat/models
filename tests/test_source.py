from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from model_registry.errors import SourcePolicyError
from model_registry.source import SourceSpec, redact_url

DIGEST = "a" * 64
REVISION = "0123456789abcdef0123456789abcdef01234567"
ROOT = Path(__file__).resolve().parents[1]
GITHUB_RELEASE_FILE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-obb.pt"
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


def test_github_release_file_addressed_by_tag_is_a_source() -> None:
    source = SourceSpec(uri=GITHUB_RELEASE_FILE, sha256=DIGEST, size_bytes=10)
    assert source.resolve_url() == GITHUB_RELEASE_FILE
    assert source.filename == "yolov8n-obb.pt"


@pytest.mark.parametrize(
    ("uri", "reason"),
    [
        ("https://api.github.com/repos/ultralytics/assets/releases/assets/508126877", "host is not approved"),
        ("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8n-obb.pt", "host is not approved"),
        ("https://github.com/ultralytics/assets/raw/main/yolov8n-obb.pt", "by its tag"),
        ("https://github.com/ultralytics/assets/archive/refs/tags/v8.4.0.zip", "by its tag"),
        ("https://github.com/ultralytics/assets/releases/latest/download/yolov8n-obb.pt", "by its tag"),
        ("https://github.com/ultralytics/assets/releases/download/v8.4.0/nested/yolov8n-obb.pt", "by its tag"),
        ("https://github.com/ultralytics/assets/releases/download/v8.4.0/..%2Fyolov8n-obb.pt", "by its tag"),
        ("https://github.com/ultralytics/assets/releases/download/v8.4.0/../yolov8n-obb.pt", "dot or parent segments"),
        (f"{GITHUB_RELEASE_FILE}?token=secret", "credentials, query parameters, or fragments"),
        (f"{GITHUB_RELEASE_FILE}#sha256", "credentials, query parameters, or fragments"),
        (GITHUB_RELEASE_FILE.replace("github.com", "user:secret@github.com"), "credentials, query parameters"),
        (GITHUB_RELEASE_FILE.replace("github.com", "github.com:444"), "default TLS port"),
        (GITHUB_RELEASE_FILE.replace("https://", "http://"), "scheme must be hf or https"),
    ],
)
def test_other_github_urls_are_not_sources(uri: str, reason: str) -> None:
    with pytest.raises(SourcePolicyError, match=reason):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()


def test_scripts_and_shared_tooling_download_nothing_from_github_releases() -> None:
    # A source is pinned in model.yaml and fetched by the registry, and datasets come from the Dataset
    # Registry, so no script or tool names a release file itself.
    offenders = [
        path.relative_to(ROOT).as_posix()
        for top in ("models", "scripts", "tools")
        for path in sorted((ROOT / top).rglob("*"))
        if path.is_file()
        and path.name != "model.yaml"
        and "__pycache__" not in path.parts
        and GITHUB_RELEASE_DOWNLOAD.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def pins_release_file_outside_conversion(model_yaml: str) -> bool:
    document = yaml.safe_load(model_yaml)
    document.get("conversion", {}).pop("source", None)
    return bool(GITHUB_RELEASE_DOWNLOAD.search(yaml.safe_dump(document, width=float("inf"))))


@pytest.mark.parametrize(
    ("section", "expected"),
    [("conversion", False), ("compilation", True)],
)
def test_release_file_pin_is_found_outside_the_conversion_source(section: str, expected: bool) -> None:
    model_yaml = yaml.safe_dump({section: {"source": {"uri": GITHUB_RELEASE_FILE, "sha256": DIGEST, "size_bytes": 10}}})
    assert pins_release_file_outside_conversion(model_yaml) is expected


def test_only_conversion_sources_pin_github_release_files() -> None:
    # A conversion reads its checkpoint once and the registry keeps the converted model. A compile source is
    # fetched on every build, so it stays on a host whose pinned files cannot be replaced or deleted.
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "models").glob("*/model.yaml"))
        if pins_release_file_outside_conversion(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_vulcan_onnx_source() -> None:
    uri = f"https://artifacts.stg.neat.sima.ai/models-onnx/rf_detr_small/{DIGEST}/123-1/model.onnx"
    assert SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url() == uri


@pytest.mark.parametrize("name", ["source.pth", "weights.pth", "source.pt"])
def test_vulcan_does_not_admit_unused_checkpoint_transports(name: str) -> None:
    uri = f"https://artifacts.stg.neat.sima.ai/models-onnx/ssd_mobilenet_v3/{DIGEST}/123-2/{name}"
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=DIGEST, size_bytes=10).resolve_url()


TORCHVISION_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
TORCHVISION_PATH = "/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


def test_official_torchvision_weights_are_content_pinned() -> None:
    uri = f"https://download.pytorch.org{TORCHVISION_PATH}"
    source = SourceSpec.from_mapping({"uri": uri, "sha256": TORCHVISION_SHA256, "size_bytes": 14069355})
    assert source.resolve_url() == uri
    assert source.filename == "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


@pytest.mark.parametrize(
    "uri",
    [
        f"http://download.pytorch.org{TORCHVISION_PATH}",
        f"https://download.pytorch.org:443{TORCHVISION_PATH}",
        f"https://download.pytorch.org:444{TORCHVISION_PATH}",
        f"https://user:secret@download.pytorch.org{TORCHVISION_PATH}",
        f"https://download.pytorch.org{TORCHVISION_PATH}?download=1",
        f"https://download.pytorch.org{TORCHVISION_PATH}#weights",
        f"https://download.pytorch.org/other{TORCHVISION_PATH}",
        f"https://download.pytorch.org/../{TORCHVISION_PATH.lstrip('/')}",
        "https://download.pytorch.org/models/%2e%2e/ssdlite-a79551df.pth",
        "https://download.pytorch.org/models/ssdlite%2da79551df.pth",
        "https://download.pytorch.org/models/nested/ssdlite-a79551df.pth",
        "https://download.pytorch.org/models/ssdlite-a79551df.onnx",
        "https://download.pytorch.org/models/ssdlite-a79551df.tar.gz",
        "https://download.pytorch.org/models/ssdlite.pth",
        "https://download.pytorch.org/models/ssdlite-00000000.pth",
        f"https://download.pytorch.org.example.com{TORCHVISION_PATH}",
        f"https://pytorch.org{TORCHVISION_PATH}",
    ],
)
def test_torchvision_weight_source_is_narrowly_bounded(uri: str) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=uri, sha256=TORCHVISION_SHA256, size_bytes=14069355).resolve_url()


@pytest.mark.parametrize(("digest", "size"), [("a79551df", 1), (TORCHVISION_SHA256, 0)])
def test_direct_torchvision_source_cannot_omit_integrity(digest: str, size: int) -> None:
    with pytest.raises(SourcePolicyError):
        SourceSpec(uri=f"https://download.pytorch.org{TORCHVISION_PATH}", sha256=digest, size_bytes=size).resolve_url()
