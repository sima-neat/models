from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.errors import FetchError, SourcePolicyError
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

BASE = "https://artifacts.stg.neat.sima.ai/"
PREFIX = "models-onnx/example_model/" + "a" * 64 + "/export-1/"


def metadata(payload):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def fixture_bundle(mutator=None):
    files = {"model.onnx": b"onnx", "weights/data.bin": b"weights", "validation.json": b'{"status":"passed"}'}
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        for name, body in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            stream.addfile(member, io.BytesIO(body))
    body = archive.getvalue()
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "files": {name: metadata(value) for name, value in files.items()},
        "bundle": {"name": "onnx-bundle.tar", **metadata(body)},
        "objects": [
            {"key": PREFIX + "onnx-bundle.tar.part00000", "version_id": "v1", **metadata(body[:5000])},
            {"key": PREFIX + "onnx-bundle.tar.part00001", "version_id": "v2", **metadata(body[5000:])},
        ],
    }
    payloads = {BASE + manifest["objects"][0]["key"]: body[:5000], BASE + manifest["objects"][1]["key"]: body[5000:]}
    if mutator:
        mutator(manifest)
    raw = json.dumps(manifest).encode()
    uri = BASE + PREFIX + "manifest.json"
    payloads[uri] = raw
    source = {"type": "onnx_bundle", "uri": uri, **metadata(raw)}
    return source, payloads


def test_downloads_multpart_bundle_and_revalidates_cache(tmp_path):
    source, payloads = fixture_bundle()
    requests = []

    def opener(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(payloads[request.full_url])

    fetcher = Fetcher(tmp_path, opener=opener)
    path, hit, digest, size = fetcher.fetch_source(source)
    assert not hit
    assert path.read_bytes() == b"onnx"
    assert (path.parent / "weights/data.bin").read_bytes() == b"weights"
    assert digest == source["sha256"]
    assert len(requests) == 3
    assert fetcher.fetch_source(source)[1]
    assert len(requests) == 3
    (path.parent / "weights/data.bin").write_bytes(b"corrupt")
    with pytest.raises(FetchError, match="checksum/size mismatch"):
        fetcher.fetch_source(source)


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "traversal",
        "missing",
        "corrupt",
        "outside_prefix",
        "duplicate_part",
        "missing_version",
        "bad_size",
        "collision",
    ],
)
def test_rejects_invalid_bundle_before_exposing_entrypoint(tmp_path, case):
    def mutate(manifest):
        if case == "incomplete":
            manifest["status"] = "uploading"
        elif case == "traversal":
            manifest["files"]["../escape"] = metadata(b"x")
        elif case == "missing":
            manifest["files"]["missing.bin"] = metadata(b"x")
        elif case == "corrupt":
            manifest["files"]["weights/data.bin"]["sha256"] = "0" * 64
        elif case == "outside_prefix":
            manifest["objects"][0]["key"] = "models-onnx/other/file"
        elif case == "duplicate_part":
            manifest["objects"][1] = manifest["objects"][0]
        elif case == "missing_version":
            manifest["objects"][0]["version_id"] = "null"
        elif case == "bad_size":
            manifest["files"]["model.onnx"]["size_bytes"] = -1
        elif case == "collision":
            manifest["files"]["MODEL.onnx"] = metadata(b"onnx")

    source, payloads = fixture_bundle(mutate)
    fetcher = Fetcher(tmp_path, opener=lambda request, timeout: io.BytesIO(payloads[request.full_url]))
    with pytest.raises(FetchError):
        fetcher.fetch_source(source)
    assert not (tmp_path / "onnx-bundles" / source["sha256"]).exists()


@pytest.mark.parametrize(
    "uri",
    [
        BASE + "models-onnx/_smoke/run/manifest.json",
        BASE + "models-onnx/model/latest/manifest.json",
        BASE + "other/model.onnx",
        BASE + PREFIX + "nested/model.onnx",
    ],
)
def test_vulcan_host_is_limited_to_versioned_artifacts(uri):
    with pytest.raises(SourcePolicyError):
        SourceSpec.from_mapping({"uri": uri, **metadata(b"x")})


def test_vendored_reader_import_does_not_require_aws_dependencies():
    import ast

    from model_registry import _vulcan_onnx_artifacts

    tree = ast.parse(Path(_vulcan_onnx_artifacts.__file__).read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "botocore.exceptions" for node in tree.body)


@pytest.mark.parametrize("case", ["symlink", "traversal", "undeclared", "duplicate"])
def test_rejects_unsafe_archive_members(tmp_path, case):
    source, payloads = fixture_bundle()
    manifest = json.loads(payloads[source["uri"]])
    original = b"".join(payloads[BASE + obj["key"]] for obj in manifest["objects"])
    target = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original)) as old, tarfile.open(fileobj=target, mode="w") as new:
        for member in old:
            new.addfile(member, old.extractfile(member))
        member = tarfile.TarInfo(
            {"symlink": "link", "traversal": "../escape", "undeclared": "unexpected", "duplicate": "model.onnx"}[case]
        )
        if case == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
        new.addfile(member, io.BytesIO())
    body = target.getvalue()
    key = PREFIX + "onnx-bundle.tar"
    manifest["bundle"] = {"name": "onnx-bundle.tar", **metadata(body)}
    manifest["objects"] = [{"key": key, "version_id": "v1", **metadata(body)}]
    raw = json.dumps(manifest).encode()
    payloads = {source["uri"]: raw, BASE + key: body}
    source.update(metadata(raw))
    fetcher = Fetcher(tmp_path, opener=lambda request, timeout: io.BytesIO(payloads[request.full_url]))
    with pytest.raises(FetchError):
        fetcher.fetch_source(source)
    assert not (tmp_path / "onnx-bundles" / source["sha256"]).exists()


def test_rejects_tampered_part(tmp_path):
    source, payloads = fixture_bundle()
    part = next(url for url in payloads if "part00000" in url)
    payloads[part] = b"x" * len(payloads[part])
    fetcher = Fetcher(tmp_path, opener=lambda request, timeout: io.BytesIO(payloads[request.full_url]))
    with pytest.raises(FetchError, match="SHA-256 mismatch"):
        fetcher.fetch_source(source)


def test_rejects_symlink_in_verified_cache(tmp_path):
    source, payloads = fixture_bundle()
    fetcher = Fetcher(tmp_path, opener=lambda request, timeout: io.BytesIO(payloads[request.full_url]))
    path = fetcher.fetch_source(source)[0]
    sidecar = path.parent / "weights/data.bin"
    sidecar.unlink()
    sidecar.symlink_to(path)
    with pytest.raises(FetchError, match="non-regular"):
        fetcher.fetch_source(source)
