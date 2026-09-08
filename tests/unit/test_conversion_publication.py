from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from model_registry.fetch import FetchError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def publication(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("conversion_publisher", ROOT / "scripts/publish_onnx_conversion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.onnx").write_bytes(b"fixture graph")
    (export / "validation.json").write_text(json.dumps({"status": "passed"}))
    (export / "provenance.json").write_text("{}")
    for key, value in {
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REPOSITORY": "sima-neat/models",
        "GITHUB_SHA": "a" * 40,
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda service: object()))
    uploads = []

    def upload(client, bucket, prefix, bundle, manifest, kms, journal):
        uploads.append(prefix)
        obj = {"key": prefix + "manifest.json", "sha256": "b" * 64, "size_bytes": 100, "version_id": "version"}
        journal.append(obj)
        return manifest, obj

    monkeypatch.setattr(module.artifacts, "publish", upload)
    monkeypatch.setattr(module.Fetcher, "fetch_source", lambda *args: (export / "model.onnx", False, "b" * 64, 100))
    result = {"output_dir": str(export), "model_id": "rf_detr_small", "fingerprint": "c" * 64}
    return module, result, export, uploads


def test_publication_emits_pin_after_verified_download(publication, tmp_path):
    module, result, export, uploads = publication
    output = tmp_path / "publication"
    source = module.publish(result, "kms-key", output)
    assert uploads == [f"models-onnx/rf_detr_small/{'c' * 64}/123-1/"]
    assert yaml.safe_load((output / "source.yaml").read_text()) == {"source": source}
    assert source["type"] == "onnx_bundle"
    assert source["sha256"] == "b" * 64


def test_failed_validation_is_never_published(publication, tmp_path):
    module, result, export, uploads = publication
    (export / "validation.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="Only validated exports"):
        module.publish(result, "kms-key", tmp_path / "publication")
    assert uploads == []


def test_failed_download_does_not_emit_source_pin(publication, tmp_path, monkeypatch):
    module, result, export, uploads = publication

    def corrupt(*args):
        raise FetchError("checksum mismatch")

    monkeypatch.setattr(module.Fetcher, "fetch_source", corrupt)
    output = tmp_path / "publication"
    with pytest.raises(FetchError, match="checksum mismatch"):
        module.publish(result, "kms-key", output)
    assert uploads
    assert (output / "publication.json").is_file()
    assert not (output / "source.yaml").exists()
