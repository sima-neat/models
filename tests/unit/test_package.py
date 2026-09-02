from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

from model_registry.package import assemble_available_packages, assemble_package

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"


def _compiled_component(tmp_path: Path) -> Path:
    output = tmp_path / "compiled" / "resnet_50" / "modalix_bf16" / "model"
    output.mkdir(parents=True)
    archive = output / "resnet_50_mpk.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        payload = b"compiled model"
        info = tarfile.TarInfo("model.elf")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    result = {
        "archive": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "size_bytes": archive.stat().st_size,
        "provenance": {
            "model_id": "resnet_50",
            "build_profile_id": "modalix_bf16",
            "component_id": "model",
            "target": "modalix",
            "source_sha256": SOURCE_SHA,
            "git_sha": "a" * 40,
            "environment": "staging",
            "compiler_version": "2.1.3",
            "compiler_label": "model-compiler-2.1.3",
        },
    }
    (output / "compile-result.json").write_text(json.dumps(result), encoding="utf-8")
    return tmp_path / "compiled"


def test_assembles_deterministic_retrieval_bundle_with_intact_component_mpk(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    first = assemble_package(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages-a",
        model_id="resnet_50",
        build_profile_id="modalix_bf16",
        package_id="default",
    )
    second = assemble_package(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages-b",
        model_id="resnet_50",
        build_profile_id="modalix_bf16",
        package_id="default",
    )
    assert first["sha256"] == second["sha256"]
    bundle = tmp_path / "packages-a" / "resnet_50" / "modalix_bf16" / "packages" / "default" / first["archive"]
    with tarfile.open(bundle) as stream:
        assert stream.getnames() == [
            "resnet_50/manifest.json",
            "resnet_50/components/model/resnet_50_mpk.tar.gz",
        ]
        manifest = json.load(stream.extractfile("resnet_50/manifest.json"))
    assert manifest["components"][0]["id"] == "model"
    assert manifest["composition"] == {
        "id": "default",
        "type": "single",
        "component": "model",
        "entrypoint": "default",
    }


def test_assemble_available_packages_skips_incomplete_builds(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    results = assemble_available_packages(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages",
    )
    assert [(item["model_id"], item["build_profile_id"]) for item in results] == [
        ("resnet_50", "modalix_bf16")
    ]
