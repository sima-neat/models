from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from model_registry import package as package_module
from model_registry.errors import ValidationError
from model_registry.package import assemble_available_packages, assemble_package

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
MPK_NAME = "resnet_50_w16a16_mpk.tar.gz"


def _compiled_component(tmp_path: Path) -> Path:
    output = tmp_path / "compiled" / "resnet_50" / "w16a16" / "model"
    output.mkdir(parents=True)
    archive = output / MPK_NAME
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
            "build_profile_id": "w16a16",
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


def _model_with_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, install_path: str
) -> tuple[dict, dict]:
    model = copy.deepcopy(
        package_module.load_and_validate_model(
            ROOT / "models" / "resnet_50", ROOT / "schemas" / "model.schema.json"
        )
    )
    payload = tmp_path / "resource.json"
    payload.write_bytes(b'{"fixture":true}\n')
    resource_license = {
        "spdx": "LicenseRef-Fixture-Resource",
        "redistribution": "restricted",
        "attribution": "Fixture resource attribution.",
    }
    resource = {
        "id": "runtime_config",
        "source": {
            "type": "file",
            "uri": "https://example.invalid/resource.json",
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "size_bytes": payload.stat().st_size,
        },
        "install_path": install_path,
        "license": resource_license,
    }
    model["resources"] = [resource]
    model["packages"][0]["resources"] = [resource["id"]]

    class FakeFetcher:
        def __init__(self, _cache_dir: str | Path | None) -> None:
            pass

        def fetch_source(self, _source: dict) -> tuple[Path, str, str, int]:
            return (
                payload,
                resource["source"]["uri"],
                resource["source"]["sha256"],
                payload.stat().st_size,
            )

    monkeypatch.setattr(package_module, "load_and_validate_model", lambda *_args: model)
    monkeypatch.setattr(package_module, "Fetcher", FakeFetcher)
    return model, resource_license


def test_single_component_package_is_the_compiled_mpk(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    result = assemble_package(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages",
        model_id="resnet_50",
        build_profile_id="w16a16",
        package_id="default",
    )
    package_dir = tmp_path / "packages" / "resnet_50" / "w16a16" / "packages" / "default"
    compiled_mpk = compiled / "resnet_50" / "w16a16" / "model" / MPK_NAME
    assert result["archive"] == MPK_NAME
    assert sorted(path.name for path in package_dir.iterdir()) == ["package-result.json", MPK_NAME]
    assert (package_dir / MPK_NAME).read_bytes() == compiled_mpk.read_bytes()
    manifest = result["manifest"]
    assert manifest["components"][0]["id"] == "model"
    assert manifest["components"][0]["path"] == MPK_NAME
    assert manifest["composition"] == {
        "id": "default",
        "type": "single",
        "component": "model",
        "entrypoint": "default",
    }
    assert manifest["files"] == [{
        "path": MPK_NAME,
        "sha256": hashlib.sha256(compiled_mpk.read_bytes()).hexdigest(),
        "size_bytes": compiled_mpk.stat().st_size,
    }]


def test_assemble_available_packages_skips_incomplete_builds(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    results = assemble_available_packages(
        repository=ROOT,
        compiled_dir=compiled,
        output_dir=tmp_path / "packages",
    )
    assert [(item["model_id"], item["build_profile_id"]) for item in results] == [
        ("resnet_50", "w16a16")
    ]


def test_package_with_resources_is_a_flat_deterministic_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, resource_license = _model_with_resource(monkeypatch, tmp_path, "resources/runtime.json")
    compiled = _compiled_component(tmp_path)
    results = [
        assemble_package(
            repository=ROOT,
            compiled_dir=compiled,
            output_dir=tmp_path / output,
            model_id="resnet_50",
            build_profile_id="w16a16",
            package_id="default",
        )
        for output in ("packages-a", "packages-b")
    ]

    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["archive"] == "resnet_50_w16a16_default.zip"
    bundle = tmp_path / "packages-a" / "resnet_50" / "w16a16" / "packages" / "default" / results[0]["archive"]
    with zipfile.ZipFile(bundle) as stream:
        assert stream.namelist() == [MPK_NAME, "resources/runtime.json"]
    assert results[0]["manifest"]["resources"][0]["license"] == resource_license


def _declarative_composition() -> dict:
    return {
        "id": "default",
        "type": "declarative",
        "components": [{"id": "only", "component": "model", "entrypoint": "default"}],
        "connections": [],
        "inputs": [{"name": "input", "to": {"component": "only", "input": "input"}}],
        "outputs": [{"name": "output", "from": {"component": "only", "output": "output"}}],
    }


def test_package_records_declarative_composition_without_shipping_a_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = copy.deepcopy(
        package_module.load_and_validate_model(ROOT / "models" / "resnet_50", ROOT / "schemas" / "model.schema.json")
    )
    model["compositions"] = [_declarative_composition()]
    monkeypatch.setattr(package_module, "load_and_validate_model", lambda *args: model)

    result = assemble_package(
        repository=ROOT,
        compiled_dir=_compiled_component(tmp_path),
        output_dir=tmp_path / "packages",
        model_id="resnet_50",
        build_profile_id="w16a16",
        package_id="default",
    )

    assert result["archive"] == MPK_NAME
    assert result["manifest"]["composition"] == {"id": "default", "type": "declarative"}
    assert [item["path"] for item in result["manifest"]["files"]] == [MPK_NAME]


def test_package_rejects_resource_path_that_collides_with_a_component_mpk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model_with_resource(monkeypatch, tmp_path, MPK_NAME)

    with pytest.raises(ValidationError, match=f"package path is declared more than once: {MPK_NAME}"):
        assemble_package(
            repository=ROOT,
            compiled_dir=_compiled_component(tmp_path),
            output_dir=tmp_path / "packages",
            model_id="resnet_50",
            build_profile_id="w16a16",
            package_id="default",
        )
