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
from model_registry.artifact import deployment_files
from model_registry.errors import ValidationError
from model_registry.package import assemble_available_packages, assemble_package

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
MPK_NAME = "resnet_50_w16a16_mpk.tar.gz"


def _compiled_component(tmp_path: Path, component_id: str = "model", archive_name: str = MPK_NAME) -> Path:
    output = tmp_path / "compiled" / "resnet_50" / "w16a16" / component_id
    output.mkdir(parents=True)
    archive = output / archive_name
    with tarfile.open(archive, "w:gz") as stream:
        payload = f"compiled {component_id}".encode()
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
            "component_id": component_id,
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


def _two_component_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, head_archive: str) -> Path:
    """Package resnet_50 with a second component whose compiled MPK is named head_archive."""
    model = copy.deepcopy(
        package_module.load_and_validate_model(ROOT / "models" / "resnet_50", ROOT / "schemas" / "model.schema.json")
    )
    model["components"].append({**copy.deepcopy(model["components"][0]), "id": "head"})
    model["packages"][0]["components"] = ["head", "model"]
    monkeypatch.setattr(package_module, "load_and_validate_model", lambda *_args: model)
    _compiled_component(tmp_path)
    return _compiled_component(tmp_path, "head", head_archive)


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


def test_multi_component_package_is_a_flat_deterministic_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _two_component_model(monkeypatch, tmp_path, "head_mpk.tar.gz")
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
        assert stream.namelist() == ["head_mpk.tar.gz", MPK_NAME]
    assert results[0]["manifest"]["resources"] == []


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


def test_package_rejects_components_whose_mpks_share_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _two_component_model(monkeypatch, tmp_path, MPK_NAME)

    with pytest.raises(ValidationError, match=f"package path is declared more than once: {MPK_NAME}"):
        assemble_package(
            repository=ROOT,
            compiled_dir=compiled,
            output_dir=tmp_path / "packages",
            model_id="resnet_50",
            build_profile_id="w16a16",
            package_id="default",
        )


def _model_with_resource(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, install_path: str) -> Path:
    """resnet_50 with one pinned two-file resource, fetched from tmp_path instead of the network."""
    model = copy.deepcopy(
        package_module.load_and_validate_model(ROOT / "models" / "resnet_50", ROOT / "schemas" / "model.schema.json")
    )
    source_root = tmp_path / "resource"
    (source_root / "voices").mkdir(parents=True)
    (source_root / "config.json").write_bytes(b'{"fixture":true}\n')
    (source_root / "voices" / "a.json").write_bytes(b"[1, 2]\n")
    files = {
        name: {"sha256": hashlib.sha256((source_root / name).read_bytes()).hexdigest(),
               "size_bytes": (source_root / name).stat().st_size}
        for name in ("config.json", "voices/a.json")
    }
    model["resources"] = {
        "runtime": {
            "source": {"repository": "example/runtime", "revision": "0" * 40, "main_file": "config.json",
                       "files": files},
            "install_path": install_path,
        }
    }

    class FakeFetcher:
        def __init__(self, _cache_dir: str | Path | None) -> None:
            pass

        def fetch_source(self, _source: dict) -> tuple[Path, bool, str, int]:
            return source_root / "config.json", True, "b" * 64, sum(item["size_bytes"] for item in files.values())

    monkeypatch.setattr(package_module, "load_and_validate_model", lambda *_args: model)
    monkeypatch.setattr(package_module, "Fetcher", FakeFetcher)
    return _compiled_component(tmp_path)


def _declare_runtime_file(compiled: Path, name: str, payload: bytes) -> None:
    profile_dir = compiled / "resnet_50" / "w16a16"
    (profile_dir / name).write_bytes(payload)
    (profile_dir / "compile-outputs.json").write_text(json.dumps({
        "runtime_files": [{"path": name, "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")


def test_package_with_resources_and_runtime_files_is_a_deterministic_folder_of_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _model_with_resource(monkeypatch, tmp_path, "runtime")
    _declare_runtime_file(compiled, "tables.npz", b"tables")
    results = [
        assemble_package(
            repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / output, model_id="resnet_50",
            build_profile_id="w16a16", package_id="default", resource_cache_dir=tmp_path / "cache",
        )
        for output in ("packages-a", "packages-b")
    ]

    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["layout"] == "directory"
    assert results[0]["manifest_path"] == "deployment/manifest.json"
    assert "archive" not in results[0]
    package_dir = tmp_path / "packages-a" / "resnet_50" / "w16a16" / "packages" / "default"
    assert not list(package_dir.rglob("*.zip"))
    manifest = results[0]["manifest"]
    assert [item["path"] for item in manifest["files"]] == [
        MPK_NAME, "runtime/config.json", "runtime/voices/a.json", "tables.npz",
    ]
    assert deployment_files(manifest) == manifest["files"]
    deployment = package_dir / "deployment"
    assert (deployment / "tables.npz").read_bytes() == b"tables"
    assert (deployment / MPK_NAME).read_bytes() == (compiled / "resnet_50/w16a16/model" / MPK_NAME).read_bytes()
    assert json.loads((deployment / "manifest.json").read_text()) == manifest
    assert results[0]["size_bytes"] == sum(item["size_bytes"] for item in manifest["files"])
    assert manifest["components"][0]["path"] == MPK_NAME
    assert manifest["resources"] == [{
        "id": "runtime", "paths": ["runtime/config.json", "runtime/voices/a.json"],
        "source_sha256": "b" * 64, "size_bytes": 24,
    }]


def test_package_of_an_mpk_and_a_runtime_file_is_a_folder(tmp_path: Path) -> None:
    compiled = _compiled_component(tmp_path)
    _declare_runtime_file(compiled, "tables.npz", b"tables")

    result = assemble_package(
        repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages", model_id="resnet_50",
        build_profile_id="w16a16", package_id="default",
    )

    assert result["layout"] == "directory"
    assert [item["path"] for item in result["manifest"]["files"]] == [MPK_NAME, "tables.npz"]


def test_build_on_demand_package_rejects_files_beyond_its_mpks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _model_with_resource(monkeypatch, tmp_path, "runtime")
    # The patched loader returns the same model object on every call.
    package_module.load_and_validate_model()["build-on-demand"] = True

    with pytest.raises(ValidationError, match="build-on-demand package carries only its MPKs"):
        assemble_package(
            repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages", model_id="resnet_50",
            build_profile_id="w16a16", package_id="default", resource_cache_dir=tmp_path / "cache",
        )


def test_package_rejects_a_runtime_file_that_changed_after_the_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _model_with_resource(monkeypatch, tmp_path, "runtime")
    _declare_runtime_file(compiled, "tables.npz", b"tables")
    (compiled / "resnet_50" / "w16a16" / "tables.npz").write_bytes(b"edited")

    with pytest.raises(ValidationError, match="runtime file tables.npz is missing or changed"):
        assemble_package(
            repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages", model_id="resnet_50",
            build_profile_id="w16a16", package_id="default",
        )


def test_package_rejects_a_runtime_file_that_collides_with_a_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _model_with_resource(monkeypatch, tmp_path, "runtime")
    (compiled / "resnet_50" / "w16a16" / "runtime").mkdir()
    _declare_runtime_file(compiled, "runtime/config.json", b"x")

    with pytest.raises(ValidationError, match="package path is declared more than once: runtime/config.json"):
        assemble_package(
            repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages", model_id="resnet_50",
            build_profile_id="w16a16", package_id="default",
        )


@pytest.mark.parametrize("grouped", [False, True])
def test_package_preserves_runner_audit(tmp_path, monkeypatch, grouped):
    compiled = (_two_component_model(monkeypatch, tmp_path, "head.tar.gz")
                if grouped else _compiled_component(tmp_path))
    runner = {
        "requested_runner_class": "expensive", "runner_class": "expensive",
        "requested_runner_market": "on-demand", "runner_market": "on-demand",
        "runner_actual_market": "on-demand", "runner_name": "runner-test",
        "runner_instance_id": "i-test", "runner_instance_type": "c8gd.16xlarge",
        "runner_approval_id": "sima-neat/models:resnet_50", "runner_approval_spec_revision": "a" * 64,
    }
    for path in compiled.rglob("compile-result.json"):
        result = json.loads(path.read_text())
        result["provenance"].update(runner)
        path.write_text(json.dumps(result))
    result = assemble_package(
        repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages",
        model_id="resnet_50", build_profile_id="w16a16", package_id="default",
    )
    assert {field: result["manifest"]["provenance"][field] for field in runner} == runner
    if grouped:
        path = compiled / "resnet_50/w16a16/head/compile-result.json"
        data = json.loads(path.read_text())
        data["provenance"]["runner_instance_id"] = "i-other"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError, match="incompatible"):
            assemble_package(
                repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages",
                model_id="resnet_50", build_profile_id="w16a16", package_id="default",
            )
