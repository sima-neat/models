from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.dependencies import resolve_benchmark_runtime, resolve_catalog_inputs, resolve_model_compiler
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def test_repository_model_compiler_dependency_is_pinned() -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["model-compiler"]["version"]
    assert resolve_model_compiler(ROOT) == {
        "identity": version,
        "label": f"model-compiler-{version}",
        "version": version,
    }


def test_repository_benchmark_runtime_uses_source_revision_package() -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    assert "platform_version" not in manifest["devkit"]

    runtime = resolve_benchmark_runtime(ROOT)
    assert runtime["devkit"] == {**manifest["devkit"], "platform_version": manifest["platform-version"]}
    assert runtime["neat_core"] == {"install_target": manifest["neat-core"]["install_target"]}
    assert runtime["llima"] == manifest["llima"]
    assert "benchmarker" not in runtime


def test_benchmark_runtime_requires_a_package_name(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    manifest["neat-core"].pop("install_target")
    path = tmp_path / "deps" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="neat-core.install_target"):
        resolve_benchmark_runtime(tmp_path)


@pytest.mark.parametrize("version", ["2.1", "2.1.3.1.5", "main", "2.1.3-develop", 213, None])
def test_model_compiler_dependency_requires_supported_version(tmp_path: Path, version: object) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"model-compiler": {"version": version}}), encoding="utf-8")

    with pytest.raises(ValidationError, match="MAJOR.MINOR.PATCH"):
        resolve_model_compiler(tmp_path)


def test_model_compiler_dependency_accepts_commit_qualified_develop_build(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "model-compiler": {
                    "channel": "develop",
                    "revision": "e94f03a8f59c",
                    "version": "3.0.0",
                }
            }
        ),
        encoding="utf-8",
    )

    assert resolve_model_compiler(tmp_path) == {
        "identity": "3.0.0+develop.e94f03a8f59c",
        "label": "model-compiler-develop-e94f03a8f59c",
        "version": "3.0.0",
    }


def test_model_compiler_dependency_accepts_service_release(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"model-compiler": {"version": "2.1.3.1"}}),
        encoding="utf-8",
    )

    assert resolve_model_compiler(tmp_path) == {
        "identity": "2.1.3.1",
        "label": "model-compiler-2.1.3.1",
        "version": "2.1.3.1",
    }


@pytest.mark.parametrize(
    "dependency",
    [
        {"channel": "develop", "revision": "e94f03a8f59", "version": "3.0.0"},
        {"channel": "Develop", "revision": "e94f03a8f59c", "version": "3.0.0"},
        {"channel": "develop", "version": "3.0.0"},
    ],
)
def test_model_compiler_dependency_rejects_incomplete_develop_identity(
    tmp_path: Path, dependency: dict[str, str]
) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"model-compiler": dependency}), encoding="utf-8")

    with pytest.raises(ValidationError):
        resolve_model_compiler(tmp_path)


def test_missing_model_compiler_dependency_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must define model-compiler"):
        resolve_model_compiler(tmp_path)


def _manifest_with_core(tmp_path: Path, core: dict[str, object]) -> Path:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    manifest["neat-core"] = core
    path = tmp_path / "deps" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_catalog_inputs_name_the_platform_and_the_core_release(tmp_path: Path) -> None:
    revision = "b1f9556caec1897cbc4bfc7df5cdc699cad370ec"
    repository = _manifest_with_core(tmp_path, {"install_target": "core@v0.6.2", "revision": revision})
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))

    assert resolve_catalog_inputs(repository) == {
        "platform_version": manifest["platform-version"],
        "neat_core_version": "0.6.2",
        "neat_core_revision": revision,
    }


def test_a_core_branch_pin_has_no_catalog_inputs(tmp_path: Path) -> None:
    assert resolve_catalog_inputs(_manifest_with_core(tmp_path, {"install_target": "core@develop"})) is None


@pytest.mark.parametrize("revision", [None, "b1f9556caec1", "B1F9556CAEC1897CBC4BFC7DF5CDC699CAD370EC"])
def test_a_core_release_pin_requires_its_full_commit(tmp_path: Path, revision: object) -> None:
    core = {"install_target": "core@v0.6.2", **({} if revision is None else {"revision": revision})}

    with pytest.raises(ValidationError, match="neat-core.revision must be the full commit"):
        resolve_catalog_inputs(_manifest_with_core(tmp_path, core))


def test_a_core_branch_pin_rejects_a_revision(tmp_path: Path) -> None:
    core = {"install_target": "core@develop", "revision": "b1f9556caec1897cbc4bfc7df5cdc699cad370ec"}

    with pytest.raises(ValidationError, match="applies only to a Core release pin"):
        resolve_catalog_inputs(_manifest_with_core(tmp_path, core))


def test_the_core_revision_changes_no_benchmark_runtime(tmp_path: Path) -> None:
    pinned = _manifest_with_core(tmp_path / "a", {"install_target": "core@v0.6.2", "revision": "0" * 40})
    other = _manifest_with_core(tmp_path / "b", {"install_target": "core@v0.6.2", "revision": "1" * 40})

    assert resolve_benchmark_runtime(pinned) == resolve_benchmark_runtime(other)
