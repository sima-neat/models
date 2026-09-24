from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.dependencies import resolve_benchmark_runtime, resolve_model_compiler
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def test_repository_model_compiler_dependency_is_pinned() -> None:
    assert resolve_model_compiler(ROOT) == {
        "identity": "3.0.0+develop.e94f03a8f59c",
        "label": "model-compiler-develop-e94f03a8f59c",
        "version": "3.0.0",
    }


def test_repository_benchmark_runtime_uses_source_revision_package() -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["platform-version"] == "3.0.0"
    assert "platform_version" not in manifest["devkit"]

    runtime = resolve_benchmark_runtime(ROOT)
    assert runtime["devkit"] == {"board": "modalix", "os_type": "elxr", "platform_version": "3.0.0"}
    assert runtime["neat_core"] == {"version": "0.4.0", "install_target": "core@release-3.0.0-prep:96d8805fe3c1"}
    assert runtime["benchmarker"]["vulcan"] == {
        "environment": "stg",
        "name": "models-utils/benchmarker",
    }


def test_benchmark_runtime_requires_a_package_name(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    manifest["benchmarker"]["vulcan"].pop("name")
    path = tmp_path / "deps" / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValidationError, match="benchmarker.vulcan is missing: name"):
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
