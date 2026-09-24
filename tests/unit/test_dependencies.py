from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.dependencies import resolve_benchmark_runtime, resolve_model_compiler
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]


def test_repository_model_compiler_dependency_is_pinned() -> None:
    assert resolve_model_compiler(ROOT) == {
        "label": "model-compiler-2.1.3.1",
        "version": "2.1.3.1",
    }


def test_repository_benchmark_runtime_uses_source_revision_package() -> None:
    manifest = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["platform-version"] == "2.1.3"
    assert "platform_version" not in manifest["devkit"]

    runtime = resolve_benchmark_runtime(ROOT)
    assert runtime["devkit"] == {"board": "modalix", "os_type": "elxr", "platform_version": "2.1.3"}
    assert runtime["neat_core"] == {"version": "0.4.0", "install_target": "core@v0.4.0"}
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


@pytest.mark.parametrize("version", ["2.1", "2.1.3.1.5", "develop", "2.1.3-develop", 213, None])
def test_model_compiler_dependency_requires_semantic_version(tmp_path: Path, version: object) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"model-compiler": {"version": version}}), encoding="utf-8")

    with pytest.raises(ValidationError, match="MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH.SERVICE"):
        resolve_model_compiler(tmp_path)


def test_model_compiler_dependency_accepts_service_release(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps({"model-compiler": {"version": "2.1.3.1"}}),
        encoding="utf-8",
    )

    assert resolve_model_compiler(tmp_path) == {
        "label": "model-compiler-2.1.3.1",
        "version": "2.1.3.1",
    }


def test_missing_model_compiler_dependency_fails(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must define model-compiler"):
        resolve_model_compiler(tmp_path)
