from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.dependencies import resolve_benchmark_runtime, resolve_model_compiler, resolve_model_sdk_version
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
    assert runtime["neat_core"] == manifest["neat-core"]
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


def test_sdk_version_is_not_the_compiler_service_release() -> None:
    assert resolve_model_compiler(ROOT)["version"] == "3.0.0.1"
    assert resolve_model_sdk_version(ROOT) == "3.0.0"


@pytest.mark.parametrize("sdk_version", [None, "3.0", "3.0.0.1", "3.0.0-dev", 300])
def test_sdk_version_requires_an_explicit_semantic_version(tmp_path: Path, sdk_version: object) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"model-compiler": {"version": "3.0.0.1", "sdk_version": sdk_version}}))
    with pytest.raises(ValidationError, match="model-compiler.sdk_version"):
        resolve_model_sdk_version(tmp_path)


def test_sdk_version_does_not_make_an_invalid_package_identity_valid(tmp_path: Path) -> None:
    manifest = tmp_path / "deps" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"model-compiler": {"version": "wrong", "sdk_version": "3.0.0"}}))
    with pytest.raises(ValidationError, match="model-compiler.version"):
        resolve_model_sdk_version(tmp_path)
