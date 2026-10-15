from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from model_registry.artifact import (
    pack_llima,
    record_compile_outputs,
    validate_compile_outputs,
    verify_artifact,
    verify_llima,
)
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model
from model_registry.package import assemble_package
from model_registry.registry import S3Object, publish_compilations, publish_packages
from model_registry.source import source_identity
from model_registry.stage_keys import compile_key

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3_5_2b"
PROFILE = "modalix_a16w4_4k"


def deployment(path: Path) -> Path:
    (path / "devkit").mkdir(parents=True)
    (path / "elf_files").mkdir()
    (path / "npy_files").mkdir()
    (path / "devkit/vlm_config.json").write_text(
        json.dumps(
            {
                "lm_cfg": {},
                "pipeline_cfg": {"max_num_tokens": 4096, "language_group_size": 128},
                "vm_cfg": {"image_size": [448, 448]},
            }
        )
    )
    (path / "devkit/tokenizer.json").write_text("{}")
    (path / "elf_files/language.elf").write_bytes(b"\x7fELFsynthetic-language")
    (path / "elf_files/vision.elf").write_bytes(b"\x7fELFsynthetic-vision")
    (path / "npy_files/embeddings.npy").write_bytes(b"synthetic-weights")
    return path


def test_native_artifact_and_package_preserve_the_complete_deployment(tmp_path):
    source = deployment(tmp_path / "deployed")
    dist = tmp_path / "compiled"
    profile = dist / MODEL / PROFILE
    component = profile / "model"
    archive = component / f"{MODEL}_llima.zip"
    pack_llima(source, archive)
    model = load_and_validate_model(ROOT / "models" / MODEL)
    compiler = resolve_model_compiler(ROOT)
    provenance = {
        "model_id": MODEL,
        "build_profile_id": PROFILE,
        "component_id": "model",
        "target": "modalix",
        "source_sha256": source_identity(model["components"][0]["source"]),
        "git_sha": "a" * 40,
        "environment": "staging",
        "compiler_version": compiler["version"],
        "compiler_label": compiler["label"],
        "build_fingerprint": compile_key(ROOT, MODEL, PROFILE, "model", compiler["identity"]),
    }
    result = verify_artifact(component, write_manifest=True, provenance=provenance)
    assert result["format"] == "llima"
    record_compile_outputs(
        profile, model_id=MODEL, build_profile_id=PROFILE, compile_invocation_id="model", component_ids=["model"]
    )
    manifest = validate_compile_outputs(
        profile / "compile-outputs.json",
        profile,
        model_id=MODEL,
        build_profile_id=PROFILE,
        compile_invocation_id="model",
        component_ids=["model"],
    )
    assert manifest["outputs"][0]["artifact_path"] == str(archive)
    calls = []

    def request(method, path, body, key):
        calls.append((method, path, body))
        return {"run_id": "native-run"} if path == "/v1/runs" else {"ok": True}

    class Store:
        def ensure_native_sha256(self, path, key, sha256, *, media_type="application/gzip"):
            return S3Object(key=key, version_id="1", sha256=sha256, size_bytes=path.stat().st_size)

    publish_args = dict(
        api_url="https://registry.example.invalid",
        aws_region="us-west-2",
        bucket="models",
        kms_key_id=None,
        environment="staging",
        branch="feature/llima-models",
        git_sha="a" * 40,
        github_run_id="1",
        github_run_attempt="1",
        github_run_url="https://example.invalid/run/1",
        request=request,
        artifact_store=Store(),
    )
    published = publish_compilations(repository=ROOT, dist_dir=dist, **publish_args)
    assert published["count"] == 1
    assert any(path.endswith("/artifacts/llima") and body["media_type"] == "application/zip" for _, path, body in calls)
    assert any(
        path.endswith("/inputs/source-component") and body["input_type"] == "model-source-file-set"
        for _, path, body in calls
    )
    (component / "registry-publication.json").write_text(json.dumps(published))
    package = assemble_package(
        repository=ROOT,
        compiled_dir=dist,
        output_dir=tmp_path / "packages",
        model_id=MODEL,
        build_profile_id=PROFILE,
        package_id="default",
    )
    package_path = tmp_path / "packages" / MODEL / PROFILE / "packages/default" / package["archive"]
    with zipfile.ZipFile(package_path) as contents:
        assert not any(name.endswith((".zip", ".tar.gz")) for name in contents.namelist())
        for path in source.rglob("*"):
            if path.is_file():
                assert contents.read(f"{MODEL}/{path.relative_to(source).as_posix()}") == path.read_bytes()
        metadata = json.loads(contents.read(f"{MODEL}/manifest.json"))
        assert metadata["components"][0]["format"] == "llima"
        assert metadata["components"][0]["path"] == "."
    assert publish_packages(repository=ROOT, package_dir=tmp_path / "packages", **publish_args)["count"] == 1


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "./alias", "elf_files//alias", "bad\\path", "manifest.json"]
)
def test_native_archive_rejects_unsafe_or_reserved_paths(tmp_path, unsafe):
    archive = tmp_path / "bad_llima.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(unsafe, b"invalid")
    with pytest.raises(ValidationError, match="unsafe|duplicate"):
        verify_llima(archive)


def test_native_pack_rejects_links_and_incomplete_deployments(tmp_path):
    source = deployment(tmp_path / "deployed")
    archive = tmp_path / "model_llima.zip"
    (source / "link").symlink_to(source / "devkit/tokenizer.json")
    with pytest.raises(ValidationError, match="symbolic link"):
        pack_llima(source, archive)
    (source / "link").unlink()
    (source / "devkit/vlm_config.json").unlink()
    with pytest.raises(ValidationError, match="invalid LLiMa deployment"):
        pack_llima(source, archive)
    assert not archive.exists()
