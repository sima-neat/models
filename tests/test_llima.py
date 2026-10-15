from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from threading import Lock

import pytest
from model_registry.artifact import (
    deployment_artifact_name,
    deployment_files,
    record_compile_outputs,
    validate_compile_outputs,
    verify_artifact,
    verify_llima,
)
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model
from model_registry.package import assemble_package
from model_registry.registry import (
    S3Object,
    _download_registry_file,
    publish_compilations,
    publish_packages,
    stage_catalog_build,
)
from model_registry.source import source_identity
from model_registry.stage_keys import compile_key

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3_5_2b"
PROFILE = "modalix_a16w4_4k"


def deployment(path: Path) -> Path:
    (path / "devkit").mkdir(parents=True)
    (path / "elf_files").mkdir()
    (path / "npy_files").mkdir()
    (path / "devkit/vlm_config.json").write_text(json.dumps({
        "lm_cfg": {},
        "pipeline_cfg": {"max_num_tokens": 4096, "language_group_size": 128},
        "vm_cfg": {"image_size": [448, 448]},
    }))
    (path / "devkit/tokenizer.json").write_text("{}")
    (path / "elf_files/language.elf").write_bytes(b"\x7fELFsynthetic-language")
    (path / "elf_files/vision.elf").write_bytes(b"\x7fELFsynthetic-vision")
    (path / "npy_files/embeddings.npy").write_bytes(b"synthetic-weights")
    (path / "empty.txt").write_bytes(b"")
    return path


class Registry:
    """Capture the real publisher requests and serve their immutable artifacts back."""
    def __init__(self):
        self.calls = []
        self.runs = {}
        self.artifacts = {}
        self.objects = {}
        self.uploads = 0
        self.lock = Lock()

    def request(self, method, path, body, key):
        self.calls.append((method, path, body))
        if path == "/v1/runs":
            run_id = f"run-{len(self.runs)}"
            self.runs[run_id] = body
            return {"run_id": run_id}
        if "/artifacts/" in path:
            self.artifacts[path] = {"name": path.rsplit("/", 1)[1], **body}
        return {"ok": True}

    def ensure_native_sha256(self, path, key, sha256, *, media_type="application/gzip"):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256
        with self.lock:
            if key not in self.objects:
                self.objects[key] = data
                self.uploads += 1
        return S3Object(key=key, version_id="1", sha256=sha256, size_bytes=len(data))

    def args(self):
        return dict(
            api_url="https://registry.example.invalid", aws_region="us-west-2", bucket="models",
            kms_key_id=None, environment="staging", branch="feature/llima-models", git_sha="a" * 40,
            github_run_id="1", github_run_attempt="1", github_run_url="https://example.invalid/run/1",
            request=self.request, artifact_store=self,
        )


@pytest.fixture
def compiled(tmp_path):
    dist = tmp_path / "compiled"
    profile = dist / MODEL / PROFILE
    component = profile / "model"
    deployment(component / "deployment")
    model = load_and_validate_model(ROOT / "models" / MODEL)
    compiler = resolve_model_compiler(ROOT)
    provenance = {
        "model_id": MODEL, "build_profile_id": PROFILE, "component_id": "model", "target": "modalix",
        "source_sha256": source_identity(model["components"][0]["source"]),
        "git_sha": "a" * 40, "environment": "staging", "compiler_version": compiler["version"],
        "compiler_label": compiler["label"],
        "build_fingerprint": compile_key(ROOT, MODEL, PROFILE, "model", compiler["identity"]),
    }
    verify_artifact(component, write_manifest=True, provenance=provenance)
    record_compile_outputs(
        profile, model_id=MODEL, build_profile_id=PROFILE, compile_invocation_id="model", component_ids=["model"]
    )
    return dist


def test_native_deployment_publish_package_and_retrieve_without_archiving(compiled, tmp_path):
    component = compiled / MODEL / PROFILE / "model"
    source = component / "deployment"
    manifest = validate_compile_outputs(
        component.parent / "compile-outputs.json", component.parent,
        model_id=MODEL, build_profile_id=PROFILE, compile_invocation_id="model", component_ids=["model"],
    )
    assert manifest["outputs"][0]["artifact_path"] == str(component / "deployment-manifest.json")
    registry = Registry()
    published = publish_compilations(repository=ROOT, dist_dir=compiled, **registry.args())
    compile_run = published["runs"][0]["run_id"]
    source_input = next(body for _, path, body in registry.calls if path.endswith("/inputs/source-component"))
    assert source_input["metadata"]["revision"] == "185ccd6cb4e420e47cc7b6c4113618b02ec4226d"
    (component / "registry-publication.json").write_text(json.dumps(published))
    package = assemble_package(
        repository=ROOT, compiled_dir=compiled, output_dir=tmp_path / "packages",
        model_id=MODEL, build_profile_id=PROFILE, package_id="default",
    )
    package_dir = tmp_path / "packages" / MODEL / PROFILE / "packages/default"
    assert package["layout"] == "directory"
    assert not list(tmp_path.rglob("*.zip"))
    for path in source.rglob("*"):
        if path.is_file():
            assert (package_dir / "deployment" / path.relative_to(source)).read_bytes() == path.read_bytes()
    payload_uploads = registry.uploads
    packages = publish_packages(repository=ROOT, package_dir=tmp_path / "packages", **registry.args())
    # Only the new package manifest uploads; every compiled payload object is reused.
    assert registry.uploads == payload_uploads + 1
    package_run = packages["packages"][0]["run_id"]
    artifact = registry.artifacts[f"/v1/runs/{package_run}/artifacts/package"]
    assert artifact["media_type"] == "application/json"
    assert artifact["metadata"]["deployment_size_bytes"] == sum(path.stat().st_size for path in source.rglob("*")
                                                                  if path.is_file())
    for item in package["manifest"]["files"]:
        registered = registry.artifacts[f"/v1/runs/{package_run}/artifacts/{item['artifact']}"]
        assert (registered["sha256"], registered["size_bytes"]) == (item["sha256"], item["size_bytes"])
    assert all(artifact["retention_class"] == "on-demand" for artifact in registry.artifacts.values())
    assert all(run["metadata"]["build_on_demand"] for run in registry.runs.values())

    catalog = {"catalog_revision": "b" * 40, "models": [{"id": MODEL, "builds": [{
        "build_profile_id": PROFILE, "compile_invocation_id": "model", "component_id": "model",
        "run_id": compile_run, "artifact_sha256": published["runs"][0]["artifact_sha256"],
    }]}]}

    def read(url):
        if "/catalogs/current?" in url:
            return {"catalog": {"document": catalog}}
        path = url.removeprefix("https://registry.example.invalid")
        if path in registry.artifacts:
            record = registry.artifacts[path]
            return {"artifact": {**record, "download_url": record["s3_key"]}}
        return {"run": {"toolchain_version": resolve_model_compiler(ROOT)["version"]}}

    restore_args = dict(
        repository=ROOT, output_dir=tmp_path / "restored", api_url="https://registry.example.invalid",
        branch="develop", catalog_revision="b" * 40, model_id=MODEL, build_profile_id=PROFILE,
        compile_invocation_id="model", read_request=read, download_request=registry.objects.__getitem__,
    )
    stage_catalog_build(**restore_args)
    restored = tmp_path / "restored" / MODEL / PROFILE / "model/deployment"
    assert verify_llima(restored) == verify_llima(source)
    assert json.loads((restored / "devkit/vlm_config.json").read_text())["vm_cfg"]["image_size"] == [448, 448]
    stale = restored / "removed-by-newer-build.bin"
    stale.write_bytes(b"old payload")
    downloads = []

    def download(url):
        downloads.append(url)
        return registry.objects[url]

    stage_catalog_build(**{**restore_args, "download_request": download})
    assert not stale.exists()
    assert len(downloads) == 1  # Only the manifest is fetched; verified payload files are reused.
    assert verify_llima(restored) == verify_llima(source)


def test_native_publication_rejects_changed_payload_before_creating_runs(compiled):
    (compiled / MODEL / PROFILE / "model/deployment/elf_files/language.elf").write_bytes(b"corrupt")
    registry = Registry()
    with pytest.raises(ValidationError, match="manifest"):
        publish_compilations(repository=ROOT, dist_dir=compiled, **registry.args())
    assert registry.calls == []


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "./alias", "elf_files//alias", "bad\\path",
                                    "manifest.json", "manifest.json/alias", ".", "foo.", "a:b"])
def test_deployment_manifest_rejects_unsafe_paths(unsafe):
    item = {"path": unsafe, "artifact": deployment_artifact_name(unsafe), "sha256": "a" * 64, "size_bytes": 1}
    with pytest.raises(ValidationError, match="unsafe"):
        deployment_files({"files": [item]})


def test_deployment_manifest_rejects_aliases_and_overlapping_paths():
    def item(path):
        return {"path": path, "artifact": deployment_artifact_name(path), "sha256": "a" * 64, "size_bytes": 1}
    for files in ([item("a"), item("A")], [item("a"), item("a/b")]):
        with pytest.raises(ValidationError, match="duplicate|overlap"):
            deployment_files({"files": files})


def test_native_deployment_rejects_links_and_incomplete_configuration(tmp_path):
    source = deployment(tmp_path / "deployed")
    (source / "link").symlink_to(source / "devkit/tokenizer.json")
    with pytest.raises(ValidationError, match="unsafe"):
        verify_llima(source)
    (source / "link").unlink()
    (source / "devkit/vlm_config.json").unlink()
    with pytest.raises(ValidationError, match="invalid LLiMa deployment"):
        verify_llima(source)


def test_registry_download_streams_and_replaces_only_verified_files(tmp_path, monkeypatch):
    payload = b"deployment-payload" * (200 * 1024)
    destination = tmp_path / "weights.npy"
    previous = b"previous verified payload"
    destination.write_bytes(previous)
    reads = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            assert size == 1024 * 1024
            assert destination.read_bytes() == previous
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr("model_registry.registry.urlopen", lambda *args, **kwargs: Response(payload))
    artifact = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    _download_registry_file("https://example.invalid/weights", destination, artifact)
    assert destination.read_bytes() == payload
    assert len(reads) > 3
    assert sorted(tmp_path.iterdir()) == [destination]

    destination.write_bytes(previous)
    with pytest.raises(ValidationError, match="SHA-256"):
        _download_registry_file("https://example.invalid/weights", destination,
                                {**artifact, "sha256": "0" * 64})
    assert destination.read_bytes() == previous
    assert sorted(tmp_path.iterdir()) == [destination]
