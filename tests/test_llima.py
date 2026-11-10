from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from threading import Lock

import pytest
import yaml
from model_registry.artifact import (
    deployment_artifact_name,
    deployment_files,
    record_compile_outputs,
    save_llima_recipe,
    validate_compile_outputs,
    verify_artifact,
    verify_llima,
)
from model_registry.candidates import DEPLOYMENTS, publish_candidate
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.fingerprints import COMPILE_TOOLING, compile_fingerprint
from model_registry.model import load_and_validate_model, load_models
from model_registry.registry import (
    RegistryPublicationError,
    S3ArtifactStore,
    S3Object,
    _download_registry_file,
    _preflight_compilations,
    canonical_sha256,
    publish_compilations,
    stage_catalog_build,
)
from model_registry.source import source_identity

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3_5_0_8b"
MTP_MODEL = "gemma_4_e2b_it_textonly_mtp"
PROFILE = "w4a16"


def deployment(path: Path) -> Path:
    (path / "devkit").mkdir(parents=True)
    (path / "elf_files").mkdir()
    (path / "npy_files").mkdir()
    (path / "devkit/vlm_config.json").write_text(json.dumps({
        "lm_cfg": {},
        "pipeline_cfg": {"max_num_tokens": 4096, "input_token_group_size": 128},
        "vm_cfg": {"image_size": [448, 448]},
    }))
    (path / "devkit/tokenizer.json").write_text("{}")
    (path / "devkit/backup").mkdir()
    (path / "devkit/backup/tokenizer.json").write_text("{}")
    (path / "elf_files/language.elf").write_bytes(b"\x7fELFsynthetic-language")
    (path / "elf_files/vision.elf").write_bytes(b"\x7fELFsynthetic-vision")
    (path / "npy_files/embeddings.npy").write_bytes(b"synthetic-weights")
    (path / "compile.sh").write_text('#!/usr/bin/env bash\nllima-compile "$@"\n')
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
        self.candidate = None
        self.bindings = {}
        self.completed = set()
        self.responses = {}
        self.fail_once = None

    def request(self, method, path, body, key):
        self.calls.append((method, path, body))
        if method != "GET" and key in self.responses:
            previous, response = self.responses[key]
            assert previous == (method, path, body)
            return response
        if path == DEPLOYMENTS and method == "POST":
            self.candidate = {
                **body, "deployment_id": "11111111-1111-4111-8111-111111111111", "state": "assembling",
                "contract_digest": canonical_sha256(body["contract"]), "manifest_digest": None,
                "upload_prefix": "models/candidates/11111111-1111-4111-8111-111111111111/",
                "assembly_expires_at": "2099-01-01T00:00:00Z", "availability": "assembling",
                "protection": [{"reason": "assembly", "expires_at": "2099-01-01T00:00:00Z"}],
            }
            response = dict(self.candidate)
        elif path.startswith(DEPLOYMENTS) and method == "GET":
            return dict(self.candidate)
        elif path.startswith(DEPLOYMENTS) and "/runs/" in path:
            run_id = path.rsplit("/", 1)[1]
            role = body["role"]
            assert self.candidate["contract"]["roles"][role]["run_type"] == self.runs[run_id]["run_type"]
            assert role not in self.bindings or self.bindings[role] == run_id
            self.bindings[role] = run_id
            response = {"ok": True}
        elif path.startswith(DEPLOYMENTS) and path.endswith("/finalize"):
            assert body["contract_digest"] == self.candidate["contract_digest"]
            assert set(self.bindings) == set(self.candidate["contract"]["roles"])
            for role, run_id in self.bindings.items():
                spec = self.candidate["contract"]["roles"][role]
                assert run_id in self.completed
                assert all(f"/v1/runs/{run_id}/artifacts/{name}" in self.artifacts for name in spec["artifacts"])
                assert all(any(path == f"/v1/runs/{run_id}/inputs/{name}" for _, path, _ in self.calls)
                           for name in spec["inputs"])
            self.candidate.update(state="finalized", manifest_digest="f" * 64, availability="registered",
                                  protection=[{"reason": "latest_successful", "expires_at": None}])
            response = dict(self.candidate)
        elif path == "/v1/runs":
            run_id = f"run-{len(self.runs)}"
            self.runs[run_id] = body
            response = {"run_id": run_id}
        else:
            if "/artifacts/" in path:
                if body["retention_class"] == "on-demand":
                    assert path.split("/")[3] in self.bindings.values()
                if self.fail_once and path.endswith(self.fail_once):
                    self.fail_once = None
                    raise RegistryPublicationError("temporary publication failure")
                self.artifacts[path] = {"name": path.rsplit("/", 1)[1], **body}
            if path.endswith("/finalize"):
                self.completed.add(path.split("/")[3])
            response = {"ok": True}
        self.responses[key] = ((method, path, body), response)
        return response

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

COMPILE_TIMING = json.dumps({
    "scope": "compile_recipe", "started_at": "2026-08-13T00:00:00Z", "finished_at": "2026-08-13T01:00:00Z",
})


@pytest.fixture
def compiled(tmp_path, request):
    dist = tmp_path / "compiled"
    profile = dist / MODEL / PROFILE
    component = profile / "model"
    deployment(component / "deployment")
    model = load_and_validate_model(ROOT / "models" / MODEL)
    compiler = resolve_model_compiler(ROOT)
    provenance = {
        "model_id": MODEL, "build_profile_id": PROFILE, "component_id": "model", "target": "modalix",
        "source_sha256": source_identity(model["components"][0]["source"]),
        "git_sha": "a" * 40, "environment": getattr(request, "param", "staging"),
        "compiler_version": compiler["version"],
        "compiler_label": compiler["label"],
        "compile_fingerprint": compile_fingerprint(ROOT, MODEL, PROFILE, PROFILE, compiler["identity"]),
    }
    verify_artifact(component, write_manifest=True, provenance=provenance)
    record_compile_outputs(
        profile, model_id=MODEL, build_profile_id=PROFILE, compile_invocation_id=PROFILE, component_ids=["model"]
    )
    (profile / "compile-timing.json").write_text(COMPILE_TIMING)
    return dist


@pytest.mark.parametrize("compiled,environment", [("staging", "staging"), ("production", "production")],
                         indirect=["compiled"])
def test_native_deployment_publish_package_and_retrieve_without_archiving(compiled, tmp_path, environment):
    component = compiled / MODEL / PROFILE / "model"
    source = component / "deployment"
    manifest = validate_compile_outputs(
        component.parent / "compile-outputs.json", component.parent,
        model_id=MODEL, build_profile_id=PROFILE, compile_invocation_id=PROFILE, component_ids=["model"],
    )
    assert manifest["outputs"][0]["artifact_path"] == str(component / "deployment-manifest.json")
    registry = Registry()
    arguments = {**registry.args(), "environment": environment}
    candidate = publish_candidate(
        repository=ROOT, dist_dir=compiled, output_dir=tmp_path / "candidate",
        model_id=MODEL, build_profile_id=PROFILE, **arguments,
    )
    assert candidate["state"] == "finalized"
    published = json.loads((component / "registry-publication.json").read_text())
    compile_run = published["runs"][0]["run_id"]
    source_input = next(body for _, path, body in registry.calls if path.endswith("/inputs/source-component"))
    assert source_input["metadata"]["revision"] == (
        load_and_validate_model(ROOT / "models" / MODEL)["upstream"]["revision"]
    )
    package_dir = tmp_path / "candidate/packages" / MODEL / PROFILE / "packages/default"
    package = json.loads((package_dir / "package-result.json").read_text())
    packages = json.loads((package_dir / "registry-publication.json").read_text())
    assert package["layout"] == "directory"
    assert not list(tmp_path.rglob("*.zip"))
    for path in source.rglob("*"):
        if path.is_file():
            assert (package_dir / "deployment" / path.relative_to(source)).read_bytes() == path.read_bytes()
    # Shared payloads upload once; the two roots have different manifest contents.
    payloads = {(item["sha256"], Path(item["path"]).name) for item in package["manifest"]["files"]}
    assert registry.uploads == len(payloads) + 2
    assert all(key.startswith(candidate["upload_prefix"]) for key in registry.objects)
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
        "build_profile_id": PROFILE, "compile_invocation_id": PROFILE, "component_id": "model",
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
        compile_invocation_id=PROFILE, read_request=read, download_request=registry.objects.__getitem__,
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


def test_native_publication_rejects_wrong_compiled_settings_before_creating_runs(compiled):
    component = compiled / MODEL / PROFILE / "model"
    path = component / "deployment/devkit/vlm_config.json"
    config = json.loads(path.read_text())
    config["pipeline_cfg"]["max_num_tokens"] = 8192
    path.write_text(json.dumps(config))
    provenance = json.loads((component / "compile-result.json").read_text())["provenance"]
    verify_artifact(component, write_manifest=True, provenance=provenance)
    record_compile_outputs(component.parent, model_id=MODEL, build_profile_id=PROFILE,
                           compile_invocation_id=PROFILE, component_ids=["model"])
    registry = Registry()
    with pytest.raises(ValidationError, match="compiled max_num_tokens does not match model.yaml"):
        publish_compilations(repository=ROOT, dist_dir=compiled, **registry.args())
    assert registry.calls == []


def test_candidate_retries_resume_bound_runs_and_finalized_candidates(compiled, tmp_path):
    registry = Registry()
    registry.fail_once = "/artifacts/package"
    args = dict(repository=ROOT, dist_dir=compiled, output_dir=tmp_path / "candidate",
                model_id=MODEL, build_profile_id=PROFILE, **registry.args())
    with pytest.raises(RegistryPublicationError, match="temporary publication failure"):
        publish_candidate(**args)
    assert json.loads((tmp_path / "candidate/candidate.json").read_text())["state"] == "assembling"
    assert len(registry.runs) == 2
    result = publish_candidate(**{**args, "github_run_attempt": "2"})
    assert result["state"] == "finalized"
    count = registry.uploads
    assert publish_candidate(**args) == result
    assert len(registry.runs) == 2
    assert registry.uploads == count


def test_candidate_rejects_expired_assembly_before_publication(compiled, tmp_path):
    registry = Registry()

    def request(method, path, body, key):
        result = registry.request(method, path, body, key)
        if method == "GET":
            result["assembly_expires_at"] = "2000-01-01T00:00:00Z"
        return result

    with pytest.raises(ValidationError, match="assembly expired"):
        publish_candidate(repository=ROOT, dist_dir=compiled, output_dir=tmp_path / "candidate",
                          model_id=MODEL, build_profile_id=PROFILE, **{**registry.args(), "request": request})
    assert not registry.runs


def test_on_demand_publisher_rejects_unbound_publication(compiled):
    registry = Registry()
    with pytest.raises(ValidationError, match="publish-candidate"):
        publish_compilations(repository=ROOT, dist_dir=compiled, **registry.args())
    assert not registry.calls


def test_candidate_upload_retry_reuses_saved_exact_s3_version(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.write_bytes(b"weights")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()
    calls = []

    def head(self, key, version_id=None):
        calls.append(version_id)
        return {"ChecksumSHA256": checksum, "ContentLength": payload.stat().st_size,
                "VersionId": version_id or ("original" if len(calls) == 1 else "newer")}

    monkeypatch.setattr(S3ArtifactStore, "_head", head)
    args = dict(receipt_dir=tmp_path / "uploads")
    first = S3ArtifactStore("models", **args).ensure_native_sha256(payload, "candidate/weights", digest)
    retry = S3ArtifactStore("models", **args).ensure_native_sha256(payload, "candidate/weights", digest)
    assert first == retry
    assert calls == [None, "original"]
    receipt = json.loads(next((tmp_path / "uploads").glob("*.json")).read_text())
    assert receipt["s3_version_id"] == "original"


@pytest.mark.parametrize("model_id, image_input, group_size", [
    (MODEL, True, 64), (MODEL, False, 128), (MTP_MODEL, False, 128),
])
def test_llima_runner_passes_yaml_settings_to_the_recipe(tmp_path, model_id, image_input, group_size):
    root = tmp_path / "repository"
    for folder in ("schemas", "taxonomies", "deps", "scripts", "tools", f"models/{model_id}"):
        shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__"))
    if model_id == MTP_MODEL:
        shutil.copytree(ROOT / "models/gemma_4_e2b_it_textonly", root / "models/gemma_4_e2b_it_textonly")
    model_dir = root / "models" / model_id
    path = model_dir / "model.yaml"
    model = yaml.safe_load(path.read_text())
    parameters = model["build_profiles"][PROFILE]["components"]["model"]["parameters"]
    parameters.update(max_num_tokens=8192, language_group_size=group_size)
    if image_input:
        parameters.update(input_height=512, input_width=640)
    else:
        parameters.pop("input_height", None)
        parameters.pop("input_width", None)
    if model_id == MTP_MODEL:
        model["upstream"]["revision"] = "a" * 40
        for path_name, member in model["compilation"]["source"]["files"].items():
            repository, revision_path = member["uri"].split("@", 1)
            revision = "b" * 40 if path_name.startswith(parameters["draft_model_path"] + "/") else "a" * 40
            member["uri"] = f"{repository}@{revision}/{revision_path.split('/', 1)[1]}"
    path.write_text(yaml.safe_dump(model, sort_keys=False))
    model = load_and_validate_model(model_dir)
    source = deployment(tmp_path / "deployment")
    config_path = source / "devkit/vlm_config.json"
    config = json.loads(config_path.read_text())
    config["pipeline_cfg"].update(max_num_tokens=8192, input_token_group_size=group_size)
    config["vm_cfg"]["image_size"] = [512, 640] if image_input else 448
    if model_id == MTP_MODEL:
        config["lm_cfg"]["speculative_decoding_cfg"] = {"is_draft": False, "method": "gemma4_mtp"}
    config_path.write_text(json.dumps(config))
    checkpoint = tmp_path / "source-sets/9f" / ("9f" + "a" * 62)
    checkpoint_config = checkpoint / model["components"][0]["source"]["entrypoint"]
    checkpoint_config.parent.mkdir(parents=True)
    checkpoint_config.write_text("{}")
    checkpoint = checkpoint_config.parent
    if model_id == MTP_MODEL:
        target_name = Path(model["upstream"]["repository"]).name
        target = source / target_name
        target.mkdir()
        for child in list(source.iterdir()):
            if child != target:
                child.rename(target / child.name)
        draft_name = parameters["draft_model_path"]
        draft = deployment(source / draft_name)
        draft_config = json.loads((draft / "devkit/vlm_config.json").read_text())
        draft_config["lm_cfg"]["speculative_decoding_cfg"] = {"is_draft": True, "method": "gemma4_mtp"}
        (draft / "devkit/vlm_config.json").write_text(json.dumps(draft_config))
        (checkpoint.parent / draft_name).mkdir()
        (checkpoint.parent / draft_name / "config.json").write_text("{}")
    tools = tmp_path / "bin"
    tools.mkdir()
    programs = {
        "registry-python": '''import json, os, sys
if sys.argv[1:4] == ["-m", "model_registry", "fetch"]:
    print(json.dumps({"path": os.environ["CHECKPOINT_CONFIG"]}))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''',
        "llima-compile": '''import json, os, runpy, sys
from pathlib import Path
checkpoint = Path(sys.argv[1])
assert checkpoint.is_symlink()
assert checkpoint.name == os.environ["CHECKPOINT_NAME"]
assert checkpoint.resolve() == Path(os.environ["CHECKPOINT_CONFIG"]).parent
if "--draft_model_path" in sys.argv:
    draft = Path(sys.argv[sys.argv.index("--draft_model_path") + 1])
    assert draft.is_symlink()
    assert draft.resolve() == Path(os.environ["CHECKPOINT_CONFIG"]).parent.parent / draft.name
    configure = runpy.run_path(sys.argv[sys.argv.index("-c") + 1])["get_layer_configuration"]
    assert configure({"is_draft_model": False}, {}) == {"precision": "BF16"}
    assert configure({"is_draft_model": True}, {}) == {"precision": "A_BF16_W_INT8"}
Path(os.environ["COMPILE_ARGUMENTS"]).write_text(json.dumps(sys.argv[1:]))
Path(sys.argv[sys.argv.index("-o") + 1]).mkdir()
''',
        "llima-deploy": '''import os, shutil, sys
shutil.copytree(os.environ["DEPLOYMENT_SOURCE"], sys.argv[2])
''',
    }
    for name, code in programs.items():
        executable = tools / name
        executable.write_text(f"#!{sys.executable}\n{code}")
        executable.chmod(0o755)
    captured = tmp_path / "arguments.json"
    subprocess.run([
        "bash", str(root / "scripts/run_llima_compilation.sh"), str(model_dir), PROFILE, PROFILE,
        str(tmp_path / "output"),
    ], check=True, capture_output=True, text=True, env={
        **os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "PYTHONPATH": str(root / "tools"),
        "MODEL_COMPILER_PYTHON": str(tools / "registry-python"), "CHECKPOINT_CONFIG": str(checkpoint / "config.json"),
        "CHECKPOINT_NAME": Path(model["upstream"]["repository"]).name,
        "DEPLOYMENT_SOURCE": str(source), "COMPILE_ARGUMENTS": str(captured),
    })
    arguments = json.loads(captured.read_text())
    assert not Path(arguments[0]).exists()
    assert (checkpoint / "config.json").read_text() == "{}"
    assert arguments[arguments.index("--max_num_tokens") + 1] == "8192"
    if group_size == 128:
        assert "--language_group_size" not in arguments
    else:
        assert arguments[arguments.index("--language_group_size") + 1] == str(group_size)
    assert ("--input_height" in arguments) is image_input
    assert ("--input_width" in arguments) is image_input
    if image_input:
        assert arguments[arguments.index("--input_height") + 1] == "512"
        assert arguments[arguments.index("--input_width") + 1] == "640"
    output = tmp_path / "output"
    replay = output / model_id / PROFILE / "model/deployment/compile.sh"
    replay_arguments = tmp_path / "replay-arguments.json"
    executable = tools / "llima-compile"
    executable.write_text(f"#!{sys.executable}\nimport json, os, runpy, sys\n"
                          'if "-c" in sys.argv:\n'
                          '    configure = runpy.run_path(sys.argv[sys.argv.index("-c") + 1])["get_layer_configuration"]\n'
                          '    assert configure({"is_draft_model": True}, {}) == {"precision": "A_BF16_W_INT8"}\n'
                          'open(os.environ["COMPILE_ARGUMENTS"], "w").write(json.dumps(sys.argv[1:]))\n')
    subprocess.run(["bash", str(replay)], cwd=tmp_path, check=True, env={
        **os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "COMPILE_ARGUMENTS": str(replay_arguments),
    })
    recorded = json.loads(replay_arguments.read_text())
    assert "--revision" not in replay.read_text()
    assert "set --" not in replay.read_text()
    checkpoint_name = Path(model["upstream"]["repository"]).name
    assert recorded[0] == checkpoint_name
    assert recorded[1:3] == ["-o", checkpoint_name.removesuffix("-Safetensors") +
                           ("-MTP" if model_id == MTP_MODEL else "")]
    if model_id == MTP_MODEL:
        recipe = replay.read_text()
        for name, config in ((checkpoint_name, model["components"][0]["source"]["entrypoint"]),
                             (parameters["draft_model_path"], parameters["draft_model_path"] + "/config.json")):
            uri = next(item["uri"] for item in model["components"][0]["source"]["files"] if item["path"] == config)
            repository = uri.removeprefix("hf://models/").split("@", 1)[0]
            assert f'# hf download {repository} --local-dir {name}' in recipe
        assert recorded[recorded.index("--draft_model_path") + 1] == parameters["draft_model_path"]
        arguments[arguments.index("--draft_model_path") + 1] = parameters["draft_model_path"]
        for values in (recorded, arguments):
            index = values.index("-c")
            del values[index:index + 2]
    assert recorded[recorded.index("--jobs") + 1] == "4"
    assert arguments[arguments.index("--jobs") + 1] == "20"
    for values in (recorded, arguments):
        index = values.index("--jobs")
        del values[index:index + 2]
    assert recorded[3:] == arguments[3:]
    manifest = json.loads((replay.parent.parent / "deployment-manifest.json").read_text())
    assert "compile.sh" in {item["path"] for item in manifest["files"]}
    if model_id == MTP_MODEL:
        assert "config.py" in {item["path"] for item in manifest["files"]}
    compiler = resolve_model_compiler(root)
    verify_artifact(output / model_id / PROFILE / "model", write_manifest=True, provenance={
        "model_id": model_id, "build_profile_id": PROFILE, "component_id": "model", "target": "modalix",
        "source_sha256": source_identity(model["components"][0]["source"]),
        "git_sha": "a" * 40, "environment": "staging", "compiler_version": compiler["version"],
        "compiler_label": compiler["label"],
        "compile_fingerprint": compile_fingerprint(root, model_id, PROFILE, PROFILE, compiler["identity"]),
    })
    prepared = _preflight_compilations(root, output, load_models(root), environment="staging", git_sha="a" * 40)
    assert prepared[0]["compiled_parameters"] == {
        "max_num_tokens": 8192, "language_group_size": group_size,
        "input_height": 512 if image_input else 448, "input_width": 640 if image_input else 448,
        **({"speculative_method": "gemma4_mtp"} if model_id == MTP_MODEL else {}),
    }
    if model_id == MTP_MODEL:
        registry = Registry()
        candidate = publish_candidate(repository=root, dist_dir=output, output_dir=tmp_path / "candidate",
                                      model_id=model_id, build_profile_id=PROFILE, **registry.args())
        assert candidate["state"] == "finalized"
        package_dir = tmp_path / "candidate/packages" / model_id / PROFILE / "packages/default"
        package = json.loads((package_dir / "package-result.json").read_text())
        assert {item["path"] for item in package["manifest"]["files"]} == {
            item["path"] for item in manifest["files"]
        }


@pytest.mark.parametrize("future_mask, log_probe", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("model_id", ["whisper_small", "whisper_medium", "whisper_large_v3_turbo"])
def test_whisper_recipe_and_saved_script_preserve_settings(tmp_path, future_mask, log_probe, model_id):
    work = tmp_path / "source with spaces"
    checkpoint = work / model_id.replace("_", "-")
    checkpoint.mkdir(parents=True)
    output = tmp_path / "compiled output"
    tools = tmp_path / "bin"
    tools.mkdir()
    python = tools / "python3"
    python.write_text(f"#!{sys.executable}\nimport os, sys\n"
                      "os.execv(sys.executable, [sys.executable, '-S', *sys.argv[1:]])\n")
    python.chmod(0o755)
    modules = {
        "afe/__init__.py": "",
        "afe/apis/__init__.py": "",
        "afe/apis/error_handling_variables.py": "def enable_verbose_error_messages(): pass\n",
        "sima_lmm/__init__.py": "",
        "sima_lmm/model.py": '''import json, os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

FileGenMode = SimpleNamespace(ALL="all")
FileGenPrecision = SimpleNamespace(A_BF16_W_INT8="a16w8")

class WhisperModel:
    @classmethod
    def from_hf_cache(cls, **kwargs):
        model = cls()
        model.source = kwargs
        return model

    def gen_files(self, mode, **kwargs):
        with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
            assert pool.submit(abs, -7).result() == 7
        self.source["sima_path"].mkdir(parents=True)
        Path(os.environ["COMPILER_CALLS"]).write_text(json.dumps({
            "source": self.source, "mode": mode, "generation": kwargs,
        }, default=str))
''',
    }
    for name, code in modules.items():
        path = tools / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    captured = tmp_path / "calls.json"
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "PYTHONPATH": str(tools),
           "COMPILER_CALLS": str(captured)}
    options = [
        f"--{'' if future_mask else 'no-'}use_future_token_mask",
        f"--{'' if log_probe else 'no-'}enable_log_probe",
    ]
    recipe = ROOT / "models" / model_id / "scripts/compile.sh"
    for script, arguments, destination in (
        (recipe, [str(checkpoint), str(output), *options], output),
        (output / "compile.sh", [], work / f"{checkpoint.name}-a16w8"),
    ):
        subprocess.run(["bash", str(script), *arguments], cwd=work, env=env,
                       check=True, capture_output=True, text=True)
        calls = json.loads(captured.read_text())
        assert (work / calls["source"]["hf_cache_path"]).resolve() == checkpoint
        assert calls["source"]["model_name"] == checkpoint.name
        assert (work / calls["source"]["sima_path"]).resolve() == destination / "sima_files"
        assert calls["source"]["use_future_token_mask"] is future_mask
        assert calls["source"]["enable_log_probe"] is log_probe
        assert calls["mode"] == "all"
        assert calls["generation"]["precision"] == {"all": {"precision": "a16w8"}}
        assert calls["generation"]["num_processes"] == (20 if script == recipe else 4)
        assert calls["generation"]["part"] == "all"
        assert calls["generation"]["resume"] is True
        if script == recipe:
            source = load_and_validate_model(ROOT / "models" / model_id)["components"][0]["source"]
            config = next(item for item in source["files"] if item["path"] == source["entrypoint"])
            save_llima_recipe(recipe.parent, output, [checkpoint.name, f"{checkpoint.name}-a16w8", *options],
                              {checkpoint.name: config["uri"]})
    assert os.access(output / "compile.sh", os.X_OK)
    assert (output / "compile.py").read_bytes() == recipe.with_suffix(".py").read_bytes()


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_mpk_candidate_binds_every_component_before_finalizing(tmp_path, monkeypatch, environment):
    root = tmp_path / "repository"
    model_dir = root / "models/fixture_model"
    shutil.copytree(ROOT / "tests/fixtures/fixture_model", model_dir)
    for directory in ("schemas", "taxonomies"):
        shutil.copytree(ROOT / directory, root / directory)
    for relative in ("deps/manifest.json", *COMPILE_TOOLING):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    path = model_dir / "model.yaml"
    model = yaml.safe_load(path.read_text())
    model["build-on-demand"] = True
    model["compilation"]["components"] = ["backbone", "decoder"]
    model["build_profiles"]["fixture"]["components"] = {
        "backbone": {"parameters": {}}, "decoder": {"parameters": {}},
    }
    path.write_text(yaml.safe_dump(model, sort_keys=False))
    compiler = resolve_model_compiler(root)
    key = compile_fingerprint(root, "fixture_model", "fixture", "fixture", compiler["identity"])
    registry = Registry()
    # The profile's one compile invocation compiles both components.
    invocation = tmp_path / "invocations" / "fixture" / "fixture_model/fixture"
    outputs = []
    for component in ("backbone", "decoder"):
        output = invocation / component
        output.mkdir(parents=True)
        archive = output / f"fixture_model_fixture_{component}_mpk.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, payload in (("layer.elf", b"synthetic-elf"), ("fixture_mpk.json", b"{}")):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
        verify_artifact(output, write_manifest=True, provenance={
            "model_id": "fixture_model", "build_profile_id": "fixture", "component_id": component,
            "target": "modalix", "source_sha256": "a" * 64, "git_sha": "a" * 40,
            "environment": environment, "compiler_version": compiler["version"], "compiler_label": compiler["label"],
            "compile_fingerprint": key,
        })
        outputs.append({"component_id": component, "mpk_path": f"{component}/{archive.name}",
                        "mpk_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    (invocation / "compile-timing.json").write_text(COMPILE_TIMING)
    (invocation / "compile-outputs.json").write_text(json.dumps({
        "model_id": "fixture_model", "build_profile_id": "fixture", "compile_invocation_id": "fixture",
        "outputs": outputs,
        "derived_graphs": [{"component_id": component, "sha256": "a" * 64} for component in ("backbone", "decoder")],
    }))
    from model_registry import stage_plan

    # A catalog build of one component at the head key cannot join a candidate built from another source.
    previous = {"models": [{"id": "fixture_model", "builds": [
        {"build_profile_id": "fixture", "component_id": "backbone", "run_id": "older-source",
         "compile_fingerprint": key},
    ]}]}
    monkeypatch.setattr(stage_plan, "read_catalog_sources", lambda *args: (None, [previous]))
    plan_args = dict(repository=root, compiler_version=compiler["identity"], branch="feature/candidate",
                     model_ids=["fixture_model"], profile_ids=["fixture"], include_on_demand=True,
                     read=lambda url: {}, read_api_url="https://registry.example.invalid")
    for component_ids in (None, ["decoder"]):
        plan = stage_plan.plan_stages(**plan_args, component_ids=component_ids)
        assert [(item["compile_invocation_id"], item["component_ids"]) for item in plan["compile"]["include"]] == [
            ("fixture", ["backbone", "decoder"]),
        ]
    arguments = {**registry.args(), "environment": environment}
    result = publish_candidate(repository=root, dist_dir=tmp_path / "invocations", output_dir=tmp_path / "candidate",
                               model_id="fixture_model", build_profile_id="fixture", **arguments)
    assert result["state"] == "finalized"
    assert set(registry.bindings) == {"compile-backbone", "compile-decoder", "package-default"}
    finalization = next(index for index, (method, path, _) in enumerate(registry.calls)
                        if path.startswith(DEPLOYMENTS) and path.endswith("/finalize"))
    assert all(index < finalization for index, (_, path, _) in enumerate(registry.calls)
               if path.startswith("/v1/runs/") and path.endswith("/finalize"))


def test_llima_compile_fingerprint_names_no_compile_invocation():
    """A LLiMa key covers the compiler, target, and each component's source and parameters, as it always has."""
    model = load_and_validate_model(ROOT / "models" / MODEL)
    profile = model["build_profiles"][0]
    compiler = resolve_model_compiler(ROOT)["identity"]
    identity = {
        "compiler_version": compiler,
        "target": profile["target"],
        "components": [{
            "source_sha256": source_identity(model["components"][0]["source"]),
            "parameters": profile["components"][0]["parameters"],
        }],
    }

    assert profile["compile_invocations"] == [{"id": PROFILE, "components": ["model"]}]
    assert compile_fingerprint(ROOT, MODEL, PROFILE, PROFILE, compiler) == hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


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


def test_native_deployment_rejects_links(tmp_path):
    source = deployment(tmp_path / "deployed")
    (source / "link").symlink_to(source / "devkit/tokenizer.json")
    with pytest.raises(ValidationError, match="unsafe"):
        verify_llima(source)


def test_speculative_deployment_manifest_preserves_nested_models(tmp_path):
    source = tmp_path / "deployment"
    for name in ("target-model", "draft-model"):
        deployment(source / name)
    config_path = source / "draft-model/devkit/vlm_config.json"
    config = json.loads(config_path.read_text())
    config["lm_cfg"]["speculative_decoding_cfg"] = {"is_draft": True}
    config_path.write_text(json.dumps(config))
    (source / "compile.sh").write_text('#!/usr/bin/env bash\nllima-compile "$@"\n')
    result = verify_artifact(tmp_path, write_manifest=True)
    files = json.loads((tmp_path / "deployment-manifest.json").read_text())["files"]
    assert {item["path"] for item in files} == {
        path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()
    }
    assert len({item["artifact"] for item in files}) == len(files)
    assert verify_artifact(tmp_path)["sha256"] == result["sha256"]
    (source / "draft-model/elf_files/language.elf").write_bytes(b"modified draft")
    with pytest.raises(ValidationError, match="do not match the manifest"):
        verify_artifact(tmp_path)


@pytest.mark.parametrize("model", ["target-model", "draft-model"])
@pytest.mark.parametrize("missing", ["configuration", "deployment", "elf_files"])
def test_speculative_deployment_rejects_incomplete_pairs(tmp_path, model, missing):
    source = tmp_path / "deployment"
    for name in ("target-model", "draft-model"):
        folder = deployment(source / name)
        path = folder / "devkit/vlm_config.json"
        config = json.loads(path.read_text())
        config["lm_cfg"]["speculative_decoding_cfg"] = {
            "method": "gemma4_mtp", "is_draft": name == "draft-model",
        }
        path.write_text(json.dumps(config))
    folder = source / model
    if missing == "configuration":
        (folder / "devkit/vlm_config.json").unlink()
    else:
        shutil.rmtree(folder if missing == "deployment" else folder / "elf_files")
    with pytest.raises(ValidationError, match="expected one .* configuration|contains no ELF files"):
        verify_artifact(tmp_path, write_manifest=True)
    assert not (tmp_path / "deployment-manifest.json").exists()


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
