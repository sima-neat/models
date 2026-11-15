"""Metadata-only staging and direct native deployment downloads on the DevKit."""

import hashlib
import json
from pathlib import Path

import pytest
from model_registry import registry
from model_registry.artifact import deployment_artifact_name
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.fingerprints import compile_fingerprint
from model_registry.model import load_and_validate_model
from model_registry.report_publication import _registered_builds

ROOT = Path(__file__).resolve().parents[1]
MODEL, PROFILE = "qwen3_5_0_8b", "w4a16"
COMPILE_RUN = "00000000-0000-4000-8000-000000000001"
PACKAGE_RUN = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
def native_input(tmp_path):
    payloads = {"target/elf_files/language.elf": b"target", "draft/elf_files/language.elf": b"draft"}
    files = [{"path": path, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "artifact": deployment_artifact_name(path)} for path, data in payloads.items()]
    component = json.dumps({"format": "llima", "files": files}).encode()
    package = json.dumps({"model_id": MODEL, "build_profile_id": PROFILE, "package_id": "default",
                          "components": [{"id": "model", "compile_run_id": COMPILE_RUN}], "files": files}).encode()
    objects = {"llima": component, "package": package}
    fingerprint = compile_fingerprint(ROOT, MODEL, PROFILE, PROFILE, resolve_model_compiler(ROOT)["identity"])
    catalog = {"catalog_revision": "a" * 40, "models": [{"id": MODEL,
        "builds": [{"component_id": "model", "build_profile_id": PROFILE, "compile_invocation_id": PROFILE,
                    "run_id": COMPILE_RUN, "compile_fingerprint": fingerprint}],
        "packages": [{"package_id": "default", "build_profile_id": PROFILE, "run_id": PACKAGE_RUN,
                      "component_run_ids": {"model": COMPILE_RUN}}]}]}
    calls, downloaded = [], []

    def read(url):
        calls.append(url)
        if "/catalogs/current?" in url:
            return {"catalog": {"document": catalog}}
        kind = url.rsplit("/", 1)[-1]
        data = objects[kind]  # Any file-level API request fails: staging must fetch only the two manifests.
        return {"artifact": {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                "download_url": kind, "metadata": {"layout": "directory",
                                                     "component_run_ids": {"model": COMPILE_RUN}}}}

    def download(url):
        downloaded.append(url)
        return objects[url]

    arguments = dict(repository=ROOT, output_dir=tmp_path / "compiled", api_url="https://registry.example.invalid",
        branch="develop", catalog_revision="a" * 40, model_id=MODEL, build_profile_id=PROFILE,
        compile_invocation_id=PROFILE, read_request=read, download_request=download, metadata_only=True)
    return arguments, package, payloads, calls, downloaded


@pytest.mark.parametrize("branch", [None, "main"])
def test_native_staging_downloads_only_manifests_and_keeps_publication_bindings(native_input, branch):
    arguments, package, _, calls, downloaded = native_input
    registry.stage_catalog_build(**arguments, deployment_branch=branch)
    profile = arguments["output_dir"] / MODEL / PROFILE
    assert downloaded == ["llima", "package"]
    assert not list(profile.rglob("*.elf"))
    plan = json.loads((profile / "benchmark-download.json").read_text())
    assert plan["branch"] == (branch or "develop")
    assert f"git_ref=refs%2Fheads%2F{branch or 'develop'}" in calls[0]
    assert plan["run_id"] == PACKAGE_RUN
    assert plan["manifest_sha256"] == hashlib.sha256(package).hexdigest()
    model = load_and_validate_model(ROOT / "models" / MODEL)
    builds, package_run = _registered_builds(arguments["output_dir"], model, PROFILE)
    assert builds["model"]["run_id"] == COMPILE_RUN and package_run == PACKAGE_RUN


@pytest.mark.parametrize("environment", ["production", "staging"])
@pytest.mark.parametrize("mismatch", [None, "run_id", "manifest_sha256"])
def test_cli_download_uses_selected_branch_environment_and_manifest(
    native_input, tmp_path, monkeypatch, environment, mismatch,
):
    arguments, package, payloads, _, _ = native_input
    registry.stage_catalog_build(**arguments, deployment_branch="main")
    downloads = tmp_path / "downloads"

    def cli(command, **kwargs):
        assert command[:3] == ["sima-cli", "models", "download"]
        assert command[command.index("--branch") + 1] == "main"
        assert command[command.index("--id") + 1] == MODEL
        assert command[command.index("--variant") + 1] == PROFILE
        assert ("--stg" in command) == (environment == "staging")
        folder = Path(command[command.index("--output") + 1]) / "CLI display name"
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_bytes(package)
        for path, data in payloads.items():
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        result = {"path": str(folder), "format": "directory", "run_id": PACKAGE_RUN,
                  "model_id": MODEL, "build_profile_id": PROFILE,
                  "manifest_sha256": hashlib.sha256(package).hexdigest()}
        if mismatch:
            result[mismatch] = "different"
        return json.dumps(result)

    monkeypatch.setattr(registry.subprocess, "check_output", cli)
    if mismatch:
        with pytest.raises(ValidationError, match="differs from the staged package"):
            registry.download_benchmark_deployments(
                compiled_dir=arguments["output_dir"], download_dir=downloads, environment=environment,
            )
        assert not (arguments["output_dir"] / MODEL / PROFILE / "model/deployment").exists()
    else:
        result = registry.download_benchmark_deployments(
            compiled_dir=arguments["output_dir"], download_dir=downloads, environment=environment,
        )
        deployment = Path(result["deployments"][0])
        assert not (deployment / "manifest.json").exists()
        assert {path: (deployment / path).read_bytes() for path in payloads} == payloads


def test_native_staging_rejects_a_package_for_different_component_builds(native_input):
    arguments, _, _, _, _ = native_input
    read = arguments["read_request"]

    def changed(url):
        result = read(url)
        if url.endswith("/package"):
            result["artifact"]["metadata"]["component_run_ids"] = {"model": "different"}
        return result

    with pytest.raises(ValidationError, match="package does not match"):
        registry.stage_catalog_build(**{**arguments, "read_request": changed})
