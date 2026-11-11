from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from model_registry import neat_core
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]
TAG = "123456789abc"
RUNNER = "0.6.1+local.abcdef123456"
SCHEDULER = "0.6.1+local.123456abcdef"


def _publish_dependencies(state):
    content = state.dependencies if isinstance(state.dependencies, str) else json.dumps(state.dependencies)
    state.metadata = {
        "resources": ["simaai-runner.deb", "simaai-scheduler.deb", neat_core.DEPENDENCIES_FILE],
        "resources-checksum": {neat_core.DEPENDENCIES_FILE: hashlib.sha256(content.encode()).hexdigest()},
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = json.loads((ROOT / "deps/manifest.json").read_text())
    (tmp_path / "deps").mkdir()
    state = SimpleNamespace(
        repository=tmp_path, target=f"core@develop:{TAG}", calls=[], urls=[], latest=TAG,
        status={"components": {
            "core": {"channel": "develop", "tag": TAG, "provenance": {"vulcanEnvironment": "prod"}},
            "runner": {"version": RUNNER}, "scheduler": {"version": SCHEDULER},
        }},
        dependencies={"runner": {"debian_version": RUNNER}, "scheduler": {"debian_version": SCHEDULER}},
    )
    _publish_dependencies(state)
    monkeypatch.delenv("NEAT_ARTIFACTS_BASE_URL", raising=False)
    monkeypatch.setenv("PYNEAT_PYTHON", "/test/pyneat/bin/python")

    def run(command, **kwargs):
        state.calls.append(command)
        if command == ["neat", "--json"]:
            if isinstance(state.status, Exception):
                raise state.status
            value = state.status if isinstance(state.status, str) else json.dumps(state.status)
            return subprocess.CompletedProcess(command, 0, stdout=value)
        return subprocess.CompletedProcess(command, 0)

    def read(url):
        state.urls.append(url)
        if url.endswith("latest.tag"):
            value = state.latest
        elif url.endswith(neat_core.DEPENDENCIES_FILE):
            value = state.dependencies
        else:
            value = state.metadata
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)

    def ensure(**kwargs):
        manifest["neat-core"]["install_target"] = state.target
        (tmp_path / "deps/manifest.json").write_text(json.dumps(manifest))
        neat_core.ensure_neat_core(tmp_path, **kwargs)

    state.ensure = ensure
    monkeypatch.setattr(neat_core.subprocess, "run", run)
    monkeypatch.setattr(neat_core, "_read_text", read)
    return state


def _installer_calls(setup):
    return [call for call in setup.calls if call[0] == "sima-cli"]


def _assert_reporting(setup):
    assert setup.calls[-2][0] == "/test/pyneat/bin/python"
    assert "import pyneat" in setup.calls[-2][-1]
    assert setup.calls[-1] == ["dpkg-query", "-W", "*neat*", "simaai-*"]


@pytest.mark.parametrize("branch,channel,version", [
    ("develop", "develop", RUNNER),
    ("v0.6.1", "v0.6.1", "0.6.1"),
    ("feature/core-artifact", "feature/core-artifact", RUNNER),
    ("feature/core-artifact", "feature-core-artifact", RUNNER),
])
@pytest.mark.parametrize("installed_environment", ["prod", "production", "prd"])
def test_matching_state_skips_install_and_still_reports(setup, capsys, branch, channel, version, installed_environment):
    setup.target = f"core@{branch}:{TAG}"
    core = setup.status["components"]["core"]
    core["channel"] = channel
    core["provenance"]["vulcanEnvironment"] = installed_environment
    setup.status["components"]["runner"]["version"] = version
    setup.dependencies["runner"]["debian_version"] = version
    _publish_dependencies(setup)

    setup.ensure()

    assert _installer_calls(setup) == []
    assert f"NEAT core already installed ({branch}/{TAG}). Skipping install." in capsys.readouterr().out
    key = branch.replace("/", "%252F")
    index = f"{neat_core.PRODUCTION_ARTIFACTS}/{key}/{TAG}"
    assert setup.urls == [f"{index}/metadata-minimal.json", f"{index}/{neat_core.DEPENDENCIES_FILE}"]
    _assert_reporting(setup)


@pytest.mark.parametrize("section,key,value", [
    ("core", "channel", "other"), ("core", "tag", "abcdef123456"),
    ("core", "provenance", {"vulcanEnvironment": "stg"}), ("core", "provenance", None),
    ("core", "tag", None), ("runner", "version", "0.6.0+older.build"),
    ("scheduler", "version", "0.6.0+older.build"), ("runner", "version", None),
])
def test_identity_or_republished_dependency_mismatch_installs(setup, section, key, value):
    setup.status["components"][section][key] = value

    setup.ensure()

    assert _installer_calls(setup) == [[
        "sima-cli", "neat", "install", "--env", "production", "--type", "minimal", f"core@develop:{TAG}",
    ]]
    _assert_reporting(setup)


@pytest.mark.parametrize("metadata", [
    OSError("unavailable"), "{", "[]", "null", {}, {"resources": None}, {"resources": [None]},
    {"resources": ["simaai-runner.deb"]},
])
def test_unverifiable_metadata_installs(setup, metadata):
    setup.metadata = metadata

    setup.ensure()

    assert len(_installer_calls(setup)) == 1
    _assert_reporting(setup)


@pytest.mark.parametrize("status", [
    FileNotFoundError("neat"), subprocess.CalledProcessError(1, ["neat", "--json"]),
    subprocess.TimeoutExpired(["neat", "--json"], 30), "{", "[]", "null", {},
    {"components": {"core": {}}},
])
def test_unreadable_installed_state_installs(setup, status):
    setup.status = status

    setup.ensure()

    assert len(_installer_calls(setup)) == 1
    _assert_reporting(setup)


@pytest.mark.parametrize("dependencies", [
    "{", None, [], {}, {"runner": {"debian_version": RUNNER}},
    {"runner": {"debian_version": None}, "scheduler": {"debian_version": SCHEDULER}},
    {"runner": {"debian_version": "invalid version"}, "scheduler": {"debian_version": SCHEDULER}},
])
def test_unverifiable_dependency_versions_install(setup, dependencies):
    setup.dependencies = dependencies
    _publish_dependencies(setup)

    setup.ensure()

    assert len(_installer_calls(setup)) == 1


@pytest.mark.parametrize("problem", [
    "checksum missing", "checksum wrong", "file undeclared", "file ambiguous", "package ambiguous",
    "package version ambiguous", "offline",
])
def test_requires_verified_unambiguous_dependency_manifest(setup, problem):
    if problem == "checksum missing":
        setup.metadata.pop("resources-checksum")
    elif problem == "checksum wrong":
        setup.metadata["resources-checksum"][neat_core.DEPENDENCIES_FILE] = "0" * 64
    elif problem == "file undeclared":
        setup.metadata["resources"].remove(neat_core.DEPENDENCIES_FILE)
    elif problem == "file ambiguous":
        setup.metadata["resources"].append(neat_core.DEPENDENCIES_FILE)
    elif problem == "package ambiguous":
        setup.metadata["resources"].append("simaai-runner.deb")
    elif problem == "package version ambiguous":
        setup.metadata["resources"].append("simaai-runner_0.6.0_arm64.deb")
    else:
        setup.dependencies = OSError("offline")

    setup.ensure()

    assert len(_installer_calls(setup)) == 1


@pytest.mark.parametrize("matches", [False, True])
def test_floating_target_resolves_once_and_installs_that_exact_tag(setup, matches):
    setup.target = "core@develop"
    if not matches:
        setup.status["components"]["core"]["tag"] = "abcdef123456"
    setup.latest = TAG + "\n"

    setup.ensure()

    index = f"{neat_core.PRODUCTION_ARTIFACTS}/develop"
    assert setup.urls == [f"{index}/latest.tag", f"{index}/{TAG}/metadata-minimal.json",
                          f"{index}/{TAG}/{neat_core.DEPENDENCIES_FILE}"]
    installs = _installer_calls(setup)
    assert len(installs) == int(not matches)
    if installs:
        assert installs[0][-1] == f"core@develop:{TAG}"
    _assert_reporting(setup)


@pytest.mark.parametrize("latest", [OSError("offline"), "", "latest", "../bad", "two tags"])
def test_unresolved_floating_target_fails_without_benchmarking_or_installing(setup, latest):
    setup.target, setup.latest = "core@develop", latest

    with pytest.raises(ValidationError, match="Core tag"):
        setup.ensure()

    assert setup.calls == []


@pytest.mark.parametrize("target,floating", [(f"core@{TAG}", False), ("core@latest", True)])
def test_core_shorthand_targets_follow_installer_resolution(setup, target, floating):
    setup.target = target
    setup.status["components"]["core"]["channel"] = "main"

    setup.ensure()

    assert setup.urls[-2] == f"{neat_core.PRODUCTION_ARTIFACTS}/main/{TAG}/metadata-minimal.json"
    assert len(setup.urls) == 2 + int(floating)
    assert _installer_calls(setup) == []


def test_core_source_stays_production_when_model_registry_uses_staging(setup, monkeypatch):
    monkeypatch.setenv("VULCAN_ENV", "staging")
    setup.status["components"]["runner"]["version"] = "0.1.0"

    setup.ensure()

    assert setup.urls[0].startswith(neat_core.PRODUCTION_ARTIFACTS)
    assert _installer_calls(setup)[0][3:5] == ["--env", "production"]


@pytest.mark.parametrize("argument", [None, "https://other-environment.test/core"])
def test_production_source_ignores_artifact_index_overrides(setup, monkeypatch, argument):
    monkeypatch.setenv("NEAT_ARTIFACTS_BASE_URL", "https://other-environment.test/core")

    setup.ensure(artifacts_base_url=argument)

    assert setup.urls == [
        f"{neat_core.PRODUCTION_ARTIFACTS}/develop/{TAG}/metadata-minimal.json",
        f"{neat_core.PRODUCTION_ARTIFACTS}/develop/{TAG}/{neat_core.DEPENDENCIES_FILE}",
    ]
    assert _installer_calls(setup) == []
    _assert_reporting(setup)


@pytest.mark.parametrize("index", [None, neat_core.PRODUCTION_ARTIFACTS])
def test_nonproduction_needs_appropriate_explicit_index_to_skip(setup, index):
    setup.status["components"]["core"]["provenance"]["vulcanEnvironment"] = "stg"

    setup.ensure(environment="staging", artifacts_base_url=index)

    assert setup.urls == []
    assert _installer_calls(setup)[0][3:5] == ["--env", "staging"]
    _assert_reporting(setup)


def test_nonproduction_explicit_index_can_verify_a_match(setup, monkeypatch):
    monkeypatch.setenv("NEAT_ARTIFACTS_BASE_URL", "https://core.staging.test/")
    setup.status["components"]["core"]["provenance"]["vulcanEnvironment"] = "staging"

    setup.ensure(environment="stg")

    assert setup.urls[0] == f"https://core.staging.test/develop/{TAG}/metadata-minimal.json"
    assert _installer_calls(setup) == []
    _assert_reporting(setup)


def test_nonproduction_floating_target_without_index_fails(setup):
    setup.target = "core@develop"

    with pytest.raises(ValidationError, match="NEAT_ARTIFACTS_BASE_URL"):
        setup.ensure(environment="staging")

    assert setup.calls == []
    assert setup.urls == []


def test_setup_failure_does_not_continue_to_reporting(setup, monkeypatch):
    def fail(command, **kwargs):
        setup.calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(neat_core.subprocess, "run", fail)
    with pytest.raises(ValidationError, match="benchmark setup failed"):
        setup.ensure()

    assert [call[0] for call in setup.calls] == ["neat", "sima-cli"]


def test_cli_dispatches_setup_without_exposing_it_to_authors(setup, monkeypatch):
    from model_registry.cli import AUTHOR_COMMANDS, main

    calls = []
    monkeypatch.setattr(neat_core, "ensure_neat_core", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert main(["ensure-neat-core", "--repository", str(setup.repository), "--environment", "production"]) == 0
    assert calls == [((setup.repository,), {"environment": "production", "artifacts_base_url": None})]
    assert "ensure-neat-core" not in AUTHOR_COMMANDS
