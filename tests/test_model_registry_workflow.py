from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "vulcan-model-registry.yml"
PACKAGE_WORKFLOW = WORKFLOWS / "model-registry-package.yml"


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_model_registry_workflow_is_change_scoped_and_publishes_subpackage() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = load(WORKFLOW)
    assert text.startswith("name: Build and Publish Model Registry\n")
    # Pull requests run the package checks through Target Branch Checks; this workflow publishes.
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    for path in (
        "tools/model_registry/**",
        "tools/model_report/**",
        "packaging/model-registry/**",
        "tests/**",
        "conftest.py",
        "scripts/**",
        ".github/workflows/**",
        "models/**",
        "deps/manifest.json",
        "schemas/**",
        "taxonomies/**",
    ):
        assert path in workflow["on"]["push"]["paths"]
    assert "website/**" not in workflow["on"]["push"]["paths"]
    jobs = workflow["jobs"]
    assert jobs["build"]["uses"] == "./.github/workflows/model-registry-package.yml"
    assert jobs["publish"]["needs"] == ["resolve-vulcan-config", "build"]
    assert "artifact_folder: model-registry" in text
    assert text.count("artifact_namespace: models-utils") == 2
    assert "artifact_pattern: model-registry-dist" in text
    assert "min_artifact_count: 4" in text
    assert "vulcan-publish-artifacts.yml@main" in text
    assert "vulcan-update-latest-artifacts.yml@main" in text


def test_model_registry_publication_uses_repository_environment() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "vars.VULCAN_ENV || 'staging'" in text
    assert "inputs.environment" not in text


def test_package_workflow_only_builds_and_checks_the_package() -> None:
    workflow = load(PACKAGE_WORKFLOW)
    text = PACKAGE_WORKFLOW.read_text(encoding="utf-8")

    assert list(workflow["on"]) == ["workflow_call"]
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"build"}
    assert workflow["jobs"]["build"]["runs-on"] == "ubuntu-latest"
    assert "secrets." not in text
    assert "packaging/model-registry/build.sh" in text
    assert "name: model-registry-dist" in text


def test_ci_runs_every_test_and_collects_every_stage_output_test() -> None:
    steps = {step["name"]: step for step in load(PACKAGE_WORKFLOW)["jobs"]["build"]["steps"]}
    collect = steps["Collect each model's stage-output tests"]["run"]

    assert steps["Test model-registry contracts and each model's scripts"]["run"] == "python -m pytest -n auto"
    assert "--collect-only" in collect
    assert "models/*/tests/build models/*/tests/benchmark" in collect
    for option in ("converted", "compiled", "accuracy", "performance"):
        assert f"--{option}=" in collect


def test_model_registry_workflow_smoke_tests_the_author_command_surface() -> None:
    steps = {step["name"]: step for step in load(PACKAGE_WORKFLOW)["jobs"]["build"]["steps"]}
    verify = (ROOT / "packaging/model-registry/verify_package.sh").read_text(encoding="utf-8")
    install = (ROOT / "scripts/install_sima_cli.sh").read_text(encoding="utf-8")

    assert steps["Install sima-cli"]["run"] == "scripts/install_sima_cli.sh"
    assert steps["Verify package installer and author command surface"]["run"] == (
        "packaging/model-registry/verify_package.sh"
    )
    assert 'help_text="$("${model_registry}" --help)"' in verify
    assert "author CLI unexpectedly exposes" in verify
    assert 'echo "SIMA_CLI_BIN=${sima_cli}" >> "${GITHUB_ENV}"' in install
