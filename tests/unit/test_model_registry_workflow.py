from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Model-Registry.yml"


def test_model_registry_workflow_is_change_scoped_and_publishes_subpackage() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.startswith("name: Model Registry Contributor Tool\n")
    for path in (
        "tools/model_registry/**",
        "packaging/model-registry/**",
        "tests/unit/**",
        "scripts/**",
        ".github/workflows/**",
        "models/**",
        "deps/manifest.json",
        "schemas/**",
        "taxonomies/**",
    ):
        assert f'      - "{path}"' in text
    assert "packaging/model-registry/build.sh" in text
    assert "artifact_folder: model-registry" in text
    assert text.count("artifact_namespace: models-utils") == 2
    assert "artifact_pattern: model-registry-dist" in text
    assert "min_artifact_count: 4" in text
    assert "vulcan-publish-artifacts.yml@main" in text
    assert "vulcan-update-latest-artifacts.yml@main" in text
    assert "github.event_name != 'pull_request'" in text


def test_model_registry_pushes_default_to_staging() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "inputs.environment || 'staging'" in text
    assert "vars.VULCAN_ENV" not in text


def test_ci_runs_every_unit_test() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pytest tests/unit\n" in text


def test_model_registry_workflow_smoke_tests_the_author_command_surface() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'help_text="$("${model_registry}" --help)"' in text
    assert "author CLI unexpectedly exposes" in text
    assert 'echo "SIMA_CLI_BIN=${sima_cli}" >> "${GITHUB_ENV}"' in text
