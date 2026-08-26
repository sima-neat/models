from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "Vulcan-Benchmarker.yml"


def test_benchmarker_workflow_is_change_scoped_and_publishes_subpackage() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for path in (
        "tools/runtime/benchmarker.py",
        "packaging/benchmarker/**",
        "scripts/build_benchmarker_vulcan_package.sh",
        "schemas/performance-*-v1.schema.json",
    ):
        assert f'      - "{path}"' in text
    assert "scripts/build_benchmarker_vulcan_package.sh" in text
    assert "artifact_folder: benchmarker" in text
    assert text.count("artifact_namespace: models-utils") == 2
    assert "artifact_pattern: benchmarker-dist" in text
    assert "vulcan-publish-artifacts.yml@main" in text
    assert "vulcan-update-latest-artifacts.yml@main" in text
    assert "github.event_name != 'pull_request'" in text


def test_benchmarker_pushes_default_to_staging() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "inputs.environment || 'staging'" in text
    assert "vars.VULCAN_ENV" not in text


def test_sima_cli_install_does_not_shadow_setup_python() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'echo "${HOME}/.sima-cli/.venv/bin" >> "${GITHUB_PATH}"' not in text
    assert 'echo "SIMA_CLI_BIN=${sima_cli}" >> "${GITHUB_ENV}"' in text
    assert 'SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"' in (
        ROOT / "scripts" / "build_benchmarker_vulcan_package.sh"
    ).read_text(encoding="utf-8")
