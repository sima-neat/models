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
    assert "artifact_pattern: benchmarker-dist" in text
    assert "vulcan-publish-artifacts.yml@main" in text
    assert "vulcan-update-latest-artifacts.yml@main" in text
    assert "github.event_name != 'pull_request'" in text
