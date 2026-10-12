from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "target-branch-checks.yml"


def load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_pull_requests_run_only_the_unprivileged_checks() -> None:
    workflow = load(WORKFLOW)

    assert set(workflow["on"]) == {"pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert "${{ secrets." not in WORKFLOW.read_text(encoding="utf-8")
    assert all("secrets" not in job for job in workflow["jobs"].values())
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert {name: job["uses"] for name, job in workflow["jobs"].items()} == {
        "package": "./.github/workflows/model-registry-package.yml",
        "portal": "./.github/workflows/developer-docs.yml",
    }


def test_only_target_branch_checks_runs_for_pull_requests() -> None:
    triggered = {
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if {"pull_request", "pull_request_target"} & set(load(path)["on"])
    }

    assert triggered == {"target-branch-checks.yml", "pr-sanitization.yml"}


def test_checked_workflows_run_only_on_hosted_runners() -> None:
    for name in ("model-registry-package.yml", "developer-docs.yml"):
        jobs = load(WORKFLOWS / name)["jobs"]
        assert all(job["runs-on"] == "ubuntu-latest" for job in jobs.values()), name
