from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "model-branch-lifecycle.yml"


def load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_branch_create_delete_and_manual_retry_are_supported() -> None:
    workflow = load_workflow()

    assert set(workflow["on"]) == {"create", "delete", "workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["operation"]["options"] == ["purge", "restore"]
    assert inputs["branch"]["required"] == "true"


def test_lifecycle_uses_repository_scoped_oidc_publisher() -> None:
    workflow = load_workflow()
    job = workflow["jobs"]["update-registry"]
    credentials = next(step for step in job["steps"] if step.get("uses", "").startswith("aws-actions/"))

    assert workflow["permissions"] == {"contents": "read", "id-token": "write"}
    assert job["environment"] == "staging"
    assert credentials["uses"] == "aws-actions/configure-aws-credentials@v6"
    assert credentials["with"]["role-to-assume"] == "${{ vars.VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN }}"
    assert "s3:" not in WORKFLOW.read_text(encoding="utf-8")


def test_only_branch_events_run_and_protected_refs_are_rejected() -> None:
    workflow = load_workflow()
    job = workflow["jobs"]["update-registry"]
    resolve = next(step for step in job["steps"] if step.get("id") == "request")

    assert "github.event.ref_type == 'branch'" in job["if"]
    assert "main|develop|release-*" in resolve["run"]
    assert 'branch="$EVENT_REF"' in resolve["run"]
    assert 'branch="$REQUESTED_BRANCH"' in resolve["run"]


def test_purge_is_polled_and_restore_is_idempotent() -> None:
    workflow = load_workflow()
    lifecycle = next(
        step
        for step in workflow["jobs"]["update-registry"]["steps"]
        if step.get("name") == "Update branch lifecycle in the model registry"
    )
    script = lifecycle["run"]

    assert "/v1/repositories/${REPOSITORY}/branches/${OPERATION}" in script
    assert "/v1/purge-jobs/${job_id}" in script
    assert "for attempt in {1..60}" in script
    assert 'if [[ "$OPERATION" == restore ]]' in script
    assert "Registered S3 object versions removed" in script
    assert lifecycle["env"]["REPOSITORY"] == "${{ github.repository }}"
    assert "--retry-connrefused" in script
    assert "--retry-all-errors" not in script


def evaluate(condition: str, event_name: str, ref_type: str) -> bool:
    expression = condition.replace("github.event_name", repr(event_name))
    expression = expression.replace("github.event.ref_type", repr(ref_type))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}}, {})


def test_only_branch_deletion_cancels_runs() -> None:
    condition = load_workflow()["jobs"]["cancel-branch-runs"]["if"]

    assert evaluate(condition, "delete", "branch") is True
    assert evaluate(condition, "delete", "tag") is False
    assert evaluate(condition, "create", "branch") is False
    assert evaluate(condition, "workflow_dispatch", "") is False


def test_cancel_job_holds_only_the_actions_permission_it_needs() -> None:
    job = load_workflow()["jobs"]["cancel-branch-runs"]
    cancel = next(step for step in job["steps"] if step.get("name") == "Cancel queued and running workflow runs")

    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"actions": "write", "contents": "read"}
    assert "environment" not in job
    assert all("aws-actions/" not in step.get("uses", "") for step in job["steps"])
    assert cancel["env"] == {"GH_TOKEN": "${{ github.token }}", "BRANCH": "${{ github.event.ref }}"}
    assert cancel["run"].startswith('scripts/cancel_branch_runs.sh "$BRANCH"')


def test_cancellation_does_not_gate_the_registry_purge() -> None:
    jobs = load_workflow()["jobs"]

    assert "needs" not in jobs["update-registry"]
    assert "needs" not in jobs["cancel-branch-runs"]


def test_cancel_script_covers_unfinished_runs_and_tolerates_races() -> None:
    script_path = ROOT / "scripts" / "cancel_branch_runs.sh"
    script = script_path.read_text(encoding="utf-8")

    assert os.access(script_path, os.X_OK)
    assert "for status in queued in_progress waiting pending requested; do" in script
    assert '-f branch="$BRANCH"' in script
    assert "--paginate" in script
    assert 'actions/runs/$run_id/cancel' in script
    # A run that finishes between listing and cancelling answers 409; any other error fails the job.
    assert "grep -q 'HTTP 409'" in script
    assert "exit 1" in script
