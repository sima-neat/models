from __future__ import annotations

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
