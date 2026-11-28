"""Check lifecycle failure routing and the target shown for default-branch events."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def evaluate(expression: str, **values: str) -> object:
    """Evaluate the small expression subset used by these routing conditions."""
    for name, value in sorted(values.items(), key=lambda pair: -len(pair[0])):
        expression = expression.replace(name, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(
        " ".join(expression.split()),
        {"__builtins__": {}},
        {
            "contains": lambda items, value: value in items,
            "fromJSON": json.loads,
            "startsWith": lambda value, prefix: value.startswith(prefix),
        },
    )


@pytest.mark.parametrize("event", ["create", "delete", "workflow_dispatch"])
@pytest.mark.parametrize(
    "conclusion,expected",
    [
        ("failure", True),
        ("timed_out", True),
        ("action_required", True),
        ("startup_failure", True),
        ("stale", True),
        ("success", False),
        ("cancelled", False),
        ("skipped", False),
        ("neutral", False),
    ],
)
def test_only_failed_lifecycle_requests_notify(event: str, conclusion: str, expected: bool) -> None:
    condition = workflow("model-branch-lifecycle-failure.yml")["jobs"]["notify-failure"]["if"]
    result = evaluate(
        condition.removeprefix("${{").removesuffix("}}"),
        **{
            "github.event.workflow_run.repository.full_name": "sima-neat/models",
            "github.repository": "sima-neat/models",
            "github.event.workflow_run.event": event,
            "github.event.workflow_run.conclusion": conclusion,
        },
    )
    assert result is expected


@pytest.mark.parametrize("repository,event", [("fork/models", "delete"), ("sima-neat/models", "push")])
def test_foreign_repositories_and_other_events_do_not_notify(repository: str, event: str) -> None:
    condition = workflow("model-branch-lifecycle-failure.yml")["jobs"]["notify-failure"]["if"]
    assert (
        evaluate(
            condition.removeprefix("${{").removesuffix("}}"),
            **{
                "github.event.workflow_run.repository.full_name": repository,
                "github.repository": "sima-neat/models",
                "github.event.workflow_run.event": event,
                "github.event.workflow_run.conclusion": "failure",
            },
        )
        is False
    )


@pytest.mark.parametrize(
    "event,operation,requested,expected",
    [
        ("delete", "", "", "Purge branch feature/model"),
        ("create", "", "", "Restore branch feature/model"),
        ("workflow_dispatch", "purge", "retry/model", "Purge branch retry/model"),
        ("workflow_dispatch", "restore", "retry/model", "Restore branch retry/model"),
    ],
)
def test_run_title_records_operation_and_requested_branch(
    event: str,
    operation: str,
    requested: str,
    expected: str,
) -> None:
    title = workflow("model-branch-lifecycle.yml")["run-name"]
    values = {
        "github.event_name": event,
        "inputs.operation": operation,
        "inputs.branch": requested,
        "github.event.ref": "feature/model",
    }
    rendered = re.sub(r"\$\{\{(.*?)\}\}", lambda match: str(evaluate(match[1], **values)), title)
    assert rendered == expected


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Purge branch feature/model", "Purge branch feature/model"),
        ("Restore branch feature/model", "Restore branch feature/model"),
        ("Model branch lifecycle", "Target unavailable; see failed jobs"),
    ],
)
def test_notification_uses_target_title_instead_of_default_execution_ref(title: str, expected: str) -> None:
    expression = workflow("model-branch-lifecycle-failure.yml")["jobs"]["notify-failure"]["with"]["branch"]
    assert "head_branch" not in expression
    assert (
        evaluate(
            expression.removeprefix("${{").removesuffix("}}"),
            **{"github.event.workflow_run.display_title": title},
        )
        == expected
    )


def test_listener_is_independent_and_passes_the_exact_failed_attempt() -> None:
    listener = workflow("model-branch-lifecycle-failure.yml")
    assert listener["on"] == {
        "workflow_run": {"workflows": ["Model branch lifecycle"], "types": ["completed"]},
    }
    assert listener["permissions"] == {"actions": "read", "contents": "read"}
    job = listener["jobs"]["notify-failure"]
    assert "steps" not in job  # Never check out or execute the failed revision with the Slack secret.
    assert job["uses"] == "sima-neat/.github/.github/workflows/vulcan-notify-slack.yml@main"
    assert job["with"]["run_id"] == "${{ github.event.workflow_run.id }}"
    assert job["with"]["run_attempt"] == "${{ github.event.workflow_run.run_attempt }}"
    assert job["with"]["run_url"] == "${{ github.event.workflow_run.html_url }}"
    assert job["secrets"] == {"slack_bot_token": "${{ secrets.SLACK_BOT_TOKEN }}"}
    assert job["with"].get("analyze_failure", "false") == "false"
    assert "github.event.workflow_run.id" in listener["concurrency"]["group"]
    assert "github.event.workflow_run.run_attempt" in listener["concurrency"]["group"]
    assert listener["concurrency"]["cancel-in-progress"] == "false"
