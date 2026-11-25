from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md"))
WORKFLOW = ROOT / ".github" / "workflows" / "issue-triage.yml"


def triage_path() -> Path:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return ROOT / workflow["jobs"]["triage"]["with"]["repo_triage_path"]


def allowed_labels() -> set[str]:
    config = json.loads((triage_path() / "config.json").read_text(encoding="utf-8"))
    return set(config["labels"]["allowed"])


def template_labels(template: Path) -> set[str]:
    front_matter = template.read_text(encoding="utf-8").split("---")[1]
    labels = yaml.safe_load(front_matter)["labels"]
    return {label.strip() for label in labels.split(",")} if isinstance(labels, str) else set(labels)


def test_triage_reads_a_config_and_a_skill() -> None:
    assert allowed_labels()
    assert list((triage_path() / "skills").glob("*/SKILL.md"))


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda template: template.name)
def test_template_labels_are_triage_labels(template: Path) -> None:
    labels = template_labels(template)

    assert labels
    assert labels <= allowed_labels()
