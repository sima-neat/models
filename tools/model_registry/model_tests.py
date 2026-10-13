"""Each model's tests: one file per stage under models/<id>/tests/, switched by tests/test-scope.yaml.

build/ checks conversion and compilation; measure/ checks DevKit accuracy and performance. No stage key
reads these files, so changing a test reruns nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .accuracy_config import DEVKIT_ACCURACY_ENGINE
from .errors import ValidationError

TESTS_DIR = "tests"
TEST_SCOPE_FILE = "test-scope.yaml"
TEST_SCOPE_SCHEMA = "test-scope.schema.json"
# Each stage's test file, by the folder that names what it checks.
TEST_GROUPS = {"build": ("conversion", "compile"), "measure": ("accuracy", "performance")}
STAGE_GROUP = {stage: group for group, stages in TEST_GROUPS.items() for stage in stages}
TESTS_DIR_ENTRIES = {TEST_SCOPE_FILE, *TEST_GROUPS, "fixtures"}
# What scripts/run_model_tests.sh prints for a stage's tests; a measurement publishes it as its test status.
TEST_OUTCOMES = ("passed", "failed", "skipped")


def stage_test_file(stage: str) -> str:
    """The stage's test file, relative to the model's tests/ folder."""
    return f"{STAGE_GROUP[stage]}/test_{stage}.py"


def required_stages(model: dict[str, Any]) -> list[str]:
    """Stages whose tests the model must switch: what it converts, compiles and measures on a DevKit."""
    return [
        *(["conversion"] if model.get("conversion") else []),
        "compile",
        *(["accuracy"] if any(entry.get("engine") == DEVKIT_ACCURACY_ENGINE for entry in model["accuracy"]) else []),
        *(["performance"] if model["benchmark"] else []),
    ]


def require_test_outcome(outcome: str) -> str:
    if outcome not in TEST_OUTCOMES:
        raise ValidationError(f"test outcome must be one of {', '.join(TEST_OUTCOMES)}, not {outcome!r}")
    return outcome


def load_test_scope(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / TESTS_DIR / TEST_SCOPE_FILE
    try:
        scope = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load test scope: {exc}") from exc
    return scope if isinstance(scope, dict) else {}


def disabled_reason(scope: dict[str, Any], stage: str) -> str | None:
    """Why the stage's tests are switched off, or None when they run."""
    switch = (scope.get(STAGE_GROUP[stage]) or {}).get(stage, True)
    return None if switch is True else switch["reason"]


def validate_test_scope(model_dir: str | Path, model: dict[str, Any], schema_path: str | Path) -> None:
    """Require one switch per required stage, the file of every enabled stage, and nothing else in tests/."""
    directory = Path(model_dir).expanduser().resolve()
    tests = directory / TESTS_DIR
    if not (tests / TEST_SCOPE_FILE).is_file():
        raise ValidationError(f"{directory}: {TESTS_DIR}/{TEST_SCOPE_FILE} is required")
    scope = load_test_scope(directory)
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_path}: could not load schema: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(scope), key=lambda error: list(error.absolute_path))
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "test scope"
        raise ValidationError(f"{tests / TEST_SCOPE_FILE}: {location}: {errors[0].message}")

    switched = {stage for group in TEST_GROUPS for stage in scope.get(group) or {}}
    required = set(required_stages(model))
    if switched != required:
        detail = [
            *(f"missing {stage}" for stage in sorted(required - switched)),
            *(f"unknown {stage}" for stage in sorted(switched - required)),
        ]
        raise ValidationError(
            f"{tests / TEST_SCOPE_FILE}: switch exactly the stages this model declares: {', '.join(detail)}"
        )
    for stage in sorted(required):
        if disabled_reason(scope, stage) is None and not (tests / stage_test_file(stage)).is_file():
            raise ValidationError(
                f"{directory}: {stage} tests are enabled but {TESTS_DIR}/{stage_test_file(stage)} is missing"
            )

    allowed = {
        tests: TESTS_DIR_ENTRIES,
        **{tests / group: {f"test_{stage}.py" for stage in stages} for group, stages in TEST_GROUPS.items()},
    }
    unexpected = sorted(
        entry.relative_to(directory).as_posix()
        for folder, names in allowed.items()
        if folder.is_dir()
        for entry in folder.iterdir()
        if entry.name not in names and entry.name != "__pycache__" and not entry.name.startswith(".")
    )
    if unexpected:
        raise ValidationError(
            f"{directory}: unexpected test files: {', '.join(unexpected)}; "
            f"each stage has one test file, build/test_<stage>.py or measure/test_<stage>.py"
        )
