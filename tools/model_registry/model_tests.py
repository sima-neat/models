"""Each model's tests: one file per stage under models/<id>/tests/, set up by tests/tests.yaml.

build/ checks conversion and compilation; measure/ checks DevKit accuracy and performance. tests.yaml says
whether each stage's tests run and what the model is expected to deliver. No stage key reads these files, so
changing a test reruns nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .accuracy_config import DEVKIT_ACCURACY_ENGINE
from .benchmark_metrics import registry_metrics
from .errors import ValidationError
from .published_metrics import load_accuracy_metrics
from .taxonomy import find_taxonomy

# How far a measured value may fall below the value tests.yaml expects and still pass, relative to that value.
ACCURACY_ALLOWED_DROP = 0.05
PERFORMANCE_ALLOWED_DROP = 0.10

TESTS_DIR = "tests"
TESTS_FILE = "tests.yaml"
TESTS_SCHEMA = "model-tests.schema.json"
# Each stage's test file, by the folder that names what it checks.
TEST_GROUPS = {"build": ("conversion", "compile"), "measure": ("accuracy", "performance")}
STAGE_GROUP = {stage: group for group, stages in TEST_GROUPS.items() for stage in stages}
TESTS_DIR_ENTRIES = {TESTS_FILE, *TEST_GROUPS, "fixtures"}
# What scripts/run_model_tests.sh prints for a stage's tests; a measurement publishes it as its test status.
TEST_OUTCOMES = ("passed", "failed", "skipped")


def stage_test_file(stage: str) -> str:
    """The stage's test file, relative to the model's tests/ folder."""
    return f"{STAGE_GROUP[stage]}/test_{stage}.py"


def required_stages(model: dict[str, Any]) -> list[str]:
    """Stages tests.yaml must set up: what the model converts, compiles and measures on a DevKit."""
    return [
        *(["conversion"] if model.get("conversion") else []),
        "compile",
        *(["accuracy"] if _devkit_accuracy(model) else []),
        *(["performance"] if model["benchmark"] else []),
    ]


def require_test_outcome(outcome: str) -> str:
    if outcome not in TEST_OUTCOMES:
        raise ValidationError(f"test outcome must be one of {', '.join(TEST_OUTCOMES)}, not {outcome!r}")
    return outcome


def load_model_tests(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / TESTS_DIR / TESTS_FILE
    try:
        tests = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load the model's tests: {exc}") from exc
    return tests if isinstance(tests, dict) else {}


def skip_reason(tests: dict[str, Any], stage: str) -> str | None:
    """Why tests.yaml skips the stage's tests, or None when they run."""
    entry = tests.get(stage) or {}
    return None if entry.get("run_tests", True) else entry["skip_reason"]


def measured_shortfalls(model_dir: str | Path, stage: str, result: dict[str, Any]) -> list[str]:
    """Each value tests.yaml expects of this measured result that the result misses by more than the stage's
    allowed drop."""
    drop = ACCURACY_ALLOWED_DROP if stage == "accuracy" else PERFORMANCE_ALLOWED_DROP
    build = [result["model"]["build_profile_id"]]
    if stage == "performance":
        # A package result is expected by its package ID, a component result by its component ID.
        build.append(result["model"].get("package_id", result["model"]["component_id"]))
    expected = load_model_tests(model_dir)[stage]["expected"]
    for key in build:
        expected = expected.get(key, {})
    if not expected:
        return [f"{TESTS_FILE} expects no {stage} values of {'/'.join(build)}"]
    measured_metrics = registry_metrics(result) if stage == "performance" else result["metrics"]
    shortfalls = []
    for name, value in expected.items():
        metric = measured_metrics.get(name) or {}
        measured = metric.get("value")
        if measured is None:
            shortfalls.append(f"{name} was not measured")
        elif metric.get("direction") == "lower" and measured > value * (1 + drop):
            shortfalls.append(f"{name} {measured:g} is above {value:g} by more than {drop:.0%}")
        elif metric.get("direction") != "lower" and measured < value * (1 - drop):
            shortfalls.append(f"{name} {measured:g} is below {value:g} by more than {drop:.0%}")
    return shortfalls


def validate_model_tests(model_dir: str | Path, model: dict[str, Any], schema_path: str | Path) -> None:
    """Require an entry per required stage, the file of every stage whose tests run, the values every measured
    build is expected to deliver, and nothing else in tests/."""
    directory = Path(model_dir).expanduser().resolve()
    tests_dir = directory / TESTS_DIR
    path = tests_dir / TESTS_FILE
    if not path.is_file():
        raise ValidationError(f"{directory}: {TESTS_DIR}/{TESTS_FILE} is required")
    tests = load_model_tests(directory)
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{schema_path}: could not load schema: {exc}") from exc
    errors = sorted(Draft202012Validator(schema).iter_errors(tests), key=lambda error: list(error.absolute_path))
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or TESTS_FILE
        raise ValidationError(f"{path}: {location}: {errors[0].message}")

    required = set(required_stages(model))
    if set(tests) != required:
        detail = [
            *(f"missing {stage}" for stage in sorted(required - set(tests))),
            *(f"unknown {stage}" for stage in sorted(set(tests) - required)),
        ]
        raise ValidationError(f"{path}: set up exactly the stages this model has: {', '.join(detail)}")
    for stage in sorted(required):
        if skip_reason(tests, stage) is not None:
            continue
        if not (tests_dir / stage_test_file(stage)).is_file():
            raise ValidationError(f"{directory}: {stage} tests run but {TESTS_DIR}/{stage_test_file(stage)} is missing")
        if stage in TEST_GROUPS["measure"]:
            _require_expected_values(path, stage, tests[stage]["expected"], _measured_builds(model, stage))

    allowed = {
        tests_dir: TESTS_DIR_ENTRIES,
        **{tests_dir / group: {f"test_{stage}.py" for stage in stages} for group, stages in TEST_GROUPS.items()},
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


def _devkit_accuracy(model: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in model["accuracy"] if entry.get("engine") == DEVKIT_ACCURACY_ENGINE]


def _measured_builds(model: dict[str, Any], stage: str) -> dict[tuple[str, ...], list[str]]:
    """Each measured build and its mandatory expectations: accuracy per profile, performance per scope."""
    if stage == "accuracy":
        return {
            (profile,): list(entry.get("metrics", entry.get("required_metrics", ())))
            for entry in _devkit_accuracy(model)
            for profile in entry["build_profiles"]
        }
    return {
        (profile, case["scope"][case["scope"]["type"]]): ["throughput_fps"]
        for case in model["benchmark"]
        for profile in case["build_profiles"]
    }


def _require_expected_values(
    path: Path, stage: str, expected: dict[str, Any], builds: dict[tuple[str, ...], list[str]]
) -> None:
    for build, metrics in builds.items():
        values = expected
        for key in build:
            values = values.get(key, {})
        missing = [metric for metric in metrics if metric not in values]
        if missing:
            raise ValidationError(f"{path}: {stage}.expected.{'.'.join(build)} lacks {', '.join(missing)}")
        if not values:
            raise ValidationError(f"{path}: {stage}.expected.{'.'.join(build)} must contain expected values")
    if stage == "accuracy":
        vocabulary = load_accuracy_metrics(find_taxonomy(path).parent / "accuracy-metrics.yaml")
        for profile, values in expected.items():
            unknown = set(values) - set(vocabulary)
            if unknown:
                raise ValidationError(
                    f"{path}: accuracy.expected.{profile} has unregistered metrics: {', '.join(sorted(unknown))}"
                )
            for metric, value in values.items():
                if not math.isfinite(value):
                    raise ValidationError(f"{path}: accuracy.expected.{profile}.{metric} must be a finite number")
