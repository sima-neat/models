"""Each model's tests: one file per stage under models/<id>/tests/, set up by tests/tests.yaml.

build/ checks conversion and compilation; benchmark/ checks DevKit accuracy and performance. tests.yaml says
whether each stage's tests run and what the model is expected to deliver. No fingerprint reads these files, so
changing a test reruns nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from model_report import METRICS

from .benchmark_declarations import benchmarked_profiles
from .errors import ValidationError

# How far a measured value may fall below the value tests.yaml expects and still pass, relative to that value.
ACCURACY_ALLOWED_DROP = 0.05
PERFORMANCE_ALLOWED_DROP = 0.10
# How far a build profile's accuracy, expected or measured, may fall below the source model's reference score,
# relative to that score. A profile further below fails model-registry validate and its DevKit accuracy test, so
# model.yaml comments it out.
ACCURACY_ALLOWED_REFERENCE_DROP = 0.08

TESTS_DIR = "tests"
TESTS_FILE = "tests.yaml"
TESTS_SCHEMA = "model-tests.schema.json"
# Each stage's test file, by the folder that names what it checks.
TEST_GROUPS = {"build": ("conversion", "compile"), "benchmark": ("accuracy", "performance")}
STAGE_GROUP = {stage: group for group, stages in TEST_GROUPS.items() for stage in stages}
TESTS_DIR_ENTRIES = {TESTS_FILE, *TEST_GROUPS, "fixtures"}
# What scripts/run_model_tests.sh prints for a stage's tests; a benchmark publishes it as its test status.
TEST_OUTCOMES = ("passed", "failed", "skipped")


def stage_test_file(stage: str) -> str:
    """The stage's test file, relative to the model's tests/ folder."""
    return f"{STAGE_GROUP[stage]}/test_{stage}.py"


def required_stages(model: dict[str, Any]) -> list[str]:
    """Stages tests.yaml must set up: what the model converts, compiles and measures on a DevKit."""
    return [
        *(["conversion"] if model.get("conversion") else []),
        "compile",
        *(["accuracy"] if "accuracy" in model else []),
        *(["performance"] if "performance" in model else []),
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


def measured_shortfalls(model_dir: str | Path, stage: str, profile_id: str, report: dict[str, Any]) -> list[str]:
    """Each value tests.yaml expects of a build profile's report that it misses by more than the stage's allowed
    drop. Accuracy expects values per dataset, named as in model.yaml, and a dataset's measured score also misses
    when it falls below the model's reference score by more than ACCURACY_ALLOWED_REFERENCE_DROP."""
    expected = load_model_tests(model_dir)[stage]["expected"].get(profile_id, {})
    if not expected:
        return [f"{TESTS_FILE} expects no {stage} values of {profile_id}"]
    if stage == "performance":
        if "components" in expected:
            if report.get("scope") != "components":
                return ["performance scope must be components"]
            return [
                shortfall
                for component, values in expected["components"].items()
                for shortfall in (
                    _shortfalls(
                        f"{component}: ", values, report["components"][component]["metrics"], PERFORMANCE_ALLOWED_DROP
                    )
                    if component in report.get("components", {})
                    else [f"{component} was not measured"]
                )
            ]
        return _shortfalls("", expected, report["metrics"], PERFORMANCE_ALLOWED_DROP)

    # model.py imports this module, so its loader is imported where the reference scores are needed.
    from .model import load_model

    model = load_model(model_dir)
    measured = report["datasets"]
    shortfalls = []
    for dataset, values in expected.items():
        if dataset not in measured:
            shortfalls.append(f"{dataset} was not evaluated")
            continue
        metrics = measured[dataset]["metrics"]
        shortfalls += _shortfalls(f"{dataset}: ", values, metrics, ACCURACY_ALLOWED_DROP)
        if reference := _reference_shortfall(model, dataset, values, metrics):
            shortfalls.append(f"{dataset}: {reference}")
    return shortfalls


def _shortfalls(label: str, expected: dict[str, float], metrics: dict[str, float], drop: float) -> list[str]:
    shortfalls = []
    for name, value in expected.items():
        measured = metrics.get(name)
        if measured is None:
            shortfalls.append(f"{label}{name} was not measured")
        elif side := _missed_side(name, measured, value, drop):
            shortfalls.append(f"{label}{name} {measured:g} is {side} {value:g} by more than {drop:.0%}")
    return shortfalls


def _missed_side(name: str, value: float, target: float, drop: float) -> str | None:
    """The side, "above" or "below", on which a metric's value is worse than the target by more than the drop,
    relative to the target and in the metric's better direction, or None when it is not."""
    if METRICS[name]["direction"] == "lower":
        return "above" if value > target * (1 + drop) else None
    return "below" if value < target * (1 - drop) else None


def _reference_shortfall(
    model: dict[str, Any], dataset_name: str, expected: dict[str, float], metrics: dict[str, float]
) -> str | None:
    """How a build profile's accuracy on one dataset falls below the dataset's reference score by more than
    ACCURACY_ALLOWED_REFERENCE_DROP, or None. tests.yaml's `expected` values choose the metric, and `metrics` holds
    the values compared: the expected ones themselves, or a report's. A dataset without a comparable reference, or
    a metric `metrics` lacks, decides nothing."""
    reference = _reference_score(model, dataset_name, expected)
    if reference is None:
        return None
    name, score = reference
    drop = ACCURACY_ALLOWED_REFERENCE_DROP
    side = _missed_side(name, metrics[name], score, drop) if name in metrics else None
    return f"{name} {metrics[name]:g} is {side} the reference {score:g} by more than {drop:.0%}" if side else None


def _reference_score(model: dict[str, Any], dataset_name: str, expected: dict[str, float]) -> tuple[str, float] | None:
    """The metric and reference score a build profile's accuracy on one dataset is compared with, or None. The
    reference is the model's FP32 baseline on the dataset, else the score published for the split the model
    evaluates, and the metric is the first of that reference that tests.yaml expects."""
    dataset = model["accuracy"]["datasets"][dataset_name]
    on_dataset = [entry for entry in model.get("reference_metrics", []) if entry["dataset"]["id"] == dataset["id"]]
    baselines = [entry for entry in on_dataset if entry.get("origin") == "fp32_baseline"]
    published = [
        entry
        for entry in on_dataset
        if entry.get("origin", "published") == "published" and entry["dataset"]["split"] == dataset["variant"]
    ]
    reference = next(iter([*baselines, *published]), None)
    if reference is None:
        return None

    name = next((name for name in reference["metrics"] if name in expected), None)
    return (name, reference["metrics"][name]) if name else None


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
        if stage in TEST_GROUPS["benchmark"]:
            _require_expected_values(path, stage, tests[stage]["expected"], model)

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
            f"each stage has one test file, build/test_<stage>.py or benchmark/test_<stage>.py"
        )


def _require_expected_values(path: Path, stage: str, expected: dict[str, Any], model: dict[str, Any]) -> None:
    """Every benchmarked build profile expects its required metrics, for accuracy on each dataset (performance: its
    throughput), and no other declared profile expects any. A profile model.yaml leaves out, such as one commented
    out until it works, keeps its expectations unchecked."""
    profiles = benchmarked_profiles(model, stage)
    declared = {profile["id"] for profile in model["build_profiles"]}
    unknown_profiles = (set(expected) & declared) - set(profiles)
    if unknown_profiles:
        names = ", ".join(sorted(unknown_profiles))
        raise ValidationError(f"{path}: {stage}.expected names build profiles that are not benchmarked: {names}")
    for profile in profiles:
        values = expected.get(profile, {})
        if stage == "performance":
            if model[stage].get("scope") == "components":
                components = {component["id"] for component in model["components"]}
                if set(values.get("components", {})) != components:
                    raise ValidationError(
                        f"{path}: performance.expected.{profile}.components must name {sorted(components)}"
                    )
                for component in components:
                    _require_metric_values(
                        path,
                        stage,
                        f"performance.expected.{profile}.components.{component}",
                        values["components"][component],
                        ["throughput_fps"],
                    )
                continue
            _require_metric_values(path, stage, f"{stage}.expected.{profile}", values, ["throughput_fps"])
            continue
        datasets = model["accuracy"]["datasets"]
        unknown = sorted(set(values) - set(datasets))
        if unknown:
            names = ", ".join(unknown)
            raise ValidationError(f"{path}: accuracy.expected.{profile} names undeclared datasets: {names}")
        for name, dataset in datasets.items():
            _require_metric_values(
                path, stage, f"accuracy.expected.{profile}.{name}", values.get(name, {}), dataset["required_metrics"]
            )


def require_reference_accuracy(model_dir: str | Path, model: dict[str, Any]) -> None:
    """Require every benchmarked build profile to expect accuracy within ACCURACY_ALLOWED_REFERENCE_DROP of each
    dataset's reference score. model.yaml comments a profile further below out instead of shipping it.

    `model` is the model validate_model_tests accepted. model-registry validate and validate-models apply this rule
    to the checkout a contributor changes. Loading a model does not, so tooling still reads another revision of a
    model that declares such a profile, such as the target branch's during promotion.
    """
    tests = load_model_tests(model_dir)
    if "accuracy" not in model or skip_reason(tests, "accuracy") is not None:
        return
    path = Path(model_dir).expanduser().resolve() / TESTS_DIR / TESTS_FILE
    expected = tests["accuracy"]["expected"]
    for profile in benchmarked_profiles(model, "accuracy"):
        for dataset, values in expected[profile].items():
            if shortfall := _reference_shortfall(model, dataset, values, values):
                raise ValidationError(
                    f"{path}: accuracy.expected.{profile}.{dataset}: {shortfall}; "
                    f"comment the {profile} profile out in model.yaml"
                )


def _require_metric_values(path: Path, stage: str, label: str, values: dict[str, Any], required: list[str]) -> None:
    missing = [metric for metric in required if metric not in values]
    if missing:
        raise ValidationError(f"{path}: {label} lacks {', '.join(missing)}")
    unknown = sorted(name for name in values if METRICS.get(name, {}).get("stage") != stage)
    if unknown:
        raise ValidationError(f"{path}: {label} has unknown {stage} metrics: {', '.join(unknown)}")
    for metric, value in values.items():
        if not math.isfinite(value):
            raise ValidationError(f"{path}: {label}.{metric} must be a finite number")
