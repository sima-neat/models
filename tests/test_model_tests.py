from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.model_tests import (
    TEST_GROUPS,
    load_model_tests,
    measured_shortfalls,
    skip_reason,
    stage_test_file,
    validate_model_tests,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL_TESTS = ROOT / "scripts" / "run_model_tests.sh"
PERFORMANCE_EXPECTED = """\
performance:
  run_tests: true
  expected:
    w8a8:
      throughput_fps: 100.0
"""
PERFORMANCE_TESTS = '''
def test_needs_no_output() -> None:
    assert True


def test_throughput_meets_its_minimum(performance, build_profile) -> None:
    assert build_profile == "w8a8"
    assert performance["metrics"]["throughput_fps"] >= 100
'''


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A repository holding one model with performance tests, configured as this repository runs them."""
    for name in ("pyproject.toml", "conftest.py"):
        shutil.copy(ROOT / name, tmp_path / name)
    model = tmp_path / "models" / "demo"
    (model / "tests" / "benchmark").mkdir(parents=True)
    (model / "model.yaml").write_text("id: demo\n", encoding="utf-8")
    (model / "tests" / "tests.yaml").write_text(PERFORMANCE_EXPECTED, encoding="utf-8")
    (model / "tests" / "benchmark" / "test_performance.py").write_text(PERFORMANCE_TESTS, encoding="utf-8")
    return model


def performance_report(tmp_path: Path, fps: float) -> Path:
    path = tmp_path / "model-report.json"
    report = {"schema": "sima.model-report", "stage": "performance", "metrics": {"throughput_fps": fps}}
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def pytest_in(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(repository / "pyproject.toml"), "--rootdir", str(repository),
         "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, check=False,
    )  # fmt: skip


def run_model_tests(stage: str, model: Path, output: Path, *profile: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUN_MODEL_TESTS), stage, str(model), str(output), *profile],
        capture_output=True, text=True, check=False, env={**os.environ, "REGISTRY_PYTHON": sys.executable},
    )  # fmt: skip


def test_plain_run_deselects_the_tests_that_check_stage_output(model_dir: Path) -> None:
    result = pytest_in(model_dir.parents[1], str(model_dir / "tests"))

    assert result.returncode == 0, result.stdout
    assert "1 passed, 1 deselected" in result.stdout


def test_given_stage_output_runs_only_the_tests_that_check_it(model_dir: Path, tmp_path: Path) -> None:
    output = performance_report(tmp_path, 150)
    result = pytest_in(
        model_dir.parents[1], str(model_dir / "tests"), f"--performance={output}", "--build-profile=w8a8", "-v"
    )

    assert result.returncode == 0, result.stdout
    assert "test_throughput_meets_its_minimum PASSED" in result.stdout
    assert "1 passed, 1 deselected" in result.stdout


@pytest.mark.parametrize(("fps", "outcome", "exit_code"), [(150, "passed", 0), (50, "failed", 1)])
def test_stage_runner_prints_the_outcome_of_the_stage_tests(
    model_dir: Path, tmp_path: Path, fps: float, outcome: str, exit_code: int
) -> None:
    result = run_model_tests("performance", model_dir, performance_report(tmp_path, fps), "w8a8")

    assert result.stdout == f"{outcome}\n", result.stderr
    assert result.returncode == exit_code


def test_stage_runner_skips_a_stage_tests_yaml_skips(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "tests.yaml").write_text(
        "performance:\n  run_tests: false\n  skip_reason: no published result\n", encoding="utf-8"
    )

    result = run_model_tests("performance", model_dir, performance_report(tmp_path, 50), "w8a8")

    assert result.stdout == "skipped\n"
    assert result.returncode == 0
    assert "no published result" in result.stderr
    # A plain run skips the file's tests with the same reason.
    plain = pytest_in(model_dir.parents[1], str(model_dir / "tests"), f"--performance={tmp_path / 'none.json'}")
    assert "1 skipped, 1 deselected" in plain.stdout
    assert "skipped in tests.yaml: no published result" in plain.stdout


def test_stage_runner_passes_a_stage_whose_tests_take_no_output(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "benchmark" / "test_performance.py").write_text(
        "def test_needs_no_output() -> None:\n    assert True\n", encoding="utf-8"
    )

    result = run_model_tests("performance", model_dir, performance_report(tmp_path, 50), "w8a8")

    assert result.stdout == "passed\n"
    assert "no performance test of demo takes the performance output" in result.stderr


@pytest.mark.parametrize(
    ("stage", "output", "profile", "message"),
    [
        ("latency", "result.json", ("w8a8",), "unknown stage: latency"),
        ("performance", "missing.json", ("w8a8",), "performance output does not exist"),
        ("performance", "result.json", (), "performance tests need the build profile the report measured"),
        ("accuracy", "result.json", ("w8a8",), "tests/benchmark/test_accuracy.py is missing"),
    ],
)
def test_stage_runner_reports_no_outcome_when_it_cannot_run_the_tests(
    model_dir: Path, tmp_path: Path, stage: str, output: str, profile: tuple[str, ...], message: str
) -> None:
    performance_report(tmp_path, 150).rename(tmp_path / "result.json")

    result = run_model_tests(stage, model_dir, tmp_path / output, *profile)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_stage_output_tests_stop_on_an_invalid_tests_file(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "tests.yaml").write_text("performance: [\n", encoding="utf-8")

    result = pytest_in(
        model_dir.parents[1], str(model_dir / "tests"), f"--performance={performance_report(tmp_path, 150)}"
    )

    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "could not load the model's tests" in result.stderr


@pytest.mark.parametrize("model", sorted(path.name for path in (ROOT / "models").iterdir() if path.is_dir()))
def test_each_running_benchmark_file_checks_its_stage_result(model: str) -> None:
    model_dir = ROOT / "models" / model
    tests = load_model_tests(model_dir)
    for stage in TEST_GROUPS["benchmark"]:
        if stage not in tests or skip_reason(tests, stage):
            continue
        path = model_dir / "tests" / stage_test_file(stage)
        tests = [
            node
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        assert any(stage in {argument.arg for argument in test.args.args} for test in tests), (
            f"{path.relative_to(ROOT)} has no test that takes the {stage} result"
        )


def test_a_test_with_its_own_fixture_of_a_stage_output_name_still_runs(model_dir: Path) -> None:
    (model_dir / "tests" / "benchmark" / "test_performance.py").write_text(
        "import pytest\n\n\n@pytest.fixture\ndef accuracy():\n    return 1\n\n\n"
        "def test_own_fixture(accuracy):\n    assert accuracy == 1\n\n\n"
        "@pytest.mark.parametrize('performance', [1, 2])\n"
        "def test_own_parameter(performance):\n    assert performance\n",
        encoding="utf-8",
    )

    result = pytest_in(model_dir.parents[1], str(model_dir / "tests"))

    assert result.returncode == 0, result.stdout
    assert "3 passed" in result.stdout
    assert "deselected" not in result.stdout


@pytest.mark.parametrize(
    ("datasets", "shortfalls"),
    [
        ({"imagenet": {"top1_accuracy": 0.80, "top5_accuracy": 0.95}, "coco": {"coco_map": 0.4}}, []),
        ({"imagenet": {"top1_accuracy": 0.7601, "top5_accuracy": 0.95}, "coco": {"coco_map": 0.4}}, []),
        (
            {"imagenet": {"top1_accuracy": 0.7599, "top5_accuracy": 0.95}, "coco": {"coco_map": 0.3}},
            ["imagenet: top1_accuracy 0.7599 is below 0.8 by more than 5%",
             "coco: coco_map 0.3 is below 0.4 by more than 5%"],
        ),
        (
            {"imagenet": {"top1_accuracy": 0.80}, "coco": {"coco_map": 0.4}},
            ["imagenet: top5_accuracy was not measured"],
        ),
        ({"imagenet": {"top1_accuracy": 0.80, "top5_accuracy": 0.95}}, ["coco was not evaluated"]),
    ],
    ids=["equal", "within the allowed drop", "below the allowed drop", "missing metric", "missing dataset"],
)
def test_measured_shortfalls_compare_each_expected_value_on_each_dataset(
    tmp_path: Path, datasets: dict[str, dict[str, float]], shortfalls: list[str]
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "tests.yaml").write_text(
        "accuracy:\n  run_tests: true\n  expected:\n    w8a8:\n"
        "      imagenet: {top1_accuracy: 0.80, top5_accuracy: 0.95}\n      coco: {coco_map: 0.4}\n",
        encoding="utf-8",
    )
    report = {"datasets": {name: {"metrics": metrics} for name, metrics in datasets.items()}}

    assert measured_shortfalls(tmp_path, "accuracy", "w8a8", report) == shortfalls


def test_measured_shortfalls_need_values_expected_of_the_build_profile(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "tests.yaml").write_text(PERFORMANCE_EXPECTED, encoding="utf-8")

    assert measured_shortfalls(tmp_path, "performance", "w16a16", {"metrics": {"throughput_fps": 100.0}}) == [
        "tests.yaml expects no performance values of w16a16"
    ]


def test_measured_shortfalls_allow_performance_its_own_drop_and_rise(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    expected = PERFORMANCE_EXPECTED + "      latency_avg_ms: 2.0\n"
    (tmp_path / "tests" / "tests.yaml").write_text(expected, encoding="utf-8")

    def shortfalls(metrics: dict[str, float]) -> list[str]:
        return measured_shortfalls(tmp_path, "performance", "w8a8", {"metrics": metrics})

    assert shortfalls({"throughput_fps": 90, "latency_avg_ms": 2.19}) == []
    assert shortfalls({"throughput_fps": 89, "latency_avg_ms": 2.21}) == [
        "throughput_fps 89 is below 100 by more than 10%",
        "latency_avg_ms 2.21 is above 2 by more than 10%",
    ]


ACCURACY_MODEL = {
    "build_profiles": [{"id": "w8a8"}],
    "accuracy": {"datasets": {
        "imagenet": {"required_metrics": ["top1_accuracy"]}, "coco": {"required_metrics": ["coco_map"]},
    }},
}
COCO = {"coco": {"coco_map": 0.4}}


@pytest.mark.parametrize(
    ("expected", "error"),
    [
        ({"w8a8": {"imagenet": {"top1_accuracy": 0.8, "top5_accuracy": 0.95}, **COCO}}, None),
        ({}, "accuracy.expected.w8a8.imagenet lacks top1_accuracy"),
        ({"w8a8": {}}, "accuracy.expected.w8a8"),
        ({"w8a8": {"imagenet": {}, **COCO}}, "accuracy.expected.w8a8.imagenet"),
        ({"w8a8": {"imagenet": {"top5_accuracy": 0.95}, **COCO}}, "w8a8.imagenet lacks top1_accuracy"),
        ({"w8a8": {"imagenet": {"top1_accuracy": 0.8}}}, "accuracy.expected.w8a8.coco lacks coco_map"),
        (
            {"w8a8": {"imagenet": {"top1_accuracy": 0.8}, "other": {"top1_accuracy": 0.8}, **COCO}},
            "accuracy.expected.w8a8 names undeclared datasets: other",
        ),
        # model.yaml leaves w16a16 out, as when it is commented out, so its expectations wait unchecked.
        ({"w8a8": {"imagenet": {"top1_accuracy": 0.8}, **COCO}, "w16a16": {"imagenet": {"top1_accuracy": 0.8}}}, None),
        (
            {"w8a8": {"imagenet": {"top1_accuracy": 0.8, "unknown_metric": 0.9}, **COCO}},
            "unknown accuracy metrics: unknown_metric",
        ),
        (
            {"w8a8": {"imagenet": {"top1_accuracy": 0.8, "throughput_fps": 9.0}, **COCO}},
            "unknown accuracy metrics: throughput_fps",
        ),
        ({"w8a8": {"imagenet": {"top1_accuracy": float("nan")}, **COCO}}, "must be a finite number"),
        ({"w8a8": {"imagenet": {"top1_accuracy": float("inf")}, **COCO}}, "must be a finite number"),
    ],
)
def test_accuracy_expectations_give_each_measured_build_profile_its_required_metrics(
    model_dir: Path, expected: dict, error: str | None
) -> None:
    tests = model_dir / "tests"
    (tests / "build").mkdir()
    (tests / "build" / "test_compile.py").write_text("", encoding="utf-8")
    (tests / "benchmark" / "test_accuracy.py").write_text("", encoding="utf-8")
    (tests / "tests.yaml").write_text(
        yaml.safe_dump({"compile": {"run_tests": True}, "accuracy": {"run_tests": True, "expected": expected}}),
        encoding="utf-8",
    )
    if error is None:
        validate_model_tests(model_dir, ACCURACY_MODEL, ROOT / "schemas" / "model-tests.schema.json")
    else:
        with pytest.raises(ValidationError, match=error):
            validate_model_tests(model_dir, ACCURACY_MODEL, ROOT / "schemas" / "model-tests.schema.json")


def test_accuracy_expectations_reject_a_declared_profile_that_is_not_measured(model_dir: Path) -> None:
    model = {**ACCURACY_MODEL, "build_profiles": [{"id": "w8a8"}, {"id": "w16a16"}],
             "accuracy": {**ACCURACY_MODEL["accuracy"], "build_profiles": ["w8a8"]}}
    tests = model_dir / "tests"
    (tests / "build").mkdir()
    (tests / "build" / "test_compile.py").write_text("", encoding="utf-8")
    (tests / "benchmark" / "test_accuracy.py").write_text("", encoding="utf-8")
    expected = {"w8a8": {"imagenet": {"top1_accuracy": 0.8}, **COCO}, "w16a16": {"imagenet": {"top1_accuracy": 0.8}}}
    (tests / "tests.yaml").write_text(
        yaml.safe_dump({"compile": {"run_tests": True}, "accuracy": {"run_tests": True, "expected": expected}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="names build profiles that are not benchmarked: w16a16"):
        validate_model_tests(model_dir, model, ROOT / "schemas" / "model-tests.schema.json")


def test_accuracy_expectations_may_be_skipped_explicitly(model_dir: Path) -> None:
    (model_dir / "tests" / "build").mkdir()
    (model_dir / "tests" / "build" / "test_compile.py").write_text("", encoding="utf-8")
    (model_dir / "tests" / "tests.yaml").write_text(
        "compile:\n  run_tests: true\naccuracy:\n  run_tests: false\n  skip_reason: no published measurement\n",
        encoding="utf-8",
    )

    validate_model_tests(model_dir, ACCURACY_MODEL, ROOT / "schemas" / "model-tests.schema.json")


def test_component_expectations_gate_each_result_without_combining(model_dir):
    tests = {"performance": {"run_tests": True, "expected": {"w8a16": {"components": {
        "image_encoder": {"throughput_fps": 200.0}, "text_encoder": {"throughput_fps": 600.0},
    }}}}}
    (model_dir / "tests/tests.yaml").write_text(yaml.safe_dump(tests))
    report = {"scope": "components", "components": {
        "image_encoder": {"metrics": {"throughput_fps": 195.0}},
        "text_encoder": {"metrics": {"throughput_fps": 500.0}},
    }}
    shortfalls = measured_shortfalls(model_dir, "performance", "w8a16", report)
    assert len(shortfalls) == 1 and shortfalls[0].startswith("text_encoder: throughput_fps")
    report["components"]["text_encoder"]["metrics"]["throughput_fps"] = 600.0
    assert measured_shortfalls(model_dir, "performance", "w8a16", report) == []
    del report["components"]["image_encoder"]
    assert measured_shortfalls(model_dir, "performance", "w8a16", report) == ["image_encoder was not measured"]
