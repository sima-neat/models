from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from model_registry.model_tests import (
    TEST_GROUPS,
    load_model_tests,
    measured_shortfalls,
    skip_reason,
    stage_test_file,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_MODEL_TESTS = ROOT / "scripts" / "run_model_tests.sh"
PERFORMANCE_EXPECTED = """\
performance:
  run_tests: true
  expected:
    modalix_int8:
      model:
        throughput_fps: 100.0
"""
PERFORMANCE_TESTS = '''
def test_needs_no_output() -> None:
    assert True


def test_throughput_meets_its_minimum(performance, build_profile) -> None:
    assert build_profile == "modalix_int8"
    assert performance["benchmark_report"]["fps"] >= 100
'''


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A repository holding one model with performance tests, configured as this repository runs them."""
    for name in ("pyproject.toml", "conftest.py"):
        shutil.copy(ROOT / name, tmp_path / name)
    model = tmp_path / "models" / "demo"
    (model / "tests" / "measure").mkdir(parents=True)
    (model / "model.yaml").write_text("id: demo\n", encoding="utf-8")
    (model / "tests" / "tests.yaml").write_text(PERFORMANCE_EXPECTED, encoding="utf-8")
    (model / "tests" / "measure" / "test_performance.py").write_text(PERFORMANCE_TESTS, encoding="utf-8")
    return model


def performance_result(tmp_path: Path, fps: float) -> Path:
    path = tmp_path / "performance-result.json"
    result = {"model": {"build_profile_id": "modalix_int8"}, "benchmark_report": {"fps": fps}}
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def pytest_in(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(repository / "pyproject.toml"), "--rootdir", str(repository),
         "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, check=False,
    )  # fmt: skip


def run_model_tests(stage: str, model: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUN_MODEL_TESTS), stage, str(model), str(output)],
        capture_output=True, text=True, check=False, env={**os.environ, "REGISTRY_PYTHON": sys.executable},
    )  # fmt: skip


def test_plain_run_deselects_the_tests_that_check_stage_output(model_dir: Path) -> None:
    result = pytest_in(model_dir.parents[1], str(model_dir / "tests"))

    assert result.returncode == 0, result.stdout
    assert "1 passed, 1 deselected" in result.stdout


def test_given_stage_output_runs_only_the_tests_that_check_it(model_dir: Path, tmp_path: Path) -> None:
    output = performance_result(tmp_path, 150)
    result = pytest_in(model_dir.parents[1], str(model_dir / "tests"), f"--performance={output}", "-v")

    assert result.returncode == 0, result.stdout
    assert "test_throughput_meets_its_minimum PASSED" in result.stdout
    assert "1 passed, 1 deselected" in result.stdout


@pytest.mark.parametrize(("fps", "outcome", "exit_code"), [(150, "passed", 0), (50, "failed", 1)])
def test_stage_runner_prints_the_outcome_of_the_stage_tests(
    model_dir: Path, tmp_path: Path, fps: float, outcome: str, exit_code: int
) -> None:
    result = run_model_tests("performance", model_dir, performance_result(tmp_path, fps))

    assert result.stdout == f"{outcome}\n", result.stderr
    assert result.returncode == exit_code


def test_stage_runner_skips_a_stage_tests_yaml_skips(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "tests.yaml").write_text(
        "performance:\n  run_tests: false\n  skip_reason: no published result\n", encoding="utf-8"
    )

    result = run_model_tests("performance", model_dir, performance_result(tmp_path, 50))

    assert result.stdout == "skipped\n"
    assert result.returncode == 0
    assert "no published result" in result.stderr
    # A plain run skips the file's tests with the same reason.
    plain = pytest_in(model_dir.parents[1], str(model_dir / "tests"), f"--performance={tmp_path / 'none.json'}")
    assert "1 skipped, 1 deselected" in plain.stdout
    assert "skipped in tests.yaml: no published result" in plain.stdout


def test_stage_runner_passes_a_stage_whose_tests_take_no_output(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "measure" / "test_performance.py").write_text(
        "def test_needs_no_output() -> None:\n    assert True\n", encoding="utf-8"
    )

    result = run_model_tests("performance", model_dir, performance_result(tmp_path, 50))

    assert result.stdout == "passed\n"
    assert "no performance test of demo takes the performance output" in result.stderr


@pytest.mark.parametrize(
    ("stage", "output", "message"),
    [
        ("latency", "result.json", "unknown stage: latency"),
        ("performance", "missing.json", "performance output does not exist"),
        ("accuracy", "result.json", "tests/measure/test_accuracy.py is missing"),
    ],
)
def test_stage_runner_reports_no_outcome_when_it_cannot_run_the_tests(
    model_dir: Path, tmp_path: Path, stage: str, output: str, message: str
) -> None:
    performance_result(tmp_path, 150).rename(tmp_path / "result.json")

    result = run_model_tests(stage, model_dir, tmp_path / output)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_stage_output_tests_stop_on_an_invalid_tests_file(model_dir: Path, tmp_path: Path) -> None:
    (model_dir / "tests" / "tests.yaml").write_text("performance: [\n", encoding="utf-8")

    result = pytest_in(
        model_dir.parents[1], str(model_dir / "tests"), f"--performance={performance_result(tmp_path, 150)}"
    )

    assert result.returncode == pytest.ExitCode.USAGE_ERROR
    assert "could not load the model's tests" in result.stderr


@pytest.mark.parametrize("model", sorted(path.name for path in (ROOT / "models").iterdir() if path.is_dir()))
def test_each_running_measure_file_checks_its_stage_result(model: str) -> None:
    model_dir = ROOT / "models" / model
    tests = load_model_tests(model_dir)
    for stage in TEST_GROUPS["measure"]:
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
    (model_dir / "tests" / "measure" / "test_performance.py").write_text(
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


def accuracy_result(**values: float) -> dict:
    metrics = {name: {"value": value, "unit": "ratio", "direction": "higher"} for name, value in values.items()}
    return {"model": {"build_profile_id": "modalix_int8"}, "metrics": metrics}


@pytest.mark.parametrize(
    ("result", "shortfalls"),
    [
        (accuracy_result(top1=0.80, top5=0.95), []),
        (accuracy_result(top1=0.7601, top5=0.95), []),
        (accuracy_result(top1=0.7599, top5=0.95), ["top1 0.7599 is below 0.8 by more than 5%"]),
        (accuracy_result(top1=0.80), ["top5 was not measured"]),
        (
            {**accuracy_result(top1=0.80, top5=0.95), "model": {"build_profile_id": "modalix_bf16"}},
            ["tests.yaml expects no accuracy values of modalix_bf16"],
        ),
    ],
    ids=["equal", "within the allowed drop", "below the allowed drop", "missing metric", "unexpected build"],
)
def test_measured_shortfalls_compare_each_expected_value(tmp_path: Path, result: dict, shortfalls: list[str]) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "tests.yaml").write_text(
        "accuracy:\n  run_tests: true\n  expected:\n    modalix_int8:\n"
        "      top1: 0.80\n      top5: 0.95\n",
        encoding="utf-8",
    )

    assert measured_shortfalls(tmp_path, "accuracy", result) == shortfalls


def test_measured_shortfalls_hold_a_lower_is_better_metric_under_its_allowed_rise(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "tests.yaml").write_text(
        "accuracy:\n  run_tests: true\n  expected:\n    modalix_int8:\n      error: 0.20\n",
        encoding="utf-8",
    )
    result = accuracy_result(error=0.2101)
    result["metrics"]["error"]["direction"] = "lower"

    assert measured_shortfalls(tmp_path, "accuracy", result) == ["error 0.2101 is above 0.2 by more than 5%"]


def test_measured_shortfalls_take_the_performance_component(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "tests.yaml").write_text(PERFORMANCE_EXPECTED, encoding="utf-8")
    result = json.loads((ROOT / "artifact-browser/fixtures/performance-result.json").read_text(encoding="utf-8"))
    result["model"].update(build_profile_id="modalix_int8", component_id="model")

    def measured(fps: float) -> dict:
        return {**result, "benchmark_report": {**result["benchmark_report"], "fps": fps}}

    assert measured_shortfalls(tmp_path, "performance", measured(90)) == []
    assert measured_shortfalls(tmp_path, "performance", measured(89)) == [
        "throughput_fps 89 is below 100 by more than 10%"
    ]
