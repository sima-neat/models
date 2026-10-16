import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from model_registry import core_benchmark
from model_registry.benchmark_metrics import registry_metrics
from model_registry.benchmark_publication import load_benchmark_result
from model_registry.core_benchmark import _require_internals_runtime, import_core_benchmark
from model_registry.errors import ValidationError
from model_registry.stage_keys import PERFORMANCE_FINGERPRINT_VERSION

PLAN = {"schema": "sima.neat.execution_plan", "schema_version": 1, "model": {"name": "resnet_50", "batch": 4},
        "nodes": [{"id": "p0", "kind": "Input", "runs_on": "A65"}], "edges": []}


@pytest.fixture
def core_inputs(tmp_path):
    archive = tmp_path / "model.tar.gz"
    archive.write_bytes(b"test archive")
    script = tmp_path / "benchmark.py"
    script.write_text("report = model.benchmark()")
    manifest = {
        "model": {
            "id": "resnet_50",
            "build_profile_id": "modalix_int8",
            "component_id": "model",
            "path": str(archive),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "uri": None,
        },
        "configuration": {"samples": 100, "include_plugin_latency": False},
        "workload": {"kind": "model.synthetic", "batch_size": 4, "inputs": [{"shape": [4, 3, 224, 224]}]},
        "script": str(script),
        "compatibility": {"platform_version": "3.0.0"},
        "provenance": {"compile_run_id": "00000000-0000-4000-8000-000000000001", "models_commit": "a" * 40},
    }
    manifest_path, report_path = tmp_path / "manifest.json", tmp_path / "report.json"
    context_path = tmp_path / "context.json"
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(saved_report()))
    context = {"core_version": "0.5.0+test", "num_samples": 100, "min_duration_ms": 60000, "execution_plan": PLAN}
    context_path.write_text(json.dumps(context))
    return manifest_path, report_path, context_path


def summary(key, unit, mean, maximum, samples=15):
    return {"key": key, "label": key, "unit": unit, "samples": samples,
            "mean": mean, "minimum": mean - 1.0, "maximum": maximum}


def no_samples(key, unit):
    return summary(key, unit, 0.0, 0.0, samples=0)


def sentinel_telemetry():
    """Telemetry in the shape BenchmarkReport.save() writes from Sentinel on a DevKit."""
    return {
        "status": "ok", "trace_id": "neat-benchmark-test", "sentinel_version": "main:test",
        "sample_interval_ms": 2000, "samples": 15,
        "power_som_watts": summary("power_current_watts", "W", 11.3, 11.7),
        "power_mla_watts": summary("power_rail_mla_watts", "W", 1.9, 2.0),
        "power_rails_watts": [summary("soc", "W", 3.5, 3.8), summary("mla", "W", 1.9, 2.0)],
        "die_temperatures_c": [summary("rtsn_0", "C", 41.9, 42.7), summary("rtsn_1", "C", 42.2, 43.3)],
        "die_temperature_max_c": summary("die_temperature_max_c", "C", 42.3, 43.3),
        "board_temperatures_c": [summary("temp1", "C", 40.9, 42.0), summary("temp2", "C", 43.9, 44.6)],
        "cpu_usage_pct": summary("cpu_usage_pct", "%", 7.8, 8.7),
        "cpu_cores_usage_pct": [summary("cpu_core_0_usage_pct", "%", 6.0, 7.0),
                                summary("cpu_core_1_usage_pct", "%", 61.0, 98.5)],
        "mla_memory_mib": summary("mla_memory_mib", "MiB", 29.6, 29.7),
        "cma_memory_mib": summary("cma_memory_mib", "MiB", 863.8, 867.2),
        "linux_memory_used_mib": summary("linux_memory_used_mib", "MiB", 600.0, 607.5),
    }


def unavailable_telemetry():
    """What save() writes when Sentinel did not answer: every reading present with no samples."""
    telemetry = sentinel_telemetry()
    for name, value in telemetry.items():
        if isinstance(value, dict):
            telemetry[name] = no_samples(value["key"], value["unit"])
        elif isinstance(value, list):
            telemetry[name] = []
    return {**telemetry, "status": "sentinel_unreachable", "samples": 0}


def saved_report(**fields):
    """A report in the shape BenchmarkReport.save() writes; Core writes 0 for what it did not measure."""
    return {
        "latency_ms": 2.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0,
        "fps": 2.0, "avg_power_watts": 0.0, "energy_joules": 2.0, "peak_power_watts": 0.0,
        "throughput_duration_ms": 60000.0, "throughput_samples": 120,
        "telemetry": unavailable_telemetry(), "layer_stats": [], **fields,
    }


def package(core_inputs, tmp_path):
    output = tmp_path / "performance-result.json"
    import_core_benchmark(
        *core_inputs,
        output,
        schema_path=Path(__file__).resolve().parents[1] / "schemas/performance-result.schema.json",
        environment={
            "buildinfo": "DISTRO_VERSION = 3.0.0\nBUILD = B1800\n",
            "packages": {"neat-runtime": "0.5.0+test"},
            "runner": "test",
            "fingerprint_version": 2,
            "kernel_release": "test",
            "cpu_count": 8,
            "cpu_governor": None,
            "board_model": "Modalix",
            "machine": "aarch64",
        },
    )
    return load_benchmark_result(
        output, Path(__file__).resolve().parents[1] / "schemas/performance-result.schema.json"
    )


@pytest.mark.parametrize(
    "packages",
    [
        {"neat-runtime": "0.5.0+release-3.0.0-prep.464889585182"},
        {"neat-runtime:arm64": "0.5.0+release-3.0.0-prep.464889585182"},
    ],
)
def test_accepts_versioned_internals_runtime_package(packages):
    _require_internals_runtime(packages)


@pytest.mark.parametrize(
    "packages",
    [
        {},
        {"neat-runtime": ""},
        {"neat-internals": "0.4.0+legacy"},
        {"neat-internals-dev": "0.5.0+test"},
        {"sima-neat": "0.4.0+test"},
    ],
)
def test_rejects_missing_internals_runtime_package(packages):
    with pytest.raises(ValidationError, match="installed Neat Internals version"):
        _require_internals_runtime(packages)


def test_runner_environment_records_only_installed_packages(monkeypatch):
    package_output = "\n".join([
        "ii \tneat-runtime:arm64\t0.5.0+release-3.0.0-prep.464889585182",
        "ii \tsima-neat\t0.4.0+release-3.0.0-prep.02a5ca96b124",
        "rc \tneat-gst-plugins\t0.4.0+removed",
        "un \tneat-internals\t",
    ])

    def run(command, **kwargs):
        assert command == [
            "dpkg-query", "-W",
            "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n",
            "*neat*",
        ]
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return SimpleNamespace(stdout=package_output)

    monkeypatch.setattr(Path, "read_text", lambda self, encoding: "DISTRO_VERSION = 3.0.0\nBUILD = B1800\n")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(core_benchmark.subprocess, "run", run)

    environment = core_benchmark.runner_environment()

    assert environment["packages"] == {
        "neat-runtime:arm64": "0.5.0+release-3.0.0-prep.464889585182",
        "sima-neat": "0.4.0+release-3.0.0-prep.02a5ca96b124",
    }
    assert environment["fingerprint_version"] == PERFORMANCE_FINGERPRINT_VERSION
    assert environment["mla_frequency_hz"] is None
    assert environment["cpu_frequency_khz"] is None


def test_runner_environment_records_the_mla_and_cpu_clocks(monkeypatch):
    files = {
        "/etc/buildinfo": "DISTRO_VERSION = 3.0.0\nBUILD = B1800\n",
        str(core_benchmark.MLA_FREQUENCY): "1000000000\n",
        str(core_benchmark.CPU_FREQUENCY): "1400000\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: files[str(self)])
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in files)
    packages = SimpleNamespace(stdout="ii \tneat-runtime\t0.5.0+test\n")
    monkeypatch.setattr(core_benchmark.subprocess, "run", lambda command, **kwargs: packages)

    environment = core_benchmark.runner_environment()

    assert environment["mla_frequency_hz"] == 1_000_000_000
    assert environment["cpu_frequency_khz"] == 1_400_000


def test_publishes_the_saved_report_unchanged_with_its_context(core_inputs, tmp_path):
    result = package(core_inputs, tmp_path)
    assert result["schema_version"] == 6
    assert result["benchmark_report"] == json.loads(core_inputs[1].read_text())
    assert result["execution_plan"] == PLAN
    assert result["configuration"]["method"] == "core_model_benchmark"
    assert result["environment"]["core_version"] == "0.5.0+test"
    assert "B1800" in result["environment"]["buildinfo"]
    assert result["compatibility"]["expected_neat_version"] is None
    for removed in ("metrics", "trials", "power_measurement", "core_report"):
        assert removed not in result


def test_registry_metrics_read_the_saved_report(core_inputs, tmp_path):
    metrics = registry_metrics(package(core_inputs, tmp_path))
    assert metrics["energy_per_inference_joules"]["value"] == pytest.approx(2.0 / 400)
    assert metrics["throughput_fps"]["value"] == 2.0
    assert metrics["throughput_fps"]["unit"] == "inferences/s"
    assert metrics["throughput_fps"]["direction"] == "higher"
    # Core writes 0 for what it did not measure; the registry leaves those out.
    for name in ("avg_power_watts", "latency_p99_ms", "avg_cpu_usage_pct", "peak_power_watts"):
        assert name not in metrics, name


def test_rejects_a_benchmark_without_the_execution_plan_the_browser_draws(core_inputs, tmp_path):
    context = json.loads(core_inputs[2].read_text())
    del context["execution_plan"]
    core_inputs[2].write_text(json.dumps(context))
    with pytest.raises(ValidationError, match="execution_plan"):
        package(core_inputs, tmp_path)


def test_every_performance_script_records_the_execution_plan_of_the_model_it_benchmarks():
    scripts = sorted((Path(__file__).resolve().parents[1] / "models").glob("*/scripts/performance.py"))
    assert scripts
    for script in scripts:
        assert '"execution_plan": json.loads(model.execution_plan().to_json()),' in script.read_text(), script


@pytest.mark.parametrize(
    "failure", [
        "nonfinite", "negative_power", "missing_metric", "unmeasured_latency",
        "wrong_artifact", "missing_core", "wrong_platform", "missing_workload",
    ]
)
def test_rejects_invalid_report_or_provenance(core_inputs, tmp_path, failure):
    manifest_path, report_path, context_path = core_inputs
    report = json.loads(report_path.read_text())
    if failure == "nonfinite":
        report["latency_ms"] = float("nan")
    elif failure == "negative_power":
        report["avg_power_watts"] = -1.0
    elif failure == "missing_metric":
        del report["latency_p50_ms"]
    elif failure == "unmeasured_latency":
        report["latency_ms"] = 0.0
    elif failure == "missing_core":
        context_path.write_text(json.dumps({"num_samples": 100}))
    else:
        manifest = json.loads(manifest_path.read_text())
        if failure == "wrong_artifact":
            Path(manifest["model"]["path"]).write_bytes(b"different archive")
        elif failure == "wrong_platform":
            manifest["compatibility"]["platform_version"] = "2.1.3"
        else:
            del manifest["workload"]
        manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValidationError):
        package(core_inputs, tmp_path)


def with_report(core_inputs, **fields):
    core_inputs[1].write_text(json.dumps(saved_report(**fields)))


def test_sentinel_telemetry_becomes_board_metrics(core_inputs, tmp_path):
    with_report(core_inputs, avg_power_watts=11.3, peak_power_watts=11.7, telemetry=sentinel_telemetry())
    metrics = registry_metrics(package(core_inputs, tmp_path))

    assert metrics["peak_power_watts"]["value"] == 11.7
    assert metrics["peak_power_watts"]["aggregation"] == "max"
    assert metrics["avg_mla_power_watts"]["value"] == 1.9
    assert metrics["avg_mla_power_watts"]["aggregation"] == "mean"
    assert metrics["peak_die_temperature_c"]["unit"] == "°C"
    assert metrics["avg_cpu_usage_pct"]["value"] == 7.8
    assert metrics["mla_memory_avg_mib"]["value"] == 29.6
    assert metrics["mla_memory_peak_mib"]["value"] == 29.7
    assert metrics["avg_linux_memory_mib"]["value"] == 600.0
    assert metrics["peak_linux_memory_mib"]["sample_count"] == 15
    # Only the power rails are kept; the sensors and cores stay in the saved report.
    assert [rail["key"] for rail in metrics["peak_power_watts"]["details"]["breakdown"]] == ["soc", "mla"]
    assert [name for name, metric in metrics.items() if "details" in metric] == ["peak_power_watts"]
    # The busiest core stands out even though the all-core average stays low.
    assert metrics["avg_cpu_core_usage_pct"]["value"] == 61.0
    assert metrics["peak_cpu_core_usage_pct"]["value"] == 98.5
    assert metrics["avg_cma_memory_mib"]["value"] == 863.8
    assert metrics["peak_cma_memory_mib"]["value"] == 867.2
    # The hotter SoM board sensor stands for the board.
    assert metrics["avg_board_temperature_c"]["value"] == 43.9
    assert metrics["peak_board_temperature_c"]["value"] == 44.6


def test_latency_percentiles_come_from_the_saved_report(core_inputs, tmp_path):
    with_report(core_inputs, latency_p50_ms=1.9, latency_p95_ms=2.6, latency_p99_ms=3.1)
    metrics = registry_metrics(package(core_inputs, tmp_path))

    assert {name: metrics[name]["value"] for name in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")} == {
        "latency_p50_ms": 1.9, "latency_p95_ms": 2.6, "latency_p99_ms": 3.1}


def test_a_reading_sentinel_did_not_sample_is_not_a_metric(core_inputs, tmp_path):
    telemetry = {**sentinel_telemetry(), "cpu_usage_pct": no_samples("cpu_usage_pct", "%")}
    with_report(core_inputs, telemetry=telemetry)
    metrics = registry_metrics(package(core_inputs, tmp_path))
    assert "avg_cpu_usage_pct" not in metrics
    assert "avg_mla_power_watts" in metrics


def test_rejects_telemetry_in_a_unit_the_metric_does_not_carry(core_inputs, tmp_path):
    telemetry = sentinel_telemetry()
    telemetry["die_temperature_max_c"] = summary("die_temperature_max_c", "K", 315.0, 316.0)
    with_report(core_inputs, telemetry=telemetry)
    with pytest.raises(ValidationError, match="must be in 'C'"):
        registry_metrics(package(core_inputs, tmp_path))


def test_every_performance_script_saves_the_core_report():
    scripts = sorted((Path(__file__).resolve().parents[1] / "models").glob("*/scripts/performance.py"))
    assert scripts
    for script in scripts:
        source = script.read_text()
        assert 'model.benchmark(benchmark_options).save(output / "benchmark-report.json")' in source, script
        assert "report_json" not in source, script
