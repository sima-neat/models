import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from model_registry import core_benchmark
from model_registry.benchmark_publication import load_benchmark_result
from model_registry.core_benchmark import _require_internals_runtime, import_core_benchmark
from model_registry.errors import ValidationError


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
    report = {
        "core_version": "0.5.0+test",
        "latency_ms": 2.0,
        "fps": 2.0,
        "avg_power_watts": 0.0,
        "energy_joules": 2.0,
    }
    manifest_path, report_path = tmp_path / "manifest.json", tmp_path / "core.json"
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(report))
    return manifest_path, report_path


def package(core_inputs, tmp_path):
    output = tmp_path / "performance-result-v5.json"
    import_core_benchmark(
        *core_inputs,
        output,
        schema_path=Path(__file__).resolve().parents[2] / "schemas/performance-result-v5.schema.json",
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
        output, Path(__file__).resolve().parents[2] / "schemas/performance-result-v5.schema.json"
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


def test_core_report_keeps_values_units_and_unavailable_metrics(core_inputs, tmp_path):
    result = package(core_inputs, tmp_path)
    assert result["metrics"]["energy_per_inference_joules"]["value"] == pytest.approx(2.0 / 400)
    assert result["metrics"]["throughput_fps"]["value"] == 2.0
    assert result["metrics"]["throughput_fps"]["unit"] == "inferences/s"
    assert result["metrics"]["avg_power_watts"]["value"] is None
    assert result["metrics"]["avg_power_watts"]["reason"] == "not_measured_by_core_benchmark"
    assert result["metrics"]["mla_memory_peak_delta_mib"]["status"] == "unavailable"
    assert result["trials"] == []
    assert result["core_report"] == json.loads(core_inputs[1].read_text())
    assert result["configuration"]["method"] == "core_model_benchmark"
    assert result["environment"]["core_version"] == "0.5.0+test"
    assert "B1800" in result["environment"]["buildinfo"]
    assert result["compatibility"]["expected_neat_version"] is None


@pytest.mark.parametrize(
    "failure", [
        "nonfinite", "negative_power", "negative_energy", "missing_metric",
        "wrong_artifact", "missing_core", "wrong_platform", "missing_context",
    ]
)
def test_rejects_invalid_report_or_provenance(core_inputs, tmp_path, failure):
    manifest_path, report_path = core_inputs
    report = json.loads(report_path.read_text())
    if failure == "nonfinite":
        report["latency_ms"] = float("nan")
    elif failure == "negative_power":
        report["avg_power_watts"] = -1.0
    elif failure == "negative_energy":
        report["energy_joules"] = -1.0
    elif failure == "missing_metric":
        del report["latency_ms"]
    elif failure == "missing_core":
        report["core_version"] = ""
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
