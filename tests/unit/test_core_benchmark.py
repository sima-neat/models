import hashlib
import json
from pathlib import Path

import pytest
from model_registry.benchmark_publication import load_benchmark_result
from model_registry.core_benchmark import CORE_UNITS, import_core_benchmark
from model_registry.errors import ValidationError


@pytest.fixture
def core_inputs(tmp_path):
    archive = tmp_path / "model.tar.gz"
    archive.write_bytes(b"test archive")
    script = tmp_path / "benchmark.py"
    script.write_text("model.benchmark(output_path=output)")
    manifest = {
        "model": {
            "id": "resnet_50",
            "build_profile_id": "modalix_int8",
            "component_id": "model",
            "path": str(archive),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "uri": None,
        },
        "script": str(script),
        "runtime": {"devkit": {"build": "B1748"}},
        "compatibility": {"platform_version": "3.0.0", "neat_version": "0.4.0"},
        "provenance": {"compile_run_id": "00000000-0000-4000-8000-000000000001", "models_commit": "a" * 40},
    }
    report = {
        "schema_version": 1,
        "tool": {"name": "pyneat.Model.benchmark", "version": "0.4.0+test"},
        "configuration": {"samples": 100, "include_plugin_latency": False},
        "workload": {"kind": "model.synthetic", "batch_size": 1, "inputs": [{"shape": [1, 3, 224, 224]}]},
        "metrics": {name: {"status": "measured", "value": 2.0, "unit": unit} for name, unit in CORE_UNITS.items()},
    }
    report["metrics"]["avg_power_watts"] = {
        "status": "unavailable",
        "value": None,
        "unit": "W",
        "reason": "power_not_available",
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
            "buildinfo": "DISTRO_VERSION = 3.0.0\nBUILD = B1748\n",
            "packages": {"neat-internals": "0.4.0+test"},
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


def test_core_report_keeps_values_units_and_unavailable_metrics(core_inputs, tmp_path):
    result = package(core_inputs, tmp_path)
    assert result["metrics"]["throughput_fps"]["value"] == 2.0
    assert result["metrics"]["throughput_fps"]["unit"] == "inferences/s"
    assert result["metrics"]["avg_power_watts"]["value"] is None
    assert result["metrics"]["avg_power_watts"]["reason"] == "power_not_available"
    assert result["metrics"]["mla_memory_peak_delta_mib"]["status"] == "unavailable"
    assert result["trials"] == []
    assert result["core_report"] == json.loads(core_inputs[1].read_text())
    assert result["configuration"]["method"] == "core_model_benchmark"
    assert result["environment"]["core_version"] == "0.4.0+test"


@pytest.mark.parametrize("failure", ["nonfinite", "wrong_unit", "wrong_artifact", "wrong_core", "wrong_build"])
def test_rejects_invalid_report_or_provenance(core_inputs, tmp_path, failure):
    manifest_path, report_path = core_inputs
    report = json.loads(report_path.read_text())
    if failure == "nonfinite":
        report["metrics"]["latency_avg_ms"]["value"] = float("nan")
    elif failure == "wrong_unit":
        report["metrics"]["latency_avg_ms"]["unit"] = "seconds"
    elif failure == "wrong_core":
        report["tool"]["version"] = "0.3.0"
    else:
        manifest = json.loads(manifest_path.read_text())
        if failure == "wrong_artifact":
            Path(manifest["model"]["path"]).write_bytes(b"different archive")
        else:
            manifest["runtime"]["devkit"]["build"] = "B1493"
        manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValidationError):
        package(core_inputs, tmp_path)
