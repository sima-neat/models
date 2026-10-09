"""Package Core's measurements for registry publication; never execute a model."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .benchmark_publication import benchmark_identity_document, load_benchmark_result
from .errors import ValidationError
from .fetch import sha256_file
from .registry import canonical_sha256
from .stage_keys import PERFORMANCE_FINGERPRINT_VERSION

CORE_UNITS = {
    "latency_avg_ms": "ms",
    "throughput_fps": "inferences/s",
    "avg_power_watts": "W",
    "energy_joules": "J",
}
UNAVAILABLE_UNITS = {
    "latency_p50_ms": "ms",
    "latency_p95_ms": "ms",
    "model_load_ms": "ms",
    "time_to_first_inference_ms": "ms",
    "mla_memory_baseline_mib": "MiB",
    "mla_memory_peak_mib": "MiB",
    "mla_memory_peak_delta_mib": "MiB",
    "mla_memory_final_mib": "MiB",
}


def _require_internals_runtime(installed: dict[str, str]) -> None:
    """Require a versioned Internals runtime package in the dpkg inventory."""
    if any(
        name.partition(":")[0] == "neat-runtime" and version.strip()
        for name, version in installed.items()
    ):
        return
    raise ValidationError("could not establish installed Neat Internals version")


def runner_environment() -> dict[str, Any]:
    """Record installation identity, without taking performance measurements."""
    buildinfo = Path("/etc/buildinfo").read_text(encoding="utf-8")
    packages = subprocess.run(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n", "*neat*"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    installed = {}
    for line in packages.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0] == "ii " and fields[2].strip():
            installed[fields[1]] = fields[2]
    _require_internals_runtime(installed)
    return {
        "buildinfo": buildinfo,
        "packages": installed,
        "board_model": "Modalix",
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "fingerprint_version": PERFORMANCE_FINGERPRINT_VERSION,
        "cpu_count": os.cpu_count(),
        "cpu_governor": (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
            if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists()
            else None
        ),
        "runner": os.environ.get("RUNNER_NAME", platform.node()),
    }


def _metric(value: dict[str, Any], unit: str) -> dict[str, Any]:
    measured = value.get("status") == "measured"
    number = value.get("value")
    if value.get("unit") != unit or value.get("status") not in {"measured", "unavailable"}:
        raise ValidationError("invalid Core metric status or unit")
    if measured and (
        isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0
    ):
        raise ValidationError("Core measured metrics must be finite positive numbers")
    if not measured and (number is not None or not value.get("reason")):
        raise ValidationError("unavailable Core metrics require a reason and null value")
    return {
        "status": value["status"],
        "reason": value.get("reason"),
        "unit": unit,
        "aggregation": "raw",
        "value": number,
        "minimum": None,
        "maximum": None,
        "standard_deviation": None,
        "sample_count": 1 if measured else 0,
    }


def import_core_benchmark(
    manifest_path: Path,
    report_path: Path,
    output_path: Path,
    *,
    schema_path: Path,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValidationError("benchmark report must be an object")
    configuration = manifest.get("configuration", {})
    workload = manifest.get("workload", {})
    if (
        type(configuration.get("samples")) is not int
        or configuration["samples"] <= 0
        or workload.get("kind") != "model.synthetic"
        or not workload.get("inputs")
        or type(workload.get("batch_size")) is not int
        or workload["batch_size"] <= 0
    ):
        raise ValidationError("benchmark manifest is missing configuration or inspected workload")
    actual_core = report.get("core_version", "")
    if not isinstance(actual_core, str) or not actual_core:
        raise ValidationError("benchmark report must identify the Core version")
    expected = manifest["compatibility"]
    digest, _ = sha256_file(Path(manifest["model"]["path"]))
    if digest != manifest["model"]["sha256"]:
        raise ValidationError("benchmark input digest does not match the compilation")
    environment = dict(runner_environment() if environment is None else environment)
    buildinfo = environment["buildinfo"]
    found = re.search(r"^\s*DISTRO_VERSION\s*=\s*(\S+)", buildinfo, re.MULTILINE)
    actual_platform = found.group(1) if found else None
    if actual_platform != expected["platform_version"]:
        raise ValidationError("benchmark environment does not match the required platform")
    environment["core_version"] = actual_core
    environment["fingerprint"] = canonical_sha256(environment)
    metrics = {}
    fields = {"latency_avg_ms": "latency_ms", "throughput_fps": "fps"}
    for name, unit in CORE_UNITS.items():
        field = fields.get(name, name)
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError(f"benchmark report requires a finite number for {field}")
        if value < 0:
            raise ValidationError(f"benchmark report requires a nonnegative number for {field}")
        measured = value > 0
        metrics[name] = _metric({
            "status": "measured" if measured else "unavailable",
            "value": value if measured else None,
            "unit": unit,
            "reason": None if measured else "not_measured_by_core_benchmark",
        }, unit)
    for name in ("latency_avg_ms", "throughput_fps"):
        if metrics[name]["status"] != "measured":
            raise ValidationError(f"benchmark did not measure {name}")
    energy = metrics["energy_joules"]
    metrics["energy_per_inference_joules"] = _metric(
        {
            "status": energy["status"],
            "value": (
                energy["value"] / (configuration["samples"] * workload["batch_size"])
                if energy["status"] == "measured" else None
            ),
            "unit": "J/inference",
            "reason": energy["reason"],
        },
        "J/inference",
    )
    for name, unit in UNAVAILABLE_UNITS.items():
        metrics[name] = _metric(
            {"status": "unavailable", "value": None, "unit": unit, "reason": "not_reported_by_core_benchmark"}, unit
        )
    result = {
        "schema_version": 5,
        "status": "passed",
        "benchmark_type": "model.synthetic",
        "tool": {"name": "pyneat.Model.benchmark", "version": actual_core},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity": {"algorithm": "sha256", "canonicalization_version": 5, "fingerprint": None},
        "model": {**manifest["model"], "contract": None},
        "compatibility": {
            "expected_platform_version": expected["platform_version"],
            "actual_platform_version": actual_platform,
            "expected_neat_version": None,
            "actual_neat_version": actual_core,
        },
        "configuration": {
            **configuration,
            "workload": workload,
            "method": "core_model_benchmark",
            "script_sha256": sha256_file(Path(manifest["script"]))[0],
        },
        "environment": environment,
        "power_measurement": {"source": "Core benchmark report"},
        "metrics": metrics,
        "trials": [],
        "core_report": report,
        "orchestration": {
            "provider": "github-actions-devkit-runner" if os.environ.get("GITHUB_ACTIONS") else "manual",
            "session_id": os.environ.get("GITHUB_RUN_ID"),
            "task_id": os.environ.get("GITHUB_JOB"),
            "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "target_correlation_id": os.environ.get("RUNNER_NAME"),
        },
        "provenance": manifest["provenance"],
    }
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_benchmark_result(output_path, schema_path)
    return {"result": str(output_path), "fingerprint": result["identity"]["fingerprint"]}
