"""Package Core's saved benchmark report for registry publication; never execute a model."""

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

# run_model_performance.sh sets the MLA and CPU clocks before measuring; the result records the clocks it ran at.
MLA_FREQUENCY = Path("/sys/class/devfreq/5000000.mla/cur_freq")
CPU_FREQUENCY = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")


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
        "mla_frequency_hz": (
            int(MLA_FREQUENCY.read_text(encoding="utf-8")) if MLA_FREQUENCY.exists() else None
        ),
        "cpu_frequency_khz": (
            int(CPU_FREQUENCY.read_text(encoding="utf-8")) if CPU_FREQUENCY.exists() else None
        ),
        "runner": os.environ.get("RUNNER_NAME", platform.node()),
    }


def _reject_nonfinite(value: str) -> None:
    raise ValidationError(f"benchmark report contains non-finite JSON number: {value}")


def import_core_benchmark(
    manifest_path: Path,
    report_path: Path,
    context_path: Path,
    output_path: Path,
    *,
    schema_path: Path,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the report BenchmarkReport.save() wrote, unchanged, with the script's context and the
    registry identity, provenance, and runner environment around it.

    For a package, the report is the pipeline's, written in the same fields by scripts/performance.py, and the
    context holds each component's layer stats and execution plan.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(context, dict):
        raise ValidationError("benchmark report and context must be objects")
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
    actual_core = context.get("core_version", "")
    if not isinstance(actual_core, str) or not actual_core:
        raise ValidationError("benchmark context must identify the Core version")
    include_layer_stats = context.get("include_layer_stats", False)
    if type(include_layer_stats) is not bool:
        raise ValidationError("benchmark context include_layer_stats must be a boolean")
    if include_layer_stats and not report.get("layer_stats"):
        raise ValidationError("benchmark requested layer stats but the report contains none")
    expected = manifest["compatibility"]
    package = "components" in manifest
    for build in manifest["components"] if package else [manifest["model"]]:
        digest, _ = sha256_file(Path(build["path"]))
        if digest != build["sha256"]:
            raise ValidationError("benchmark input digest does not match the compilation")
    if package and set(context.get("components", {})) != {build["component_id"] for build in manifest["components"]}:
        raise ValidationError("benchmark context must give each package component's layer stats and execution plan")
    environment = dict(runner_environment() if environment is None else environment)
    buildinfo = environment["buildinfo"]
    found = re.search(r"^\s*DISTRO_VERSION\s*=\s*(\S+)", buildinfo, re.MULTILINE)
    actual_platform = found.group(1) if found else None
    if actual_platform != expected["platform_version"]:
        raise ValidationError("benchmark environment does not match the required platform")
    environment["core_version"] = actual_core
    environment["fingerprint"] = canonical_sha256(environment)
    for field in ("latency_ms", "fps"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValidationError(f"benchmark did not measure {field}")
    result = {
        "schema_version": 6,
        "status": "passed",
        "benchmark_type": "model.orchestration" if package else "model.synthetic",
        "tool": {"name": "pyneat.ModelRunner" if package else "pyneat.Model.benchmark", "version": actual_core},
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
            "include_layer_stats": include_layer_stats,
            "workload": workload,
            "method": "model_orchestration" if package else "core_model_benchmark",
            "script_sha256": sha256_file(Path(manifest["script"]))[0],
        },
        "environment": environment,
        "benchmark_report": report,
        "execution_plan": context.get("execution_plan"),
        "orchestration": {
            "provider": "github-actions-devkit-runner" if os.environ.get("GITHUB_ACTIONS") else "manual",
            "session_id": os.environ.get("GITHUB_RUN_ID"),
            "task_id": os.environ.get("GITHUB_JOB"),
            "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "target_correlation_id": os.environ.get("RUNNER_NAME"),
        },
        "provenance": manifest["provenance"],
    }
    if package:
        result["components"] = [
            {
                **build,
                "layer_stats": context["components"][build["component_id"]]["layer_stats"],
                "execution_plan": context["components"][build["component_id"]]["execution_plan"],
            }
            for build in manifest["components"]
        ]
    result["identity"]["fingerprint"] = canonical_sha256(benchmark_identity_document(result))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_benchmark_result(output_path, schema_path)
    return {"result": str(output_path), "fingerprint": result["identity"]["fingerprint"]}
