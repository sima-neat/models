#!/usr/bin/env python3
"""Benchmark one compiled MPK with the installed public pyneat runtime."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sys
import tarfile
import tempfile
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean, pstdev
from types import ModuleType
from typing import Any

EXIT_CONFIGURATION = 2
EXIT_COMPATIBILITY = 3
EXIT_BENCHMARK = 4
SUPPORTED_PLATFORM_VERSION = "2.1.3"
SUPPORTED_NEAT_VERSION = "0.4.0"
MAX_MPK_MANIFEST_BYTES = 32 * 1024 * 1024
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OPTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MODEL_ID = re.compile(r"^[a-z][a-z0-9_]*$")
VARIANT_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
REFERENCE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")
PACKAGE_DISTRIBUTION = "sima-neat-benchmarker"
DEFAULT_TRIALS = 3
POWER_DEVICE_PATHS = (Path("/dev/i2c-3"), Path("/dev/i2c-4"))
RESULT_SCHEMA_VERSION = 2


def tool_version() -> str:
    try:
        return version(PACKAGE_DISTRIBUTION)
    except PackageNotFoundError:
        return "source"


class BenchmarkToolError(RuntimeError):
    """Base class for expected, user-actionable failures."""

    classification = "benchmark"
    exit_code = EXIT_BENCHMARK


class ConfigurationError(BenchmarkToolError):
    classification = "configuration"
    exit_code = EXIT_CONFIGURATION


class CompatibilityError(BenchmarkToolError):
    classification = "compatibility"
    exit_code = EXIT_COMPATIBILITY


class ModelError(BenchmarkToolError):
    classification = "model"


class BenchmarkExecutionError(BenchmarkToolError):
    classification = "benchmark"


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{path} must be a positive integer")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _require_pattern(value: Any, path: str, pattern: re.Pattern[str]) -> str:
    text = _require_string(value, path)
    if not pattern.fullmatch(text):
        raise ConfigurationError(f"{path} has an invalid format")
    return text


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"could not read benchmark manifest {path}: {exc}") from exc
    manifest = _require_mapping(value, "manifest")
    _reject_unknown(
        manifest,
        {"schema_version", "model", "compatibility", "benchmark", "model_options", "provenance"},
        "manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ConfigurationError("schema_version must be 1")

    model = _require_mapping(manifest.get("model"), "model")
    _reject_unknown(model, {"id", "variant_id", "path", "sha256", "uri"}, "model")
    _require_pattern(model.get("id"), "model.id", MODEL_ID)
    _require_pattern(model.get("variant_id"), "model.variant_id", VARIANT_ID)
    for field in ("path", "sha256"):
        _require_string(model.get(field), f"model.{field}")
    if not SHA256.fullmatch(model["sha256"]):
        raise ConfigurationError("model.sha256 must contain 64 lowercase hexadecimal characters")
    if "uri" in model:
        _require_string(model["uri"], "model.uri")

    compatibility = _require_mapping(manifest.get("compatibility"), "compatibility")
    _reject_unknown(compatibility, {"platform_version", "neat_version"}, "compatibility")
    for field in ("platform_version", "neat_version"):
        version = _require_string(compatibility.get(field), f"compatibility.{field}")
        if not SEMVER.fullmatch(version):
            raise ConfigurationError(f"compatibility.{field} must use MAJOR.MINOR.PATCH")
    if compatibility["platform_version"] != SUPPORTED_PLATFORM_VERSION:
        raise CompatibilityError(
            f"this benchmarker supports platform {SUPPORTED_PLATFORM_VERSION}, not {compatibility['platform_version']}"
        )
    if compatibility["neat_version"] != SUPPORTED_NEAT_VERSION:
        raise CompatibilityError(
            f"this benchmarker supports Neat {SUPPORTED_NEAT_VERSION}, not {compatibility['neat_version']}"
        )

    benchmark = _require_mapping(manifest.get("benchmark"), "benchmark")
    _reject_unknown(
        benchmark,
        {
            "mode",
            "samples",
            "trials",
            "warmup_samples",
            "include_plugin_latency",
            "original_width",
            "original_height",
        },
        "benchmark",
    )
    mode = benchmark.get("mode")
    if mode is not None and mode != "bounded_runner":
        raise ConfigurationError("benchmark.mode must be bounded_runner when provided")
    _require_positive_int(benchmark.get("samples"), "benchmark.samples")
    trials = benchmark.get("trials", DEFAULT_TRIALS)
    _require_positive_int(trials, "benchmark.trials")
    if trials > 100:
        raise ConfigurationError("benchmark.trials must be <= 100")
    include_plugin_latency = benchmark.get("include_plugin_latency", False)
    if not isinstance(include_plugin_latency, bool):
        raise ConfigurationError("benchmark.include_plugin_latency must be a boolean")
    width, height = benchmark.get("original_width"), benchmark.get("original_height")
    if (width is None) != (height is None):
        raise ConfigurationError("benchmark.original_width and original_height must be provided together")
    if width is not None:
        _require_positive_int(width, "benchmark.original_width")
        _require_positive_int(height, "benchmark.original_height")
    if mode == "bounded_runner":
        _require_positive_int(benchmark.get("warmup_samples"), "benchmark.warmup_samples")
        if width is not None:
            raise ConfigurationError("bounded_runner does not support original input geometry overrides")

    options = manifest.get("model_options", {})
    _require_mapping(options, "model_options")
    for name, option_value in options.items():
        if not isinstance(name, str) or not OPTION_NAME.fullmatch(name):
            raise ConfigurationError(f"invalid model option name: {name!r}")
        if option_value is not None and not isinstance(option_value, (str, int, float, bool)):
            raise ConfigurationError(f"model_options.{name} must be a scalar")
        if isinstance(option_value, float) and not math.isfinite(option_value):
            raise ConfigurationError(f"model_options.{name} must be finite")
    _require_mapping(manifest.get("provenance", {}), "provenance")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelError(f"could not read MPK {path}: {exc}") from exc
    return digest.hexdigest()


def inspect_mpk(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelError(f"MPK does not exist: {path}")
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            elf_names = sorted(member.name for member in members if member.name.endswith(".elf"))
            manifests: list[tuple[str, dict[str, Any]]] = []
            for member in members:
                if not member.name.endswith("_mpk.json"):
                    continue
                if member.size > MAX_MPK_MANIFEST_BYTES:
                    raise ModelError(f"MPK manifest is unexpectedly large: {member.name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    payload = json.loads(extracted.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ModelError(f"invalid MPK manifest {member.name}: {exc}") from exc
                if isinstance(payload, dict) and isinstance(payload.get("plugins"), list):
                    manifests.append((member.name, payload))
    except (OSError, tarfile.TarError) as exc:
        raise ModelError(f"could not inspect MPK archive {path}: {exc}") from exc

    if len(manifests) != 1:
        raise ModelError(f"expected one model MPK manifest, found {len(manifests)}")
    manifest_name, mpk = manifests[0]
    mla_plugins = [
        plugin
        for plugin in mpk["plugins"]
        if isinstance(plugin, dict) and str(plugin.get("processor", "")).lower() == "mla"
    ]
    if not mla_plugins:
        raise CompatibilityError("MPK does not contain an MLA stage")

    stage_names: list[str] = []
    batch_sizes: set[int] = set()
    for stage in mla_plugins:
        resources = stage.get("resources") if isinstance(stage.get("resources"), dict) else {}
        executable = resources.get("executable")
        if not isinstance(executable, str) or executable not in elf_names:
            raise ModelError(f"MLA stage references an ELF not present in the archive: {executable!r}")
        stage_names.append(str(stage.get("name") or executable))
        config = stage.get("config_params") if isinstance(stage.get("config_params"), dict) else {}
        batch = config.get("actual_batch_size")
        if isinstance(batch, int) and not isinstance(batch, bool) and batch > 0:
            batch_sizes.add(batch)

    if len(batch_sizes) > 1:
        raise CompatibilityError(f"MPK MLA stages declare inconsistent compiled batch sizes: {sorted(batch_sizes)}")
    return {
        "name": mpk.get("name") if isinstance(mpk.get("name"), str) else None,
        "manifest": manifest_name,
        "model_sdk_version": (mpk.get("model_sdk_version") if isinstance(mpk.get("model_sdk_version"), str) else None),
        "elf_count": len(elf_names),
        "mla_stage_count": len(mla_plugins),
        "compiled_batch_size": next(iter(batch_sizes)) if batch_sizes else None,
        "mla_stages": stage_names,
    }


def detect_platform_version(buildinfo: Path = Path("/etc/buildinfo")) -> str | None:
    override = os.environ.get("SIMA_PLATFORM_VERSION", "").strip()
    if override:
        return override
    try:
        lines = buildinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "DISTRO_VERSION":
            return value.strip() or None
    return None


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip().strip("\x00") or None
    except (OSError, UnicodeDecodeError):
        return None


def _build_information(path: Path = Path("/etc/buildinfo")) -> dict[str, str]:
    text = _read_text(path)
    if text is None:
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() and value.strip():
            result[key.strip()] = value.strip()
    return result


def _temperature_snapshot(hwmon_root: Path = Path("/sys/class/hwmon")) -> list[dict[str, Any]]:
    readings: list[dict[str, Any]] = []
    for input_path in sorted(hwmon_root.glob("hwmon*/temp*_input")):
        try:
            celsius = float(input_path.read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not math.isfinite(celsius):
            continue
        prefix = input_path.name.removesuffix("_input")
        label = _read_text(input_path.with_name(f"{prefix}_label")) or prefix
        source = _read_text(input_path.parent / "name") or input_path.parent.name
        readings.append(
            {
                "sensor_id": f"{input_path.parent.name}/{prefix}",
                "sensor": label,
                "source": source,
                "celsius": celsius,
            }
        )
    return readings


def runtime_observation() -> dict[str, Any]:
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "load_average": load_average,
        "temperatures_c": _temperature_snapshot(),
    }


def runtime_environment(actual_platform: str | None, actual_neat: str | None) -> dict[str, Any]:
    stable = {
        "board_model": _read_text(Path("/proc/device-tree/model")),
        "machine": platform.machine() or None,
        "kernel_release": platform.release() or None,
        "platform_version": actual_platform,
        "neat_version": actual_neat,
        "cpu_count": os.cpu_count(),
        "cpu_governor": _read_text(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")),
        "build_information": _build_information(),
    }
    start = runtime_observation()
    return {
        "fingerprint_version": 1,
        "fingerprint": canonical_sha256(stable),
        **stable,
        "observations": {"start": start, "end": None},
        "temperature_samples": [_temperature_sample(start, phase="start")],
    }


def _temperature_sample(
    observation: dict[str, Any],
    *,
    phase: str,
    trial: int | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": observation["timestamp_utc"],
        "phase": phase,
        "trial": trial,
        "temperatures_c": observation["temperatures_c"],
    }


def record_temperature_sample(
    environment: dict[str, Any],
    *,
    phase: str,
    trial: int | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    captured = observation or runtime_observation()
    environment.setdefault("temperature_samples", []).append(
        _temperature_sample(captured, phase=phase, trial=trial)
    )
    return captured


def power_telemetry_preflight(device_paths: tuple[Path, ...] = POWER_DEVICE_PATHS) -> dict[str, Any]:
    devices = []
    for path in device_paths:
        if path.exists():
            devices.append(
                {
                    "path": str(path),
                    "readable": os.access(path, os.R_OK),
                    "writable": os.access(path, os.W_OK),
                }
            )
    accessible = any(item["readable"] and item["writable"] for item in devices)
    if accessible:
        status, reason = "available", None
    elif devices:
        status, reason = "unavailable", "permission_denied"
    else:
        status, reason = "unavailable", "i2c_devices_not_found"
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        username = str(os.geteuid())
    return {
        "status": status,
        "reason": reason,
        "runtime_user": username,
        "required_group": "i2c",
        "devices": devices,
    }


def warn_power_preflight(preflight: dict[str, Any]) -> None:
    if preflight["status"] == "available":
        return
    user = preflight["runtime_user"]
    if preflight["reason"] == "permission_denied":
        message = (
            f"warning: power telemetry is unavailable because user {user!r} cannot read and write the Modalix "
            "I2C devices; add the runtime user to the i2c group and start a new login session"
        )
    else:
        message = "warning: power telemetry is unavailable because the Modalix I2C devices were not found"
    print(message, file=sys.stderr)


def base_version(version: str) -> str:
    return version.split("+", 1)[0]


def require_runtime_compatibility(
    expected_platform: str,
    expected_neat: str,
    pyneat: ModuleType,
    actual_platform: str | None,
) -> dict[str, str | None]:
    actual_neat_value = getattr(pyneat, "__version__", None)
    actual_neat = actual_neat_value if isinstance(actual_neat_value, str) else None
    if actual_platform is None:
        raise CompatibilityError("could not determine platform version from /etc/buildinfo")
    if actual_platform != expected_platform:
        raise CompatibilityError(f"platform version mismatch: expected {expected_platform}, found {actual_platform}")
    if actual_neat is None:
        raise CompatibilityError("pyneat does not expose __version__")
    if base_version(actual_neat) != expected_neat:
        raise CompatibilityError(f"Neat version mismatch: expected {expected_neat}, found {actual_neat}")
    return {
        "expected_platform_version": expected_platform,
        "actual_platform_version": actual_platform,
        "expected_neat_version": expected_neat,
        "actual_neat_version": actual_neat,
    }


def import_pyneat() -> ModuleType:
    try:
        return importlib.import_module("pyneat")
    except ImportError as exc:
        raise CompatibilityError("pyneat is not importable; install the matching wheel or activate ~/pyneat") from exc


def resolve_pyneat_reference(pyneat: ModuleType, reference: str) -> Any:
    if not REFERENCE.fullmatch(reference):
        raise ConfigurationError(f"invalid pyneat reference: {reference!r}")
    value: Any = pyneat
    for part in reference.split("."):
        if not hasattr(value, part):
            raise CompatibilityError(f"installed pyneat does not expose {reference}")
        value = getattr(value, part)
    return value


def build_model(pyneat: ModuleType, model_path: Path, options: dict[str, Any]) -> Any:
    try:
        if not options:
            return pyneat.Model(str(model_path))
        model_options = pyneat.ModelOptions()
        for name, value in options.items():
            if not hasattr(model_options, name):
                raise CompatibilityError(f"installed pyneat ModelOptions has no field {name!r}")
            resolved = (
                resolve_pyneat_reference(pyneat, value.removeprefix("pyneat:"))
                if isinstance(value, str) and value.startswith("pyneat:")
                else value
            )
            setattr(model_options, name, resolved)
        return pyneat.Model(str(model_path), model_options)
    except BenchmarkToolError:
        raise
    except Exception as exc:
        raise ModelError(f"could not load model: {exc}") from exc


def _spec_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_spec_value(item) for item in value]
    name = getattr(value, "name", None)
    return name if isinstance(name, str) and name else str(value)


def _spec_records(model: Any, method: str) -> list[dict[str, Any]]:
    try:
        records = []
        for spec in getattr(model, method)():
            record = {"type": type(spec).__name__}
            for field in (
                "shape",
                "rank",
                "dtypes",
                "image_format",
                "device",
                "preferred_device",
                "allowed_devices",
                "allow_composite",
                "required_segment_names",
            ):
                if hasattr(spec, field):
                    record[field] = _spec_value(getattr(spec, field))
            records.append(record)
        return records
    except Exception as exc:
        return [{"type": "unavailable", "error": str(exc)}]


def resolved_model(model: Any) -> dict[str, Any]:
    postprocess = None
    output_topology = None
    try:
        info = model.info()
        postprocess = str(info.selection.selected_post_kind)
        topology = info.output_topology
        output_topology = {
            "physical": int(topology.physical_outputs),
            "logical": int(topology.logical_outputs),
            "packed": bool(topology.packed_outputs),
        }
    except Exception:
        pass
    return {
        "postprocess": postprocess,
        "output_topology": output_topology,
        "input_specs": _spec_records(model, "input_specs"),
        "output_specs": _spec_records(model, "output_specs"),
    }


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result) and result > 0:
            return result
    return None


def execute_trials(
    pyneat: ModuleType,
    model: Any,
    benchmark: dict[str, Any],
    *,
    results: list[dict[str, Any]] | None = None,
    on_trial_complete: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    trials = benchmark.get("trials", DEFAULT_TRIALS)
    completed = results if results is not None else []
    for trial_index in range(1, trials + 1):
        try:
            options = pyneat.BenchmarkOptions()
            options.num_samples = benchmark["samples"]
            options.include_plugin_latency = benchmark.get("include_plugin_latency", False)
            if benchmark.get("original_width") is not None:
                options.original_width = benchmark["original_width"]
                options.original_height = benchmark["original_height"]
            report = model.benchmark(options)
        except Exception as exc:
            raise BenchmarkExecutionError(f"trial {trial_index} failed: {exc}") from exc

        latency_avg = _finite_positive(getattr(report, "latency_avg_ms", None))
        if latency_avg is None:
            latency_avg = _finite_positive(getattr(report, "latency_ms", None))
        throughput = _finite_positive(getattr(report, "throughput_fps", None))
        if throughput is None:
            throughput = _finite_positive(getattr(report, "fps", None))
        if latency_avg is None or throughput is None:
            raise BenchmarkExecutionError(f"trial {trial_index} did not produce positive latency and throughput")
        power = _finite_positive(getattr(report, "avg_power_watts", None))
        energy = _finite_positive(getattr(report, "energy_joules", None))
        completed.append(
            {
                "trial": trial_index,
                "latency_avg_ms": latency_avg,
                "latency_p50_ms": _finite_positive(getattr(report, "latency_p50_ms", None)),
                "latency_p95_ms": _finite_positive(getattr(report, "latency_p95_ms", None)),
                "throughput_fps": throughput,
                "avg_power_watts": power,
                "energy_joules": energy,
                "energy_per_inference_joules": None,
            }
        )
        if on_trial_complete is not None:
            on_trial_complete(trial_index)
    return completed


def _synthetic_runner_inputs(model: Any) -> list[Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise CompatibilityError("bounded_runner requires numpy in the pyneat environment") from exc

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "uint64": np.uint64,
        "int64": np.int64,
    }
    inputs = []
    try:
        specs = model.input_specs()
    except Exception as exc:
        raise ModelError(f"could not inspect model input specifications: {exc}") from exc
    if not specs:
        raise ModelError("bounded_runner requires at least one model input")
    for index, spec in enumerate(specs):
        shape = list(getattr(spec, "shape", []))
        if not shape or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape):
            raise ModelError(f"model input {index} has an unsupported shape: {shape}")
        dtypes = list(getattr(spec, "dtypes", []))
        if not dtypes:
            raise ModelError(f"model input {index} does not declare a dtype")
        dtype_name = getattr(dtypes[0], "name", str(dtypes[0])).split(".")[-1].lower()
        dtype = dtype_map.get(dtype_name)
        if dtype is None:
            raise ModelError(f"model input {index} has an unsupported dtype: {dtype_name}")
        inputs.append(np.zeros(tuple(shape), dtype=dtype))
    return inputs


def _bounded_measure_options(
    pyneat: ModuleType,
    *,
    title: str,
    logical_batch_size: int,
    include_plugin_latency: bool,
    include_power: bool,
) -> Any:
    options = pyneat.MeasureOptions()
    options.duration_ms = 120000
    options.warmup_ms = 0
    options.timeout_ms = 120000
    options.include_plugin_latency = include_plugin_latency
    options.include_edge_latency = False
    options.include_message_latency = False
    options.include_power = include_power
    options.logical_batch_size = logical_batch_size
    options.title = title
    options.input = "synthetic"
    return options


def _measurement_payload(report: Any, description: str) -> dict[str, Any]:
    try:
        value = json.loads(report.to_json())
    except (AttributeError, TypeError, json.JSONDecodeError) as exc:
        raise BenchmarkExecutionError(f"{description} did not return a JSON measurement report") from exc
    if not isinstance(value, dict):
        raise BenchmarkExecutionError(f"{description} measurement report must be an object")
    return value


def execute_bounded_trials(
    pyneat: ModuleType,
    model: Any,
    benchmark: dict[str, Any],
    *,
    logical_batch_size: int,
    include_power: bool,
    results: list[dict[str, Any]] | None = None,
    on_trial_complete: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    inputs = _synthetic_runner_inputs(model)
    samples = benchmark["samples"]
    warmup_samples = benchmark["warmup_samples"]
    trials = benchmark.get("trials", DEFAULT_TRIALS)
    include_plugin_latency = benchmark.get("include_plugin_latency", False)
    completed = results if results is not None else []

    for trial_index in range(1, trials + 1):
        runner = None
        try:
            run_options = pyneat.RunOptions()
            run_options.startup_preflight = False
            if include_power:
                run_options.enable_board_power()
            runner = model.build(inputs, run_options=run_options)
            for _ in range(warmup_samples):
                runner.run(inputs, timeout_ms=120000)

            latency_scope = runner.start_measurement(
                _bounded_measure_options(
                    pyneat,
                    title=f"NEAT bounded latency trial {trial_index}",
                    logical_batch_size=logical_batch_size,
                    include_plugin_latency=include_plugin_latency,
                    include_power=False,
                )
            )
            for _ in range(samples):
                runner.run(inputs, timeout_ms=120000)
            latency = _measurement_payload(latency_scope.stop(), "latency")

            outputs: list[Any] = []
            pull_errors: list[Exception] = []

            def pull_outputs() -> None:
                try:
                    for _ in range(samples):
                        outputs.append(runner.pull(120000))
                except Exception as exc:  # Keep the worker error for the controlling thread.
                    pull_errors.append(exc)

            throughput_scope = runner.start_measurement(
                _bounded_measure_options(
                    pyneat,
                    title=f"NEAT bounded throughput trial {trial_index}",
                    logical_batch_size=logical_batch_size,
                    include_plugin_latency=include_plugin_latency,
                    include_power=include_power,
                )
            )
            puller = threading.Thread(target=pull_outputs, name=f"benchmark-pull-{trial_index}")
            puller.daemon = True
            puller.start()
            for _ in range(samples):
                if not runner.push(inputs):
                    raise BenchmarkExecutionError("throughput measured push failed")
            puller.join(timeout=125)
            if puller.is_alive():
                raise BenchmarkExecutionError("throughput output collection timed out")
            if pull_errors:
                raise pull_errors[0]
            if len(outputs) != samples:
                raise BenchmarkExecutionError(
                    f"throughput measured output count mismatch: expected {samples}, found {len(outputs)}"
                )
            throughput = _measurement_payload(throughput_scope.stop(), "throughput")
        except BenchmarkToolError:
            raise
        except Exception as exc:
            raise BenchmarkExecutionError(f"trial {trial_index} failed: {exc}") from exc
        finally:
            if runner is not None:
                runner.close()

        latency_stats = latency.get("measurement", {}).get("end_to_end", {})
        throughput_measurement = throughput.get("measurement", {})
        power = throughput.get("graph_metrics", {}).get("power", {})
        latency_avg = _finite_positive(latency_stats.get("avg_ms"))
        throughput_fps = _finite_positive(throughput_measurement.get("throughput_inferences_per_s"))
        if latency_avg is None or throughput_fps is None:
            raise BenchmarkExecutionError(
                f"trial {trial_index} did not produce positive latency and throughput"
            )
        completed.append(
            {
                "trial": trial_index,
                "latency_avg_ms": latency_avg,
                "latency_p50_ms": _finite_positive(latency_stats.get("p50_ms")),
                "latency_p95_ms": _finite_positive(latency_stats.get("p95_ms")),
                "throughput_fps": throughput_fps,
                "avg_power_watts": _finite_positive(power.get("total_avg_watts")),
                "energy_joules": _finite_positive(power.get("energy_joules")),
                "energy_per_inference_joules": None,
            }
        )
        print(
            f"trial={trial_index}/{trials} samples={samples} warmup={warmup_samples} "
            f"latency_ms={latency_avg:.3f} throughput_fps={throughput_fps:.6f}",
            flush=True,
        )
        if on_trial_complete is not None:
            on_trial_complete(trial_index)
    return completed


def metric(
    trials: list[dict[str, Any]],
    field: str,
    unit: str,
    missing: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    values = [trial[field] for trial in trials if trial.get(field) is not None]
    if not values:
        return {
            "status": missing,
            "reason": reason,
            "unit": unit,
            "aggregation": "mean",
            "value": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
            "sample_count": 0,
        }
    return {
        "status": "measured",
        "reason": None,
        "unit": unit,
        "aggregation": "mean",
        "value": fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": pstdev(values),
        "sample_count": len(values),
    }


def _configuration(manifest: dict[str, Any]) -> dict[str, Any]:
    benchmark = manifest["benchmark"]
    configuration = {
        "samples": benchmark["samples"],
        "trials": benchmark.get("trials", DEFAULT_TRIALS),
        "include_plugin_latency": benchmark.get("include_plugin_latency", False),
        "original_width": benchmark.get("original_width"),
        "original_height": benchmark.get("original_height"),
        "model_options": manifest.get("model_options", {}),
    }
    if benchmark.get("mode") == "bounded_runner":
        configuration["mode"] = "bounded_runner"
        configuration["warmup_samples"] = benchmark["warmup_samples"]
    return configuration


def benchmark_identity_document(result: dict[str, Any]) -> dict[str, Any] | None:
    model = result["model"]
    compatibility = result["compatibility"]
    required = (
        model.get("id"),
        model.get("variant_id"),
        model.get("sha256"),
        compatibility.get("expected_platform_version"),
        compatibility.get("expected_neat_version"),
        result.get("configuration"),
        result.get("environment", {}).get("fingerprint"),
    )
    if any(value is None for value in required):
        return None
    return {
        "schema_version": result["schema_version"],
        "benchmark_type": result["benchmark_type"],
        "model": {
            "id": model["id"],
            "variant_id": model["variant_id"],
            "sha256": model["sha256"],
        },
        "compatibility": {
            "platform_version": compatibility["expected_platform_version"],
            "neat_version": compatibility["expected_neat_version"],
        },
        "configuration": result["configuration"],
        "environment_fingerprint": result["environment"]["fingerprint"],
        "tool": result["tool"],
    }


def update_result_identity(result: dict[str, Any]) -> None:
    document = benchmark_identity_document(result)
    result["identity"] = {
        "algorithm": "sha256",
        "canonicalization_version": result["schema_version"],
        "fingerprint": canonical_sha256(document) if document is not None else None,
    }


def _partial_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _partial_configuration(manifest: dict[str, Any]) -> dict[str, Any] | None:
    benchmark = _partial_mapping(manifest.get("benchmark"))
    samples = benchmark.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        return None
    trials = benchmark.get("trials", DEFAULT_TRIALS)
    mode = benchmark.get("mode")
    warmup_samples = benchmark.get("warmup_samples")
    include_plugin_latency = benchmark.get("include_plugin_latency", False)
    width, height = benchmark.get("original_width"), benchmark.get("original_height")
    model_options = manifest.get("model_options", {})
    if (
        not isinstance(trials, int)
        or isinstance(trials, bool)
        or not 1 <= trials <= 100
        or not isinstance(include_plugin_latency, bool)
        or mode not in {None, "bounded_runner"}
        or (
            mode == "bounded_runner"
            and (
                not isinstance(warmup_samples, int)
                or isinstance(warmup_samples, bool)
                or warmup_samples <= 0
            )
        )
        or (width is None) != (height is None)
        or (width is not None and (not isinstance(width, int) or isinstance(width, bool) or width <= 0))
        or (height is not None and (not isinstance(height, int) or isinstance(height, bool) or height <= 0))
        or not isinstance(model_options, dict)
    ):
        return None
    configuration = {
        "samples": samples,
        "trials": trials,
        "include_plugin_latency": include_plugin_latency,
        "original_width": width,
        "original_height": height,
        "model_options": model_options,
    }
    if mode == "bounded_runner":
        configuration["mode"] = mode
        configuration["warmup_samples"] = warmup_samples
    return configuration


def _result_base(
    manifest: dict[str, Any],
    model_path: Path | None,
    actual_platform: str | None,
    *,
    preflight: dict[str, Any] | None = None,
    actual_neat: str | None = None,
) -> dict[str, Any]:
    model = _partial_mapping(manifest.get("model"))
    expected = _partial_mapping(manifest.get("compatibility"))
    model_sha = model.get("sha256")
    telemetry = preflight or power_telemetry_preflight()
    power_reason = telemetry.get("reason") or "runtime_did_not_return_metric"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "error",
        "benchmark_type": "model.synthetic",
        "tool": {"name": "benchmarker", "version": tool_version()},
        "identity": {
            "algorithm": "sha256",
            "canonicalization_version": RESULT_SCHEMA_VERSION,
            "fingerprint": None,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failure_classification": None,
        "model": {
            "id": model.get("id") if isinstance(model.get("id"), str) else None,
            "variant_id": model.get("variant_id") if isinstance(model.get("variant_id"), str) else None,
            "path": str(model_path) if model_path is not None else None,
            "sha256": model_sha if isinstance(model_sha, str) and SHA256.fullmatch(model_sha) else None,
            "uri": model.get("uri") if isinstance(model.get("uri"), str) else None,
            "contract": None,
        },
        "compatibility": {
            "expected_platform_version": (
                expected.get("platform_version")
                if isinstance(expected.get("platform_version"), str) and SEMVER.fullmatch(expected["platform_version"])
                else None
            ),
            "actual_platform_version": actual_platform,
            "expected_neat_version": (
                expected.get("neat_version")
                if isinstance(expected.get("neat_version"), str) and SEMVER.fullmatch(expected["neat_version"])
                else None
            ),
            "actual_neat_version": actual_neat,
        },
        "configuration": _partial_configuration(manifest),
        "environment": runtime_environment(actual_platform, actual_neat),
        "power_measurement": {
            "scope": "board_pmbus_rails",
            "phase": "throughput",
            "source": (
                "pyneat.ModelRunner.measure"
                if _partial_mapping(manifest.get("benchmark")).get("mode") == "bounded_runner"
                else "pyneat.Model.benchmark"
            ),
            "idle_baseline_subtracted": False,
            "telemetry_preflight": telemetry,
        },
        "metrics": {
            "latency_avg_ms": metric([], "latency_avg_ms", "ms", "unavailable", reason="benchmark_not_completed"),
            "latency_p50_ms": metric(
                [], "latency_p50_ms", "ms", "unsupported", reason="neat_0.4.0_does_not_expose_percentiles"
            ),
            "latency_p95_ms": metric(
                [], "latency_p95_ms", "ms", "unsupported", reason="neat_0.4.0_does_not_expose_percentiles"
            ),
            "throughput_fps": metric(
                [], "throughput_fps", "fps", "unavailable", reason="benchmark_not_completed"
            ),
            "avg_power_watts": metric([], "avg_power_watts", "W", "unavailable", reason=power_reason),
            "energy_joules": metric([], "energy_joules", "J", "unavailable", reason=power_reason),
            "energy_per_inference_joules": metric(
                [], "energy_per_inference_joules", "J/inference", "unavailable", reason=power_reason
            ),
        },
        "trials": [],
        "orchestration": orchestration_context(),
        "provenance": manifest.get("provenance", {}) if isinstance(manifest.get("provenance", {}), dict) else {},
    }
    update_result_identity(result)
    return result


def orchestration_context() -> dict[str, Any]:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        runner = os.environ.get("RUNNER_NAME", "")
        correlation = hashlib.sha256(runner.encode()).hexdigest()[:16] if runner else None
        attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
        return {
            "provider": "github-actions-devkit-runner",
            "session_id": os.environ.get("GITHUB_RUN_ID"),
            "task_id": os.environ.get("GITHUB_JOB"),
            "attempt": int(attempt) if attempt.isdigit() else None,
            "target_correlation_id": correlation,
        }
    return {
        "provider": "manual",
        "session_id": None,
        "task_id": None,
        "attempt": None,
        "target_correlation_id": None,
    }


def run_benchmark(
    manifest: dict[str, Any],
    *,
    pyneat: ModuleType | None = None,
    actual_platform: str | None = None,
    power_preflight: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_path = Path(manifest["model"]["path"]).expanduser().resolve()
    if actual_platform is None:
        actual_platform = detect_platform_version()
    preflight = power_preflight or power_telemetry_preflight()
    warn_power_preflight(preflight)
    if result is None:
        result = _result_base(manifest, model_path, actual_platform, preflight=preflight)
    contract = inspect_mpk(model_path)
    result["model"]["contract"] = {key: value for key, value in contract.items() if key != "manifest"}
    actual_sha = sha256_file(model_path)
    if actual_sha != manifest["model"]["sha256"]:
        raise ModelError(f"MPK checksum mismatch: expected {manifest['model']['sha256']}, found {actual_sha}")
    expected_platform = manifest["compatibility"]["platform_version"]
    if contract["model_sdk_version"] and contract["model_sdk_version"] != expected_platform:
        raise CompatibilityError(
            f"MPK Model SDK version mismatch: expected {expected_platform}, found {contract['model_sdk_version']}"
        )

    runtime = pyneat or import_pyneat()
    result["compatibility"] = require_runtime_compatibility(
        expected_platform,
        manifest["compatibility"]["neat_version"],
        runtime,
        actual_platform,
    )
    result["environment"] = runtime_environment(actual_platform, result["compatibility"]["actual_neat_version"])
    update_result_identity(result)
    model = build_model(runtime, model_path, manifest.get("model_options", {}))
    result["model"]["resolved"] = resolved_model(model)
    trials = result["trials"]
    batch_size = contract["compiled_batch_size"] or 1
    logical_inferences = manifest["benchmark"]["samples"] * batch_size
    power_reason = preflight.get("reason") or "runtime_did_not_return_metric"
    try:
        def trial_callback(trial: int) -> None:
            record_temperature_sample(
                result["environment"],
                phase="trial_complete",
                trial=trial,
            )
        if manifest["benchmark"].get("mode") == "bounded_runner":
            execute_bounded_trials(
                runtime,
                model,
                manifest["benchmark"],
                logical_batch_size=batch_size,
                include_power=preflight.get("status") == "available",
                results=trials,
                on_trial_complete=trial_callback,
            )
        else:
            execute_trials(
                runtime,
                model,
                manifest["benchmark"],
                results=trials,
                on_trial_complete=trial_callback,
            )
    finally:
        for trial in trials:
            if trial["energy_joules"] is not None:
                trial["energy_per_inference_joules"] = trial["energy_joules"] / logical_inferences
        result["metrics"] = {
            "latency_avg_ms": metric(trials, "latency_avg_ms", "ms", "unavailable"),
            "latency_p50_ms": metric(
                trials,
                "latency_p50_ms",
                "ms",
                "unsupported",
                reason="neat_0.4.0_does_not_expose_percentiles",
            ),
            "latency_p95_ms": metric(
                trials,
                "latency_p95_ms",
                "ms",
                "unsupported",
                reason="neat_0.4.0_does_not_expose_percentiles",
            ),
            "throughput_fps": metric(trials, "throughput_fps", "fps", "unavailable"),
            "avg_power_watts": metric(
                trials, "avg_power_watts", "W", "unavailable", reason=power_reason
            ),
            "energy_joules": metric(
                trials, "energy_joules", "J", "unavailable", reason=power_reason
            ),
            "energy_per_inference_joules": metric(
                trials,
                "energy_per_inference_joules",
                "J/inference",
                "unavailable",
                reason=power_reason,
            ),
        }
    end_observation = record_temperature_sample(result["environment"], phase="end")
    result["environment"]["observations"]["end"] = end_observation
    result["status"] = "passed"
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {tool_version()}")
    parser.add_argument("--manifest", type=Path, required=True, help="performance-test-v1 JSON manifest")
    parser.add_argument("--output", type=Path, required=True, help="performance-result-v2 JSON output")
    return parser


def _read_partial_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _partial_model_path(manifest: dict[str, Any]) -> Path | None:
    model = _partial_mapping(manifest.get("model"))
    value = model.get("path")
    return Path(value).expanduser().resolve() if isinstance(value, str) and value else None


def _record_error(result: dict[str, Any], exc: Exception, classification: str) -> None:
    result["failure_classification"] = classification
    result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    end_observation = record_temperature_sample(result["environment"], phase="end")
    result["environment"]["observations"]["end"] = end_observation


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    actual_platform = detect_platform_version()
    try:
        manifest = load_manifest(args.manifest)
    except BenchmarkToolError as exc:
        partial = _read_partial_manifest(args.manifest)
        result = _result_base(partial, _partial_model_path(partial), actual_platform)
        _record_error(result, exc, exc.classification)
        atomic_write_json(args.output, result)
        print(f"{exc.classification} error: {exc}", file=sys.stderr)
        print(f"report_json={args.output}")
        return exc.exit_code

    model_path = Path(manifest["model"]["path"]).expanduser().resolve()
    preflight = power_telemetry_preflight()
    result = _result_base(manifest, model_path, actual_platform, preflight=preflight)
    try:
        run_benchmark(
            manifest,
            actual_platform=actual_platform,
            power_preflight=preflight,
            result=result,
        )
    except BenchmarkToolError as exc:
        _record_error(result, exc, exc.classification)
        atomic_write_json(args.output, result)
        print(f"benchmark error: {exc}", file=sys.stderr)
        print(f"report_json={args.output}")
        return exc.exit_code
    except Exception as exc:
        _record_error(result, exc, "infrastructure")
        atomic_write_json(args.output, result)
        print(f"unexpected benchmark error: {exc}", file=sys.stderr)
        print(f"report_json={args.output}")
        return EXIT_BENCHMARK

    atomic_write_json(args.output, result)
    print(f"latency_avg_ms={result['metrics']['latency_avg_ms']['value']}")
    print(f"throughput_fps={result['metrics']['throughput_fps']['value']}")
    print(f"report_json={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
