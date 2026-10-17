"""Read the registry's benchmark metrics from a performance result's saved Core report."""

from __future__ import annotations

from typing import Any

from .errors import ValidationError

# Each registry metric names the report field it reads, its unit, and its registered Vulcan measurement.
REPORT_METRICS = {
    "latency_avg_ms": ("latency_ms", "ms", "model.latency.avg"),
    "latency_p50_ms": ("latency_p50_ms", "ms", "model.latency.p50"),
    "latency_p95_ms": ("latency_p95_ms", "ms", "model.latency.p95"),
    "latency_p99_ms": ("latency_p99_ms", "ms", "model.latency.p99"),
    "throughput_fps": ("fps", "inferences/s", "model.inference_rate"),
    "avg_power_watts": ("avg_power_watts", "W", "board.power.avg"),
    "energy_joules": ("energy_joules", "J", "board.energy"),
}
# Board telemetry Core reads from Sentinel over the throughput window: each registry metric names its
# report telemetry field, the statistic, the unit Sentinel reports, its registered Vulcan measurement,
# and the field whose entries break it down. A list field (one entry per CPU core) reports its busiest
# entry for that statistic, so the average and the peak of the busiest core can come from different cores.
TELEMETRY_METRICS = {
    "peak_power_watts": ("power_som_watts", "maximum", "W", "board.power.peak", "power_rails_watts"),
    "avg_mla_power_watts": ("power_mla_watts", "mean", "W", "board.mla_power.avg", None),
    "peak_mla_power_watts": ("power_mla_watts", "maximum", "W", "board.mla_power.peak", None),
    "avg_die_temperature_c": ("die_temperature_max_c", "mean", "C", "board.die_temperature.avg", None),
    "peak_die_temperature_c": ("die_temperature_max_c", "maximum", "C", "board.die_temperature.peak", None),
    "avg_board_temperature_c": ("board_temperatures_c", "mean", "C", "board.som_temperature.avg", None),
    "peak_board_temperature_c": ("board_temperatures_c", "maximum", "C", "board.som_temperature.peak", None),
    "avg_cpu_usage_pct": ("cpu_usage_pct", "mean", "%", "board.cpu_usage.avg", None),
    "peak_cpu_usage_pct": ("cpu_usage_pct", "maximum", "%", "board.cpu_usage.peak", None),
    "avg_cpu_core_usage_pct": ("cpu_cores_usage_pct", "mean", "%", "board.cpu_core_usage.avg", None),
    "peak_cpu_core_usage_pct": ("cpu_cores_usage_pct", "maximum", "%", "board.cpu_core_usage.peak", None),
    "mla_memory_avg_mib": ("mla_memory_mib", "mean", "MiB", "model.mla_memory.avg", None),
    "mla_memory_peak_mib": ("mla_memory_mib", "maximum", "MiB", "model.mla_memory.peak", None),
    "avg_cma_memory_mib": ("cma_memory_mib", "mean", "MiB", "board.cma_memory.avg", None),
    "peak_cma_memory_mib": ("cma_memory_mib", "maximum", "MiB", "board.cma_memory.peak", None),
    "avg_linux_memory_mib": ("linux_memory_used_mib", "mean", "MiB", "board.linux_memory.avg", None),
    "peak_linux_memory_mib": ("linux_memory_used_mib", "maximum", "MiB", "board.linux_memory.peak", None),
}
ENERGY_PER_INFERENCE = "energy_per_inference_joules"
AGGREGATIONS = {"mean": "mean", "maximum": "max"}
# Sentinel writes Celsius as "C"; the registry spells the unit out.
REGISTRY_UNITS = {"C": "°C"}


def measurement_name(metric: str) -> str:
    """The Vulcan measurement a registry metric is published as."""
    if metric == ENERGY_PER_INFERENCE:
        return "model.energy_per_inference"
    return REPORT_METRICS[metric][2] if metric in REPORT_METRICS else TELEMETRY_METRICS[metric][3]


def registry_metrics(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each metric the result's Core report measured, by registry name. Core reports 0 for a headline
    value it did not measure, and 0 samples for a telemetry reading Sentinel did not provide."""
    report = result["benchmark_report"]
    metrics = {}
    for name, (field, unit, _) in REPORT_METRICS.items():
        if report[field] > 0:
            metrics[name] = _metric(name, report[field], unit, "raw", 1)
    if "energy_joules" in metrics:
        inferences = result["configuration"]["samples"] * result["configuration"]["workload"]["batch_size"]
        energy = report["energy_joules"] / inferences
        metrics[ENERGY_PER_INFERENCE] = _metric(ENERGY_PER_INFERENCE, energy, "J/inference", "raw", 1)
    telemetry = report["telemetry"]
    for name, (field, statistic, unit, _, breakdown) in TELEMETRY_METRICS.items():
        value = telemetry[field]
        entries = [entry for entry in (value if isinstance(value, list) else [value]) if entry["samples"]]
        if not entries:
            continue
        if any(entry["unit"] != unit for entry in entries):
            raise ValidationError(f"benchmark telemetry {field} must be in {unit!r}")
        selected = max(entries, key=lambda entry: entry[statistic])
        metric = _metric(
            name, selected[statistic], REGISTRY_UNITS.get(unit, unit), AGGREGATIONS[statistic], selected["samples"]
        )
        metric["minimum"], metric["maximum"] = selected["minimum"], selected["maximum"]
        if breakdown:
            metric["details"] = {"breakdown": [
                {key: entry[key] for key in ("key", "label", "unit", "samples", "mean", "maximum")}
                for entry in telemetry[breakdown]
                if entry["samples"]
            ]}
        metrics[name] = metric
    return metrics


def _metric(name: str, value: float, unit: str, aggregation: str, sample_count: int) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "aggregation": aggregation,
        "sample_count": sample_count,
        "minimum": None,
        "maximum": None,
        "direction": "higher" if name == "throughput_fps" else "lower",
    }
