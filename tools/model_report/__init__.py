"""Write a model's report from what Neat Core returned.

A model's scripts hand Core's results to from_performance() or from_accuracy(). Independent component
benchmarks combine their single-component reports with from_component_performance().
Report.save() writes model-report.json, which the registry
publishes unchanged and the model browser reads. metrics.json names every metric a report may hold, with
its unit, direction, and the Vulcan measurement name (registry_name) it is published under.

Needs only the Python standard library, so it runs in PyNeat's environment on a DevKit:
``PYTHONPATH=tools ~/pyneat/bin/python models/<model>/scripts/performance.py --mpk ... --output out``.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "sima.model-report"
SCHEMA_VERSION = 1
REPORT_FILE = "model-report.json"
# Core's own saved BenchmarkReport, kept beside the report as evidence.
CORE_REPORT_FILE = "benchmark-report.json"
METRICS: dict[str, dict[str, Any]] = json.loads(Path(__file__).with_name("metrics.json").read_text(encoding="utf-8"))
# Every performance result must hold these; each accuracy dataset's datasets/<id>.yaml names its own.
REQUIRED_PERFORMANCE_METRICS = [name for name, metric in METRICS.items() if metric.get("required")]
# Derived from the measured power and rate rather than read from Core.
ENERGY_PER_INFERENCE = "energy_per_inference_joules"


def accuracy_value_error(metric: str, value: float) -> str | None:
    """Why an accuracy score is outside its bounds, or None.

    Bounds default to 0..1 for ratio scores, 0..100 for percentages, and nonnegative
    for other units. A metric can override these with bounds; null means no upper limit.
    """
    definition = METRICS[metric]
    unit = definition["unit"]
    minimum, maximum = definition.get("bounds", (0, {"ratio": 1, "%": 100}.get(unit)))
    if math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum):
        return None
    if maximum is None:
        return f"{metric} must be at least {minimum} {unit}; found {value}"
    label = {"ratio": "a ratio", "%": "a percentage"}.get(unit, unit)
    return f"{metric} must be {label} from {minimum} to {maximum}; found {value}"


class Report:
    """A model report and the Core evidence saved beside it."""

    def __init__(self, document: dict[str, Any], evidence: dict[str, dict[str, Any]] | None = None) -> None:
        self.document = document
        self.evidence = evidence or {}

    def save(self, output: str | Path) -> Path:
        """Write model-report.json and its evidence into the output directory; return the report's path."""
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        for name, document in self.evidence.items():
            _write(directory / name, document)
        path = directory / REPORT_FILE
        _write(path, self.document)
        return path


def from_generation(generation: dict[str, Any], runtime: dict[str, Any]) -> Report:
    """A native LLiMa performance report, with timings and telemetry per text/image workload."""
    return Report({
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "stage": "performance",
        "runtime": runtime, "metrics": {},
        "workload": "Full dataset comparisons" if generation["settings"].get("decoding") == "speculative"
        else "Text buckets and an encoder-sized image when supported",
        "telemetry": {}, "components": {}, "generation": generation,
    }, {CORE_REPORT_FILE: generation})


def telemetry_metrics(telemetry: dict[str, Any]) -> dict[str, float]:
    """The measured board metrics in the same units as Core's performance report."""
    return _performance_metrics({}, telemetry)


def from_performance(
    benchmark: Any,
    models: dict[str, Any],
    *,
    workload: str,
    telemetry: Any | None = None,
    mla_layer_stats: dict[str, dict[str, Any]] | None = None,
    mla_stage_stats: dict[str, dict[str, Any]] | None = None,
) -> Report:
    """A performance report from a Core ``BenchmarkReport`` of the whole model.

    ``models`` maps each component to the ``pyneat.Model`` that ran it, in execution order. A script that
    times the model itself fills a ``pyneat.BenchmarkReport`` with what it measured, passes the
    ``TelemetryMonitor.stop()`` result as ``telemetry``, and gives each component's MLA stats as
    ``mla_layer_stats`` and ``mla_stage_stats``, in the shape Core saves them; otherwise all three come from
    ``benchmark``. Core reports 0 for a value it did not measure, and a metric it did not measure is absent
    from the report.
    """
    if not models:
        raise ValueError("models must name at least one component")
    with tempfile.TemporaryDirectory() as scratch:
        core_report = _saved(benchmark, Path(scratch) / CORE_REPORT_FILE)
        measured_telemetry = (
            core_report["telemetry"] if telemetry is None else _saved(telemetry, Path(scratch) / "telemetry.json")
        )
    layer_stats = _component_stats("mla_layer_stats", mla_layer_stats, core_report, models)
    stage_stats = _component_stats("mla_stage_stats", mla_stage_stats, core_report, models)
    return Report(
        {
            **_header("performance"),
            "workload": workload,
            "metrics": _performance_metrics(core_report, measured_telemetry),
            "telemetry": measured_telemetry,
            "components": {
                component: {
                    "execution_plan": json.loads(model.execution_plan().to_json()),
                    "mla_layer_stats": layer_stats[component],
                    "mla_stage_stats": stage_stats[component],
                }
                for component, model in models.items()
            },
        },
        {CORE_REPORT_FILE: core_report},
    )


def from_component_performance(reports: dict[str, Report]) -> Report:
    """Independent component benchmarks, without a combined throughput or latency.

    Build each entry with ``from_performance(benchmark, {component: model}, workload=...)``.
    Every component retains its own metrics, telemetry, workload and Core evidence.
    """
    if not reports:
        raise ValueError("reports must name at least one component")
    components, evidence = {}, {}
    header = _header("performance")
    for name, report in reports.items():
        document = report.document
        if document.get("stage") != "performance" or set(document.get("components", {})) != {name}:
            raise ValueError(f"{name}: expected a performance report of exactly that component")
        if document["runtime"] != header["runtime"]:
            raise ValueError(f"{name}: component benchmarks must use the same runtime")
        components[name] = {
            **document["components"][name],
            **{key: document[key] for key in ("metrics", "workload", "telemetry")},
        }
        evidence.update({f"{name}-{filename}": value for filename, value in report.evidence.items()})
    return Report({**header, "scope": "components", "components": components}, evidence)


def from_accuracy(datasets: dict[str, dict[str, Any]]) -> Report:
    """An accuracy report: per dataset, each metric's score over its evaluated part, in the metric's unit.

    ``datasets`` maps each dataset name model.yaml's accuracy lists to its result: the ``metrics``, the
    ``split`` evaluated, ``evaluated_samples``, ``available_samples``, and optional ``details``.
    """
    if not datasets:
        raise ValueError("datasets must name at least one dataset")
    return Report(
        {
            **_header("accuracy"),
            "datasets": {name: _dataset_result(name, result) for name, result in datasets.items()},
        }
    )


def _component_stats(
    name: str, given: dict[str, dict[str, Any]] | None, core_report: dict[str, Any], models: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Each component's ``name`` stats: the script's, or for a single component, those in Core's report."""
    if given is None:
        if len(models) != 1:
            raise ValueError(f"{name} must give each component's stats")
        return {component: core_report[name] for component in models}
    if set(given) != set(models):
        raise ValueError(f"{name} must name exactly the components in models")
    return given


def _dataset_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    fields = {"metrics", "split", "evaluated_samples", "available_samples"}
    if not fields <= set(result) <= {*fields, "details"}:
        raise ValueError(f"{name} must give metrics, split, evaluated_samples, available_samples, and optional details")
    metrics = result["metrics"]
    unknown = sorted(metric for metric in metrics if METRICS.get(metric, {}).get("stage") != "accuracy")
    if unknown:
        raise ValueError(f"{name}: unknown accuracy metrics: {', '.join(unknown)}")
    for metric, value in metrics.items():
        if error := accuracy_value_error(metric, value):
            raise ValueError(f"{name}: {error}")
    if not 0 < result["evaluated_samples"] <= result["available_samples"]:
        raise ValueError(f"{name}: evaluated_samples must be positive and at most available_samples")
    return {
        "split": result["split"],
        "evaluated_samples": result["evaluated_samples"],
        "available_samples": result["available_samples"],
        "metrics": {metric: float(value) for metric, value in metrics.items()},
        "details": result.get("details") or {},
    }


def _header(stage: str) -> dict[str, Any]:
    import pyneat

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "runtime": {"pyneat": pyneat.__version__, "build": dict(pyneat.build_info())},
    }


def _performance_metrics(report: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for name, metric in METRICS.items():
        if "report" in metric and report.get(metric["report"], 0) > 0:
            metrics[name] = float(report[metric["report"]])
        elif "telemetry" in metric:
            field, statistic, unit = metric["telemetry"]
            value = telemetry.get(field, [])
            # A list field has one entry per core or sensor; the busiest entry represents it.
            entries = [entry for entry in (value if isinstance(value, list) else [value]) if entry["samples"]]
            if any(entry["unit"] != unit for entry in entries):
                raise ValueError(f"telemetry {field} must be in {unit!r}")
            if entries:
                metrics[name] = float(max(entry[statistic] for entry in entries))
    # Core's throughput counts every inference in a batch, so power over it is already energy per inference.
    if "avg_power_watts" in metrics and "throughput_fps" in metrics:
        metrics[ENERGY_PER_INFERENCE] = metrics["avg_power_watts"] / metrics["throughput_fps"]
    return metrics


def _saved(value: Any, path: Path) -> dict[str, Any]:
    """The JSON Core writes for one of its results."""
    value.save(path)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _reject_nonfinite(value: str) -> float:
    raise ValueError(f"Core result contains a non-finite number: {value}")


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
