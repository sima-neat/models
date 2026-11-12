#!/usr/bin/env python3
"""Serve the artifact browser with a local registry API built from the models in this repository.

The catalog, runs, documentation, and provenance come from models/ through the registry tooling, so the
preview shows exactly the models, builds, and packages the repository declares. Only what a board or a
registry would measure or assign is synthetic: run IDs, timestamps, sizes, and measured values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import posixpath
import re
import sys
import uuid
from contextlib import contextmanager
from functools import cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
sys.path.insert(0, str(ROOT / "tools"))

from model_registry.benchmark_declarations import benchmarked_profiles, reference_metrics_by_model  # noqa: E402
from model_registry.compile_matrix import build_matrix  # noqa: E402
from model_registry.conversion import apply_selection, conversion_fingerprint  # noqa: E402
from model_registry.dependencies import resolve_benchmark_runtime, resolve_model_compiler  # noqa: E402
from model_registry.documentation import load_model_documentation  # noqa: E402
from model_registry.fingerprints import BENCHMARK_RUN_TYPES, benchmark_fingerprint  # noqa: E402
from model_registry.model import component_pipelines, load_models, mpk_filename, profile_component_for  # noqa: E402
from model_registry.model_tests import load_model_tests  # noqa: E402
from model_registry.registry import build_catalog_snapshot  # noqa: E402
from model_report import METRICS  # noqa: E402

UI_MODE = os.environ.get("MODEL_BROWSER_UI_MODE", "developer")
BRANCH = "main" if UI_MODE == "customer" else "develop"
GIT_REF = f"refs/heads/{BRANCH}"
# Two catalog revisions, so every variant has an older build in its build history.
REVISIONS = (
    ("2026-09-02", hashlib.sha1(b"preview-previous-revision").hexdigest()),
    ("2026-10-01", hashlib.sha1(b"preview-current-revision").hexdigest()),
)
REPOSITORY_API = "/v1/repositories/sima-neat/models"


def _id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "preview/" + "/".join(parts)))


def _sha256(*parts: str) -> str:
    return hashlib.sha256("/".join(parts).encode()).hexdigest()


def _fraction(*parts: str) -> float:
    """A stable value in [0, 1) for synthetic measurements, so each build differs but never changes."""
    return int(_sha256(*parts)[:8], 16) / 0x1_0000_0000


def _timestamp(day: str, position: int) -> str:
    return f"{day}T{8 + position // 60:02d}:{position % 60:02d}:00+00:00"


@contextmanager
def _preview_conversions(models: dict):
    """Synthetic content identities for the local preview, never registry credentials or writes."""
    selected = {}
    for model_id, model in models.items():
        if model["components"][0]["source"]["type"] != "conversion":
            continue
        selected[model_id] = {
            "sha256": _sha256("preview-conversion", model_id), "size_bytes": 1000,
            "conversion_fingerprint": conversion_fingerprint(ROOT, model),
        }
        apply_selection(model, selected, root=ROOT)
    previous = os.environ.get("MODEL_REGISTRY_CONVERSIONS")
    os.environ["MODEL_REGISTRY_CONVERSIONS"] = json.dumps(selected)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MODEL_REGISTRY_CONVERSIONS", None)
        else:
            os.environ["MODEL_REGISTRY_CONVERSIONS"] = previous


@cache
def _repository() -> dict[str, Any]:
    models = load_models(ROOT)
    compiler = resolve_model_compiler(ROOT)
    with _preview_conversions(models):
        matrix = build_matrix(models, sorted(models), repository=ROOT, compiler_version=compiler["identity"])["include"]
        documentation = {
            model_id: load_model_documentation(ROOT / "models" / model_id, model["display_name"], repository=ROOT,
                is_genai=any(c["interface"]["format"] == "huggingface" for c in model["components"]))
            for model_id, model in models.items()
        }
        revisions = [_revision(models, matrix, documentation, compiler, day, sha) for day, sha in REVISIONS]
        runs = sorted((run for revision in revisions for run in revision["runs"]), key=lambda run: run["created_at"])
        runs = _old_layout_runs(runs[0]) + runs
        return {"models": models, "compiler": compiler, "revisions": revisions, "runs": runs[::-1]}

# Compile runs older Models revisions wrote, keyed by the field they lack; build history hides them.
OLD_LAYOUT_RUN_IDS = {
    "build_profile_id": "00000000-0000-4000-8000-0000000000a1",
    "compile_timing": "00000000-0000-4000-8000-0000000000a2",
}


def _old_layout_runs(run: dict) -> list[dict]:
    day = REVISIONS[0][0]
    return [{
        **run, "id": run_id, "created_at": f"{day}T00:00:00+00:00",
        "started_at": f"{day}T00:00:00+00:00", "finished_at": f"{day}T00:00:42+00:00",
        "metadata": {**{key: value for key, value in run["metadata"].items() if key != field}, "github_run_id": run_id},
    } for field, run_id in OLD_LAYOUT_RUN_IDS.items()]


def _documentation_receipt(model_id: str, documentation: dict[str, Any], sha: str) -> dict[str, Any]:
    """The catalog documentation receipt the registry publishes for a model README and hero."""
    hero = documentation["hero"]
    return {
        "format": "markdown",
        "overview_markdown": documentation["overview_markdown"],
        "run_id": _id("documentation", sha, model_id),
        "artifact_name": "readme",
        "artifact_sha256": documentation["readme_sha256"],
        "size_bytes": Path(documentation["path"]).stat().st_size,
        "media_type": "text/markdown; charset=utf-8",
        "source_path": documentation["source_path"],
        "source_commit": sha,
        "hero": {
            "artifact_name": hero["artifact_name"],
            "markdown_path": hero["markdown_path"],
            "alt_text": hero["alt_text"],
            "artifact_sha256": hero["sha256"],
            "size_bytes": Path(hero["path"]).stat().st_size,
            "media_type": hero["media_type"],
            "source_path": hero["source_path"],
        },
    }


def _published_builds(models: dict, matrix: list[dict], sha: str) -> dict[str, Any]:
    """A previous catalog whose builds match every compile fingerprint, as if each invocation had been published.

    An ONNX model moved to on-demand builds keeps the benchmarks of its earlier automatic builds, as the registry
    carries them forward; an on-demand LLiMa model was never benchmarked.
    """
    builds: dict[str, list[dict]] = {model_id: [] for model_id in models}
    packages: dict[str, list[dict]] = {model_id: [] for model_id in models}
    benchmarks: dict[str, dict[str, dict[str, str]]] = {model_id: {} for model_id in models}
    for entry in matrix:
        profile_builds = {}
        for component_id in entry["component_ids"]:
            build = {
                "build_profile_id": entry["build_profile_id"],
                "component_id": component_id,
                "compile_fingerprint": entry["compile_fingerprint"],
                "compile_invocation_id": entry["compile_invocation_id"],
                "run_id": _id("compile", sha, entry["model_id"], entry["build_profile_id"], component_id),
                "artifact_sha256": _sha256("mpk", sha, entry["model_id"], entry["build_profile_id"], component_id),
            }
            builds[entry["model_id"]].append(build)
            profile_builds[component_id] = build
        model = models[entry["model_id"]]
        if (entry.get("build_on_demand") and entry["stages"]
                and model["components"][0]["interface"]["format"] == "onnx"):
            benchmarks[entry["model_id"]][entry["build_profile_id"]] = {
                stage: benchmark_fingerprint(ROOT, model, entry["build_profile_id"], profile_builds, stage)
                for stage in entry["stages"]
            }
    for model_id, model in models.items():
        for declaration in model["packages"]:
            for profile_id in declaration["build_profiles"]:
                parts = (sha, model_id, profile_id, declaration["id"])
                manifest = _sha256("package-manifest", *parts)
                packages[model_id].append({
                    "package_id": declaration["id"],
                    "build_profile_id": profile_id,
                    "run_id": _id("package", *parts),
                    "artifact_sha256": _sha256("package", *parts),
                    "size_bytes": 20_000_000 + int(_fraction("package-size", *parts) * 180_000_000),
                    "package_fingerprint": manifest,
                    "component_run_ids": {
                        component_id: _id("compile", sha, model_id, profile_id, component_id)
                        for component_id in declaration["components"]
                    },
                })
    return {"models": [
        {"id": model_id, "builds": builds[model_id], "packages": packages[model_id],
         **({"benchmark_fingerprints": benchmarks[model_id]} if benchmarks[model_id] else {})}
        for model_id in models
    ]}


def _revision(
    models: dict, matrix: list[dict], documentation: dict, compiler: dict, day: str, sha: str
) -> dict[str, Any]:
    document = build_catalog_snapshot(
        repository=ROOT,
        branch=BRANCH,
        git_sha=sha,
        previous=_published_builds(models, matrix, sha),
        documentation_receipts={
            model_id: _documentation_receipt(model_id, entry, sha) for model_id, entry in documentation.items()
        },
    )
    runs = []
    for model in document["models"]:
        for build in model["builds"]:
            runs.append(_compile_run(models[model["id"]], model, build, compiler, day, sha, len(runs)))
        for package in model["packages"]:
            runs.append(_package_run(model, package, compiler, day, sha, len(runs)))
    return {"sha": sha, "document": document, "runs": runs}


def _model_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """Run and model-list fields the registry derives from a catalog entry's categories."""
    categories = entry["categories"]
    return {
        "model_type": categories["model_type"],
        "architecture_id": categories["architecture"]["id"],
        "architecture_paradigm": categories["architecture"]["paradigm"],
        "tasks": categories["tasks"],
        "input_modalities": categories["modalities"]["inputs"],
        "output_modalities": categories["modalities"]["outputs"],
        "model_tags": categories["tags"],
    }


def _run(entry: dict, run_id: str, run_type: str, compiler: dict, day: str, sha: str, position: int) -> dict:
    created = _timestamp(day, position)
    return {
        "id": run_id,
        "repository": "sima-neat/models",
        "run_type": run_type,
        "commit_sha": sha,
        "git_ref": GIT_REF,
        "status": "completed",
        "fingerprint_version": 1,
        "run_fingerprint": _sha256("run", run_id),
        "created_at": created,
        "started_at": created,
        "finished_at": created.replace(":00+00:00", ":42+00:00"),
        "toolchain_name": "sima-model-compiler",
        "toolchain_version": compiler["version"],
        "target_platform": "modalix",
        **_model_fields(entry),
        "model_categories": entry["categories"],
        "metadata": {"model_id": entry["id"], "license": entry["license"], "github_run_id": _id("workflow", sha)},
    }


def _compile_run(model: dict, entry: dict, build: dict, compiler: dict, day: str, sha: str, position: int) -> dict:
    profile, _ = profile_component_for(model, build["build_profile_id"], build["component_id"])
    run = _run(entry, build["run_id"], "model_compile", compiler, day, sha, position)
    run["target_platform"] = profile["target"]
    run["metadata"].update({
        "build_profile_id": build["build_profile_id"],
        "component_id": build["component_id"],
        "compile_invocation_id": build["compile_invocation_id"],
        "compile_timing": {
            "scope": "compile_recipe", "started_at": run["started_at"], "finished_at": run["finished_at"],
        },
        "artifact_kind": "llima" if next(
            item for item in model["components"] if item["id"] == build["component_id"]
        )["interface"]["format"] == "huggingface" else "mpk",
    })
    return run


def _package_run(entry: dict, package: dict, compiler: dict, day: str, sha: str, position: int) -> dict:
    run = _run(entry, package["run_id"], "model_package", compiler, day, sha, position)
    run["metadata"].update({
        "build_profile_id": package["build_profile_id"],
        "package_id": package["package_id"],
        "component_run_ids": package["component_run_ids"],
    })
    return run


def _models_list(document: dict, runs: dict[str, dict], compiler: str) -> list[dict]:
    """The registry's /models entries: catalog entries whose builds carry their compile run's fields."""
    entries = []
    for model in document["models"]:
        builds = [{
            **build,
            **{key: runs[build["run_id"]][key] for key in (
                "status", "commit_sha", "created_at", "finished_at", "toolchain_name", "toolchain_version",
                "target_platform",
            )},
        } for build in model["builds"]]
        if compiler and not any(build["toolchain_version"] == compiler for build in builds):
            continue
        entries.append({
            "model_id": model["id"],
            **{key: model[key] for key in ("display_name", "description", "maturity", "license", "categories")},
            **_model_fields(model),
            "documentation": model["documentation"],
            **({"build_on_demand": True} if model.get("build_on_demand") else {}),
            **({"benchmark_fingerprints": model["benchmark_fingerprints"]}
               if model.get("benchmark_fingerprints") else {}),
            "catalog_revision": document["catalog_revision"],
            "taxonomy_sha256": document["taxonomy_sha256"],
            "builds": builds,
            "pending_builds": [],
            "packages": model["packages"],
        })
    return entries


def _component_source(model: dict, component_id: str) -> dict[str, Any]:
    return next(component for component in model["components"] if component["id"] == component_id)["source"]


def _provenance(model: dict, build: dict, compiler: dict) -> dict[str, Any]:
    profile, binding = profile_component_for(model, build["build_profile_id"], build["component_id"])
    component = next(item for item in model["components"] if item["id"] == build["component_id"])
    source = component["source"]
    return {
        "inputs": [{
            "name": "source-component",
            "input_type": source.get("media_type", "application/onnx").removeprefix("application/"),
            "source_reference": source.get("uri"),
            "sha256": source.get("sha256"),
            "size_bytes": source.get("size_bytes"),
            "media_type": source.get("media_type"),
            "metadata": {
                "model_id": model["id"],
                "component_id": build["component_id"],
                "source_type": source.get("type"),
                "model_card": model["upstream"].get("model_card"),
            },
        }],
        "parameters": {"parameters": {
            "schema_version": 2,
            "model_id": model["id"],
            "build_profile_id": build["build_profile_id"],
            "compile_invocation_id": build["compile_invocation_id"],
            "component_id": build["component_id"],
            "target": profile["target"],
            "compiler_version": compiler["version"],
            "compiler_label": compiler["label"],
            "source_sha256": source.get("sha256"),
            "build_fingerprint": build["compile_fingerprint"],
            "recipe": component.get("recipe"),
            "compile_parameters": binding.get("parameters", {}),
            "calibration": binding.get("calibration"),
        }},
    }


@cache
def _report_template(stage: str) -> dict[str, Any]:
    # Performance is a ResNet-50 DevKit Core benchmark converted to a model report, so board telemetry keeps
    # real proportions.
    return json.loads((BROWSER / "fixtures" / f"model-report-{stage}.json").read_text(encoding="utf-8"))


# Synthetic compiler-style names cover the ten recognized families in one profiling panel.
# The first ten entries show one of each; nested transforms and helpers exercise
# classification without substituting measured data or model-specific graph metadata.
_PREVIEW_LAYER_NAMES = (
    "MLA_{stage}/conv2d_add_relu_0",
    "MLA_{stage}/swish_1",
    "MLA_{stage}/reshape_2/reshape_channel/pad_c",
    "MLA_{stage}/tuple_einsum_3",
    "MLA_{stage}/max_pool2d_4",
    "MLA_{stage}/add_relu_5",
    "MLA_{stage}/softmax_6",
    "MLA_{stage}/image.resize2d_7/0",
    "MLA_{stage}/sum_8",
    "MLA_{stage}/conv2d_add_9_output",
    "MLA_{stage}/multiply_10",
    "MLA_{stage}/reshape_11/reshape_spatial",
    "MLA_{stage}/reshape_12/reshape_channel/split_and_concat",
    "MLA_{stage}/reshape_13",
    "MLA_{stage}/reshape_14",
    "MLA_{stage}/reshape_15",
    "MLA_{stage}/reshape_16",
    "MLA_{stage}/reshape_17",
    "slice_MLA_{stage}/transpose_20_0_MLA_{stage}/tuple_einsum_3_0",
    "concat_MLA_{stage}/softmax_6_MLA_{stage}/transpose_20_1",
    "MLA_{stage}/transpose_20",
    "MLA_{stage}/copy_21",
    "MLA_{stage}/conv2d_add_22_output",
    "MLA_{stage}/copy_23",
)
# Nonconsecutive, as when an A65 stage runs between two MLA stages.
_PREVIEW_MLA_STAGES = (0, 2)


def _components_in_order(model: dict) -> list[str]:
    """The model's components in the order its pipeline runs them."""
    pipeline = [stage["component"] for stage in model.get("pipeline", []) if "component" in stage]
    return pipeline or [component["id"] for component in model["components"]]


def _performance_report(model: dict, profile_id: str) -> dict[str, Any]:
    """A whole-model performance report: the template's DevKit readings at a latency of this build's own.

    A multi-component model, such as RF-DETR, reports each component's layers and execution plan in the
    order its pipeline runs them, and names that pipeline as its workload.
    """
    report = copy.deepcopy(_report_template("performance"))
    # Readings vary per build profile of the whole model.
    key = (model["id"], profile_id, "model")
    metrics = report["metrics"]
    latency = round(2.5 + _fraction("latency", *key) * 40.0, 3)
    scale = latency / metrics["latency_avg_ms"]
    # Pipelining makes throughput exceed one input per mean latency; keep the template's ratio.
    metrics["throughput_fps"] = round(metrics["throughput_fps"] / scale, 1)
    for name in ("latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
        metrics[name] = round(metrics[name] * scale, 3)
    metrics["energy_per_inference_joules"] = metrics["avg_power_watts"] / metrics["throughput_fps"]
    memory = report["telemetry"]["mla_memory_mib"]
    memory_scale = (16.0 + _fraction("memory", *key) * 240.0) / memory["maximum"]
    memory.update({name: round(memory[name] * memory_scale, 1) for name in ("mean", "minimum", "maximum")})
    metrics.update({"mla_memory_avg_mib": memory["mean"], "mla_memory_peak_mib": memory["maximum"]})
    components = _components_in_order(model)
    if len(components) > 1:
        stages = model.get("pipeline") or [{"component": component} for component in components]
        workload = ", ".join(stage.get("component") or f"{stage['cpu']} on the CPU" for stage in stages)
        report["workload"] = workload[:1].upper() + workload[1:]
    plan = report["components"]["model"]["execution_plan"]
    # Illustrative compiled layers exercise filtering, no-ops and nonconsecutive MLA stage indices.
    # Preview timings are generated, not published measurements of these model definitions.
    report["components"] = {}
    for component in components:
        layers = [
            {"stage_index": stage,
             "name": name.format(stage=stage),
             "latency_us": 0.0 if index in (21, 23) else round(
                 latency / len(components) * 1000 * (0.065 / (1 + index / 4))
                 * (0.8 + _fraction("layer", *key, component, str(index))), 2)}
            for stage in _PREVIEW_MLA_STAGES for index, name in enumerate(_PREVIEW_LAYER_NAMES)
        ]
        # A stage's measured latency also holds the time between its layers.
        stages = [{"stage_index": stage, "latency_us": round(1.05 * sum(
            layer["latency_us"] for layer in layers if layer["stage_index"] == stage), 2)}
            for stage in _PREVIEW_MLA_STAGES]
        report["components"][component] = {
            "execution_plan": plan,
            "mla_layer_stats": {"status": "ok", "layers": layers},
            "mla_stage_stats": {"status": "ok", "stages": stages},
        }
    if model.get("performance", {}).get("scope") == "components":
        for index, (name, component) in enumerate(report["components"].items(), 1):
            measured = copy.deepcopy(metrics)
            measured["throughput_fps"] *= index
            for key in ("latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"):
                measured[key] /= index
            measured["energy_per_inference_joules"] = measured["avg_power_watts"] / measured["throughput_fps"]
            component.update(metrics=measured, telemetry=copy.deepcopy(report["telemetry"]),
                             workload=f"Independent {name.replace('_', ' ')} benchmark; illustrative preview data")
        for key in ("metrics", "telemetry", "workload"):
            del report[key]
        report["scope"] = "components"
    # The registry stores the report as PostgreSQL jsonb, which orders keys shorter first, then bytewise.
    report["components"] = dict(sorted(report["components"].items(), key=lambda item: (len(item[0]), item[0])))
    # The rest of the board telemetry keeps the template's DevKit readings.
    return report


def _accuracy_report(model: dict, profile_id: str) -> dict[str, Any]:
    """A whole-model accuracy report on each dataset the model declares, for the metrics its tests expect there,
    else the ones the dataset requires.

    Each metric lands a little worse than the source model's reference score on that dataset, when there is one:
    below it, or above it for an error metric such as EPE. One profile in two evaluates a sampled subset of each
    split, so both coverages appear.
    """
    template = _report_template("accuracy")
    sample = next(iter(template["datasets"].values()))
    key = (model["id"], profile_id)
    tests = load_model_tests(ROOT / "models" / model["id"])
    expected = tests.get("accuracy", {}).get("expected", {}).get(profile_id, {})
    datasets = {}
    for name, dataset in model["accuracy"]["datasets"].items():
        # The FP32 baseline, measured on the package's own inputs, is the closer reference: it wins over a
        # published score of the same metric.
        entries = [entry for entry in model.get("reference_metrics", []) if entry["dataset"]["id"] == dataset["id"]]
        entries.sort(key=lambda entry: entry.get("origin") == "fp32_baseline")
        reference = {metric: value for entry in entries for metric, value in entry["metrics"].items()}
        available = 500 + int(_fraction("samples", *key, name) * 4500)
        datasets[name] = {
            **copy.deepcopy(sample),
            "evaluated_samples": available if _fraction("coverage", *key) < 0.5 else available // 5,
            "available_samples": available,
            "metrics": {
                metric: round(reference.get(metric, 0.3 + _fraction("reference", *key, metric) * 0.6)
                              * _worse(metric, _fraction("accuracy", *key, name, metric)), 4)
                for metric in expected.get(name) or dataset["required_metrics"]
            },
        }
    return {**copy.deepcopy(template), "datasets": datasets}


def _worse(metric: str, fraction: float) -> float:
    """The factor that makes a measured score 2-10% worse than its reference, in the metric's direction."""
    shortfall = 0.02 + (1 - fraction) * 0.08
    return 1 + shortfall if METRICS[metric]["direction"] == "lower" else 1 - shortfall


def _benchmark_parameters(model: dict, benchmark: dict) -> dict[str, Any]:
    """The run parameters report publication stores: the stage's declaration, the builds it benchmarked, and the
    runtime it ran on."""
    return {
        benchmark["stage"]: model[benchmark["stage"]],
        "builds": {build["component_id"]: {"run_id": build["run_id"], "artifact_sha256": build["artifact_sha256"]}
                   for build in benchmark["builds"]},
        "runtime": resolve_benchmark_runtime(ROOT),
    }


def _mla_stage_simulation_test(model: dict, package: dict) -> dict[str, Any] | None:
    """The MLA stage simulation a package run carries: each preview stage at a fraction of its measured latency.
    A LLiMa build has no MPK to simulate."""
    if any(component["interface"]["format"] == "huggingface" for component in model["components"]):
        return None
    profile_id = package["build_profile_id"]
    report = _performance_report(model, profile_id)
    components = {}
    for component, stats in report["components"].items():
        stages = []
        for stage in stats["mla_stage_stats"]["stages"]:
            key = (model["id"], profile_id, component, str(stage["stage_index"]))
            effective_tops = 4.0 + _fraction("tops", *key) * 20.0
            effective_ops = round(stage["latency_us"] * 1e-6 * effective_tops * 1e12)
            stages.append({
                "stage_index": stage["stage_index"],
                "effective_ops": effective_ops,
                "total_ops": round(effective_ops / (0.7 + _fraction("packing", *key) * 0.25)),
                "simulated_latency_us": round(stage["latency_us"] * (0.8 + _fraction("simulated", *key) * 0.18), 3),
            })
        components[component] = {"status": "ok", "stages": stages}
    return {
        "suite": "performance",
        "name": "mla-stage-simulation",
        "status": "passed",
        "version": "1",
        "summary": {"schema_version": 1, "model_id": model["id"], "build_profile_id": profile_id,
                    "components": components},
    }


def _fidelity_test(model: dict, package: dict) -> dict[str, Any] | None:
    """The numerical fidelity a package run carries when model.yaml declares it."""
    config = model.get("numerical_fidelity")
    if config is None:
        return None
    key = (model["id"], package["build_profile_id"])
    return {
        "suite": "correctness",
        "name": "numerical-fidelity",
        "status": "passed",
        "version": "1",
        "summary": {
            "schema_version": 1,
            "model_id": model["id"],
            "build_profile_id": package["build_profile_id"],
            "status": config["status"],
            "reference": "Original ONNX (ONNX Runtime)",
            "candidate": "Quantized AFE network (JAX execute)",
            "dataset": {field: config["dataset"][field] for field in ("id", "variant", "sha256")},
            "selection": config["selection"],
            "samples": config["selection"]["samples"],
            "metrics": [{
                "name": output,
                "cosine_similarity": round(0.99 + _fraction("cosine", *key, output) * 0.0099, 6),
                "normalized_rmse": round(_fraction("rmse", *key, output) * 0.02, 5),
                "mean_absolute_error": round(_fraction("mae", *key, output) * 0.01, 5),
                "max_absolute_error": round(_fraction("max", *key, output) * 0.05, 5),
            } for output in config["outputs"]],
        },
    }


_UNBENCHMARKED = {("yolo26n", "w16a16")}


@cache
def _index() -> dict[str, Any]:
    """Lookups from every identifier the API serves to what it describes."""
    repository = _repository()
    runs = {run["id"]: run for run in repository["runs"]}
    builds, packages, documentation, benchmarks = {}, {}, {}, {}
    for revision in repository["revisions"]:
        for model in revision["document"]["models"]:
            source = repository["models"][model["id"]]
            documentation[model["documentation"]["run_id"]] = model["id"]
            for build in model["builds"]:
                builds[build["run_id"]] = (source, build)
            for package in model["packages"]:
                packages[package["run_id"]] = (source, package)
            # Each stage of a profile the catalog lists a benchmark fingerprint for is one DevKit run, except one
            # the DevKit has not benchmarked yet, which the browser shows as not measured.
            for profile_id, fingerprints in model.get("benchmark_fingerprints", {}).items():
                for stage, fingerprint in fingerprints.items():
                    if (profile_id not in benchmarked_profiles(source, stage)
                            or (model["id"], profile_id) in _UNBENCHMARKED):
                        continue
                    profile_builds = [build for build in model["builds"] if build["build_profile_id"] == profile_id]
                    package = next((item for item in model["packages"] if item["build_profile_id"] == profile_id), None)
                    benchmarks[fingerprint] = {
                        "model": source, "profile": profile_id, "stage": stage, "builds": profile_builds,
                        "parent_run_id": (package or profile_builds[0])["run_id"],
                    }
    return {
        "runs": runs,
        "builds": builds,
        "packages": packages,
        "documentation": documentation,
        "benchmarks": benchmarks,
        "benchmark_run_fingerprints": {
            _id(entry["stage"], fingerprint): fingerprint for fingerprint, entry in benchmarks.items()
        },
    }


def _benchmark_run(run_id: str) -> dict:
    """A DevKit run that benchmarked a profile at the latest revision, as CI benchmarks a reused build, under the
    profile's package run."""
    fingerprint = _index()["benchmark_run_fingerprints"][run_id]
    benchmark = _index()["benchmarks"][fingerprint]
    workflow = int(_fraction("workflow", run_id) * 10**11)
    return {
        **_index()["runs"][benchmark["parent_run_id"]],
        "id": run_id,
        "run_type": BENCHMARK_RUN_TYPES[benchmark["stage"]],
        "parent_run_id": benchmark["parent_run_id"],
        "run_fingerprint": fingerprint,
        "commit_sha": _repository()["revisions"][-1]["sha"],
        "ci_run_url": f"https://github.com/sima-neat/models/actions/runs/{workflow}",
        "metadata": {
            "model_id": benchmark["model"]["id"], "build_profile_id": benchmark["profile"],
            "stage": benchmark["stage"], "package_run_id": benchmark["parent_run_id"],
        },
    }


def _deployment_artifacts(archive: dict, run_id: str) -> list[dict]:
    """Native deployments use a small JSON index and separate path-preserving payload files."""
    paths = ["model.json", "weights/model.bin", "runtime/config.json"]
    total = archive["size_bytes"]
    files = [{
        "name": f"file-{hashlib.sha256(path.encode()).hexdigest()}",
        "artifact_type": "model-file", "media_type": "application/octet-stream",
        "sha256": _sha256("deployment-file", run_id, path),
        "size_bytes": total // len(paths) + (total % len(paths) if index == 0 else 0),
        "metadata": {"path": path},
    } for index, path in enumerate(paths)]
    return [{
        **archive, "name": "package" if archive["artifact_type"] == "model-package" else "llima",
        "size_bytes": 2048, "media_type": "application/json",
        "metadata": {**archive.get("metadata", {}), "layout": "directory", "storage_backend": "s3",
                     "file_count": len(files), "deployment_size_bytes": total},
    }, *files]


class PreviewHandler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self, code: str) -> None:
        self._json({"error": {"code": code, "message": "Not found in the local preview"}}, 404)

    def _bytes(self, body: bytes, media_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if re.fullmatch(r"/config(\.[0-9a-f]+)?\.js", path):
            self._bytes(
                b"window.MODEL_REGISTRY_CONFIG = Object.freeze({"
                b'apiBase: location.origin, environment: "Local preview", '
                + f'uiMode: "{UI_MODE}", '.encode()
                + b'repository: Object.freeze({organization: "sima-neat", name: "models"})});\n',
                "text/javascript",
            )
        elif path == "/reference-metrics.json":
            self._json(reference_metrics_by_model(_repository()["models"]))
        elif path == "/component-pipelines.json":
            # The unbuilt sources request this name; an assembled portal requests the file it ships.
            self._json(component_pipelines(_repository()["models"]))
        elif path.startswith(f"{REPOSITORY_API}/"):
            self._repository_api(path.removeprefix(REPOSITORY_API), query)
        elif path.startswith("/v1/runs/"):
            self._run_api(path.strip("/").split("/")[2:])
        else:
            super().do_GET()

    def _repository_api(self, path: str, query: dict[str, list[str]]) -> None:
        repository = _repository()
        current = repository["revisions"][-1]
        git_ref = query.get("git_ref", [GIT_REF])[0]
        if path == "/branches":
            runs = repository["runs"]
            self._json({"branches": [{
                "name": BRANCH, "git_ref": GIT_REF, "latest_run_at": runs[0]["created_at"], "run_count": len(runs),
            }]})
        elif path == "/runs":
            run_type = query.get("run_type", [""])[0]
            runs = [run for run in repository["runs"] if run["git_ref"] == git_ref and (
                run["run_type"] == run_type if run_type else run["run_type"] == "model_compile"
            )]
            self._json({"runs": runs, "has_more": False, "next_cursor": None})
        elif path == "/models" and git_ref == GIT_REF:
            compiler = query.get("toolchain_version", [""])[0]
            models = _models_list(current["document"], _index()["runs"], compiler)
            self._json({"models": models, "count": len(models), "has_more": False, "next_cursor": None})
        elif path.startswith("/catalogs/") and git_ref == GIT_REF:
            revision = path.removeprefix("/catalogs/")
            match = current if revision == "current" else next(
                (item for item in repository["revisions"] if item["sha"] == revision), None
            )
            if match is None:
                self._not_found("catalog_not_found")
            else:
                self._json({"catalog": {
                    "git_ref": GIT_REF, "catalog_revision": match["sha"], "document": match["document"],
                }})
        elif path == "/resolve":
            fingerprint = query.get("fingerprint", [""])[0]
            benchmark = _index()["benchmarks"].get(fingerprint)
            run_type = query.get("run_type", [""])[0]
            if benchmark and BENCHMARK_RUN_TYPES[benchmark["stage"]] == run_type:
                self._json({"run": {"id": _id(benchmark["stage"], fingerprint), "run_type": run_type}})
            else:
                self._not_found("fingerprint_not_found")
        elif path in {"/models", "/resolve"} or path.startswith("/catalogs/"):
            self._not_found("catalog_not_found" if path != "/resolve" else "fingerprint_not_found")
        else:
            self._not_found("not_found")

    def _run_api(self, parts: list[str]) -> None:
        index = _index()
        compiler = _repository()["compiler"]
        run_id, rest = (parts[0], parts[1:]) if parts else ("", [])
        if run_id in index["documentation"] and len(rest) == 3 and rest[0] == "artifacts" and rest[2] == "content":
            model_id = index["documentation"][run_id]
            files = {"readme": ("README.md", "text/markdown; charset=utf-8"), "hero": ("hero.png", "image/png")}
            if rest[1] not in files:
                self._not_found("artifact_not_found")
                return
            name, media_type = files[rest[1]]
            self._bytes((ROOT / "models" / model_id / name).read_bytes(), media_type)
        elif run_id in index["benchmark_run_fingerprints"]:
            self._benchmark_run_api(run_id, rest)
        elif run_id in index["builds"]:
            self._compile_run_api(run_id, rest, *index["builds"][run_id], compiler)
        elif run_id in index["packages"] and not rest:
            model, package = index["packages"][run_id]
            _, media_type = _package_file(model, package)
            # The same artifact shape the registry publisher records for a full package.
            artifact = {
                "name": "package",
                "artifact_type": "model-package",
                "sha256": package["artifact_sha256"],
                "size_bytes": package["size_bytes"],
                "media_type": media_type,
                "metadata": {
                    "model_id": model["id"],
                    "build_profile_id": package["build_profile_id"],
                    "package_id": package["package_id"],
                    "manifest_sha256": package["package_fingerprint"],
                    "component_run_ids": package["component_run_ids"],
                },
            }
            artifacts = _deployment_artifacts(artifact, run_id) if media_type == "application/json" else [artifact]
            self._json({"run": index["runs"][run_id], "artifacts": artifacts})
        elif run_id in index["packages"] and rest == ["test-results"]:
            fidelity = _fidelity_test(*index["packages"][run_id])
            simulation = _mla_stage_simulation_test(*index["packages"][run_id])
            self._json({"run_id": run_id, "test_results": [test for test in (fidelity, simulation) if test]})
        elif run_id in index["packages"] and len(rest) == 2 and rest[0] == "artifacts":
            model, package = index["packages"][run_id]
            filename, _ = _package_file(model, package)
            path = f"{model['id']}/{package['build_profile_id']}/packages/{package['package_id']}/{filename}"
            self._download_link(f"models/develop/{path}")
        elif run_id in index["runs"] and len(rest) == 2 and rest[0] == "artifacts":
            self._download_link()
        else:
            self._not_found("run_not_found")

    def _benchmark_run_api(self, run_id: str, rest: list[str]) -> None:
        benchmark = _index()["benchmarks"][_index()["benchmark_run_fingerprints"][run_id]]
        model, stage = benchmark["model"], benchmark["stage"]
        run = _benchmark_run(run_id)
        if not rest:
            self._json({"run": run, "artifacts": []})
        elif rest == ["provenance"]:
            parameters = _benchmark_parameters(model, benchmark)
            self._json({"run": run, "inputs": [], "parameters": {"parameters": parameters}})
        elif rest == ["test-results"]:
            read = _performance_report if stage == "performance" else _accuracy_report
            report = read(model, benchmark["profile"])
            # The model browser reads the whole report from the stage's model test result.
            self._json({"run_id": run_id, "test_results": [
                {"suite": stage, "name": "model", "status": "passed", "version": "1", "summary": report},
            ]})
        elif len(rest) == 2 and rest[0] == "artifacts":
            self._download_link()
        else:
            self._not_found("not_found")

    def _compile_run_api(self, run_id: str, rest: list[str], model: dict, build: dict, compiler: dict) -> None:
        run = _index()["runs"][run_id]
        native = run["metadata"].get("artifact_kind") == "llima"
        if not rest:
            artifact = {
                "name": "llima" if native else "mpk",
                "artifact_type": "model-pack",
                "sha256": build["artifact_sha256"],
                "size_bytes": 8_000_000 + int(_fraction("mpk-size", run_id) * 300_000_000),
                "media_type": "application/zip" if native else "application/gzip",
            }
            self._json({"run": run, "artifacts": _deployment_artifacts(artifact, run_id) if native else [artifact]})
        elif rest == ["provenance"]:
            self._json({"run": run, **_provenance(model, build, compiler)})
        elif rest == ["test-results"]:
            self._json({"run_id": run_id, "test_results": [
                {"suite": "model", "name": "manifest-validation", "status": "passed", "duration_ms": 82},
                {"suite": "model", "name": "deployment-validation" if native else "mpk-validation",
                 "status": "passed", "duration_ms": 418},
            ]})
        elif rest == ["metrics"]:
            self._json({"run_id": run_id, "measurements": [
                {"name": "compile.duration", "numeric_value": round(180.0 + _fraction("compile", run_id) * 900.0, 1)},
            ]})
        elif len(rest) == 2 and rest[0] == "artifacts":
            filename = mpk_filename(model, build["build_profile_id"], build["component_id"])
            self._download_link(f"models/develop/{model['id']}/{build['build_profile_id']}/{build['component_id']}/{filename}")
        else:
            self._not_found("not_found")

    def _download_link(self, s3_key: str | None = None) -> None:
        self._json({"artifact": {
            "download_url": "/fixtures/model-report-performance.json", "download_url_expires_in": 300,
            **({"s3_key": s3_key} if s3_key else {}),
        }})

    def translate_path(self, path: str) -> str:
        # The fixtures stay in artifact-browser/ when --root serves an assembled portal. Normalizing
        # first resolves "..", so a path that leaves the fixtures is served from the root, or not at all.
        request = posixpath.normpath(unquote(urlparse(path).path))
        if request.startswith("/fixtures/"):
            return str(BROWSER / request.lstrip("/"))
        return super().translate_path(path)

    def log_message(self, message: str, *args: object) -> None:
        print(f"preview: {message % args}")


def _package_file(model: dict, package: dict) -> tuple[str, str]:
    """The published path and media type of a native folder index or ZIP."""
    if any(component["interface"]["format"] == "huggingface" for component in model["components"]):
        return "deployment/manifest.json", "application/json"
    return f"{model['id']}_{package['build_profile_id']}_{package['package_id']}.zip", "application/zip"


def first_variant_run_id() -> str:
    """The compile run of the first model's first build, which the startup link opens."""
    return _repository()["revisions"][-1]["document"]["models"][0]["builds"][0]["run_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=BROWSER, help="directory to serve, such as website/portal-build")
    args = parser.parse_args()
    run_id = first_variant_run_id()
    os.chdir(args.root)
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Model browser preview: http://{args.host}:{args.port}/?run={run_id}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
