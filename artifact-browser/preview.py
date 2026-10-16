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
from functools import cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
sys.path.insert(0, str(ROOT / "tools"))

from model_registry.accuracy_config import find_devkit_accuracy_config  # noqa: E402
from model_registry.compile_matrix import build_matrix  # noqa: E402
from model_registry.dependencies import resolve_model_compiler  # noqa: E402
from model_registry.documentation import load_model_documentation  # noqa: E402
from model_registry.fidelity_config import find_fidelity_config  # noqa: E402
from model_registry.model import component_pipelines, load_models, mpk_filename, profile_component_for  # noqa: E402
from model_registry.registry import build_catalog_snapshot  # noqa: E402

BRANCH = "develop"
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


@cache
def _repository() -> dict[str, Any]:
    models = load_models(ROOT)
    compiler = resolve_model_compiler(ROOT)
    matrix = build_matrix(models, sorted(models), repository=ROOT, compiler_version=compiler["identity"])["include"]
    documentation = {
        model_id: load_model_documentation(ROOT / "models" / model_id, model["display_name"], repository=ROOT)
        for model_id, model in models.items()
    }
    revisions = [_revision(models, matrix, documentation, compiler, day, sha) for day, sha in REVISIONS]
    runs = sorted((run for revision in revisions for run in revision["runs"]), key=lambda run: run["created_at"])
    return {"models": models, "compiler": compiler, "revisions": revisions, "runs": runs[::-1]}


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
    """A previous catalog whose builds match every compile key, as if each invocation had been published."""
    builds: dict[str, list[dict]] = {model_id: [] for model_id in models}
    packages: dict[str, list[dict]] = {model_id: [] for model_id in models}
    for entry in matrix:
        for component_id in entry["component_ids"]:
            builds[entry["model_id"]].append({
                "build_profile_id": entry["build_profile_id"],
                "component_id": component_id,
                "compile_fingerprint": entry["build_fingerprint"],
                "run_id": _id("compile", sha, entry["model_id"], entry["build_profile_id"], component_id),
                "artifact_sha256": _sha256("mpk", sha, entry["model_id"], entry["build_profile_id"], component_id),
            })
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
        {"id": model_id, "builds": builds[model_id], "packages": packages[model_id]} for model_id in models
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
        "artifact_kind": "mpk",
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
            "input_type": source.get("media_type", "application/octet-stream").removeprefix("application/"),
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
            "shape_profile": binding.get("shape_profile"),
            "calibration": binding.get("calibration"),
        }},
    }


@cache
def _benchmark_template() -> dict[str, Any]:
    # A ResNet-50 result imported from a DevKit Core benchmark, so board telemetry keeps real proportions.
    return json.loads((BROWSER / "fixtures" / "performance-result.json").read_text(encoding="utf-8"))


def _benchmark_result(model: dict, build: dict, compiler: dict) -> dict[str, Any]:
    result = copy.deepcopy(_benchmark_template())
    key = (model["id"], build["build_profile_id"], build["component_id"])
    result["model"].update({
        "id": model["id"], "build_profile_id": key[1], "component_id": key[2], "sha256": build["artifact_sha256"],
    })
    result["compatibility"]["actual_platform_version"] = compiler["version"]
    latency = round(2.5 + _fraction("latency", *key) * 40.0, 3)
    power = result["metrics"]["avg_power_watts"]["value"]
    samples = result["configuration"]["samples"]
    # Pipelining makes asynchronous throughput exceed one input per mean latency; keep the template's ratio.
    pipelining = result["metrics"]["throughput_fps"]["value"] * result["metrics"]["latency_avg_ms"]["value"] / 1000.0
    throughput = round(pipelining * 1000.0 / latency, 1)
    values = {
        "latency_avg_ms": latency,
        "throughput_fps": throughput,
        "energy_joules": round(power * samples / throughput, 2),
        "energy_per_inference_joules": round(power / throughput, 5),
        "mla_memory_peak_mib": round(16.0 + _fraction("memory", *key) * 240.0, 1),
    }
    memory = result["metrics"]["mla_memory_peak_mib"]
    values["mla_memory_avg_mib"] = round(
        values["mla_memory_peak_mib"] * result["metrics"]["mla_memory_avg_mib"]["value"] / memory["value"], 1,
    )
    # The rest of the board telemetry keeps the template's DevKit readings.
    for name, value in values.items():
        metric = result["metrics"][name]
        scale = value / metric["value"]
        metric.update({
            "value": value,
            "minimum": metric["minimum"] and round(metric["minimum"] * scale, 5),
            "maximum": metric["maximum"] and round(metric["maximum"] * scale, 5),
        })
    return result


def _accuracy_result(model: dict, build: dict) -> dict[str, Any]:
    """A DevKit accuracy result for the metrics, dataset, and reference the model's accuracy entry declares."""
    config = find_devkit_accuracy_config(model, build["build_profile_id"])
    reference = config.get("reference", {}).get("metrics", {})
    key = (model["id"], build["build_profile_id"])
    metrics = {}
    for name in config["metrics"]:
        expected = reference.get(name, 0.3 + _fraction("reference", *key, name) * 0.6)
        metric = {"value": round(expected * (0.9 + _fraction("accuracy", *key, name) * 0.08), 4)}
        metrics[name] = {**metric, "unit": "ratio", "direction": "higher", **(
            {"reference": reference[name]} if name in reference else {}
        )}
    samples = 500 + int(_fraction("samples", *key) * 4500)
    return {
        "schema_version": 2,
        "model": {
            "id": model["id"],
            "build_profile_id": build["build_profile_id"],
            "component_id": build["component_id"],
            "package_id": None,
            "artifacts": {build["component_id"]: build["artifact_sha256"]},
        },
        "validation": {"id": config["id"], "engine": config["engine"], "status": config["status"]},
        "dataset": {
            **{field: config["dataset"][field] for field in ("id", "variant", "sha256")},
            "split": "validation",
            "evaluated_samples": samples,
            "available_samples": samples,
        },
        "metrics": metrics,
        "execution": {"core_version": "3.0.0", "script_sha256": _sha256("script", config["id"]), "models_commit": ""},
    }


def _fidelity_test(model: dict, build: dict) -> dict[str, Any] | None:
    config = find_fidelity_config(
        ROOT / "models" / model["id"], model,
        build_profile_id=build["build_profile_id"], component_id=build["component_id"],
    )
    if config is None or config["result_component"] != build["component_id"]:
        return None
    key = (model["id"], build["build_profile_id"])
    return {
        "suite": "correctness",
        "name": "numerical-fidelity",
        "status": "passed",
        "version": "1",
        "summary": {
            "schema_version": 1,
            "model_id": model["id"],
            "build_profile_id": build["build_profile_id"],
            "component_id": build["component_id"],
            "scope": "full_pipeline",
            "status": config["status"],
            "reference": "Original ONNX (ONNX Runtime)",
            "candidate": "Quantized AFE network (JAX execute)",
            "dataset": {field: config["dataset"][field] for field in ("id", "variant")},
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


@cache
def _index() -> dict[str, Any]:
    """Lookups from every identifier the API serves to what it describes."""
    repository = _repository()
    runs = {run["id"]: run for run in repository["runs"]}
    builds, packages, documentation, accuracy = {}, {}, {}, {}
    for revision in repository["revisions"]:
        for model in revision["document"]["models"]:
            source = repository["models"][model["id"]]
            documentation[model["documentation"]["run_id"]] = model["id"]
            for build in model["builds"]:
                builds[build["run_id"]] = (source, build)
                if build["validation_fingerprint"]:
                    accuracy[build["validation_fingerprint"]] = (source, build)
            for package in model["packages"]:
                packages[package["run_id"]] = (source, package)
    return {
        "runs": runs,
        "builds": builds,
        "packages": packages,
        "documentation": documentation,
        "accuracy": accuracy,
        "benchmarks": {_id("benchmark", run_id): run_id for run_id in builds},
        "accuracy_runs": {_id("accuracy", key): key for key in accuracy},
    }


def _measurement_run(run_id: str, run_type: str, build_run_id: str) -> dict:
    """A DevKit run that measured a build at the latest revision, as CI measures a reused build."""
    workflow = int(_fraction("workflow", run_id) * 10**11)
    return {
        **_index()["runs"][build_run_id],
        "id": run_id,
        "run_type": run_type,
        "commit_sha": _repository()["revisions"][-1]["sha"],
        "ci_run_url": f"https://github.com/sima-neat/models/actions/runs/{workflow}",
    }


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
                b'repository: Object.freeze({organization: "sima-neat", name: "models"})});\n',
                "text/javascript",
            )
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
        elif path == "/resolve" and query.get("run_type") == ["model_accuracy"]:
            key = query.get("fingerprint", [""])[0]
            if key in _index()["accuracy"]:
                self._json({"run": {"id": _id("accuracy", key), "run_type": "model_accuracy"}})
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
        elif run_id in index["benchmarks"] and not rest:
            run = _measurement_run(run_id, "model_benchmark", index["benchmarks"][run_id])
            self._json({"run": run, "artifacts": []})
        elif run_id in index["accuracy_runs"] and not rest:
            _, build = index["accuracy"][index["accuracy_runs"][run_id]]
            self._json({"run": _measurement_run(run_id, "model_accuracy", build["run_id"]), "artifacts": []})
        elif run_id in index["benchmarks"] and rest == ["test-results"]:
            self._json({"run_id": run_id, "test_results": [
                {"suite": "performance", "name": "model-synthetic", "status": "passed"},
            ]})
        elif run_id in index["accuracy_runs"] and rest == ["accuracy", "latest"]:
            model, build = index["accuracy"][index["accuracy_runs"][run_id]]
            result = _accuracy_result(model, build)
            self._json({
                "accuracy_test": {"name": result["validation"]["id"], "status": "passed", "attempt": 1},
                "result": result,
                "result_artifact_url": None,
            })
        elif run_id in index["builds"]:
            self._compile_run_api(run_id, rest, *index["builds"][run_id], compiler)
        elif run_id in index["packages"] and not rest:
            model, package = index["packages"][run_id]
            _, media_type = _package_file(model, package)
            # The same artifact shape the registry publisher records for a full package.
            self._json({"run": index["runs"][run_id], "artifacts": [{
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
            }]})
        elif run_id in index["packages"] and len(rest) == 2 and rest[0] == "artifacts":
            model, package = index["packages"][run_id]
            filename, _ = _package_file(model, package)
            path = f"{model['id']}/{package['build_profile_id']}/packages/{package['package_id']}/{filename}"
            self._download_link(f"models/develop/{path}")
        elif run_id in index["runs"] and len(rest) == 2 and rest[0] == "artifacts":
            self._download_link()
        else:
            self._not_found("run_not_found")

    def _compile_run_api(self, run_id: str, rest: list[str], model: dict, build: dict, compiler: dict) -> None:
        run = _index()["runs"][run_id]
        if not rest:
            self._json({"run": run, "artifacts": [{
                "name": "mpk",
                "artifact_type": "model-pack",
                "sha256": build["artifact_sha256"],
                "size_bytes": 8_000_000 + int(_fraction("mpk-size", run_id) * 300_000_000),
                "media_type": "application/gzip",
            }]})
        elif rest == ["provenance"]:
            self._json({"run": run, **_provenance(model, build, compiler)})
        elif rest == ["test-results"]:
            fidelity = _fidelity_test(model, build)
            self._json({"run_id": run_id, "test_results": [
                {"suite": "model", "name": "manifest-validation", "status": "passed", "duration_ms": 82},
                {"suite": "model", "name": "mpk-validation", "status": "passed", "duration_ms": 418},
                *([fidelity] if fidelity else []),
            ]})
        elif rest == ["metrics"]:
            self._json({"run_id": run_id, "measurements": [
                {"name": "compile.duration", "numeric_value": round(180.0 + _fraction("compile", run_id) * 900.0, 1)},
            ]})
        elif rest == ["benchmarks", "latest"] and build["benchmark_fingerprint"]:
            self._json({
                "benchmark_run": {"id": _id("benchmark", run_id)},
                "result": _benchmark_result(model, build, compiler),
                "result_artifact_url": "/fixtures/performance-result.json",
            })
        elif rest == ["benchmarks", "latest"]:
            self._not_found("benchmark_not_found")
        elif rest == ["accuracy", "latest"]:
            # Accuracy lives on its own model_accuracy run, resolved by the build's accuracy key.
            self._not_found("accuracy_not_found")
        elif len(rest) == 2 and rest[0] == "artifacts":
            filename = mpk_filename(model, build["build_profile_id"], build["component_id"])
            self._download_link(f"models/develop/{model['id']}/{build['build_profile_id']}/{build['component_id']}/{filename}")
        else:
            self._not_found("not_found")

    def _download_link(self, s3_key: str | None = None) -> None:
        self._json({"artifact": {
            "download_url": "/fixtures/performance-result.json", "download_url_expires_in": 300,
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
    """The published filename and media type of a package: its lone MPK, or a ZIP of everything."""
    declaration = next(item for item in model["packages"] if item["id"] == package["package_id"])
    if len(declaration["components"]) == 1 and not declaration["resources"]:
        return mpk_filename(model, package["build_profile_id"], declaration["components"][0]), "application/gzip"
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
