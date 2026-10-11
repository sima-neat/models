#!/usr/bin/env python3
"""Serve the artifact browser with a realistic local benchmark API fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
MODELS = ROOT / "models"
RUN_ID = "74b48a12-6088-42a0-b881-5925a5272fae"
COMMIT = "244c03323cefc96a6baaa097f0a4bb008e9c39ae"
PREVIEW_RUNS = (
    ("resnet_50", "modalix_int8", "model", "main", "2.1.3", 165_482_104),
    ("resnet_50", "modalix_bf16", "model", "main", "2.1.3", 284_103_221),
    ("ssd_mobilenet_v1", "modalix_bf16_tess_mla", "model", "develop", "2.1.3", 132_714_990),
    ("ssd_mobilenet_v1", "modalix_int8_tess_mla", "model", "develop", "2.1.3", 78_204_118),
    ("ssd_mobilenet_v2", "modalix_bf16_tess_mla", "model", "develop", "2.1.3", 148_821_633),
    ("ssd_mobilenet_v2", "modalix_int8_tess_off_mla", "model", "develop", "2.1.3", 86_902_412),
    ("ssd_mobilenet_v3", "modalix_bf16_tess_mla", "model", "develop", "2.1.3", 119_443_210),
    ("yolo_v8n", "modalix_int8", "model", "main", "2.1.3", 41_883_092),
    ("yolo_v8s", "modalix_int8", "model", "main", "2.1.3", 96_771_405),
    ("depth_anything_v2_vits", "modalix_bf16", "model", "develop", "2.1.3", 207_614_992),
    ("fcn_hrnet48", "modalix_bf16", "model", "develop", "2.1.3", 312_045_118),
    ("midas_v21_small", "modalix_bf16", "model", "main", "2.1.2", 91_352_776),
    ("detr_resnet50", "modalix_bf16", "model", "develop", "2.1.3", 251_770_364),
    (
        "resnet_50",
        "modalix_bf16_batch4",
        "model",
        "main",
        "2.1.3",
        296_411_087,
    ),
    ("rf_detr_small", "modalix_int8_bf16", "backbone", "main", "2.1.3", 104_312_000),
    ("rf_detr_small", "modalix_int8_bf16", "transformer", "main", "2.1.3", 168_774_000),
    # An older build of the same variant, so build history shows two builds of two components each.
    ("rf_detr_small", "modalix_int8_bf16", "backbone", "main", "2.1.2", 104_298_000),
    ("rf_detr_small", "modalix_int8_bf16", "transformer", "main", "2.1.2", 168_701_000),
)

PREVIEW_TASKS = {
    "resnet_50": "image_classification",
    "depth_anything_v2_vits": "depth_estimation",
    "midas_v21_small": "depth_estimation",
    "fcn_hrnet48": "semantic_segmentation",
}
# Names and descriptions for fixture models that have no models/<id>/model.yaml.
PREVIEW_ONLY_MODELS = {
    "yolo_v8n": ("YOLOv8n", "The smallest YOLOv8 detector for COCO objects."),
    "yolo_v8s": ("YOLOv8s", "A small YOLOv8 detector for COCO objects."),
    "depth_anything_v2_vits": ("Depth Anything V2 Small", "Estimates relative depth from a single image."),
    "fcn_hrnet48": ("FCN HRNet-W48", "Labels every pixel with one of the Cityscapes classes."),
    "midas_v21_small": ("MiDaS v2.1 Small", "A compact monocular depth estimator."),
    "detr_resnet50": ("DETR ResNet-50", "A transformer detector for COCO objects."),
}


def _build_index(index: int) -> int:
    """The first run of the same model, profile, branch, and compiler; its components share one workflow."""
    model_id, variant_id, _, branch, compiler, _ = PREVIEW_RUNS[index]
    return next(
        position for position, other in enumerate(PREVIEW_RUNS)
        if (other[0], other[1], other[3], other[4]) == (model_id, variant_id, branch, compiler)
    )


def _read_result() -> dict:
    return json.loads((BROWSER / "fixtures" / "performance-result-v4.json").read_text(encoding="utf-8"))


def _run(index: int = 0) -> dict:
    model_id, variant_id, component_id, branch, compiler, _ = PREVIEW_RUNS[index]
    run_id = RUN_ID if index == 0 else f"74b48a12-6088-42a0-b881-5925a5272f{index:02x}"
    build = _build_index(index)
    commit = COMMIT if build == 0 else hashlib.sha1(f"preview-commit-{build}".encode()).hexdigest()
    architecture = "resnet_50" if "resnet" in model_id else model_id
    task = PREVIEW_TASKS.get(model_id, "object_detection")
    return {
        "id": run_id,
        "repository": "sima-neat/models",
        "run_type": "model_compile",
        "commit_sha": commit,
        "git_ref": f"refs/heads/{branch}",
        "status": "completed",
        "created_at": f"2026-08-{26 - index // 5:02d}T{19 - index % 5:02d}:30:00Z",
        "started_at": f"2026-08-{26 - index // 5:02d}T{19 - index % 5:02d}:30:08Z",
        "finished_at": f"2026-08-{26 - index // 5:02d}T{19 - index % 5:02d}:38:44Z",
        "run_fingerprint": hashlib.sha256(f"preview-run-{index}".encode()).hexdigest(),
        "toolchain_name": "model-compiler",
        "toolchain_version": compiler,
        "target_platform": "modalix",
        "metadata": {
            "model_id": model_id,
            "build_profile_id": variant_id,
            "component_id": component_id,
            "github_run_id": f"preview-workflow-{build}",
            "model_name": model_id,
            "license": {
                "spdx": "Apache-2.0",
                "redistribution": "allowed",
                "attribution": "Derived from microsoft/resnet-50 and distributed under Apache-2.0.",
            },
        },
        "model_type": "vision",
        "architecture_id": architecture,
        "architecture_paradigm": "cnn",
        "tasks": [task],
        "input_modalities": ["image"],
        "output_modalities": ["classification" if task == "image_classification" else "bounding_boxes"],
        "model_tags": ["reference"] if index == 0 else [],
        "model_categories": {
            "model_type": "vision",
            "tasks": [task],
            "architecture": {"id": architecture, "paradigm": "cnn"},
        },
    }


def _runs() -> list[dict]:
    return [_run(index) for index in range(len(PREVIEW_RUNS))]


def _run_index(run_id: str) -> int | None:
    return next((index for index, run in enumerate(_runs()) if run["id"] == run_id), None)


def _artifact(index: int) -> dict:
    model_id, variant_id, _, _, _, size_bytes = PREVIEW_RUNS[index]
    name = f"{model_id}_{variant_id}_mpk.tar.gz"
    return {
        "name": name,
        "artifact_type": "model-pack",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
        "size_bytes": size_bytes,
        "media_type": "application/gzip",
    }


def _benchmark_result(index: int) -> dict:
    result = _read_result()
    model_id, variant_id, component_id, _, compiler, _ = PREVIEW_RUNS[index]
    result["model"]["id"] = model_id
    result["model"]["build_profile_id"] = variant_id
    result["model"]["component_id"] = component_id
    result["compatibility"]["actual_platform_version"] = compiler
    latency = round(3.15 + index * 0.37, 3)
    throughput = round(510.0 - index * 21.5, 1)
    model_load = round(108.0 + index * 31.7, 1)
    first_inference = round(model_load + 214.0 + index * 8.4, 1)
    for key, value in {
        "latency_avg_ms": latency,
        "throughput_fps": max(42.0, throughput),
        "model_load_ms": model_load,
        "time_to_first_inference_ms": first_inference,
        "mla_memory_peak_delta_mib": 128.0 + index * 32.0,
    }.items():
        metric = result["metrics"][key]
        metric.update({"value": value, "minimum": value, "maximum": value, "standard_deviation": 0.0})
    if index in {5, 11}:
        result["metrics"]["time_to_first_inference_ms"].update({
            "status": "unavailable",
            "reason": "startup_probe_failed",
            "value": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
            "sample_count": 0,
        })
    return result


def _accuracy_result(index: int) -> dict | None:
    model_id, variant_id, _, _, _, _ = PREVIEW_RUNS[index]
    if (model_id, variant_id) != ("resnet_50", "modalix_int8"):
        return None
    return {
        "schema_version": 1,
        "report_scope": "sampled_accuracy_smoke",
        "model": {
            "id": model_id,
            "build_profile_id": variant_id,
            "component_id": "model",
            "source_sha256": "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50",
        },
        "dataset": {
            "id": "imagenet-2012",
            "variant": "mini",
            "split": "validation",
            "sha256": "d769e759abba1fe6e03db704020dcff3f1e8309a8cac8885ab60f5a0e512c239",
            "evaluated_samples": 500,
            "available_samples": 5000,
        },
        "evaluation": {"profile": "regular", "status": "provisional", "config": "imagenet_top5_smoke"},
        "execution": {"workers": 6, "samples_per_second": 2.589, "elapsed_seconds": 193.09},
        "metrics": {
            "candidate": {"top1_accuracy": 0.762, "top5_accuracy": 0.934},
            "reference": {"top1_accuracy": 0.798, "top5_accuracy": 0.95},
            "comparison": {"top1_agreement": 0.886, "top1_delta": -0.036, "top5_delta": -0.016},
        },
    }


def _benchmark_run_id(index: int) -> str:
    return f"9bd7b220-478f-4d5f-a02f-5fca756305{index:02x}"


def _accuracy_key(index: int) -> str:
    return hashlib.sha256(f"accuracy-{index}".encode()).hexdigest()


def _accuracy_run_id(index: int) -> str:
    return f"7bd7b220-478f-4d5f-a02f-5fca756305{index:02x}"


def _devkit_accuracy_result(index: int) -> dict | None:
    """Schema-2 accuracy measured on the DevKit and published as its own model_accuracy run."""
    model_id, variant_id, component_id, _, _, _ = PREVIEW_RUNS[index]
    if model_id != "detr_resnet50" and (model_id, component_id) != ("rf_detr_small", "transformer"):
        return None
    return {
        "schema_version": 2,
        "model": {
            "id": model_id,
            "build_profile_id": variant_id,
            "component_id": component_id,
            "package_id": None,
            "artifacts": {component_id: "4f1e5a9c7d3b2e8f6a0c1d9e7b5a3f2c4e6d8b0a1c3e5f7a9b2d4c6e8f0a1b3c"},
        },
        "validation": {"id": "coco_accuracy", "engine": "devkit_script", "status": "ready"},
        "dataset": {
            "id": "coco-2017",
            "variant": "mini",
            "split": "val2017",
            "sha256": "abb616d0eaf3d55d98d44ed092e05e2df6f87264d7f8426550d62402ef222e78",
            "evaluated_samples": 5000,
            "available_samples": 5000,
        },
        "metrics": {
            "coco_map": {"value": 0.4694, "unit": "ratio", "direction": "higher", "reference": 0.530},
            "coco_ap50": {"value": 0.6537, "unit": "ratio", "direction": "higher", "reference": 0.721},
            **{
                name: {"value": value, "unit": "ratio", "direction": "higher"}
                for name, value in {
                    "coco_ap75": 0.5017,
                    "coco_ap_small": 0.2687,
                    "coco_ap_medium": 0.5132,
                    "coco_ap_large": 0.6571,
                    "coco_ar1": 0.3611,
                    "coco_ar10": 0.6012,
                    "coco_ar100": 0.6593,
                    "coco_ar_small": 0.4598,
                    "coco_ar_medium": 0.7121,
                    "coco_ar_large": 0.8340,
                }.items()
            },
        },
        "execution": {"core_version": "3.0.0", "script_sha256": "c" * 64, "models_commit": "a" * 40},
    }


def _model_yaml_value(model_id: str, key: str) -> str | None:
    """A top-level scalar from models/<id>/model.yaml, read without a YAML parser so the preview stays stdlib-only."""
    path = MODELS / model_id / "model.yaml"
    if not path.is_file():
        return None
    match = re.search(rf"^{key}: (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1).strip().strip("\"'") if match else None


def _documentation_run_id(model_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"preview-documentation/{model_id}"))


def _documentation_files(model_id: str) -> dict[str, Path] | None:
    files = {"readme": MODELS / model_id / "README.md", "hero": MODELS / model_id / "hero.png"}
    return files if all(path.is_file() for path in files.values()) else None


def _model_documentation(model_id: str) -> dict | None:
    """The catalog documentation entry for a model in models/, shaped like the registry's published receipt."""
    files = _documentation_files(model_id)
    if files is None:
        return None
    readme = files["readme"].read_bytes()
    hero = files["hero"].read_bytes()
    overview = re.search(r"^## Overview\n(.*?)^## ", readme.decode("utf-8"), re.MULTILINE | re.DOTALL)
    return {
        "format": "markdown",
        "overview_markdown": overview.group(1).strip() if overview else "",
        "run_id": _documentation_run_id(model_id),
        "artifact_name": "readme",
        "artifact_sha256": hashlib.sha256(readme).hexdigest(),
        "size_bytes": len(readme),
        "media_type": "text/markdown; charset=utf-8",
        "source_path": f"models/{model_id}/README.md",
        "source_commit": COMMIT,
        "hero": {
            "artifact_name": "hero",
            "markdown_path": "hero.png",
            "alt_text": f"{_model_yaml_value(model_id, 'display_name')} model hero",
            "artifact_sha256": hashlib.sha256(hero).hexdigest(),
            "size_bytes": len(hero),
            "media_type": "image/png",
            "source_path": f"models/{model_id}/hero.png",
        },
    }


def _catalog_models(runs: list[dict]) -> list[dict]:
    """Catalog model entries whose builds are the given compile runs."""
    grouped: dict[str, dict] = {}
    for run in runs:
        model_id = run["metadata"]["model_id"]
        model = grouped.setdefault(model_id, {
            "model_id": model_id,
            "display_name": _model_yaml_value(model_id, "display_name") or PREVIEW_ONLY_MODELS[model_id][0],
            "description": _model_yaml_value(model_id, "description") or PREVIEW_ONLY_MODELS[model_id][1],
            "documentation": _model_documentation(model_id),
            "catalog_revision": run["commit_sha"],
            "model_type": run["model_type"],
            "architecture_id": run["architecture_id"],
            "architecture_paradigm": run["architecture_paradigm"],
            "tasks": run["tasks"],
            "input_modalities": run["input_modalities"],
            "output_modalities": run["output_modalities"],
            "model_tags": run["model_tags"],
            "categories": run.get("model_categories", {}),
            "builds": [],
        })
        # A catalog names one build per profile and component; runs are listed newest first.
        if any(
            (build["build_profile_id"], build["component_id"])
            == (run["metadata"]["build_profile_id"], run["metadata"]["component_id"])
            for build in model["builds"]
        ):
            continue
        model["builds"].append({
            "run_id": run["id"],
            "build_profile_id": run["metadata"]["build_profile_id"],
            "component_id": run["metadata"]["component_id"],
            "compile_invocation_id": run["metadata"]["github_run_id"],
            "status": run["status"],
            "commit_sha": run["commit_sha"],
            "created_at": run["created_at"],
            "finished_at": run["finished_at"],
            "toolchain_name": run["toolchain_name"],
            "toolchain_version": run["toolchain_version"],
            "target_platform": run["target_platform"],
            "validation_fingerprint": (
                _accuracy_key(_run_index(run["id"])) if _devkit_accuracy_result(_run_index(run["id"])) else None
            ),
        })
    return list(grouped.values())


class PreviewHandler(SimpleHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _documentation_content(self, run_id: str, artifact_name: str) -> None:
        model_id = next((item[0] for item in PREVIEW_RUNS if _documentation_run_id(item[0]) == run_id), None)
        files = _documentation_files(model_id) if model_id else None
        if files is None or artifact_name not in files:
            self._json({"error": {"code": "artifact_not_found", "message": "No documentation artifact"}}, 404)
            return
        body = files[artifact_name].read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png" if artifact_name == "hero" else "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        runs = _runs()
        if re.fullmatch(r"/config(\.[0-9a-f]+)?\.js", path):
            body = (
                b"window.MODEL_REGISTRY_CONFIG = Object.freeze({"
                b'apiBase: location.origin, environment: "Local preview", '
                b'repository: Object.freeze({organization: "sima-neat", name: "models"})});\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/v1/repositories/sima-neat/models/runs":
            self._json({"runs": runs, "has_more": False})
            return
        if path == "/v1/repositories/sima-neat/models/models":
            git_ref = query.get("git_ref", ["refs/heads/main"])[0]
            compiler = query.get("toolchain_version", [""])[0]
            branch_models = _catalog_models([item for item in runs if item["git_ref"] == git_ref])
            models = [model for model in branch_models if not compiler or any(
                build["toolchain_version"] == compiler for build in model["builds"]
            )]
            self._json({"models": models, "count": len(models), "has_more": False, "next_cursor": None})
            return
        catalog_prefix = "/v1/repositories/sima-neat/models/catalogs/"
        if path.startswith(catalog_prefix) and path != f"{catalog_prefix}current":
            # The catalog at a revision lists the builds compiled at that commit.
            commit = path.removeprefix(catalog_prefix)
            git_ref = query.get("git_ref", [""])[0]
            builds = [item for item in runs if item["commit_sha"] == commit and item["git_ref"] == git_ref]
            self._json({"catalog": {"document": {"models": _catalog_models(builds)}}})
            return
        if path == "/v1/repositories/sima-neat/models/branches":
            branches = sorted({item["git_ref"] for item in runs})
            self._json({"branches": [
                {
                    "name": branch.removeprefix("refs/heads/"),
                    "git_ref": branch,
                    "latest_run_at": next(item["created_at"] for item in runs if item["git_ref"] == branch),
                    "run_count": sum(item["git_ref"] == branch for item in runs),
                }
                for branch in branches
            ]})
            return
        if path == "/v1/repositories/sima-neat/models/catalogs/current":
            self._json({"catalog": {"document": {"taxonomy": {
                "model_types": {"vision": {"display_name": "Vision"}},
                "tasks": {
                    "image_classification": {"display_name": "Image classification"},
                    "object_detection": {"display_name": "Object detection"},
                    "depth_estimation": {"display_name": "Depth estimation"},
                    "semantic_segmentation": {"display_name": "Semantic segmentation"},
                },
                "architectures": {"resnet_50": {"display_name": "ResNet-50"}},
                "architecture_paradigms": {"cnn": {"display_name": "CNN"}},
                "modalities": {
                    "image": {"display_name": "Image"},
                    "classification": {"display_name": "Classification"},
                },
            }}}})
            return
        if path == "/v1/repositories/sima-neat/models/resolve":
            fingerprint = query.get("fingerprint", [""])[0]
            accuracy_index = next(
                (item for item in range(len(PREVIEW_RUNS)) if _accuracy_key(item) == fingerprint), None
            )
            if query.get("run_type") != ["model_accuracy"] or accuracy_index is None:
                self._json({"error": {"code": "fingerprint_not_found", "message": "No completed run"}}, 404)
            else:
                self._json({"run": {"id": _accuracy_run_id(accuracy_index), "run_type": "model_accuracy"}})
            return
        if path.startswith("/v1/runs/"):
            parts = path.strip("/").split("/")
            requested_run_id = parts[2] if len(parts) >= 3 else ""
            if len(parts) == 6 and parts[3] == "artifacts" and parts[5] == "content":
                self._documentation_content(requested_run_id, parts[4])
                return
            index = _run_index(requested_run_id)
            benchmark_index = next(
                (item for item in range(len(PREVIEW_RUNS)) if _benchmark_run_id(item) == requested_run_id), None
            )
            if benchmark_index is not None and parts[3:] == ["test-results"]:
                test_results = [{"suite": "performance", "name": "model-synthetic", "status": "passed"}]
                self._json({"run_id": requested_run_id, "test_results": test_results})
                return
            accuracy_index = next(
                (item for item in range(len(PREVIEW_RUNS)) if _accuracy_run_id(item) == requested_run_id), None
            )
            if accuracy_index is not None and parts[3:] == ["accuracy", "latest"]:
                accuracy = _devkit_accuracy_result(accuracy_index)
                self._json({
                    "accuracy_test": {"name": accuracy["validation"]["id"], "status": "passed", "attempt": 1},
                    "result": accuracy,
                    "result_artifact_url": None,
                })
                return
        else:
            index = None
        if index is not None and len(parts) == 3:
            artifacts = [_artifact(index)]
            if _accuracy_result(index):
                accuracy_name = "accuracy-result"
                artifacts.append({
                    "name": accuracy_name,
                    "artifact_type": "accuracy-result",
                    "sha256": hashlib.sha256(json.dumps(_accuracy_result(index), sort_keys=True).encode()).hexdigest(),
                    "size_bytes": 2048,
                    "media_type": "application/json",
                })
            self._json({"run": runs[index], "artifacts": artifacts})
            return
        if index is not None and len(parts) == 4 and parts[3] == "provenance":
            self._json({
                "run": runs[index],
                "parameters": {"parameters": {"compile_parameters": {
                    "precision": "BF16" if "bf16" in PREVIEW_RUNS[index][1] else "INT8"
                }}},
                "inputs": [{
                    "name": "onnx-model",
                    "input_type": "model",
                    "source_reference": "hf://models/onnx-community/resnet-50-ONNX@b6ef686fe842388c9449c7acb7269d02c69bacfb/onnx/model.onnx",
                    "metadata": {
                        "model_id": PREVIEW_RUNS[index][0],
                        "model_card": "https://huggingface.co/onnx-community/resnet-50-ONNX",
                    },
                    "sha256": "9c871fc4454da4436cd52254bccd428f0fae7b0f8a7ec09681c92fc6530b237a",
                    "size_bytes": 102_438_921,
                }],
            })
            return
        if index is not None and len(parts) == 4 and parts[3] == "test-results":
            test_results = [
                {"suite": "model", "name": "manifest-validation", "status": "passed", "duration_ms": 82},
                {"suite": "model", "name": "mpk-validation", "status": "passed", "duration_ms": 418},
            ]
            accuracy = _accuracy_result(index)
            if accuracy:
                test_results.append({
                    "suite": "accuracy",
                    "name": accuracy["report_scope"],
                    "status": "passed",
                    "version": "1",
                    "summary": accuracy,
                })
            if index == 0 or PREVIEW_RUNS[index][:3] == ("rf_detr_small", "modalix_int8_bf16", "transformer"):
                test_results.append({
                    "suite": "correctness",
                    "name": "numerical-fidelity",
                    "status": "passed",
                    "version": "1",
                    "summary": {
                        "schema_version": 1,
                        "model_id": PREVIEW_RUNS[index][0],
                        "build_profile_id": "modalix_bf16",
                        "component_id": PREVIEW_RUNS[index][2],
                        "scope": "full_pipeline",
                        "status": "diagnostic",
                        "reference": "Original ONNX (ONNX Runtime)",
                        "candidate": "Quantized AFE network (JAX execute)",
                        "dataset": {"id": "imagenet-2012", "variant": "mini"},
                        "samples": 8,
                        "metrics": [{
                            "name": "logits",
                            "cosine_similarity": 0.999842,
                            "normalized_rmse": 0.00312,
                            "mean_absolute_error": 0.0048,
                            "max_absolute_error": 0.0184,
                        }],
                    },
                })
            self._json({"run_id": runs[index]["id"], "test_results": test_results})
            return
        if index is not None and len(parts) == 4 and parts[3] == "metrics":
            self._json({
                "run_id": runs[index]["id"],
                "measurements": [{"name": "compile.duration", "numeric_value": 516.0 + index * 17.0}],
            })
            return
        if index is not None and len(parts) == 5 and parts[3:] == ["benchmarks", "latest"]:
            self._json({
                "benchmark_run": {"id": _benchmark_run_id(index)},
                "result": _benchmark_result(index),
                "result_artifact_url": "/fixtures/performance-result-v4.json",
            })
            return
        if index is not None and len(parts) == 5 and parts[3:] == ["accuracy", "latest"]:
            accuracy = _accuracy_result(index)
            if accuracy is None:
                self._json({"error": {"code": "accuracy_not_found", "message": "No accuracy result"}}, 404)
            else:
                self._json({
                    "accuracy_test": {
                        "id": f"8bd7b220-478f-4d5f-a02f-5fca756305{index:02x}",
                        "name": accuracy["report_scope"],
                        "status": "passed",
                        "attempt": 1,
                        "version": "1",
                    },
                    "result": accuracy,
                    "result_artifact_url": "/v1/runs/"
                    f"{runs[index]['id']}/artifacts/accuracy-result",
                })
            return
        if index is not None and len(parts) == 5 and parts[3] == "artifacts":
            self._json({
                "artifact": {
                    "download_url": "/fixtures/performance-result-v4.json",
                    "download_url_expires_in": 300,
                }
            })
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        # The fixtures stay in artifact-browser/ when --root serves an assembled portal. Normalizing
        # first resolves "..", so a path that leaves the fixtures is served from the root, or not at all.
        request = posixpath.normpath(unquote(urlparse(path).path))
        if request.startswith("/fixtures/"):
            return str(BROWSER / request.lstrip("/"))
        return super().translate_path(path)

    def log_message(self, message: str, *args: object) -> None:
        print(f"preview: {message % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=BROWSER, help="directory to serve, such as website/portal-build")
    args = parser.parse_args()
    os.chdir(args.root)
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Model browser preview: http://{args.host}:{args.port}/?run={RUN_ID}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
