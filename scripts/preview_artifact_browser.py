#!/usr/bin/env python3
"""Serve the artifact browser with a realistic local benchmark API fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
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
    ("midas_v21_small_256", "modalix_bf16", "model", "main", "2.1.2", 91_352_776),
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
)


def _read_result() -> dict:
    return json.loads((BROWSER / "fixtures" / "performance-result-v4.json").read_text(encoding="utf-8"))


def _run(index: int = 0) -> dict:
    model_id, variant_id, component_id, branch, compiler, _ = PREVIEW_RUNS[index]
    run_id = RUN_ID if index == 0 else f"74b48a12-6088-42a0-b881-5925a5272f{index:02x}"
    commit = COMMIT if index == 0 else hashlib.sha256(f"preview-commit-{index}".encode()).hexdigest()
    architecture = "resnet_50" if "resnet" in model_id else model_id
    task = "image_classification" if model_id == "resnet_50" else "object_detection"
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
        "evaluation": {"profile": "regular", "status": "provisional", "config": "tests/correctness.yaml"},
        "execution": {"workers": 6, "samples_per_second": 2.589, "elapsed_seconds": 193.09},
        "metrics": {
            "candidate": {"top1_accuracy": 0.762, "top5_accuracy": 0.934},
            "reference": {"top1_accuracy": 0.798, "top5_accuracy": 0.95},
            "comparison": {"top1_agreement": 0.886, "top1_delta": -0.036, "top5_delta": -0.016},
        },
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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        runs = _runs()
        if path == "/config.js":
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
            grouped: dict[str, dict] = {}
            for run in (item for item in runs if item["git_ref"] == git_ref):
                model_id = run["metadata"]["model_id"]
                model = grouped.setdefault(model_id, {
                    "model_id": model_id,
                    "display_name": model_id.replace("_", " ").title(),
                    "description": f"Verified {model_id.replace('_', ' ')} builds for Modalix deployment.",
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
                model["builds"].append({
                    "run_id": run["id"],
                    "build_profile_id": run["metadata"]["build_profile_id"],
                    "component_id": run["metadata"]["component_id"],
                    "compile_invocation_id": run["metadata"].get("compile_invocation_id", run["metadata"]["component_id"]),
                    "status": run["status"],
                    "commit_sha": run["commit_sha"],
                    "created_at": run["created_at"],
                    "finished_at": run["finished_at"],
                    "toolchain_name": run["toolchain_name"],
                    "toolchain_version": run["toolchain_version"],
                    "target_platform": run["target_platform"],
                })
            models = [model for model in grouped.values() if not compiler or any(
                build["toolchain_version"] == compiler for build in model["builds"]
            )]
            self._json({"models": models, "count": len(models), "has_more": False, "next_cursor": None})
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
                "tasks": {"image_classification": {"display_name": "Image classification"}},
                "architectures": {"resnet_50": {"display_name": "ResNet-50"}},
                "architecture_paradigms": {"cnn": {"display_name": "CNN"}},
                "modalities": {
                    "image": {"display_name": "Image"},
                    "classification": {"display_name": "Classification"},
                },
            }}}})
            return
        if path.startswith("/v1/runs/"):
            parts = path.strip("/").split("/")
            requested_run_id = parts[2] if len(parts) >= 3 else ""
            index = _run_index(requested_run_id)
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
            if index == 0:
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
                "benchmark_run": {"id": f"9bd7b220-478f-4d5f-a02f-5fca756305{index:02x}"},
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

    def log_message(self, message: str, *args: object) -> None:
        print(f"preview: {message % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(BROWSER)
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Model browser preview: http://{args.host}:{args.port}/?run={RUN_ID}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
