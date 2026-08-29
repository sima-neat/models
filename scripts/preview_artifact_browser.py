#!/usr/bin/env python3
"""Serve the artifact browser with a realistic local benchmark API fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
RUN_ID = "74b48a12-6088-42a0-b881-5925a5272fae"
COMMIT = "244c03323cefc96a6baaa097f0a4bb008e9c39ae"
PREVIEW_RUNS = (
    ("resnet_50", "modalix_bf16", "feat/resnet50-post-compile-benchmark", "2.1.3", 284_103_221),
    ("resnet_50", "modalix_int8", "develop", "2.1.3", 165_482_104),
    ("ssd_mobilenet_v1", "modalix_bf16_tess_mla", "develop", "2.1.3", 132_714_990),
    ("ssd_mobilenet_v1", "modalix_int8_tess_mla", "feat/ssd-mobilenet-qualification", "2.1.3", 78_204_118),
    ("ssd_mobilenet_v2", "modalix_bf16_tess_mla", "develop", "2.1.3", 148_821_633),
    ("ssd_mobilenet_v2", "modalix_int8_tess_off_mla", "feat/tessellation-placement-study", "2.1.3", 86_902_412),
    ("ssd_mobilenet_v3", "modalix_bf16_tess_mla", "develop", "2.1.3", 119_443_210),
    ("yolo_v8n", "modalix_int8", "main", "2.1.3", 41_883_092),
    ("yolo_v8s", "modalix_int8", "main", "2.1.3", 96_771_405),
    ("depth_anything_v2_vits", "modalix_bf16", "feat/depth-runtime-validation", "2.1.3", 207_614_992),
    ("fcn_hrnet48", "modalix_bf16", "develop", "2.1.3", 312_045_118),
    ("midas_v21_small_256", "modalix_bf16", "main", "2.1.2", 91_352_776),
    ("detr_resnet50", "modalix_bf16", "feat/detection-benchmark-matrix", "2.1.3", 251_770_364),
    (
        "resnet_50",
        "modalix_bf16_batch4",
        "experiment/long-running-batch-size-throughput-comparison",
        "2.1.3",
        296_411_087,
    ),
)


def _read_result() -> dict:
    return json.loads((BROWSER / "fixtures" / "performance-result-v3.json").read_text(encoding="utf-8"))


def _run(index: int = 0) -> dict:
    model_id, variant_id, branch, compiler, _ = PREVIEW_RUNS[index]
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
        "metadata": {"model_id": model_id, "variant_id": variant_id, "model_name": model_id},
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
    model_id, variant_id, _, _, size_bytes = PREVIEW_RUNS[index]
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
    model_id, variant_id, _, compiler, _ = PREVIEW_RUNS[index]
    result["model"]["id"] = model_id
    result["model"]["variant_id"] = variant_id
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
        path = urlparse(self.path).path
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
        if path == "/v1/repositories/sima-neat/models/taxonomies/latest":
            self._json({"taxonomy": {"document": {
                "model_types": {"vision": {"display_name": "Vision"}},
                "tasks": {"image_classification": {"display_name": "Image classification"}},
                "architectures": {"resnet_50": {"display_name": "ResNet-50"}},
                "architecture_paradigms": {"cnn": {"display_name": "CNN"}},
                "modalities": {
                    "image": {"display_name": "Image"},
                    "classification": {"display_name": "Classification"},
                },
            }}})
            return
        if path.startswith("/v1/runs/"):
            parts = path.strip("/").split("/")
            requested_run_id = parts[2] if len(parts) >= 3 else ""
            index = _run_index(requested_run_id)
        else:
            index = None
        if index is not None and len(parts) == 3:
            self._json({"run": runs[index], "artifacts": [_artifact(index)]})
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
            self._json({"run_id": runs[index]["id"], "test_results": [
                {"suite": "model", "name": "manifest-validation", "status": "passed", "duration_ms": 82},
                {"suite": "model", "name": "mpk-validation", "status": "passed", "duration_ms": 418},
            ]})
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
                "result_artifact_url": "/fixtures/performance-result-v3.json",
            })
            return
        if index is not None and len(parts) == 5 and parts[3] == "artifacts":
            self._json({
                "artifact": {
                    "download_url": "/fixtures/performance-result-v3.json",
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
