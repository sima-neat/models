#!/usr/bin/env python3
"""Serve the artifact browser with a realistic local benchmark API fixture."""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BROWSER = ROOT / "artifact-browser"
RUN_ID = "74b48a12-6088-42a0-b881-5925a5272fae"
COMMIT = "244c03323cefc96a6baaa097f0a4bb008e9c39ae"


def _read_result() -> dict:
    return json.loads((BROWSER / "fixtures" / "performance-result-v2.json").read_text(encoding="utf-8"))


def _run() -> dict:
    return {
        "id": RUN_ID,
        "repository": "sima-neat/models",
        "run_type": "model_compile",
        "commit_sha": COMMIT,
        "git_ref": "refs/heads/feat/resnet50-post-compile-benchmark",
        "status": "completed",
        "created_at": "2026-08-26T19:30:00Z",
        "started_at": "2026-08-26T19:30:08Z",
        "finished_at": "2026-08-26T19:38:44Z",
        "run_fingerprint": "d4b2c4713db4d76bd681952c820189682835616a4afb2d18cda598d0803e82c8",
        "toolchain_name": "model-compiler",
        "toolchain_version": "2.1.3",
        "target_platform": "modalix",
        "metadata": {"model_id": "resnet_50", "variant_id": "modalix_bf16", "model_name": "ResNet-50"},
        "model_type": "vision",
        "architecture_id": "resnet_50",
        "architecture_paradigm": "cnn",
        "tasks": ["image_classification"],
        "input_modalities": ["image"],
        "output_modalities": ["classification"],
        "model_tags": ["reference"],
        "model_categories": {
            "model_type": "vision",
            "tasks": ["image_classification"],
            "architecture": {"id": "resnet_50", "paradigm": "cnn"},
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
        path = urlparse(self.path).path
        run = _run()
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
            self._json({"runs": [run], "has_more": False})
            return
        if path == "/v1/repositories/sima-neat/models/branches":
            self._json({"branches": [{
                "name": "feat/resnet50-post-compile-benchmark",
                "git_ref": "refs/heads/feat/resnet50-post-compile-benchmark",
                "latest_run_at": run["created_at"],
                "run_count": 1,
            }]})
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
        if path == f"/v1/runs/{RUN_ID}":
            self._json({"run": run, "artifacts": [{
                "name": "resnet_50_modalix_bf16_mpk.tar.gz",
                "artifact_type": "model-pack",
                "sha256": "62b45721ab30176da63cd9f84357dc79270fd702a7273d5d1c678d8263a9c219",
                "size_bytes": 284_103_221,
                "media_type": "application/gzip",
            }]})
            return
        if path == f"/v1/runs/{RUN_ID}/provenance":
            self._json({
                "run": run,
                "parameters": {"parameters": {"compile_parameters": {"precision": "BF16"}}},
                "inputs": [{
                    "name": "onnx-model",
                    "input_type": "model",
                    "source_reference": "hf://models/onnx-community/resnet-50-ONNX@b6ef686fe842388c9449c7acb7269d02c69bacfb/onnx/model.onnx",
                    "metadata": {
                        "model_id": "resnet_50",
                        "model_card": "https://huggingface.co/onnx-community/resnet-50-ONNX",
                    },
                    "sha256": "9c871fc4454da4436cd52254bccd428f0fae7b0f8a7ec09681c92fc6530b237a",
                    "size_bytes": 102_438_921,
                }],
            })
            return
        if path == f"/v1/runs/{RUN_ID}/test-results":
            self._json({"run_id": RUN_ID, "test_results": [
                {"suite": "model", "name": "manifest-validation", "status": "passed", "duration_ms": 82},
                {"suite": "model", "name": "mpk-validation", "status": "passed", "duration_ms": 418},
            ]})
            return
        if path == f"/v1/runs/{RUN_ID}/metrics":
            self._json({"run_id": RUN_ID, "measurements": [{"name": "compile.duration", "numeric_value": 516.0}]})
            return
        if path == f"/v1/runs/{RUN_ID}/benchmarks/latest":
            self._json({
                "benchmark_run": {"id": "9bd7b220-478f-4d5f-a02f-5fca75630539"},
                "result": _read_result(),
                "result_artifact_url": "/fixtures/performance-result-v2.json",
            })
            return
        if path == f"/v1/runs/{RUN_ID}/artifacts/resnet_50_modalix_bf16_mpk.tar.gz":
            self._json({
                "artifact": {
                    "download_url": "/fixtures/performance-result-v2.json",
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
