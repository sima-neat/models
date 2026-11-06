"""Measure deployed LLiMa models with MoLE text timings and the existing image HTTP API."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .board_telemetry import BoardTelemetry
from .errors import ValidationError
from .fetch import sha256_file

NEW_TOKENS = 256
REPETITIONS = 5
IMAGE_PROMPT = "Describe what you see in the image."
TEXT = (
    "A city library is planning a community garden. Volunteers will grow vegetables, collect rainwater, "
    "and teach children how plants develop. Explain how the garden can help the neighbourhood and "
    "describe practical steps for keeping it healthy throughout the year. "
)
GENAI_METRICS = {
    "ttft_ms": ("ms", "lower", "model.generation.ttft"),
    "prefill_tps": ("tokens/s", "higher", "model.generation.prefill_rate"),
    "decode_tps": ("tokens/s", "higher", "model.generation.decode_rate"),
}


def input_buckets(context: int) -> list[int]:
    end = context - 1024
    if end < 128:
        raise ValidationError("text buckets require at least 1152 compiled context tokens")
    buckets = []
    length = 128
    while length <= end:
        buckets.append(length)
        length *= 2
    return sorted(set([*buckets, end]))


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{label} must be a positive finite measurement")
    return value


def workload_metrics(workload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if [row["phase"] for row in workload["requests"]] != ["warmup", *(["measure"] * REPETITIONS)]:
        raise ValidationError("expected one warmup followed by five measured requests")
    measured = [row for row in workload["requests"] if row["phase"] == "measure"]
    if len(measured) != REPETITIONS:
        raise ValidationError(f"{workload['id']}: expected {REPETITIONS} measured requests")
    ttft = [_positive(row["ttft_s"], "TTFT") for row in measured]
    values = {"ttft_ms": statistics.mean(ttft) * 1000}
    if workload["kind"] == "text":
        for row in workload["requests"]:
            durations = row["token_durations_s"]
            if (
                not durations
                or len(durations) != row["generated_tokens"]
                or row["decode_tokens"] != len(durations) - 1
                or row["input_tokens"] != workload["input_tokens"]
                or row["ttft_s"] != durations[0]
                or row["decode_time_s"] != sum(durations[1:])
            ):
                raise ValidationError("text counts and timings disagree")
            for duration in durations:
                _positive(duration, "token duration")
        values["prefill_tps"] = sum(row["input_tokens"] for row in measured) / sum(ttft)
        count = sum(row["decode_tokens"] for row in measured)
        elapsed = sum(_positive(row["decode_time_s"], "decode time") for row in measured)
        if count <= 0:
            raise ValidationError("text benchmark generated no decode tokens")
        values["decode_tps"] = count / elapsed
    else:
        # This is the agreed mean of streamed TPS observations, not a token-weighted runtime aggregate.
        values["decode_tps"] = statistics.mean(_positive(row["streamed_tps_mean"], "streamed TPS") for row in measured)
    return {
        name: {
            "value": value,
            "unit": GENAI_METRICS[name][0],
            "direction": GENAI_METRICS[name][1],
            "aggregation": "mean",
            "sample_count": len(measured),
            "minimum": None,
            "maximum": None,
        }
        for name, value in values.items()
    }


def validate_report(result: dict[str, Any]) -> None:
    report = result.get("benchmark_report", result)
    settings = report["settings"]
    if "configuration" in result and result["configuration"] != settings:
        raise ValidationError("GenAI configuration disagrees with its saved report")
    expected = [f"text_n{length}" for length in input_buckets(settings["max_context"])]
    if settings.get("image_size"):
        expected.append("image")
    if [row["id"] for row in report["workloads"]] != expected:
        raise ValidationError("GenAI workloads do not cover the declared text/image inputs")
    for workload in report["workloads"]:
        if workload["metrics"] != workload_metrics(workload):
            raise ValidationError("GenAI metrics disagree with recorded request timings")
        if workload["kind"] == "image_text" and workload["image_size"] != settings["image_size"]:
            raise ValidationError("image does not match the compiled vision encoder size")


def _prompt(tokenizer, length: int) -> list[int]:
    text = TEXT
    while True:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if len(tokens) >= length:
            break
        text += TEXT
    # Preserve the native assistant prefix, following MoLE's bucket truncation policy.
    a, b = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": word}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        for word in ("alpha", "bravo")
    ]
    prefix = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and a[-suffix - 1] == b[-suffix - 1]:
        suffix += 1
    if prefix + suffix >= length:
        raise ValidationError("native chat template does not fit the requested input bucket")
    return tokens[: length - suffix] + tokens[-suffix:] if suffix else tokens[:length]


def _text(deployment: Path, output: Path, context: int) -> tuple[list[dict], dict]:
    import torch
    from sima_lmm.mole.modalixboard import ModalixBoard
    from sima_lmm.mole.modalixlm import ModalixLM

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    board = ModalixBoard(address=None, port=port, model=str(deployment))
    lm = None
    threads = torch.get_num_threads()
    workloads = []
    try:
        torch.set_num_threads(1)
        board.start_server(output / "mole-server.log")
        lm = ModalixLM(board=board, pretrained=str(deployment / "devkit"), device="cpu", max_length=context)
        config_path = deployment / "devkit/generation_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        stops = config.get("eos_token_id", lm.tokenizer.eos_token_id)
        stops = [stops] if isinstance(stops, int) else list(stops or [])
        for length in input_buckets(context):
            tokens = _prompt(lm.tokenizer, length)
            workload = {"id": f"text_n{length}", "kind": "text", "input_tokens": length, "requests": []}
            workloads.append(workload)
            telemetry = BoardTelemetry(workload["id"])
            for index in range(REPETITIONS + 1):
                with telemetry.measure(enabled=index > 0):
                    durations = (
                        lm._zmq_infer(
                            torch.tensor([tokens]),
                            "generate_for_perf",
                            max_num_tokens=length + NEW_TOKENS,
                            stop_token_ids=stops,
                        )
                        .cpu()
                        .tolist()[0]
                    )
                metadata = lm.last_profile_data
                if len(durations) != metadata.get("generated_tokens") or not durations:
                    raise ValidationError("MoLE did not return one timing per generated token")
                for duration in durations:
                    _positive(duration, "token duration")
                workload["requests"].append(
                    {
                        "phase": "warmup" if index == 0 else "measure",
                        "input_tokens": length,
                        "generated_tokens": len(durations),
                        "decode_tokens": len(durations) - 1,
                        "ttft_s": durations[0],
                        "decode_time_s": sum(durations[1:]),
                        "token_durations_s": durations,
                        "accepted_draft_tokens": metadata.get("accepted_draft_tokens"),
                    }
                )
            workload["metrics"] = workload_metrics(workload)
            workload.update(telemetry.report())
            print(f"{workload['id']}: {json.dumps(workload['metrics'])}", flush=True)
        return workloads, {"mole_version": version("sima-lmm"), "stop_token_ids": stops}
    finally:
        board.stop_server()
        if lm is not None:
            lm.zmq_socket.close(linger=0)
            lm.zmq_context.term()
        torch.set_num_threads(threads)


@contextmanager
def _web_server(deployment: Path, log_path: Path):
    try:
        with socket.create_connection(("127.0.0.1", 9998), timeout=1):
            raise ValidationError("port 9998 is already in use")
    except OSError:
        pass
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            ["/usr/bin/llima", "run", str(deployment), "--mode", "web"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise ValidationError(f"web server exited; see {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", 9998), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ValidationError(f"web server startup timed out; see {log_path}") from None
                    time.sleep(0.2)
            yield
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _image(deployment: Path, output: Path, size: list[int]) -> dict:
    from PIL import Image

    source = Path("/usr/share/sima_lmm/assets/sjc.jpg")
    height, width = size
    encoded = io.BytesIO()
    with Image.open(source) as image:
        image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC).save(encoded, "JPEG", quality=90)
    data = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode()
    body = json.dumps(
        {
            "stream": True,
            "enable_thinking": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data}},
                        {"type": "text", "text": IMAGE_PROMPT},
                    ],
                }
            ],
        }
    ).encode()
    workload = {
        "id": "image",
        "kind": "image_text",
        "image_size": size,
        "prompt": IMAGE_PROMPT,
        "source_image_sha256": sha256_file(source)[0],
        "jpeg_sha256": hashlib.sha256(encoded.getvalue()).hexdigest(),
        "jpeg_bytes": len(encoded.getvalue()),
        "generated_tokens": None,
        "requests": [],
    }
    telemetry = BoardTelemetry("image")
    for index in range(REPETITIONS + 1):
        # Restart to reset every model's KV/state cache. Loading is outside server-reported TTFT.
        with _web_server(deployment, output / f"web-server-{index}.log"):
            request = urllib.request.Request(
                "http://127.0.0.1:9998/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            ttft, rates, complete = None, [], False
            with telemetry.measure(enabled=index > 0), urllib.request.urlopen(request, timeout=180) as response:
                for line in response:
                    text = line.decode().strip()
                    if text == "data: [DONE]":
                        complete = True
                        break
                    if text.startswith("data: "):
                        chunk = json.loads(text[6:])
                        if "ttft" in chunk:
                            ttft = _positive(chunk["ttft"], "image TTFT")
                        if "tps" in chunk:
                            rates.append(_positive(chunk["tps"], "streamed TPS"))
            if not complete or ttft is None or not rates:
                raise ValidationError("image generation did not complete with TTFT and streamed TPS")
            workload["requests"].append(
                {
                    "phase": "warmup" if index == 0 else "measure",
                    "ttft_s": ttft,
                    "streamed_tps_mean": statistics.mean(rates),
                    "streamed_tps": rates,
                    "generated_tokens": None,
                }
            )
    workload["metrics"] = workload_metrics(workload)
    workload.update(telemetry.report())
    return workload


def run_benchmark(deployment: Path, output: Path) -> dict[str, Any]:
    """Run locally inside the installed MoLE environment; never download a checkpoint or dataset."""
    deployment = deployment.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark-report.json").unlink(missing_ok=True)
    config = json.loads((deployment / "devkit/vlm_config.json").read_text())
    context = config["pipeline_cfg"]["max_num_tokens"]
    workloads, runtime = _text(deployment, output, context)
    vision = config.get("vm_cfg")
    if vision:
        size = vision["image_size"]
        size = [size, size] if isinstance(size, int) else size
        workloads.append(_image(deployment, output, size))
    report = {
        "status": "passed",
        "runtime": runtime,
        "workloads": workloads,
        "settings": {
            "max_context": context,
            "max_new_tokens": NEW_TOKENS,
            "image_size": size if vision else None,
            "warmup": 1,
            "repetitions": REPETITIONS,
            "sampling": "greedy",
            "thinking": False,
            "text_cache": "reset by MoLE after every request",
            "image_cache": "fresh server for every request",
            "image_output_limit": "native EOS/context",
        },
        "timing": {
            "text_ttft": "language prefill through first token; excludes tokenization and input-buffer preparation",
            "prefill_tps": "input tokens / language TTFT",
            "text_decode_tps": "decode tokens / decode duration; excludes first token",
            "image_ttft": "server preprocessing, vision encoding and language prefill through first token",
            "image_decode_tps": "mean of streamed TPS observations, averaged across measured requests",
            "excluded": "model loading, client image resizing/encoding and HTTP upload",
        },
    }
    validate_report(report)
    (output / "benchmark-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    from model_report import from_generation

    installed = subprocess.check_output(["dpkg-query", "-W", "-f=${Version}", "sima-lmm-core"], text=True).strip()
    from_generation(report, {"llima": installed, "mole": runtime["mole_version"], "build": {}}).save(output)
    return report
