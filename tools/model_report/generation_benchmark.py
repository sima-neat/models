"""Standalone MoLE timings, dataset comparisons and telemetry for deployed LLiMa models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import statistics
import subprocess
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .board_telemetry import BoardTelemetry

NEW_TOKENS = 256
REPETITIONS = 5
TEXT_DATASETS = ("gsm8k", "bfcl-v3", "alpacaeval")
IMAGE_DATASETS = ("realworldqa", "docvqa_val", "chartqa")
IMAGE_PROMPT = "Describe what you see in the image."
TEXT = (
    "A city library is planning a community garden. Volunteers will grow vegetables, collect rainwater, "
    "and teach children how plants develop. Explain how the garden can help the neighbourhood and "
    "describe practical steps for keeping it healthy throughout the year. "
)
GENAI_METRICS = {
    "ttft_ms": ("ms", "lower"),
    "prefill_tps": ("tokens/s", "higher"),
    "decode_tps": ("tokens/s", "higher"),
    "draft_contribution_pct": ("%", "higher"),
    "decode_speedup": ("×", "higher"),
}


def input_buckets(context: int) -> list[int]:
    end = context - 1024
    if end < 128:
        raise ValueError("text buckets require at least 1152 compiled context tokens")
    buckets = []
    length = 128
    while length <= end:
        buckets.append(length)
        length *= 2
    return sorted(set([*buckets, end]))


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite measurement")
    return value


def workload_metrics(workload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    requests = workload["requests"]
    measured = [row for row in requests if row["phase"] == "measure"]
    expected = workload["dataset"]["available_samples"] if "dataset" in workload else REPETITIONS
    if [row["phase"] for row in requests] != ["warmup", *(["measure"] * expected)]:
        raise ValueError("expected one warmup followed by all measured requests")
    for row in requests:
        durations = row["token_durations_s"]
        if (row.get("status", "ok") != "ok" or not durations
            or len(durations) != row["generated_tokens"] or row["decode_tokens"] != len(durations) - 1
            or row["ttft_s"] != durations[0] or row["decode_time_s"] != sum(durations[1:])):
            raise ValueError("generation counts and timings disagree")
        if workload["kind"] == "text" and row["input_tokens"] != workload["input_tokens"]:
            raise ValueError("text input does not match its bucket")
        for duration in durations:
            _positive(duration, "token duration")
        accepted = row.get("accepted_draft_tokens")
        if accepted is not None and (type(accepted) is not int or not 0 <= accepted <= row["generated_tokens"]):
            raise ValueError("invalid accepted draft token count")
    ttft = [row["ttft_s"] for row in measured]
    values = {"ttft_ms": statistics.mean(ttft) * 1000}
    if workload["kind"] in ("text", "text_dataset"):
        values["prefill_tps"] = sum(row["input_tokens"] for row in measured) / sum(ttft)
    count = sum(row["decode_tokens"] for row in measured)
    elapsed = sum(row["decode_time_s"] for row in measured)
    values["decode_tps"] = count / _positive(elapsed, "decode time")
    accepted = [row.get("accepted_draft_tokens") for row in measured]
    if all(value is not None for value in accepted):
        values["draft_contribution_pct"] = 100 * sum(accepted) / sum(row["generated_tokens"] for row in measured)
    baseline = workload.get("baseline")
    if baseline:
        baseline_metrics = workload_metrics(baseline)
        if baseline["metrics"] != baseline_metrics:
            raise ValueError("baseline metrics disagree with recorded timings")
        def identity(row):
            return (row.get("sample_id"), row["input_tokens"], row.get("prompt_sha256"),
                    row.get("output_budget"), row.get("images"))
        if (workload.get("dataset") != baseline.get("dataset")
            or [identity(row) for row in measured] != [identity(row) for row in baseline["requests"]
                                                     if row["phase"] == "measure"]):
            raise ValueError("baseline must use the same dataset, prompts and output budgets")
        if any(row.get("accepted_draft_tokens") is not None for row in baseline["requests"]):
            raise ValueError("benchmark baseline must use ordinary decoding")
        values["decode_speedup"] = values["decode_tps"] / _positive(
            baseline_metrics["decode_tps"]["value"], "baseline decode TPS"
        )
    return {
        name: {"value": value, "unit": GENAI_METRICS[name][0], "direction": GENAI_METRICS[name][1],
               "aggregation": "mean", "sample_count": len(measured), "minimum": None, "maximum": None}
        for name, value in values.items()
    }


def validate_report(result: dict[str, Any]) -> None:
    report = result.get("benchmark_report", result)
    settings = report["settings"]
    speculative = settings.get("decoding") == "speculative"
    if report["status"] != "passed":
        raise ValueError("generation benchmark did not complete")
    if speculative:
        names = (*TEXT_DATASETS, *(IMAGE_DATASETS if settings.get("image_size") else ()))
        expected = [f"dataset_{name}" for name in names]
        if not report.get("baseline"):
            raise ValueError("speculative benchmarks require an ordinary baseline")
    else:
        expected = [f"text_n{length}" for length in input_buckets(settings["max_context"])]
        if settings.get("image_size"):
            expected.append("image")
    if [row["id"] for row in report["workloads"]] != expected:
        raise ValueError("GenAI workloads do not cover the declared text/image inputs")
    for workload in report["workloads"]:
        if speculative and (not workload.get("baseline") or any(
            row.get("accepted_draft_tokens") is None for row in workload["requests"]
        )):
            raise ValueError("speculative workloads require draft counters and an ordinary baseline")
        if workload["metrics"] != workload_metrics(workload):
            raise ValueError("GenAI metrics disagree with recorded request timings")
        if workload["kind"] == "image_text" and workload["image_size"] != settings["image_size"]:
            raise ValueError("image does not match the compiled vision encoder size")


def _target(deployment: Path) -> Path:
    from sima_lmm.devkit.model_manager import resolve_target_and_draft_paths

    return resolve_target_and_draft_paths(deployment)[0]


def _config(deployment: Path) -> dict:
    return json.loads((_target(deployment) / "devkit/vlm_config.json").read_text())


def _speculative(config: dict) -> bool:
    return bool(config["lm_cfg"].get("speculative_decoding_cfg"))


@contextmanager
def _server(deployment: Path, output: Path):
    from sima_lmm.mole.modalixboard import ModalixBoard

    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    board = ModalixBoard(address=None, port=port, model=str(deployment))
    try:
        board.start_server(output / "mole-server.log")
        yield board
    finally:
        board.stop_server()


@contextmanager
def _client(board, deployment: Path, context: int, *, image: bool = False):
    if image:
        from sima_lmm.mole.multimodal import MultimodalClient

        lm = MultimodalClient(board, context, enable_thinking=False)
    else:
        from sima_lmm.mole.modalixlm import ModalixLM

        lm = ModalixLM(board=board, pretrained=str(_target(deployment) / "devkit"), device="cpu", max_length=context)
    try:
        yield lm
    finally:
        if image:
            lm.close()
        else:
            lm.zmq_socket.close(linger=0)
            lm.zmq_context.term()


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
        raise ValueError("native chat template does not fit the requested input bucket")
    return tokens[: length - suffix] + tokens[-suffix:] if suffix else tokens[:length]


def _text(lm, context: int) -> list[dict]:
    import torch

    path = Path(lm.pretrained) / "generation_config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    stops = config.get("eos_token_id", lm.tokenizer.eos_token_id)
    stops = [stops] if isinstance(stops, int) else list(stops or [])
    workloads = []
    for length in input_buckets(context):
        tokens = _prompt(lm.tokenizer, length)
        workload = {"id": f"text_n{length}", "kind": "text", "input_tokens": length, "requests": []}
        telemetry = BoardTelemetry(workload["id"])
        for index in range(REPETITIONS + 1):
            with telemetry.measure(enabled=index > 0):
                durations = lm._zmq_infer(torch.tensor([tokens]), "generate_for_perf",
                    max_num_tokens=length + NEW_TOKENS, stop_token_ids=stops).cpu().tolist()[0]
            workload["requests"].append(_record(durations, lm.last_profile_data, length, index))
        workload["metrics"] = workload_metrics(workload)
        workload.update(telemetry.report())
        workloads.append(workload)
    return workloads


def _record(durations: list, metadata: dict, inputs: int, index: int) -> dict:
    if not durations or len(durations) != metadata.get("generated_tokens"):
        raise ValueError("MoLE did not return one timing per generated token")
    return {"phase": "warmup" if index == 0 else "measure", "input_tokens": inputs,
            "token_durations_s": durations, "generated_tokens": len(durations),
            "decode_tokens": len(durations) - 1, "ttft_s": durations[0], "decode_time_s": sum(durations[1:]),
            "accepted_draft_tokens": metadata.get("accepted_draft_tokens")}


def _image(lm, size: list[int]) -> dict:
    from PIL import Image
    from sima_lmm.mole.multimodal import ordered_content

    source = Path("/usr/share/sima_lmm/assets/sjc.jpg")
    height, width = size
    with Image.open(source) as image:
        resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        content, images = ordered_content(IMAGE_PROMPT, [resized])
    workload = {"id": "image", "kind": "image_text", "image_size": size, "prompt": IMAGE_PROMPT,
                "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "images": images, "requests": []}
    telemetry = BoardTelemetry("image")
    lm.info()
    for index in range(REPETITIONS + 1):
        with telemetry.measure(enabled=index > 0):
            metadata = lm.profile([{"role": "user", "content": content}], {"max_new_tokens": NEW_TOKENS})
        workload["requests"].append(_record(metadata["token_durations"], metadata, metadata["input_tokens"], index))
    workload["metrics"] = workload_metrics(workload)
    workload.update(telemetry.report())
    return workload


class _MeasuredClient:
    """Wrap only inference calls; MoLE's preparation and first warmup stay outside telemetry."""

    def __init__(self, client, telemetry):
        self.client, self.telemetry = client, telemetry
        self.prompts = []

    def __getattr__(self, name):
        value = getattr(self.client, name)
        if name not in ("_zmq_infer", "profile"):
            return value

        def measured(*args, **kwargs):
            prompt = hashlib.sha256(args[0].cpu().numpy().tobytes()).hexdigest() if name == "_zmq_infer" else None
            with self.telemetry.measure(enabled=bool(self.prompts)):
                result = value(*args, **kwargs)
            self.prompts.append(prompt)
            return result

        return measured


def _datasets(board, deployment: Path, output: Path, context: int, names: tuple) -> list[dict]:
    from sima_lmm.mole.perf_runner import PerformanceOptions, run_performance

    workloads = []
    for name in names:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        image = name in IMAGE_DATASETS
        telemetry = BoardTelemetry(name)
        with _client(board, deployment, context, image=image) as lm:
            measured = _MeasuredClient(lm, telemetry)
            config = SimpleNamespace(performance=PerformanceOptions(dataset=name), model_id=deployment.name,
                output_path=directory, board=board, do_start_server=True, max_num_tokens=context,
                max_new_tokens=NEW_TOKENS, limit=None, random_seed=42, input_lengths=None)
            result = run_performance(measured, config)
        requests = result["requests"]
        if not image:
            for row, prompt in zip(requests, measured.prompts, strict=True):
                row["prompt_sha256"] = prompt
        workload = {"id": f"dataset_{name}", "kind": "image_dataset" if image else "text_dataset",
                    "dataset": result["dataset"], "requests": requests, "evaluator": result["evaluator"]}
        workload["metrics"] = workload_metrics(workload)
        workload.update(telemetry.report())
        workloads.append(workload)
    return workloads


def run_benchmark(deployment: Path, output: Path, baseline: Path | None = None) -> dict[str, Any]:
    """Measure ordinary buckets/images, or full speculative datasets against an ordinary deployment."""
    import torch

    from model_report import from_generation

    deployment = deployment.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _config(deployment)
    context = config["pipeline_cfg"]["max_num_tokens"]
    speculative = _speculative(config)
    vision = config.get("vm_cfg")
    size = vision["image_size"] if vision else None
    size = [size, size] if isinstance(size, int) else size
    baseline_identity = json.loads(baseline.read_text()) if baseline else None
    baseline_deployment = (baseline.parent / baseline_identity["deployment"]).resolve() if baseline else None
    baseline_config = _config(baseline_deployment) if baseline else None
    if speculative and (baseline_config is None or _speculative(baseline_config)):
        raise ValueError("speculative benchmarks need a compiled ordinary baseline")
    if baseline_config and (baseline_config["pipeline_cfg"]["max_num_tokens"] != context
                            or baseline_config.get("vm_cfg") != vision):
        raise ValueError("baseline must have the same context and vision encoder configuration")
    names = (*TEXT_DATASETS, *(IMAGE_DATASETS if vision else ()))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with _server(deployment, output / "target") as board:
            if speculative:
                workloads = _datasets(board, deployment, output / "target", context, names)
            else:
                with _client(board, deployment, context) as lm:
                    workloads = _text(lm, context)
                if vision:
                    with _client(board, deployment, context, image=True) as lm:
                        workloads.append(_image(lm, size))
        if speculative:
            with _server(baseline_deployment, output / "baseline") as board:
                ordinary = _datasets(board, baseline_deployment, output / "baseline", context, names)
            for workload, reference in zip(workloads, ordinary, strict=True):
                workload["baseline"] = reference
                workload["metrics"] = workload_metrics(workload)
    finally:
        torch.set_num_threads(threads)
    report = {"status": "passed", "runtime": {"mole_version": version("sima-lmm")}, "workloads": workloads,
        "settings": {"max_context": context, "max_new_tokens": NEW_TOKENS, "image_size": size,
                     "decoding": "speculative" if speculative else "ordinary", "warmup": 1,
                     "repetitions": 1 if speculative else REPETITIONS, "sampling": "greedy",
                     "cache_policy": "empty target/draft KV cache before every request"},
        "timing": {"text_ttft": "language prefill through first token; excludes tokenization and input preparation",
                   "image_ttft": "native image preprocessing, vision encoding and language prefill through first token",
                   "decode_tps": "tokens after the first / sum of their durations",
                   "draft_contribution": "accepted draft tokens / all generated tokens",
                   "decode_speedup": "aggregate speculative decode TPS / aggregate ordinary decode TPS",
                   "excluded": "model loading, network transport and dataset preparation"}}
    if speculative:
        report["baseline"] = baseline_identity
    validate_report(report)
    (output / "benchmark-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    installed = subprocess.check_output(["dpkg-query", "-W", "-f=${Version}", "sima-lmm-core"], text=True).strip()
    from_generation(report, {"llima": installed, "mole": report["runtime"]["mole_version"], "build": {}}).save(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a compiled LLM/VLM with MoLE performance workloads.")
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="staged ordinary benchmark baseline receipt")
    args = parser.parse_args()
    run_benchmark(args.deployment, args.output, args.baseline)
