"""Benchmark RF-DETR Small with Neat Core and save its report.

With one --mpk, Core's Model.benchmark measures that MPK. With --mpk backbone=<path> --mpk decoder=<path>,
the script measures the whole model as it runs: the backbone, the top 300 proposals on the CPU, and the
decoder, each stage working on a different frame at the same time.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pyneat

IMAGE_SIZE = 512
QUERY_COUNT = 300
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4


def benchmark_mpk(mpk: str, output: Path) -> None:
    model = pyneat.Model(mpk)
    # Request 1000 samples and a 60 s minimum measurement duration.
    benchmark_options = pyneat.BenchmarkOptions()
    benchmark_options.num_samples = 1000
    benchmark_options.min_duration_ms = 60000
    benchmark_options.include_layer_stats = True
    model.benchmark(benchmark_options).save(output / "benchmark-report.json")
    # What the report does not record: the Core release, the requested run, and the graph that ran.
    context = {
        "core_version": pyneat.__version__,
        "num_samples": benchmark_options.num_samples,
        "min_duration_ms": benchmark_options.min_duration_ms,
        "include_layer_stats": benchmark_options.include_layer_stats,
        "execution_plan": json.loads(model.execution_plan().to_json()),
    }
    (output / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The input is already a resized, normalized image; Neat still quantizes and tessellates it.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def select_top_proposals(scores: pyneat.Tensor, boxes: pyneat.Tensor) -> pyneat.Tensor:
    """calibration.select_top_proposals, without the Pillow import this stage cannot load."""
    scores, boxes = scores.to_numpy(), boxes.to_numpy()
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("proposal scores and boxes must be finite")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :QUERY_COUNT]
    proposals = np.take_along_axis(boxes, order[..., None], axis=1)
    # The decoder may read the proposals after the backbone outputs are released, so they are copied.
    return pyneat.Tensor.from_numpy(proposals, copy=True, memory=pyneat.TensorMemory.EV74)


def pull(runner: pyneat.ModelRunner, name: str) -> list[pyneat.Tensor]:
    tensors = runner.pull(TIMEOUT_MS).tensors
    if not tensors:
        raise RuntimeError(f"the {name} produced no output within {TIMEOUT_MS} ms")
    return tensors


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def benchmark_pipeline(mpks: dict[str, Path], output: Path) -> None:
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = pyneat.Model(str(mpks["backbone"]), tensor_options())
    decoder = pyneat.Model(str(mpks["decoder"]), tensor_options())
    # One synthetic image, placed once in the memory the backbone reads, is reused for every frame.
    pixels = np.random.default_rng(0).standard_normal((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    image = pyneat.Tensor.from_numpy(pixels, copy=True, memory=pyneat.TensorMemory.EV74)
    backbone_runner = backbone.build([image], run_options=run_options)
    features, scores, boxes = backbone_runner.run([image], timeout_ms=TIMEOUT_MS)
    decoder_runner = decoder.build([features, select_top_proposals(scores, boxes)], run_options=run_options)
    del features, scores, boxes

    # Latency: one frame at a time through all three stages.
    latencies = []
    for frame in range(WARMUP_FRAMES + MEASURED_FRAMES):
        started = time.perf_counter()
        features, scores, boxes = backbone_runner.run([image], timeout_ms=TIMEOUT_MS)
        decoder_runner.run([features, select_top_proposals(scores, boxes)], timeout_ms=TIMEOUT_MS)
        if frame >= WARMUP_FRAMES:
            latencies.append(time.perf_counter() - started)
    del features, scores, boxes

    # Throughput: a feeder, a CPU bridge, and a collector keep both runners busy. Each blocks in Core
    # with the GIL released while it waits, and the bridge drops the backbone outputs as soon as the
    # decoder holds them, so the backbone's output buffers return to it.
    # One entry per frame pushed into a runner, then None once its writer stops, so each reader pulls
    # only frames that will arrive.
    backbone_frames, decoder_frames = queue.SimpleQueue(), queue.SimpleQueue()
    stop = threading.Event()
    errors: list[BaseException] = []
    proposal_seconds: list[float] = []
    window: dict[str, object] = {}

    def feed() -> None:
        while not stop.is_set():
            if not backbone_runner.push([image]):
                raise RuntimeError("the backbone rejected a frame")
            backbone_frames.put(True)

    def bridge() -> None:
        for _ in iter(backbone_frames.get, None):
            features, scores, boxes = pull(backbone_runner, "backbone")
            started = time.perf_counter()
            proposals = select_top_proposals(scores, boxes)
            proposal_seconds.append(time.perf_counter() - started)
            if not decoder_runner.push([features, proposals]):
                raise RuntimeError("the decoder rejected a frame")
            del features, scores, boxes, proposals
            decoder_frames.put(True)

    def collect() -> None:
        monitor = pyneat.TelemetryMonitor()
        outputs = 0
        for _ in iter(decoder_frames.get, None):
            pull(decoder_runner, "decoder")
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                window["started"] = time.perf_counter()
            elif "frames" not in window and outputs - WARMUP_FRAMES >= MEASURED_FRAMES:
                elapsed = time.perf_counter() - window["started"]
                if elapsed >= MIN_DURATION_S:
                    window.update(frames=outputs - WARMUP_FRAMES, seconds=elapsed, telemetry=monitor.stop())
                    stop.set()

    def stage(work, done: queue.SimpleQueue | None):
        def run() -> None:
            try:
                work()
            except BaseException as error:
                errors.append(error)
                stop.set()
                # Ending both inputs unblocks a stage waiting to push.
                backbone_runner.close_input()
                decoder_runner.close_input()
            finally:
                if done is not None:
                    done.put(None)

        return threading.Thread(target=run, name=work.__name__)

    threads = [stage(feed, backbone_frames), stage(bridge, decoder_frames), stage(collect, None)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    backbone_runner.close()
    decoder_runner.close()
    if errors:
        raise errors[0]
    if "frames" not in window:
        raise RuntimeError("the pipeline stopped before the measurement window ended")

    frames, seconds, telemetry = window["frames"], window["seconds"], window["telemetry"]
    fps = frames / seconds
    window_proposal_seconds = proposal_seconds[WARMUP_FRAMES : WARMUP_FRAMES + frames]
    telemetry.save(output / "telemetry.json")
    power = telemetry.power_som_watts
    report = {
        "latency_ms": float(np.mean(latencies) * 1000),
        "latency_p50_ms": percentile_ms(latencies, 50),
        "latency_p95_ms": percentile_ms(latencies, 95),
        "latency_p99_ms": percentile_ms(latencies, 99),
        "fps": fps,
        "avg_power_watts": power.mean if power.samples else 0.0,
        "peak_power_watts": power.maximum if power.samples else 0.0,
        # Core's convention: the energy of MEASURED_FRAMES frames at the measured rate and power.
        "energy_joules": power.mean * MEASURED_FRAMES / fps if power.samples else 0.0,
        "throughput_duration_ms": seconds * 1000,
        "throughput_samples": frames,
        "telemetry": json.loads((output / "telemetry.json").read_text(encoding="utf-8")),
        "layer_stats": [],
    }
    (output / "benchmark-report.json").write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    context = {
        "core_version": pyneat.__version__,
        "num_samples": MEASURED_FRAMES,
        "min_duration_ms": MIN_DURATION_S * 1000,
        "include_layer_stats": False,
        "warmup_frames": WARMUP_FRAMES,
        "queue_depth": QUEUE_DEPTH,
        "cpu_stage": {
            "name": f"Top {QUERY_COUNT} proposals",
            "avg_ms": float(np.mean(window_proposal_seconds) * 1000),
            "p99_ms": percentile_ms(window_proposal_seconds, 99),
        },
    }
    (output / "benchmark-context.json").write_text(json.dumps(context, allow_nan=False), encoding="utf-8")
    print(
        f"{fps:.1f} FPS over {frames} frames in {seconds:.1f} s; latency {report['latency_ms']:.2f} ms; "
        f"top {QUERY_COUNT} {context['cpu_stage']['avg_ms']:.2f} ms; power {report['avg_power_watts']:.2f} W"
    )


def parse_mpks(values: list[str]) -> dict[str, Path]:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"backbone", "decoder"}:
        raise SystemExit("--mpk must name one MPK, or exactly backbone=<path> and decoder=<path>")
    return {name: Path(path) for name, path in mpks.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="an MPK, or component=path for both")
    parser.add_argument(
        "--output", required=True, help="directory for benchmark-report.json and benchmark-context.json"
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if len(args.mpk) == 1 and "=" not in args.mpk[0]:
        benchmark_mpk(args.mpk[0], output)
    else:
        benchmark_pipeline(parse_mpks(args.mpk), output)


if __name__ == "__main__":
    main()
