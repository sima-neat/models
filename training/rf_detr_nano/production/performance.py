"""Benchmark RF-DETR Nano with Neat Core and save its report.

Given --mpk backbone=<path> --mpk decoder=<path> and --dataset coco-2017-val=<extracted directory>, the
script measures ordered AA preprocessing, both QAT components, CPU proposal selection and final boxdecode
on one seeded COCO image in a 720p frame. The CPU bridge/decoder is a NumPy oracle, not the optimized
native kernel implementation. This is local qualification, not a published production profile.
It also records each MPK's MLA layer and stage stats and its execution plan.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from contract import load_contract, output_grids, parse_components, verify_packages
from runtime import decode_outputs, proposal_inputs
from runtime import image_options as qat_image_options

IMAGE_SIZE = 384
QUERY_COUNT = 300
WARMUP_FRAMES = 100
MEASURED_FRAMES = 1000
MIN_DURATION_S = 60
TIMEOUT_MS = 10000
# Frames each runner may queue; deeper queues add latency without adding throughput.
QUEUE_DEPTH = 4
IMAGES = "val2017"
# The benchmark input is a 720p camera frame, letterboxed with grey padding.
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114


def percentile_ms(seconds: list[float], percentile: float) -> float:
    return float(np.percentile(seconds, percentile) * 1000)


def mla_stats(mpk: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Core's MLA layer and stage stats for one MPK, in the shape Core saves them. Their throughput is not the
    model's, so Core's default run serves."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(str(mpk)).benchmark(options)
    layer_stats, stage_stats = report.mla_layer_stats, report.mla_stage_stats
    layers = [
        {"stage_index": layer.stage_index, "name": layer.name, "latency_us": layer.latency_us}
        for layer in layer_stats.layers
    ]
    stages = [{"stage_index": stage.stage_index, "latency_us": stage.latency_us} for stage in stage_stats.stages]
    return {"status": layer_stats.status, "layers": layers}, {"status": stage_stats.status, "stages": stages}


def sample_image(dataset: Path) -> Path:
    """One val2017 image, chosen with a fixed seed so every run measures the same input."""
    images = sorted((dataset / IMAGES).glob("*.jpg"))
    if not images:
        raise SystemExit(f"{dataset} has no images under {IMAGES}/")
    return random.Random(0).choice(images)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox_to_720p(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit a 1280x720 frame and pad it to the frame, centered."""
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = round(image.shape[1] * scale), round(image.shape[0] * scale)
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def frame_tensor(path: Path) -> pyneat.Tensor:
    """The image's 720p frame, placed once in the memory the backbone reads, reused for every frame."""
    frame = letterbox_to_720p(read_rgb(path))
    return pyneat.Tensor.from_numpy(
        frame, copy=True, image_format=pyneat.PixelFormat.RGB, memory=pyneat.TensorMemory.EV74
    )


def build_runners(models: dict[str, pyneat.Model], image: pyneat.Tensor) -> tuple[pyneat.ModelRunner, ...]:
    """A runner for each model. The decoder's is built from one real backbone output."""
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    backbone = models["backbone"].build([image], run_options=run_options)
    try:
        outputs = backbone.run([image], timeout_ms=TIMEOUT_MS)
        decoder = models["decoder"].build(proposal_inputs(pyneat, outputs), run_options=run_options)
    except Exception:
        backbone.close()
        raise
    return backbone, decoder


def infer_one(backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor, grids: tuple) -> None:
    outputs = backbone.run([image], timeout_ms=TIMEOUT_MS)
    inputs = proposal_inputs(pyneat, outputs)
    decoded = decoder.run(inputs, timeout_ms=TIMEOUT_MS)
    decode_outputs(decoded, grids, width=FRAME_WIDTH, height=FRAME_HEIGHT, image_id=0)


def measure_latency(
    backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor, grids: tuple
) -> list[float]:
    """Seconds per frame through all three stages with nothing else in flight, after the warm-up."""
    for _ in range(WARMUP_FRAMES):
        infer_one(backbone, decoder, image, grids)
    seconds: list[float] = []
    ends = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < ends:
        started = time.perf_counter()
        infer_one(backbone, decoder, image, grids)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    """The measured span of the throughput pass: the frames the decoder output, how long they took, and the
    board telemetry sampled over the same span."""

    frames: int
    seconds: float
    telemetry: pyneat.Telemetry

    @property
    def fps(self) -> float:
        return self.frames / self.seconds


def push(runner: pyneat.ModelRunner, inputs: list[pyneat.Tensor], name: str) -> None:
    if not runner.push(inputs):
        raise RuntimeError(f"the {name} rejected a frame")


def drain(runner: pyneat.ModelRunner, name: str, pushed: dict[str, int]) -> Iterator[list[pyneat.Tensor]]:
    """Each output until the runner's end of stream. Fewer outputs than frames pushed means a stall.

    Core's pull returns no tensors both at end of stream and on a timeout, so the count tells them apart.
    """
    pulled = 0
    while tensors := runner.pull(TIMEOUT_MS).tensors:
        pulled += 1
        yield tensors
        # Return the outputs to the runner before waiting for the next frame.
        del tensors
    if pulled != pushed[name]:
        raise RuntimeError(f"the {name} stalled: {pulled} of {pushed[name]} frames came out within {TIMEOUT_MS} ms")


def run_together(*stages: Callable[[], None], runners: tuple[pyneat.ModelRunner, ...]) -> None:
    """Run each stage on its own thread. The first failure ends every runner's input, which unblocks a stage
    waiting to push, and is raised once all stages stop. Each stage waits in Core with the GIL released."""
    errors: list[BaseException] = []

    def run(stage: Callable[[], None]) -> None:
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            for runner in runners:
                runner.close_input()

    threads = [threading.Thread(target=run, args=(stage,), name=stage.__name__) for stage in stages]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def measure_throughput(
    backbone: pyneat.ModelRunner, decoder: pyneat.ModelRunner, image: pyneat.Tensor, grids: tuple
) -> Throughput:
    """Frames per second and board telemetry with every stage busy, each on a different frame. The measured span
    starts after WARMUP_FRAMES outputs and lasts at least MEASURED_FRAMES frames and MIN_DURATION_S seconds."""
    monitor = pyneat.TelemetryMonitor()
    measured = threading.Event()
    throughput: Throughput | None = None
    pushed = {"backbone": 0, "decoder": 0}

    def push_inputs() -> None:
        try:
            while not measured.is_set():
                push(backbone, [image], "backbone")
                pushed["backbone"] += 1
        finally:
            # End of stream follows the last frame through the backbone, then the decoder.
            backbone.close_input()

    def forward_proposals() -> None:
        try:
            for outputs in drain(backbone, "backbone", pushed):
                inputs = proposal_inputs(pyneat, outputs)
                push(decoder, inputs, "decoder")
                pushed["decoder"] += 1
                # The decoder owns its submitted inputs until actual completion.
                del outputs, inputs
        finally:
            decoder.close_input()

    def pull_outputs() -> None:
        nonlocal throughput
        outputs, started = 0, 0.0
        for tensors in drain(decoder, "decoder", pushed):
            # Decoded completion, not raw model completion, defines the timer.
            decode_outputs(tensors, grids, width=FRAME_WIDTH, height=FRAME_HEIGHT, image_id=0)
            del tensors
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_FRAMES and not measured.is_set():
                frames, seconds = outputs - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    throughput = Throughput(frames, seconds, monitor.stop())
                    measured.set()
        if throughput is None:
            raise RuntimeError("the pipeline stopped before the measured span ended")

    try:
        run_together(push_inputs, forward_proposals, pull_outputs, runners=(backbone, decoder))
    finally:
        # A failure inside the measured span leaves the monitor running.
        if monitor.running():
            monitor.stop()
        backbone.close()
        decoder.close()
    return throughput


def write_report(
    output: Path,
    image: str,
    latencies: list[float],
    throughput: Throughput,
    models: dict[str, pyneat.Model],
    mla_layer_stats: dict[str, dict[str, object]],
    mla_stage_stats: dict[str, dict[str, object]],
) -> None:
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    benchmark.latency_p50_ms = percentile_ms(latencies, 50)
    benchmark.latency_p95_ms = percentile_ms(latencies, 95)
    benchmark.latency_p99_ms = percentile_ms(latencies, 99)
    benchmark.fps = throughput.fps
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        models,
        workload=(
            f"{image} in a {FRAME_WIDTH}x{FRAME_HEIGHT} RGB frame, Neat preprocessing stretched to "
            f"{IMAGE_SIZE}x{IMAGE_SIZE} with ordered AA/INT8 source math, backbone, CPU NumPy oracle "
            f"TopK/Gather, decoder and final CPU boxdecode (not optimized native kernel), pipelined "
            f"{QUEUE_DEPTH} frames deep"
        ),
        telemetry=throughput.telemetry,
        mla_layer_stats=mla_layer_stats,
        mla_stage_stats=mla_stage_stats,
    )
    report.save(output)
    print(
        f"{benchmark.fps:.1f} FPS over {throughput.frames} frames in {throughput.seconds:.1f} s; "
        f"latency {benchmark.latency_ms:.2f} ms over {len(latencies)} frames"
    )


def benchmark_pipeline(mpks: dict[str, Path], dataset_id: str, dataset: Path, output: Path, contract: dict) -> None:
    verify_packages(contract, mpks)
    grids = output_grids(contract)
    mla_layer_stats, mla_stage_stats = {}, {}
    for name, mpk in mpks.items():
        mla_layer_stats[name], mla_stage_stats[name] = mla_stats(mpk)
    models = {
        "backbone": pyneat.Model(str(mpks["backbone"]), qat_image_options(pyneat, contract, FRAME_WIDTH, FRAME_HEIGHT)),
        "decoder": pyneat.Model(str(mpks["decoder"])),
    }
    path = sample_image(dataset)
    image = frame_tensor(path)
    backbone, decoder = build_runners(models, image)
    try:
        latencies = measure_latency(backbone, decoder, image, grids)
    except BaseException:
        backbone.close()
        decoder.close()
        raise
    # The throughput routine owns closure from this point, including failures.
    throughput = measure_throughput(backbone, decoder, image, grids)
    write_report(output, f"{dataset_id} {path.name}", latencies, throughput, models, mla_layer_stats, mla_stage_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="component=path, for each component")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", required=True, help="directory for model-report.json")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        parser.error("--dataset must name coco-2017-val")
    contract = load_contract(args.contract, args.contract_sha256)
    benchmark_pipeline(parse_components(args.mpk), dataset_id, Path(dataset_dir), Path(args.output), contract)


if __name__ == "__main__":
    main()
