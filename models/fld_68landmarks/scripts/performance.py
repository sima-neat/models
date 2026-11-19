"""Benchmark a known face box through FAN to final landmarks, with truthful CPU preprocessing.

A seeded AFW face is placed in a 1280x720 RGB frame once. Each timed frame uses the same exact CPU crop/resize
as accuracy.py, normalizes the prepared uint8 RGB crop to BF16 in Neat, then runs FAN and decodes its heatmaps.
Face detection and image loading are excluded. The model execution plan therefore starts at a 256x256 crop,
not at the external 720p frame. Native ROI preprocessing is not enabled until its numeric parity is qualified.
The former 720p-to-heatmap benchmark can be run separately with --heatmap-only; its FPS is not this pipeline's FPS.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import model_report
import numpy as np
import pyneat
from accuracy import heatmaps_of
from landmarks import DATASET_ID, INPUT_SIZE, crop_rgb, dataset_path, decode

SPLIT = "test_data"
FRAME_WIDTH, FRAME_HEIGHT, FILL = 1280, 720, 114
WARMUP_FRAMES, MEASURED_FRAMES, MIN_DURATION_S = 100, 1000, 60
TIMEOUT_MS, QUEUE_DEPTH = 10000, 4

# For every integer pixel u in [0,255], trunc_BF16(u * 129/32768) equals
# RNE_BF16(float32(u) / 255). Neat ImagePreproc truncates BF16 while the tensor
# cast rounds to nearest-even. This derived stddev makes its multiplier exactly
# 129/32768; it is NOT general FP32 normalization or an INT8 recipe.
U8_BF16_STDDEV = np.float32(32768 / (129 * 255))


def sample_face(dataset: Path) -> Path:
    faces = sorted(
        path
        for path in (dataset / SPLIT).glob("*.jpg")
        if not path.name.startswith(".") and path.with_suffix(".pts").is_file()
    )
    if not faces:
        raise ValueError(f"{dataset} has no annotated faces under {SPLIT}/")
    return random.Random(0).choice(faces)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"could not read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def benchmark_frame(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The real face and its box in a 720p frame; rounded resize extents determine the transformed box."""
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0 or image.dtype != np.uint8:
        raise ValueError("benchmark requires a nonempty uint8 RGB image")
    scale = min(FRAME_WIDTH / image.shape[1], FRAME_HEIGHT / image.shape[0])
    width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    left, top = (FRAME_WIDTH - width) // 2, (FRAME_HEIGHT - height) // 2
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), FILL, np.uint8)
    frame[top : top + height, left : left + width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # AFW supplies a face crop, whose original face box is the complete unpadded image.
    return frame, np.array([left, top, left + width, top + height], dtype=np.float64)


def image_options():
    """Options only for the separately labeled historical heatmap-only diagnostic."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.input_max_width = FRAME_WIDTH
    options.preprocess.input_max_height = FRAME_HEIGHT
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    return options


def prepare_input(frame, face_box):
    patch, box = crop_rgb(frame, face_box)
    tensor = pyneat.Tensor.from_numpy(
        patch, copy=True, layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB
    )
    return [tensor], box


def prepared_rgb_options():
    """Normalize an already resized UInt8 crop to the reference BF16 values, with no native resize."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.input_max_width = options.preprocess.input_max_height = INPUT_SIZE
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    options.preprocess.normalize.mean = [0.0] * 3
    options.preprocess.normalize.stddev = [float(U8_BF16_STDDEV)] * 3
    options.preprocess.normalize.has_explicit_stats = True
    return options


def consume_outputs(tensors, box):
    if len(tensors) != 1:
        raise ValueError(f"expected one heatmap output, got {len(tensors)}")
    points, scores = decode(heatmaps_of(tensors[0], copy=False), box)
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("landmark decode produced nonfinite values")
    return points, scores


def infer_one(runner, frame, face_box):
    inputs, box = prepare_input(frame, face_box)
    return consume_outputs(runner.run(inputs, timeout_ms=TIMEOUT_MS), box)


def measure_latency(runner, frame, face_box):
    for _ in range(WARMUP_FRAMES):
        infer_one(runner, frame, face_box)
    seconds = []
    deadline = time.perf_counter() + MIN_DURATION_S
    while len(seconds) < MEASURED_FRAMES or time.perf_counter() < deadline:
        started = time.perf_counter()
        infer_one(runner, frame, face_box)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class Throughput:
    frames: int
    seconds: float
    telemetry: object


def measure_throughput(runner, frame, face_box):
    """One bounded producer and consumer; count outputs only after full landmark decoding."""
    done = threading.Event()
    errors = []
    pushed = 0
    result = None
    monitor = pyneat.TelemetryMonitor()
    # Every frame uses this same box. Its derivation still runs in prepare_input for each frame.
    _, output_box = crop_rgb(frame, face_box)

    def produce():
        nonlocal pushed
        try:
            while not done.is_set():
                inputs, _ = prepare_input(frame, face_box)
                if not runner.push(inputs):
                    raise RuntimeError("FAN runner rejected a frame")
                pushed += 1
        finally:
            runner.close_input()

    def consume():
        nonlocal result
        outputs, started = 0, 0.0
        while tensors := runner.pull(TIMEOUT_MS).tensors:
            consume_outputs(tensors, output_box)
            del tensors
            outputs += 1
            if outputs == WARMUP_FRAMES:
                monitor.start()
                started = time.perf_counter()
            elif outputs > WARMUP_FRAMES and not done.is_set():
                frames, seconds = outputs - WARMUP_FRAMES, time.perf_counter() - started
                if frames >= MEASURED_FRAMES and seconds >= MIN_DURATION_S:
                    result = Throughput(frames, seconds, monitor.stop())
                    done.set()
        if outputs != pushed:
            raise RuntimeError(f"FAN stalled: completed {outputs} of {pushed} frames")
        if result is None:
            raise RuntimeError("FAN ended before the measured span completed")

    def guarded(stage):
        try:
            stage()
        except BaseException as error:
            errors.append(error)
            done.set()
            try:
                runner.close_input()
            except BaseException as close_error:
                errors.append(close_error)

    threads = [
        threading.Thread(target=guarded, args=(stage,), name=f"fan-{stage.__name__}") for stage in (produce, consume)
    ]
    started_threads = []
    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        return result
    finally:
        done.set()
        try:
            runner.close_input()
        finally:
            for thread in started_threads:
                thread.join()
            if monitor.running():
                monitor.stop()


def mla_stats(mpk):
    """Separate idle-device diagnostics, never substituted for loaded CPU/MLA/post timings."""
    options = pyneat.BenchmarkOptions()
    options.include_mla_layer_stats = True
    options.include_mla_stage_stats = True
    report = pyneat.Model(str(mpk)).benchmark(options)
    return (
        {
            "status": report.mla_layer_stats.status,
            "layers": [
                {"stage_index": item.stage_index, "name": item.name, "latency_us": item.latency_us}
                for item in report.mla_layer_stats.layers
            ],
        },
        {
            "status": report.mla_stage_stats.status,
            "stages": [
                {"stage_index": item.stage_index, "latency_us": item.latency_us}
                for item in report.mla_stage_stats.stages
            ],
        },
    )


def benchmark_pipeline(mpk, face, output):
    frame, face_box = benchmark_frame(read_rgb(face))
    model = pyneat.Model(str(mpk), prepared_rgb_options())
    requirements = model.preprocess_requirements()
    if requirements.output_dtype != "EVXX_BFLOAT16" or requirements.quant_needed:
        raise ValueError("the prepared UInt8 crop route requires BF16 model inputs without quantization")
    run_options = pyneat.RunOptions()
    run_options.queue_depth = QUEUE_DEPTH
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    inputs, _ = prepare_input(frame, face_box)
    runner = model.build(inputs, run_options=run_options)
    try:
        latencies = measure_latency(runner, frame, face_box)
        throughput = measure_throughput(runner, frame, face_box)
    finally:
        runner.close()
    layers, stages = mla_stats(mpk)
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(latencies) * 1000)
    for percentile in (50, 95, 99):
        setattr(benchmark, f"latency_p{percentile}_ms", float(np.percentile(latencies, percentile) * 1000))
    benchmark.fps = throughput.frames / throughput.seconds
    benchmark.throughput_duration_ms = throughput.seconds * 1000
    benchmark.throughput_samples = throughput.frames
    report = model_report.from_performance(
        benchmark,
        {"model": model},
        telemetry=throughput.telemetry,
        mla_layer_stats={"model": layers},
        mla_stage_stats={"model": stages},
        workload=(
            f"{DATASET_ID} {SPLIT}/{face.name}, 1280x720 RGB frame plus known face box; "
            "timed CPU crop/resize -> 256x256 RGB image -> native BF16 normalization -> Neat inference "
            "-> landmark decode; "
            f"face detection excluded; {QUEUE_DEPTH} frames deep"
        ),
    )
    report.save(output)
    print(f"Face-box-to-landmarks: {benchmark.fps:.2f} FPS; latency {benchmark.latency_ms:.2f} ms")


def benchmark_heatmaps(mpk, face, output):
    model = pyneat.Model(str(mpk), image_options())
    options = pyneat.BenchmarkOptions()
    options.num_samples, options.min_duration_ms = 1000, 60000
    options.include_mla_layer_stats = options.include_mla_stage_stats = True
    frame, _ = benchmark_frame(read_rgb(face))
    options.inputs = [pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.RGB)]
    model_report.from_performance(
        model.benchmark(options),
        {"model": model},
        workload=f"HEATMAP-ONLY diagnostic: {DATASET_ID} {SPLIT}/{face.name}, 1280x720 RGB; crop/decode excluded",
    ).save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    parser.add_argument("--heatmap-only", action="store_true", help="separate historical diagnostic, not E2E")
    args = parser.parse_args(argv)
    face = sample_face(dataset_path(args.dataset))
    (benchmark_heatmaps if args.heatmap_only else benchmark_pipeline)(args.mpk, face, args.output)


if __name__ == "__main__":
    main()
