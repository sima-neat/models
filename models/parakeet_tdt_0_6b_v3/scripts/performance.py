"""Measure complete ASR processing and live-style time to first visible text.

Usage: PYTHONPATH=tools python performance.py --mpk encoder_mpk.tar.gz \
    --dataset fleurs-en-us-test=extracted-dataset --output results

Use the same 30 s of FLEURS English test speech for both models: concatenate
recordings in filename order and retain the first 480,000 samples, without
adding silence. The report identifies each recording and the resulting PCM.
--audio optionally takes a local PCM16 or float32 WAV instead. The unpaced
pass measures full-recording latency and RTF, including log-mel extraction,
encoder calls and INT8 CPU prediction/joining. Overlapping context counts once
in audio duration. A separate paced pass waits for each chunk's input, including
lookahead, before processing it. Time to first text starts at the first audio
sample and ends at the first non-whitespace decoded text (leading silence counts).
Loading, WAV decoding and warm-up are outside both passes. Separate encoder
layer/stage profiling is not requested.

The model runs through benchmark_runtime.py, which accuracy.py shares; neither is shipped in the model download.
The first CI runs collect measurements; regression thresholds need a real baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import model_report
import numpy as np
from benchmark_runtime import SAMPLE_RATE, Transcriber, read_audio, runtime_directory

WARMUP_RECORDINGS = 2
MEASURED_RECORDINGS = 10
MIN_DURATION_S = 60
FIRST_TEXT_TRIALS = 5


def load_audio(path):
    audio = read_audio(path)
    if len(audio) < 30 * SAMPLE_RATE:
        raise ValueError("Use at least 30 seconds of real speech to exercise steady-state windows")
    return audio


def load_dataset(directory):
    rows = [line.split('\t') for line in (directory / 'test.tsv').read_text(encoding='utf-8').splitlines()]
    if not rows or any(len(row) != 7 for row in rows):
        raise ValueError("Expected the original seven-column FLEURS test.tsv")
    filenames = [row[1] for row in rows]
    if len(set(filenames)) != len(filenames) or any(
        Path(name).name != name or not name.endswith('.wav') for name in filenames
    ):
        raise ValueError("Expected unique, safe FLEURS WAV filenames")
    pieces, selected, remaining = [], [], 30 * SAMPLE_RATE
    for row in sorted(rows, key=lambda row: row[1]):
        path = directory / 'audio/test' / row[1]
        audio = read_audio(path)
        if len(audio) != int(row[5]):
            raise ValueError(f"FLEURS sample count differs from the TSV: {row[1]}")
        count = min(remaining, len(audio))
        pieces.append(audio[:count])
        selected.append({'id': f'en_us/test/{path.stem}', 'samples_used': count,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        remaining -= count
        if not remaining:
            break
    if remaining:
        raise ValueError("FLEURS input contains less than 30 seconds of audio")
    return np.concatenate(pieces), {'source': 'fleurs-en-us-test', 'recordings': selected}


def available_samples(index, total):
    """Last raw sample required by the runtime's chunk, including FFT context."""
    needed = index * 32000 + 64000
    return min(total, needed)


def transcribe_recording(transcriber, audio):
    started = time.perf_counter()
    text = ""
    for text in transcriber.transcribe(audio, "auto"):
        pass
    elapsed = time.perf_counter() - started
    if not text.strip():
        raise RuntimeError("No visible text: use speech audio and check the runtime before benchmarking")
    return elapsed, text


def time_to_first_text(transcriber, audio):
    started = time.perf_counter()
    iterator = transcriber.transcribe(audio, "auto")
    try:
        index = 0
        while True:
            ready = available_samples(index, len(audio)) / SAMPLE_RATE
            delay = started + ready - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            try:
                text = next(iterator)
            except StopIteration:
                raise RuntimeError("No visible text in the paced recording") from None
            if text.strip():
                return time.perf_counter() - started
            index += 1
    finally:
        iterator.close()


def benchmark_pipeline(transcriber, audio, input_metadata, output):
    import pyneat

    for _ in range(WARMUP_RECORDINGS):
        transcribe_recording(transcriber, audio)
    monitor = pyneat.TelemetryMonitor()
    seconds = []
    monitor.start()
    started = time.perf_counter()
    try:
        while len(seconds) < MEASURED_RECORDINGS or time.perf_counter() - started < MIN_DURATION_S:
            elapsed, transcript = transcribe_recording(transcriber, audio)
            seconds.append(elapsed)
        span = time.perf_counter() - started
    finally:
        telemetry = monitor.stop()
    # Each call resets predictor/encoder history while keeping model weights and the Run loaded.
    first_text = [time_to_first_text(transcriber, audio) for _ in range(FIRST_TEXT_TRIALS)]
    duration = len(audio) / SAMPLE_RATE
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = float(np.mean(seconds) * 1000)
    for percentile in (50, 95, 99):
        setattr(benchmark, f"latency_p{percentile}_ms", float(np.percentile(seconds, percentile) * 1000))
    benchmark.fps = len(seconds) / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = len(seconds)
    profile = '14 s windows (10 s past + 2 s new + 2 s lookahead)'
    digest = hashlib.sha256(audio.astype('<f4', copy=False).tobytes()).hexdigest()
    report = model_report.from_performance(
        benchmark, {"model": transcriber.model}, telemetry=telemetry,
        execution_plans={"model": {
            "status": "unavailable",
            "reason": "Core cannot export a Model execution plan for this inference-only Graph route",
            "evidence": "encoder-run.json",
        }},
        mla_layer_stats={"model": {"status": "not requested", "layers": []}},
        mla_stage_stats={"model": {"status": "not requested", "stages": []}},
        workload=(
            f"One {duration:.3f} s 16 kHz mono stream from {input_metadata['source']}, float32 PCM SHA-256 {digest}; "
            f"{profile}; auto language. "
            "Full log-mel frontend, BF16 encoder and INT8 CPU predictor/joiner, one recording at a time. "
            "Latency is unpaced full-recording transcription; throughput is recordings/s. "
            "RTF is unpaced processing seconds / unique audio seconds (overlap counted once). "
            "Time to first text is the mean of five paced replays from first audio sample to first visible "
            "text, including input buffering/lookahead and leading silence. Models stay loaded; "
            "WAV decoding and warm-up are excluded; separate encoder profiling is not requested. "
            "The executed inference graph is saved as encoder-run.json; Core does not expose "
            "a Model.execution_plan() for this inference-only Graph route."
        ),
    )
    report.document["metrics"].update(
        real_time_factor=span / (len(seconds) * duration),
        time_to_first_text_ms=float(np.mean(first_text) * 1000),
    )
    report.evidence["asr-timings.json"] = {
        "input": input_metadata, "audio_sha256": digest, "audio_samples": len(audio), "sample_rate": SAMPLE_RATE,
        "model_kind": "parakeet", "language": "auto", "unpaced_seconds": seconds,
        "unpaced_span_seconds": span, "paced_first_text_seconds": first_text,
        "final_transcript": transcript,
    }
    report.evidence["encoder-run.json"] = json.loads(transcriber.run.json())
    report.save(output)
    print(json.dumps(report.document["metrics"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="encoder archive, optionally model=path")
    parser.add_argument("--resources", type=Path, help="package resources; defaults to the encoder's directory")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path, help="same speech WAV for both models")
    inputs.add_argument("--dataset", help="fleurs-en-us-test=extracted-dataset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mpk = Path(args.mpk.removeprefix("model=")).resolve(strict=True)
    package = runtime_directory(mpk, args.resources.resolve(strict=True) if args.resources else None)
    if args.dataset:
        name, separator, path = args.dataset.partition('=')
        if name != 'fleurs-en-us-test' or not separator:
            parser.error("--dataset must be fleurs-en-us-test=directory")
        audio, input_metadata = load_dataset(Path(path))
    else:
        audio = load_audio(args.audio)
        input_metadata = {'source': args.audio.name, 'wav_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest()}
    with Transcriber(mpk, package) as transcriber:
        benchmark_pipeline(transcriber, audio, input_metadata, args.output)



if __name__ == "__main__":
    main()
