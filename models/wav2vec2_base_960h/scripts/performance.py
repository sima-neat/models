"""Measure warm end-to-end speech transcription on FLEURS."""

import argparse
import hashlib
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from benchmark_runtime import SAMPLE_RATE, Transcriber, read_audio, recordings, runtime_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    name, sep, path = args.dataset.partition("=")
    if name != "fleurs-en-us-test" or not sep:
        parser.error("Expected fleurs-en-us-test=directory")
    directory = Path(path)
    parts = []
    remaining = 30 * SAMPLE_RATE
    selected = []
    for row in recordings(directory):
        audio = read_audio(directory / "audio/test" / row[1])
        if len(audio) != int(row[5]):
            raise ValueError(f"Invalid sample count: {row[1]}")
        take = min(remaining, len(audio))
        parts.append(audio[:take])
        selected.append({"file": row[1], "samples_used": take})
        remaining -= take
        if not remaining:
            break
    if remaining:
        raise ValueError("Expected at least 30 seconds of speech")
    audio = np.concatenate(parts)
    mpk = Path(args.mpk.removeprefix("model=")).resolve(strict=True)
    package = runtime_directory(mpk, args.resources.resolve(strict=True) if args.resources else None)
    cold_start = time.perf_counter()
    with Transcriber(mpk, package, threads=args.threads) as transcriber:
        load_seconds = time.perf_counter() - cold_start
        first_start = time.perf_counter()
        transcriber.transcribe(audio)
        first_seconds = time.perf_counter() - first_start
        transcriber.transcribe(audio)
        timings = []
        monitor = pyneat.TelemetryMonitor()
        monitor.start()
        start = time.perf_counter()
        try:
            while len(timings) < 10 or time.perf_counter() - start < 60:
                t0 = time.perf_counter()
                transcript = transcriber.transcribe(audio)
                timings.append(time.perf_counter() - t0)
            span = time.perf_counter() - start
        finally:
            telemetry = monitor.stop()
        benchmark = pyneat.BenchmarkReport()
        benchmark.latency_ms = float(np.mean(timings) * 1000)
        for percentile in (50, 95, 99):
            setattr(benchmark, f"latency_p{percentile}_ms", float(np.percentile(timings, percentile) * 1000))
        benchmark.fps = len(timings) / span
        benchmark.throughput_duration_ms = span * 1000
        benchmark.throughput_samples = len(timings)
        report = model_report.from_performance(
            benchmark,
            {"model": transcriber.model},
            telemetry=telemetry,
            mla_layer_stats={"model": {"status": "not requested", "layers": []}},
            mla_stage_stats={"model": {"status": "not requested", "stages": []}},
            workload=(
                "30 seconds made by concatenating FLEURS recordings, five-second windows with one-second overlap; "
                f"{args.threads} CPU threads, full CPU frontend, MLA encoder and CTC decoding. "
                "Warm, unpaced processing; WAV decoding and load excluded."
            ),
        )
        report.document["metrics"]["real_time_factor"] = span / (len(timings) * 30)
        report.evidence["asr-timings.json"] = {
            "filenames": selected,
            "threads": args.threads,
            "audio_sha256": hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest(),
            "cold_model_initialization_seconds": load_seconds,
            "cold_first_transcription_seconds": first_seconds,
            "durations_seconds": timings,
            "transcript": transcript,
        }
        report.save(args.output)


if __name__ == "__main__":
    main()
