"""Time a native speech recognition deployment, such as Whisper, through Neat's GenAI server.

A model's scripts/performance.py calls main(). The runner passes --deployment (the LLiMa deployment folder),
--dataset fleurs-en-us-test=<extracted dataset> and --output. The input matches the cached ASR models'
benchmarks: the first 30 s of FLEURS English test speech, recordings in filename order. Whisper decodes one window
per request and stops at its first end of text, so it would transcribe only one recording of a concatenation; the
server instead receives each recording's part of those 30 s as its own 16 kHz PCM16 WAV upload to
/v1/audio/transcriptions, with automatic language detection, and one pass sends them all in order.

Needs only the Python standard library and PyNeat, so it runs in PyNeat's environment on a DevKit.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import socket
import statistics
import struct
import sys
import time
import urllib.request
import uuid
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16000
SECONDS = 30
WARMUP_REQUESTS = 2
MEASURED_REQUESTS = 10
MIN_DURATION_S = 60
FIRST_TEXT_TRIALS = 5
DATASET = "fleurs-en-us-test"
SERVED_NAME = "asr"
LANGUAGE = "auto"


def read_wav(path: Path) -> array:
    """A 16 kHz mono PCM16 or float32 WAV as float samples."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    offset, fmt, payload = 12, None, None
    while offset + 8 <= len(data):
        chunk, size = data[offset : offset + 4], struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body)
        elif chunk == b"data":
            payload = body
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"{path}: WAV without fmt and data chunks")
    tag, channels, rate, _, _, bits = fmt
    if channels != 1 or rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected 16 kHz mono audio")
    if (tag, bits) == (3, 32):
        samples = array("f")
        samples.frombytes(payload)
    elif (tag, bits) == (1, 16):
        samples = array("f", (value / 32768 for value in array("h", payload)))
    else:
        raise ValueError(f"{path}: expected PCM16 or float32 samples")
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def wav_bytes(samples: array) -> bytes:
    """Float samples as a 16 kHz mono PCM16 WAV."""
    pcm = array("h", (max(-32768, min(32767, round(value * 32767))) for value in samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    return buffer.getvalue()


def load_dataset(directory: Path) -> tuple[array, dict]:
    """The first 30 s of FLEURS test speech, recordings concatenated in filename order."""
    rows = [line.split("\t") for line in (directory / "test.tsv").read_text(encoding="utf-8").splitlines()]
    if not rows or any(len(row) != 7 for row in rows):
        raise ValueError("Expected the original seven-column FLEURS test.tsv")
    if any(Path(row[1]).name != row[1] or not row[1].endswith(".wav") for row in rows):
        raise ValueError("Expected safe FLEURS WAV filenames")
    samples, selected, remaining = array("f"), [], SECONDS * SAMPLE_RATE
    for row in sorted(rows, key=lambda row: row[1]):
        path = directory / "audio/test" / row[1]
        audio = read_wav(path)
        if len(audio) != int(row[5]):
            raise ValueError(f"FLEURS sample count differs from the TSV: {row[1]}")
        count = min(remaining, len(audio))
        samples.extend(audio[:count])
        selected.append({"id": f"en_us/test/{path.stem}", "samples_used": count,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        remaining -= count
        if not remaining:
            break
    if remaining:
        raise ValueError("FLEURS input contains less than 30 seconds of audio")
    return samples, {"source": DATASET, "recordings": selected}


def _request(url: str, audio: bytes, *, stream: bool) -> urllib.request.Request:
    boundary = uuid.uuid4().hex
    fields = {"model": SERVED_NAME, "language": LANGUAGE, "stream": "true" if stream else "false"}
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    )
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    return urllib.request.Request(
        f"{url}/v1/audio/transcriptions", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def pieces(audio: array, metadata: dict) -> list[bytes]:
    """Each recording's part of the selected audio as its own WAV upload."""
    uploads, start = [], 0
    for recording in metadata["recordings"]:
        uploads.append(wav_bytes(audio[start : start + recording["samples_used"]]))
        start += recording["samples_used"]
    return uploads


def transcribe_all(url: str, uploads: list[bytes]) -> tuple[float, str]:
    """One pass: every upload in order; seconds for the whole pass and the joined text."""
    started = time.perf_counter()
    texts = [transcribe(url, upload)[1] for upload in uploads]
    return time.perf_counter() - started, " ".join(text.strip() for text in texts)


def transcribe(url: str, audio: bytes) -> tuple[float, str]:
    """One unstreamed transcription: seconds from upload to the complete response, and its text."""
    started = time.perf_counter()
    with urllib.request.urlopen(_request(url, audio, stream=False), timeout=600) as response:
        text = json.loads(response.read())["text"]
    elapsed = time.perf_counter() - started
    if not text.strip():
        raise RuntimeError("No visible text: use speech audio and check the deployment before benchmarking")
    return elapsed, text


def first_text(url: str, audio: bytes) -> dict:
    """One streamed transcription: seconds to the first visible text, and the server's own TTFT and TPS."""
    started = time.perf_counter()
    result = {"seconds": None, "server_ttft_s": None, "server_tps": None}
    with urllib.request.urlopen(_request(url, audio, stream=True), timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if result["seconds"] is None and event.get("text", "").strip():
                result["seconds"] = time.perf_counter() - started
            result["server_ttft_s"] = event.get("ttft", result["server_ttft_s"])
            result["server_tps"] = event.get("tps", result["server_tps"])
    if result["seconds"] is None:
        raise RuntimeError("No visible text in the streamed transcription")
    return result


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_benchmark(deployment: Path, audio: array, input_metadata: dict, output: Path) -> dict:
    import pyneat

    import model_report

    uploads = pieces(audio, input_metadata)
    duration = len(audio) / SAMPLE_RATE
    options = pyneat.genai.GenAIServerOptions()
    options.host, options.port = "127.0.0.1", _free_port()
    server = pyneat.genai.GenAIServer(options)
    server.add_model(str(deployment), SERVED_NAME)
    server.start()
    url = f"http://{options.host}:{options.port}"
    try:
        for _ in range(WARMUP_REQUESTS):
            transcribe_all(url, uploads)
        monitor = pyneat.TelemetryMonitor()
        seconds = []
        monitor.start()
        started = time.perf_counter()
        try:
            while len(seconds) < MEASURED_REQUESTS or time.perf_counter() - started < MIN_DURATION_S:
                elapsed, transcript = transcribe_all(url, uploads)
                seconds.append(elapsed)
            span = time.perf_counter() - started
        finally:
            telemetry = monitor.stop()
        streamed = [first_text(url, uploads[0]) for _ in range(FIRST_TEXT_TRIALS)]
    finally:
        server.stop()
    ordered = sorted(seconds)
    benchmark = pyneat.BenchmarkReport()
    benchmark.latency_ms = statistics.mean(seconds) * 1000
    for percentile in (50, 95, 99):
        index = min(len(ordered) - 1, round(percentile / 100 * (len(ordered) - 1)))
        setattr(benchmark, f"latency_p{percentile}_ms", ordered[index] * 1000)
    benchmark.fps = len(seconds) / span
    benchmark.throughput_duration_ms = span * 1000
    benchmark.throughput_samples = len(seconds)
    digest = hashlib.sha256(b"".join(uploads)).hexdigest()
    report = model_report.from_performance(
        benchmark, {"model": None}, telemetry=telemetry,
        execution_plans={"model": {
            "status": "unavailable",
            "reason": "The model runs inside Neat's GenAI server, which exposes no Model execution plan",
            "evidence": "asr-timings.json",
        }},
        mla_layer_stats={"model": {"status": "not requested", "layers": []}},
        mla_stage_stats={"model": {"status": "not requested", "stages": []}},
        workload=(
            f"The same {duration:.3f} s of 16 kHz mono speech from {input_metadata['source']} as the cached ASR "
            f"models, sent as its {len(uploads)} recordings, one PCM16 WAV upload each (SHA-256 of the uploads "
            f"{digest}), to Neat's GenAI server /v1/audio/transcriptions with automatic language detection, one "
            "request at a time. Latency is one pass over all uploads, upload to complete transcript; throughput is "
            "passes/s. RTF is processing seconds / audio seconds. Time to first text is the mean of five streamed "
            "requests of the first recording, from upload to the first visible text, with that recording already "
            "available. The server stays loaded; loading and warm-up are excluded."
        ),
    )
    report.document["metrics"].update(
        real_time_factor=span / (len(seconds) * duration),
        time_to_first_text_ms=statistics.mean(row["seconds"] for row in streamed) * 1000,
    )
    report.evidence["asr-timings.json"] = {
        "input": input_metadata, "uploads_sha256": digest, "audio_samples": len(audio), "sample_rate": SAMPLE_RATE,
        "runtime": "GenAIServer /v1/audio/transcriptions", "language": LANGUAGE, "uploads": len(uploads),
        "unstreamed_seconds": seconds,
        "unstreamed_span_seconds": span, "streamed": streamed, "final_transcript": transcript,
    }
    report.save(output)
    return report.document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=extracted-dataset")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name, separator, path = args.dataset.partition("=")
    if name != DATASET or not separator:
        parser.error(f"--dataset must be {DATASET}=directory")
    audio, metadata = load_dataset(Path(path))
    document = run_benchmark(args.deployment.resolve(strict=True), audio, metadata, args.output)
    print(json.dumps(document["metrics"], indent=2))
