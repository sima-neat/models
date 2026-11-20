"""Score transcription accuracy on FLEURS English test with the complete package.

Usage: PYTHONPATH=tools python accuracy.py --mpk model=encoder_mpk.tar.gz --resources package \
    --dataset fleurs-en-us-test=extracted-dataset --output results [--limit 20]

Transcribes every recording of test.tsv in filename order with automatic language detection, one recording per
window sequence, and scores the transcripts against FLEURS's raw transcriptions with model_report.speech:
one shared normalization and corpus word and character error rates.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import model_report
from benchmark_runtime import Transcriber, read_audio, runtime_directory
from model_report.speech import char_error_rate, word_error_rate

DATASET = "fleurs-en-us-test"
LANGUAGE = "auto"


def recordings(directory):
    """test.tsv rows in filename order: id, file name, raw transcription, and four more columns."""
    rows = [line.split("\t") for line in (directory / "test.tsv").read_text(encoding="utf-8").splitlines()]
    if not rows or any(len(row) != 7 for row in rows):
        raise ValueError("Expected the original seven-column FLEURS test.tsv")
    if any(Path(row[1]).name != row[1] or not row[1].endswith(".wav") for row in rows):
        raise ValueError("Expected safe FLEURS WAV filenames")
    return sorted(rows, key=lambda row: row[1])


def transcribe(transcriber, audio):
    text = ""
    for text in transcriber.transcribe(audio, LANGUAGE):
        pass
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True, help="encoder archive, optionally model=path")
    parser.add_argument("--resources", type=Path, help="package resources; defaults to the encoder's directory")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=extracted-dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="score only the first recordings, for a quick local check")
    args = parser.parse_args()
    name, separator, path = args.dataset.partition("=")
    if name != DATASET or not separator:
        parser.error(f"--dataset must be {DATASET}=directory")
    mpk = Path(args.mpk.removeprefix("model=")).resolve(strict=True)
    package = runtime_directory(mpk, args.resources.resolve(strict=True) if args.resources else None)
    directory = Path(path)
    rows = recordings(directory)
    selected = rows[: args.limit] if args.limit else rows
    references, transcripts = [], []
    started = time.monotonic()
    with Transcriber(mpk, package) as transcriber:
        for row in selected:
            audio = read_audio(directory / "audio/test" / row[1])
            if len(audio) != int(row[5]):
                raise ValueError(f"FLEURS sample count differs from the TSV: {row[1]}")
            references.append(row[2])
            transcripts.append(transcribe(transcriber, audio))
    report = model_report.from_accuracy({
        DATASET: {
            "metrics": {
                "wer": word_error_rate(references, transcripts),
                "cer": char_error_rate(references, transcripts),
            },
            "split": "test",
            "evaluated_samples": len(selected),
            "available_samples": len(rows),
            "details": {"language": LANGUAGE, "inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)


if __name__ == "__main__":
    main()
