"""Measure corpus WER and CER on FLEURS test recordings."""

import argparse
import time
from pathlib import Path

import model_report
from benchmark_runtime import Transcriber, read_audio, recordings, runtime_directory
from model_report.speech import char_error_rate, word_error_rate

DATASET = "fleurs-en-us-test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    name, sep, path = args.dataset.partition("=")
    if name != DATASET or not sep:
        parser.error(f"Expected --dataset {DATASET}=directory")
    mpk = Path(args.mpk.removeprefix("model=")).resolve(strict=True)
    package = runtime_directory(mpk, args.resources.resolve(strict=True) if args.resources else None)
    directory = Path(path)
    all_rows = recordings(directory)
    rows = all_rows
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No recordings")
    references, transcripts = [], []
    start = time.monotonic()
    with Transcriber(mpk, package) as transcriber:
        for row in rows:
            audio = read_audio(directory / "audio/test" / row[1])
            if len(audio) != int(row[5]):
                raise ValueError(f"Invalid sample count: {row[1]}")
            references.append(row[2])
            transcripts.append(transcriber.transcribe(audio))
            if len(transcripts) % 25 == 0:
                print(f"Transcribed {len(transcripts)} of {len(rows)} recordings", flush=True)
    report = model_report.from_accuracy(
        {
            DATASET: {
                "metrics": {
                    "wer": word_error_rate(references, transcripts),
                    "cer": char_error_rate(references, transcripts),
                },
                "split": "test",
                "evaluated_samples": len(rows),
                "available_samples": len(all_rows),
                "details": {
                    "inference_seconds": round(time.monotonic() - start, 1),
                },
            }
        }
    )
    report.evidence["transcripts.json"] = [
        {"filename": row[1], "reference": reference, "hypothesis": hypothesis}
        for row, reference, hypothesis in zip(rows, references, transcripts, strict=True)
    ]
    report.save(args.output)


if __name__ == "__main__":
    main()
