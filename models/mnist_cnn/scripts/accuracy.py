"""Measure MNIST CNN top-1 accuracy on the 10,000 MNIST test digits with its MPK on a Modalix board."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import CLASSES, preprocess, read_split

SPLIT = "test"


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already scaled the digit to [0, 1]; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="mnist-default=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first digits only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)

    images, labels = read_split(Path(dataset_dir), SPLIT)
    count = min(args.limit, len(images)) if args.limit else len(images)
    inputs = preprocess(images[:count])

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    correct = 0
    started = time.monotonic()
    for index in range(count):
        tensor = pyneat.Tensor.from_numpy(inputs[index], copy=False, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        logits = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
        if logits.shape != (CLASSES,) or not np.isfinite(logits).all():
            raise ValueError(f"unexpected scores for test digit {index}: shape {logits.shape}")
        correct += int(np.argmax(logits) == labels[index])
        if index == 0 or (index + 1) % 2000 == 0 or index + 1 == count:
            log(f"{index + 1}/{count} digits, {(time.monotonic() - started) / (index + 1) * 1000:.2f} ms/digit")

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {"top1_accuracy": correct / count},
            "split": SPLIT, "evaluated_samples": count, "available_samples": len(images),
            "details": {"inference_seconds": round(time.monotonic() - started, 1)},
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
