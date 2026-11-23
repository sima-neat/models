"""Measure FresUNet's change F1 on the OSCD test cities with its MPK on a Modalix board.

Each of the 10 test.txt cities is prepared as oscd.py describes: both dates' 13 bands normalized, resized to 544x544
and stacked into 26 channels, passed on the tensor route. The change class's F1 (dice), precision and recall count
true and false positives and negatives over every pixel of every city at 544x544, as SiMa's earlier model zoo's
OSCDPerformanceAnalyzer did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from oscd import ChangeScore, change_map, cities, city_input, dataset_root, prediction

DATASET = "oscd-default"
SPLIT = "test"


def parse_mpks(values: list[str]) -> Path:
    mpks = dict(value.split("=", 1) for value in values)
    if set(mpks) != {"model"}:
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(mpks["model"])


def tensor_options() -> pyneat.ModelOptions:
    """The input is a prepared 26-channel tensor: Neat only quantizes and tessellates it."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def log(*values: object) -> None:
    print(*values, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first cities only (local runs only)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    root = dataset_root(Path(dataset_dir))
    available = cities(root, SPLIT)
    names = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    score = ChangeScore()
    started = time.monotonic()
    try:
        for count, name in enumerate(names, start=1):
            tensor = pyneat.Tensor.from_numpy(city_input(root, name), copy=True, layout=pyneat.TensorLayout.HWC)
            if runner is None:
                runner = model.build([tensor])
            output = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=True)
            if not np.isfinite(output).all():
                raise ValueError(f"{name}: the model returned non-finite values")
            score.add(prediction(output), change_map(root, name))
            log(f"{count}/{len(names)} cities, {name}: F1 so far {score.dice():.4f}")
    finally:
        if runner is not None:
            runner.close()

    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": score.metrics(),
            "split": SPLIT, "evaluated_samples": len(names), "available_samples": len(available),
            "details": {
                "true_positives": score.tp, "false_positives": score.fp, "false_negatives": score.fn,
                "inference_seconds": round(time.monotonic() - started, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 5) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
