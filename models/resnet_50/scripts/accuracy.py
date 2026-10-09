"""Measure ResNet-50 ImageNet top-1 and top-5 accuracy with its MPK on a Modalix board."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyneat
from calibration import VALIDATION_SPLIT, preprocess_image, split_names, validation_labels

# Images decode on worker threads while the MPK runs; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already resized and normalized the image; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def prefetched(load: Callable, items: Sequence) -> Iterator:
    """Yield load(item) in order, keeping at most PREFETCH items in flight."""
    with ThreadPoolExecutor(WORKERS) as pool:
        pending = deque(pool.submit(load, item) for item in items[:PREFETCH])
        for item in items[PREFETCH:]:
            yield pending.popleft().result()
            pending.append(pool.submit(load, item))
        while pending:
            yield pending.popleft().result()


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
    parser.add_argument("--dataset", type=Path, required=True, help="extracted imagenet-2012/mini directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    args = parser.parse_args(argv)

    available = split_names(args.dataset, VALIDATION_SPLIT)
    labels = validation_labels(available)
    names = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    top1 = top5 = 0
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((args.dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        logits = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
        if logits.shape != (1000,) or not np.isfinite(logits).all():
            raise ValueError(f"unexpected logits for {name}: shape {logits.shape}")
        ranked = np.argsort(-logits, kind="stable")[:5]
        top1 += int(ranked[0] == labels[name])
        top5 += int(labels[name] in ranked)
        if count == 1 or count % 500 == 0 or count == len(names):
            log(f"{count}/{len(names)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")

    result = {
        "core_version": pyneat.__version__,
        "dataset": {"split": VALIDATION_SPLIT, "evaluated_samples": len(names), "available_samples": len(available)},
        "metrics": {
            "top1_accuracy": {"value": top1 / len(names), "unit": "ratio", "direction": "higher"},
            "top5_accuracy": {"value": top5 / len(names), "unit": "ratio", "direction": "higher"},
        },
        "details": {"inference_seconds": round(time.monotonic() - started, 1)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: round(item["value"], 4) for name, item in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
