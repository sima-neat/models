"""Measure DINOv3 ConvNeXt Large k-NN top-1 accuracy on ImageNet with its MPK on a Modalix board.

The model returns embeddings, not classes. Each validation image takes the class its most similar other validation
images vote for: leave-one-out k-NN on the class embeddings, with the 20 neighbours and the 0.07 temperature of
DINO's k-NN evaluation (eval_knn.py in facebookresearch/dino). Published k-NN scores take their neighbours from
the ImageNet train set, so this score is not comparable with them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_report
import numpy as np
import pyneat
from calibration import preprocess_image

VALIDATION_SPLIT = "val2012"
CLASSES = 1000
NEIGHBOURS = 20
TEMPERATURE = 0.07
# Images scored at once; their similarities with all 50,000 images take 100 MB as float32.
BLOCK = 500
# Images decode on worker threads while the MPK runs; Pillow releases the GIL while it decodes and resizes.
WORKERS = 4
PREFETCH = 8


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


def validation_labels(dataset: Path) -> dict[str, int]:
    """Each validation image's class index, from the extracted dataset's labels file."""
    labels = json.loads((dataset / "annotations" / f"labels_{VALIDATION_SPLIT}.json").read_text(encoding="utf-8"))
    if len(labels["categories"]) != CLASSES:
        raise ValueError(f"expected {CLASSES} validation classes, found {len(labels['categories'])}")
    names = {image["id"]: f"{VALIDATION_SPLIT}/{image['file_name']}" for image in labels["images"]}
    return {names[annotation["image_id"]]: annotation["category_id"] for annotation in labels["annotations"]}


def knn_top1(embeddings: np.ndarray, classes: np.ndarray) -> float:
    """The share of images whose class wins the vote of their NEIGHBOURS most similar other images.

    Similarity is the cosine of two embeddings, and a neighbour's vote weighs exp(similarity / TEMPERATURE).
    """
    if len(embeddings) <= NEIGHBOURS:
        raise ValueError(f"k-NN needs more than {NEIGHBOURS} images, got {len(embeddings)}")
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    correct = 0
    for start in range(0, len(unit), BLOCK):
        similarity = unit[start : start + BLOCK] @ unit.T
        rows = np.arange(len(similarity))
        # An image is never its own neighbour.
        similarity[rows, start + rows] = -np.inf
        nearest = np.argpartition(similarity, -NEIGHBOURS, axis=1)[:, -NEIGHBOURS:]
        weights = np.exp(np.take_along_axis(similarity, nearest, axis=1) / TEMPERATURE)
        votes = np.zeros((len(similarity), CLASSES), dtype=np.float64)
        np.add.at(votes, (rows[:, None], classes[nearest]), weights)
        correct += int(np.count_nonzero(votes.argmax(axis=1) == classes[start : start + BLOCK]))
    return correct / len(unit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first images only (local runs only)")
    # The accuracy runner passes the package's resources, here only the license. The measurement reads none of them.
    parser.add_argument("--resources", help="directory holding the package's licenses folder (unused)")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    labels = validation_labels(dataset)
    available = sorted(labels)
    names = available[: args.limit] if args.limit else available

    model = pyneat.Model(str(parse_mpks(args.mpk)))
    runner = None
    embeddings = None
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        outputs = [output.to_numpy(copy=False) for output in runner.run([tensor], timeout_ms=120000)]
        if len(outputs) != 2:
            raise ValueError(f"expected the class and patch embeddings for {name}, got {len(outputs)} outputs")
        # One vector for the image and one per patch: the class embedding is the smaller output.
        embedding = min(outputs, key=lambda output: output.size).reshape(-1)
        if not np.isfinite(embedding).all() or not embedding.any():
            raise ValueError(f"unexpected class embedding for {name}")
        if embeddings is None:
            embeddings = np.empty((len(names), embedding.size), dtype=np.float32)
        embeddings[count - 1] = embedding
        if count == 1 or count % 500 == 0 or count == len(names):
            log(f"{count}/{len(names)} images, {(time.monotonic() - started) / count * 1000:.0f} ms/image")
    inference_seconds = time.monotonic() - started

    accuracy = knn_top1(embeddings, np.asarray([labels[name] for name in names]))
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": {"knn_top1_accuracy": accuracy},
            "split": VALIDATION_SPLIT, "evaluated_samples": len(names), "available_samples": len(available),
            "details": {
                "protocol": "leave-one-out k-NN on the class embeddings of the evaluated images",
                "neighbours": NEIGHBOURS, "temperature": TEMPERATURE,
                "inference_seconds": round(inference_seconds, 1),
                "scoring_seconds": round(time.monotonic() - started - inference_seconds, 1),
            },
        },
    })
    report.save(args.output)
    measured = report.document["datasets"][dataset_id]["metrics"]
    print(json.dumps({name: round(value, 4) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
