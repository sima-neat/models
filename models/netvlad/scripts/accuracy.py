"""Measure NetVLAD Pittsburgh 30k test Recall@1/5/10 with its MPK on a Modalix board, as Patch-NetVLAD evaluates."""

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

# pitts30k-val holds the official test split; pitts30k_test.tsv lists role, image path, UTM easting and northing
# in pitts30k_test.mat order.
LISTING = "pitts30k_test.tsv"
IMAGES = "images"
SPLIT = "test"
DESCRIPTOR_SIZE = 4096
# pitts30k_test.mat posDistThr: a retrieved database image within 25 m of the query is correct.
POSITIVE_METERS = 25.0
# Patch-NetVLAD's performance.ini n_values_all; the papers publish N = 1, 5 and 10.
RECALL_AT = (1, 5, 10, 20, 50, 100)
PUBLISHED = (1, 5, 10)
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


def read_listing(dataset: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Return the database and query image paths with their UTM positions, in the official order."""
    rows = [line.split("\t") for line in (dataset / LISTING).read_text(encoding="utf-8").splitlines()]
    split = {role: [row for row in rows if row[0] == role] for role in ("database", "query")}
    if not split["database"] or not split["query"] or len(rows) != len(split["database"]) + len(split["query"]):
        raise ValueError(f"{LISTING} must list database and query rows only")
    positions = {role: np.asarray([row[2:4] for row in items], dtype=np.float64) for role, items in split.items()}
    return (
        [row[1] for row in split["database"]],
        positions["database"],
        [row[1] for row in split["query"]],
        positions["query"],
    )


def positives(database: np.ndarray, queries: np.ndarray) -> list[np.ndarray]:
    """Database indices within POSITIVE_METERS of each query, as scikit-learn's radius_neighbors returns them."""
    return [np.flatnonzero(np.linalg.norm(database - query, axis=1) <= POSITIVE_METERS) for query in queries]


def nearest(database: np.ndarray, queries: np.ndarray, count: int) -> np.ndarray:
    """The count database indices closest to each query in squared L2 distance, nearest first (faiss IndexFlatL2).

    The query's own squared norm is the same for every database image, so the ranking leaves it out.
    """
    norms = np.einsum("ij,ij->i", database, database)
    ranked = []
    for start in range(0, len(queries), 512):
        block = queries[start : start + 512]
        distances = norms[None, :] - 2.0 * block @ database.T
        top = np.argpartition(distances, count - 1, axis=1)[:, :count]
        order = np.argsort(np.take_along_axis(distances, top, axis=1), axis=1, kind="stable")
        ranked.append(np.take_along_axis(top, order, axis=1))
    return np.concatenate(ranked)


def recall_at(predictions: np.ndarray, truth: list[np.ndarray]) -> dict[int, float]:
    """Share of queries with a true match among their first N predictions, for each N in RECALL_AT."""
    hits = [np.isin(ranked, matches) for ranked, matches in zip(predictions, truth, strict=True)]
    return {n: float(np.mean([found[:n].any() for found in hits])) for n in RECALL_AT}


def describe(model: pyneat.Model, dataset: Path, names: Sequence[str], label: str) -> np.ndarray:
    runner = None
    descriptors = np.empty((len(names), DESCRIPTOR_SIZE), dtype=np.float32)
    started = time.monotonic()
    images = prefetched(lambda name: preprocess_image((dataset / IMAGES / name).read_bytes()), names)
    for index, (name, image) in enumerate(zip(names, images, strict=True)):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if runner is None:
            runner = model.build([tensor])
        descriptor = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
        if descriptor.shape != (DESCRIPTOR_SIZE,) or not np.isfinite(descriptor).all():
            raise ValueError(f"unexpected descriptor for {name}: shape {descriptor.shape}")
        descriptors[index] = descriptor
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == len(names):
            log(f"{label}: {index + 1}/{len(names)}, {(time.monotonic() - started) / (index + 1) * 1000:.0f} ms/image")
    return descriptors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="pitts30k-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    parser.add_argument("--limit", type=int, help="measure the first queries only, against the whole database")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    database, database_utm, queries, query_utm = read_listing(dataset)
    measured = slice(0, args.limit)
    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    started = time.monotonic()
    database_descriptors = describe(model, dataset, database, "database")
    query_descriptors = describe(model, dataset, queries[measured], "queries")
    predictions = nearest(database_descriptors, query_descriptors, max(RECALL_AT))
    recalls = recall_at(predictions, positives(database_utm, query_utm[measured]))

    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": {f"recall_at_{n}": recalls[n] for n in PUBLISHED},
                "split": SPLIT,
                "evaluated_samples": len(predictions),
                "available_samples": len(queries),
                "details": {
                    "database_images": len(database),
                    "recall_at": {str(n): round(value, 4) for n, value in recalls.items()},
                    "inference_seconds": round(time.monotonic() - started, 1),
                },
            },
        }
    )
    report.save(args.output)
    print(json.dumps({str(n): round(value, 4) for n, value in recalls.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
