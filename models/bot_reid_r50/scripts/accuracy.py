"""Measure BoT ReID R50 on Market-1501 with its MPK on a Modalix board.

Every query and gallery image runs through the package, so both sides of the comparison are the compiled
model's embeddings. Each embedding is L2-normalized and the gallery is ranked by squared Euclidean distance,
which orders it as cosine similarity does: fast-reid's TEST.METRIC cosine. Scoring follows Market-1501's
protocol, as fast-reid's evaluation and SiMa's earlier ReIDPerformanceAnalyzer do: junk images (person -1)
are left out, and for each query the gallery images of the same person from the same camera are ignored.
ReID mAP is the mean over queries of the average precision of the ranked gallery; Rank-k is the fraction of
queries whose first k results hold a correct match.
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
from calibration import GALLERY, QUERY, identity, preprocess_image, split_names

WORKERS = 4
PREFETCH = 8
FEATURES = 2048
RANKS = (1, 5, 10)


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


def evaluate(query: np.ndarray, gallery: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray) -> dict:
    """ReID mAP and CMC Rank-1/5/10. The *_ids arrays hold (person, camera) rows."""
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    distances = 2.0 - 2.0 * query @ gallery.T
    order = np.argsort(distances, axis=1, kind="stable")
    cmc = np.zeros(max(RANKS), np.float64)
    precisions, valid = [], 0
    for index, ranked in enumerate(order):
        person, camera = query_ids[index]
        kept = ranked[~((gallery_ids[ranked, 0] == person) & (gallery_ids[ranked, 1] == camera))]
        matches = gallery_ids[kept, 0] == person
        if not matches.any():
            continue
        valid += 1
        cmc[np.argmax(matches):] += 1
        hits = np.cumsum(matches)
        precisions.append(float(np.sum(hits[matches] / (np.flatnonzero(matches) + 1.0)) / matches.sum()))
    if not valid:
        raise ValueError("no query identity appears in the gallery")
    cmc /= valid
    return {"reid_map": float(np.mean(precisions)), **{f"cmc_rank{k}": float(cmc[k - 1]) for k in RANKS}}


def embed(runner_cache: dict, model: pyneat.Model, dataset: Path, names: list[str], label: str) -> np.ndarray:
    started = time.monotonic()
    out = np.empty((len(names), FEATURES), np.float32)
    images = prefetched(lambda name: preprocess_image((dataset / name).read_bytes()), names)
    for count, (name, image) in enumerate(zip(names, images), start=1):
        tensor = pyneat.Tensor.from_numpy(image, copy=False)
        if "runner" not in runner_cache:
            runner_cache["runner"] = model.build([tensor])
        feature = runner_cache["runner"].run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
        if feature.shape != (FEATURES,) or not np.isfinite(feature).all():
            raise ValueError(f"unexpected embedding for {name}: shape {feature.shape}")
        out[count - 1] = feature
        if count == 1 or count % 2000 == 0 or count == len(names):
            log(f"{label} {count}/{len(names)}, {(time.monotonic() - started) / count * 1000:.1f} ms/image")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="market-1501-default=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    args = parser.parse_args(argv)
    dataset_id, dataset_dir = args.dataset.split("=", 1)
    dataset = Path(dataset_dir)

    query_names, gallery_names = split_names(dataset, QUERY), split_names(dataset, GALLERY)
    model = pyneat.Model(str(parse_mpks(args.mpk)))
    cache: dict = {}
    started = time.monotonic()
    query = embed(cache, model, dataset, query_names, "query")
    gallery = embed(cache, model, dataset, gallery_names, "gallery")
    metrics = evaluate(
        query, gallery,
        np.array([identity(name) for name in query_names]), np.array([identity(name) for name in gallery_names]),
    )
    report = model_report.from_accuracy({
        dataset_id: {
            "metrics": metrics,
            "split": QUERY,
            "evaluated_samples": len(query_names),
            "available_samples": len(query_names),
            "details": {
                "gallery_images": len(gallery_names),
                "inference_seconds": round(time.monotonic() - started, 1),
            },
        },
    })
    report.save(args.output)
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
