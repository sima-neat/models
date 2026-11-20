"""Local COCO qualification of both QAT components through one exact image route.

The normal --mpk/--dataset/--output interface is retained. The additional local
contract arguments replace source/catalog wiring, which is deliberately deferred.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import time
from pathlib import Path

import model_report
import numpy as np
from contract import load_contract, parse_components, sha256_file
from faster_coco_eval import COCO, COCOeval_faster
from runtime import Pipeline, read_rgb

ANNOTATIONS = "annotations/instances_val2017.json"
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
METRICS = {
    "coco_map": "AP_all",
    "coco_ap50": "AP_50",
    "coco_ap75": "AP_75",
    "coco_ap_small": "AP_small",
    "coco_ap_medium": "AP_medium",
    "coco_ap_large": "AP_large",
    "coco_ar1": "AR_all",
    "coco_ar10": "AR_second",
    "coco_ar100": "AR_third",
    "coco_ar_small": "AR_small",
    "coco_ar_medium": "AR_medium",
    "coco_ar_large": "AR_large",
}


def evaluate(
    packages: dict[str, Path], contract: dict, dataset_id: str, dataset: Path, output: Path, limit: int | None = None
) -> dict:
    if dataset_id != "coco-2017-val":
        raise ValueError("this accuracy entry point requires the official coco-2017-val dataset")
    annotations = dataset / ANNOTATIONS
    if sha256_file(annotations) != ANNOTATIONS_SHA256:
        raise ValueError("COCO annotation identity differs from the qualified reference")
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(annotations))
    available = sorted(truth.getImgIds())
    if len(available) != 5000:
        raise ValueError("official COCO val2017 must contain 5,000 image IDs")
    if limit is not None and not 1 <= limit <= len(available):
        raise ValueError("local screen limit must lie between 1 and 5,000")
    image_ids = sorted(random.Random(0).sample(available, limit)) if limit else available
    records = truth.loadImgs(image_ids)
    capacity = (max(record["width"] for record in records), max(record["height"] for record in records))
    pipeline = Pipeline(packages, contract, capacity)
    predictions = []
    start = time.monotonic()
    try:
        for count, record in enumerate(records, start=1):
            image = read_rgb(dataset / "val2017" / record["file_name"])
            if image.shape[:2] != (record["height"], record["width"]):
                raise ValueError(f"decoded image geometry differs from annotation: {record['id']}")
            predictions.append(pipeline(image, record["id"]))
            if count == 1 or count % 250 == 0 or count == len(records):
                print(f"{count}/{len(records)} images", flush=True)
    finally:
        pipeline.close()
    inference_seconds = time.monotonic() - start
    rows = np.concatenate(predictions)
    if not len(rows):
        raise ValueError("the compiled pipeline returned no detections")
    with contextlib.redirect_stdout(io.StringIO()):
        results = truth.loadRes(rows)
    scorer = COCOeval_faster(truth, results, "bbox")
    scorer.params.imgIds = image_ids
    scorer.params.maxDets = [1, 10, 100]
    scorer.evaluate()
    scorer.accumulate()
    scorer.summarize()
    metrics = {name: float(scorer.stats_as_dict[label]) for name, label in METRICS.items()}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
        raise ValueError("COCO metrics are incomplete or out of range")
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": metrics,
                "split": "val2017",
                "evaluated_samples": len(image_ids),
                "available_samples": len(available),
                "details": {
                    "image_ids": image_ids,
                    "annotations_sha256": ANNOTATIONS_SHA256,
                    "checkpoint_sha256": contract["checkpoint_sha256"],
                    "reference_sha256": contract["reference_sha256"],
                    "component_sha256": {key: value["sha256"] for key, value in contract["components"].items()},
                    "inference_seconds": inference_seconds,
                    "preprocessing": contract["preprocessing"],
                    "final_boxdecode_included": True,
                    "bridge": "public PyNeat + CPU NumPy oracle; not native-kernel speed qualification",
                    "official_full5000": len(image_ids) == 5000,
                },
            }
        }
    )
    report.save(output)
    return report.document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True)
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--limit", type=int, help="local screen only; never a full validation result")
    args = parser.parse_args()
    dataset_id, separator, directory = args.dataset.partition("=")
    if not separator or not directory:
        parser.error("--dataset must be coco-2017-val=<directory>")
    contract = load_contract(args.contract, args.contract_sha256)
    evaluate(parse_components(args.mpk), contract, dataset_id, Path(directory), args.output, args.limit)


if __name__ == "__main__":
    main()
