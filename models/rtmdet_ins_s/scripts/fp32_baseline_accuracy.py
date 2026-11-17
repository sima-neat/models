"""Measure the FP32 baseline: RTMDet-Ins S's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, with the same letterboxed input,
post-processing and COCO scoring. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline.
Run the model.onnx convert.py writes on a host with numpy, opencv-python-headless, faster-coco-eval and onnxruntime,
and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/rtmdet_ins_s/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx convert.py exports")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != "coco-2017-val":
        raise SystemExit(f"--dataset must be coco-2017-val, got {dataset_id}")
    dataset = Path(directory)
    ground_truth, _, image_ids, records, categories = accuracy.load_dataset(dataset, args.limit)

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    def infer(image: np.ndarray) -> list[np.ndarray]:
        # accuracy.py passes the MPK a channels-last RGB image and reads channels-last outputs. The converted ONNX
        # takes the image channels-first in BGR order, as surgery.py's RGB rewrite undoes, and returns channels-first.
        bgr = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None])
        return [np.ascontiguousarray(value.transpose(0, 2, 3, 1)) for value in session.run(None, {"input": bgr})]

    detections = accuracy.evaluate(dataset, records, categories, infer)
    metrics = accuracy.score(ground_truth, detections, image_ids)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    report = {"images": len(image_ids), "metrics": rounded}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
