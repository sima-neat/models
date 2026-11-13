"""Measure the FP32 baseline: RF-DETR Seg Small's ONNX on a host, scored exactly as accuracy.py scores the MPKs.

The baseline is the accuracy the package would reach without quantization, on the same 384x384 inputs and COCO
val2017 images. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. Take the
model.onnx that convert.py exports, then run it on a host with numpy, opencv-python-headless, Pillow,
faster-coco-eval and onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/rf_detr_seg_small/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import accuracy
import numpy as np


def onnx_pipeline(path: Path) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a function that runs one preprocessed image through the ONNX, with outputs laid out as the MPKs'."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def infer(image):
        # accuracy.preprocess returns HWC; the ONNX takes NCHW and returns masks as [1, query, height, width].
        dets, labels, masks = session.run(None, {"input": image.transpose(2, 0, 1)[None]})
        return dets, labels, masks[0].transpose(1, 2, 0)

    return infer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx convert.py exports")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    dataset_dir = args.dataset.split("=", 1)[1]
    # Only the metrics: a model report records the board it ran on, which a host lacks.
    metrics = accuracy.evaluate(onnx_pipeline(args.onnx), Path(dataset_dir), args.limit)["metrics"]
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: round(value, 4) for name, value in metrics.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
