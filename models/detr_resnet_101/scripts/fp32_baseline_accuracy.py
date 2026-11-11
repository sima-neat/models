"""Measure the FP32 baseline: the converted ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without compilation or quantization, on the same COCO
val2017 images, 1280x720 stretched inputs and decoding. model.yaml publishes the result as the
reference_metrics entry with origin fp32_baseline. Run it on a host with the packages in
requirements-accuracy.txt, onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/detr_resnet_101/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np


def onnx_inference(onnx: Path) -> accuracy.Inference:
    """Run the converted ONNX on the CPU; it reads the NCHW layout of the image the MPK reads as HWC."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    (input_name,) = [value.name for value in session.get_inputs()]

    def infer(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits, boxes = session.run(["logits", "pred_boxes"], {input_name: image.transpose(2, 0, 1)[None]})
        return logits, boxes

    return infer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the converted model.onnx")
    parser.add_argument("--dataset", required=True, help="coco-2017-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)

    result = accuracy.evaluate(Path(directory), onnx_inference(args.onnx), args.limit)
    rounded = {name: round(value, 4) for name, value in result["metrics"].items()}
    output = {"dataset": dataset_id, "images": result["evaluated_samples"], "metrics": rounded}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
