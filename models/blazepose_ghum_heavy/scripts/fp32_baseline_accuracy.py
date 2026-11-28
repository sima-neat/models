"""Measure the FP32 baseline: BlazePose GHUM Heavy 3D's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same ground-truth-box crops of
COCO val2017 people. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. Run it on
the conversion's model.onnx, on a host with numpy, opencv-python-headless, onnx, onnxruntime and
faster-coco-eval, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/blazepose_ghum_heavy/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset coco-2017-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
from surgery import INPUT, OUTPUTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the converted model.onnx")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != accuracy.DATASET:
        raise SystemExit(f"--dataset must be {accuracy.DATASET}, got {dataset_id}")
    dataset = Path(directory)
    ground_truth = accuracy.load_ground_truth(dataset)
    image_ids = accuracy.sampled(accuracy.scored_image_ids(ground_truth), args.limit)

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    def infer(pixels: np.ndarray) -> list[np.ndarray]:
        # The ONNX takes the crop channels-last, as the MPK does, with a batch axis.
        return session.run(OUTPUTS, {INPUT: pixels[None]})

    metrics, detections = accuracy.evaluate(dataset, ground_truth, image_ids, infer)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    result = {"images": len(image_ids), "people": len(detections), "metrics": rounded}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
