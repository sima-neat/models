"""Measure the FP32 baseline: MiDaS v2.1 Small's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same 256x256 inputs and NYU
Depth V2 test images. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. It scores the
ONNX convert.py writes; surgery.py's compile-time rewrite of it is exact. Run it on a host with numpy,
opencv-python-headless and onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/midas_v21_small/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset nyu-depth-v2-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
from calibration import HEIGHT, WIDTH

DATASET = "nyu-depth-v2-val"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx convert.py writes")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the two metrics")
    parser.add_argument("--limit", type=int, help="score the first images only, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    root = Path(directory)
    images = accuracy.validation_images(root)
    images = images[: args.limit] if args.limit else images

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    (input_name,) = [value.name for value in session.get_inputs()]
    totals = np.zeros(2)
    for image in images:
        tensor, depth = accuracy.load_sample(root, image)
        # accuracy.py hands the MPK a channels-last image; the ONNX takes it channels-first.
        (output,) = session.run(None, {input_name: np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])})
        if output.size != HEIGHT * WIDTH or not np.isfinite(output).all():
            raise ValueError(f"unexpected depth for {image.name}: shape {output.shape}")
        totals += accuracy.score(output.reshape(HEIGHT, WIDTH), depth)
    abs_rel, delta1 = totals / len(images)
    result = {"dataset": dataset_id, "images": len(images), "metrics": {"abs_rel": abs_rel, "delta1": delta1}}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: round(value, 4) for name, value in result["metrics"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
