"""Measure the FP32 baseline: timm's DINOv3 ConvNeXt Small as reference.onnx on a host, scored exactly as accuracy.py
scores the MPK.

The baseline is the accuracy the package would reach without compilation or quantization, on the same 5,000
imagenet-2012 val images, preprocessing and k-NN scoring. model.yaml publishes the result as the reference_metrics
entry with origin fp32_baseline. Run it on a host with the packages in requirements-accuracy.txt, onnxruntime, and
tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/dinov3_convnext_small/scripts/fp32_baseline_accuracy.py \\
        --onnx reference.onnx --dataset imagenet-2012-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np


def onnx_embedding(onnx: Path) -> accuracy.Embed:
    """Run reference.onnx on the CPU; it reads the NCHW layout of the image the MPK reads as HWC."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    (input_name,) = [value.name for value in session.get_inputs()]

    def embed(image: np.ndarray) -> np.ndarray:
        (embedding,) = session.run(["class_embedding"], {input_name: image.transpose(2, 0, 1)[None]})
        return embedding.reshape(-1)

    return embed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the converted reference.onnx")
    parser.add_argument("--dataset", required=True, help="imagenet-2012-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score the first images only, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)

    result = accuracy.evaluate(Path(directory), onnx_embedding(args.onnx), args.limit)
    rounded = {name: round(value, 4) for name, value in result["metrics"].items()}
    output = {"dataset": dataset_id, "images": result["evaluated_samples"], "metrics": rounded}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
