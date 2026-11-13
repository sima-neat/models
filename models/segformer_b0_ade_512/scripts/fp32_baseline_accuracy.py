"""Measure the FP32 baseline: the converted ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without compilation or quantization, on the same ADE20K
validation images and scoring. The ONNX has no Neat pipeline, so each image is preprocessed on the host as
the package requires (Pillow's bilinear letterbox to 640x480, RGB 114 padding, ImageNet normalization), where the board
uses Neat's preprocessing. Run it on a host with onnxruntime and tools/model_report on PYTHONPATH, on the model.onnx
that convert.py writes:

    PYTHONPATH=tools python models/segformer_b0_ade_512/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset ade20k-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import accuracy
import numpy as np
from calibration import preprocess_image


def onnx_logits(onnx: Path, images: Sequence[Path]) -> Iterator[np.ndarray]:
    """Run the converted ONNX on the CPU; it reads the NCHW layout of the image the MPK reads as HWC."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    (input_name,) = [value.name for value in session.get_inputs()]
    for image in accuracy.prefetched(lambda path: preprocess_image(path.read_bytes()), images):
        yield session.run(None, {input_name: image.transpose(2, 0, 1)[None]})[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the converted model.onnx")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score the first images only, for a quick check")
    args = parser.parse_args(argv)
    root, images, _ = accuracy.parse_dataset(parser, args.dataset, args.limit)

    measured = accuracy.scores(accuracy.accumulate(root, images, onnx_logits(args.onnx, images)))
    rounded = {"miou": round(measured["miou"], 4), "pixel_accuracy": round(measured["pixel_accuracy"], 4)}
    output = {"dataset": accuracy.DATASET, "images": len(images), "metrics": rounded}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
