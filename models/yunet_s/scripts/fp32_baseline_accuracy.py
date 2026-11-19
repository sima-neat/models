"""Measure the FP32 baseline: the upstream ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without compilation or quantization, on the same WIDER FACE
validation images, letterbox, decode, NMS and scoring. Run it on a host with onnxruntime, OpenCV, SciPy and
tools/model_report on PYTHONPATH, on the pinned upstream ONNX:

    PYTHONPATH=tools python models/yunet_s/scripts/fp32_baseline_accuracy.py \\
        --onnx face_detection_yunet_2023mar.onnx --dataset wider-face-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import accuracy
import numpy as np


def onnx_detections(onnx: Path, root: Path, items: Sequence[tuple[str, str]]) -> Iterator[np.ndarray]:
    """Run the upstream ONNX on the CPU; it reads the NCHW layout of the image the MPK reads as HWC."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_outputs()]
    (input_name,) = [value.name for value in session.get_inputs()]
    for image, width, height in accuracy.prefetched(lambda item: accuracy.load_image(root, item), items):
        outputs = dict(zip(names, session.run(None, {input_name: image.transpose(2, 0, 1)[None]}), strict=True))
        # Upstream outputs are already [1, H*W, C] in row-major cell order, as the MPK's maps flatten.
        maps = {name: outputs[name].reshape(-1, accuracy.HEADS[name.split("_")[0]]) for name in accuracy.OUTPUTS}
        yield accuracy.detect(maps, width, height)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the pinned upstream ONNX")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of images, for a quick check")
    args = parser.parse_args(argv)
    root = accuracy.parse_dataset(parser, args.dataset)
    ground_truth = accuracy.load_ground_truth(root)
    available = [(event, name) for event, name, _, _ in ground_truth]
    items = sorted(random.Random(0).sample(available, args.limit)) if args.limit else available

    metrics, _, _ = accuracy.score(root, items, onnx_detections(args.onnx, root, items), ground_truth)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    output = {"dataset": accuracy.DATASET, "images": len(items), "metrics": rounded}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
