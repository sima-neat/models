"""Measure the FP32 baseline: SuperGlue Outdoor's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same 512-keypoint inputs and
YFCC100M pairs. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. Export the ONNX
with surgery.py, then run it on a host with numpy, opencv-python-headless and onnxruntime, and tools/model_report on
PYTHONPATH:

    PYTHONPATH=tools python models/superglue_outdoor/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset yfcc100m-keypoints-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np

# The ONNX returns couplings[1, j, 1, i]: image-1 keypoints on the channel axis, image-0 keypoints on the width axis.
COUPLINGS = accuracy.KEYPOINTS + 1


def couplings_from_onnx(output: np.ndarray) -> np.ndarray:
    """The [i, j] couplings matrix, image-0 keypoints as rows, from the ONNX's [1, j, 1, i] output."""
    if output.shape != (1, COUPLINGS, 1, COUPLINGS):
        raise ValueError(f"unexpected ONNX output: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("ONNX output contains non-finite values")
    return output.reshape(COUPLINGS, COUPLINGS).T


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx surgery.py exports")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the five metrics")
    parser.add_argument("--limit", type=int, help="score the first pairs only, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != accuracy.DATASET:
        raise SystemExit(f"--dataset must be {accuracy.DATASET}, got {dataset_id}")
    dataset = accuracy.load_dataset(Path(directory))
    usable = dataset.pairs.usable
    count = min(args.limit, accuracy.PAIRS) if args.limit else accuracy.PAIRS
    pairs = [pair for pair in range(count) if usable[pair]]

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]
    if set(names) != set(accuracy.INPUTS):
        raise SystemExit(f"{args.onnx} takes {names}, expected {list(accuracy.INPUTS)}")

    def couplings_of(inputs: list[np.ndarray]) -> np.ndarray:
        # accuracy.py builds channels-last tensors for the MPK; the ONNX takes them channels-first.
        feeds = {
            name: np.ascontiguousarray(value.transpose(0, 3, 1, 2)) for name, value in zip(accuracy.INPUTS, inputs)
        }
        (output,) = session.run(None, feeds)
        return couplings_from_onnx(output)

    metrics, _ = accuracy.evaluate(dataset, pairs, couplings_of)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    args.output.write_text(json.dumps({"pairs": len(pairs), "metrics": rounded}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
