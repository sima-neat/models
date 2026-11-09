"""Measure the FP32 baseline: LightGlue's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same 512-keypoint inputs and
yfcc100m-keypoints val pairs. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. Run
it on the model.onnx surgery.py exports from the converted weights, on a host with numpy, opencv-python-headless and
onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/lightglue_superpoint/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset yfcc100m-keypoints-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
from calibration import INPUT_SHAPES, KEYPOINTS, dataset_root, usable_pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx surgery.py exports")
    parser.add_argument("--dataset", required=True, help="yfcc100m-keypoints-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score the first pairs only, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    root = dataset_root(Path(directory))
    pairs = usable_pairs(root)[: args.limit]

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_inputs()]
    if names != list(INPUT_SHAPES):
        raise SystemExit(f"{args.onnx} takes {names}, expected {list(INPUT_SHAPES)}")

    def assignment_of(inputs: dict[str, np.ndarray]) -> np.ndarray:
        # accuracy.py builds channels-last inputs for the MPK; the ONNX takes them channels-first and returns the
        # assignment as [1, j, 1, i].
        (assignment,) = session.run(
            None, {name: np.ascontiguousarray(value.transpose(0, 3, 1, 2)) for name, value in inputs.items()})
        return assignment.reshape(KEYPOINTS, KEYPOINTS).T

    metrics, mean_matches = accuracy.evaluate(root, pairs, assignment_of)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    result = {"dataset": dataset_id, "pairs": len(pairs), "metrics": rounded, "mean_matches": round(mean_matches, 1)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
