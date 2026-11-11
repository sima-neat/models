"""Measure the FP32 baseline: RAFT-Stereo realtime's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same frames and Scene Flow test
pairs, and stands in for a published score, since princeton-vl publishes none for the realtime model on Scene Flow.
Export the ONNX with surgery.py, then run it on a host with numpy, pillow and onnxruntime, and tools/model_report on
PYTHONPATH:

    PYTHONPATH=tools python models/raft_stereo_realtime/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset scene-flow-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
from scene_flow import INPUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the model.onnx surgery.py exports")
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the two metrics")
    parser.add_argument("--limit", type=int, help="score a seeded sample of pairs, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != accuracy.DATASET:
        raise SystemExit(f"--dataset must be {accuracy.DATASET}, got {dataset_id}")
    dataset = Path(directory)
    pairs = accuracy.sampled_pairs(accuracy.listed_pairs(dataset), args.limit)

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    def disparity_of(frame: np.ndarray) -> np.ndarray:
        # accuracy.py passes the MPK a channels-last frame; the ONNX takes it channels-first.
        (output,) = session.run(None, {INPUT: np.ascontiguousarray(frame.transpose(2, 0, 1)[None])})
        return output

    metrics = accuracy.evaluate(dataset, pairs, disparity_of)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    args.output.write_text(json.dumps({"pairs": len(pairs), "metrics": rounded}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
