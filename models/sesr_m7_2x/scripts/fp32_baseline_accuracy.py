"""Score the FP32 ONNX that compile.py receives, on a host, with accuracy.py's tiling and scoring.

Run it on the converted model.onnx and the extracted div2k-mini archive:

    python fp32_baseline_accuracy.py --model model.onnx --dataset <directory> [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from accuracy import evaluate, image_pairs


def main() -> None:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="the converted FP32 ONNX")
    parser.add_argument("--dataset", type=Path, required=True, help="extracted div2k-mini archive")
    parser.add_argument("--limit", type=int, help="first images only")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def infer(tile: np.ndarray) -> np.ndarray:
        return session.run(None, {name: np.ascontiguousarray(tile.transpose(2, 0, 1)[None])})[0]

    pairs = image_pairs(args.dataset)
    print(json.dumps(evaluate(pairs[: args.limit] if args.limit else pairs, infer)))


if __name__ == "__main__":
    main()
