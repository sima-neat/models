"""Score the exported FP32 FAN ONNX with the same AFW crops, decode and NME as accuracy.py.

Run on a host with numpy, OpenCV and onnxruntime:
    python fp32_baseline_accuracy.py --onnx model.onnx --dataset afw-faces-default=<directory> --output baseline.json
No credentials, registry client, PyNeat or YAML configuration are used. This is an accuracy reference, not a
performance or quantized-simulator result. model.yaml must record only an actually measured complete-dataset result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from landmarks import DATASET_ID, HEATMAP_SIZE, LANDMARKS, TEST_SPLIT, dataset_path, evaluate, test_faces


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    if [(item.name, item.shape) for item in session.get_inputs()] != [("image", [1, 3, 256, 256])]:
        raise ValueError("FP32 baseline requires image [1,3,256,256]")
    if [(item.name, item.shape) for item in session.get_outputs()] != [("heatmaps", [1, 68, 64, 64])]:
        raise ValueError("FP32 baseline requires heatmaps [1,68,64,64]")

    def infer(patch):
        (heatmaps,) = session.run(["heatmaps"], {"image": np.ascontiguousarray(patch.transpose(2, 0, 1)[None])})
        if heatmaps.shape != (1, LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE):
            raise ValueError(f"unexpected FP32 heatmaps: {heatmaps.shape}")
        return heatmaps[0]

    faces = test_faces(dataset_path(args.dataset))
    metrics, details = evaluate(faces, infer)
    result = {"dataset": DATASET_ID, "split": TEST_SPLIT, "samples": len(faces), "metrics": metrics, **details}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
