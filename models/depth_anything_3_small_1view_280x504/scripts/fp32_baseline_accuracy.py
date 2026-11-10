"""Score the converted FP32 ONNX with accuracy.py's exact NYU preprocessing, crop and alignment."""

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help=f"{accuracy.DATASET}=<directory>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != accuracy.DATASET:
        parser.error(f"--dataset must name {accuracy.DATASET}")
    images = accuracy.selected_frames(accuracy.frames(Path(directory)), args.limit)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])

    def infer(image):
        inputs = {"image": np.ascontiguousarray(image.transpose(2, 0, 1)[None])}
        return accuracy.depth_output(session.run(["depth", "depth_conf"], inputs))

    result = {"samples": len(images), "metrics": accuracy.evaluate(images, infer)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
