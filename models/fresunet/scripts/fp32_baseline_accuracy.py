"""Measure the FP32 baseline: FresUNet's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same OSCD test cities and
preprocessing. model.yaml publishes it as the reference_metrics entry with origin fp32_baseline. Run it on a host with
numpy, scipy, opencv-python-headless and onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/fresunet/scripts/fp32_baseline_accuracy.py \\
        --onnx fresunet_13ch.onnx --dataset oscd-default=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from oscd import INPUT_NAME, ChangeScore, change_map, cities, city_input, dataset_root, prediction

# As accuracy.py.
DATASET = "oscd-default"
SPLIT = "test"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the FresUNet ONNX model.yaml pins")
    parser.add_argument("--dataset", required=True, help=f"{DATASET}=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    if dataset_id != DATASET:
        raise SystemExit(f"--dataset must be {DATASET}, got {dataset_id}")
    root = dataset_root(Path(directory))

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    score = ChangeScore()
    names = cities(root, SPLIT)
    for name in names:
        # accuracy.py passes the input channels last; the ONNX takes it channels first.
        feed = np.ascontiguousarray(city_input(root, name).transpose(2, 0, 1)[None])
        (output,) = session.run(None, {INPUT_NAME: feed})
        score.add(prediction(output), change_map(root, name))
    rounded = {name: round(value, 4) for name, value in score.metrics().items()}
    args.output.write_text(json.dumps({"cities": len(names), "metrics": rounded}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
