"""Measure the FP32 baseline: SuperPoint's ONNX on a host, scored exactly as accuracy.py scores the MPK.

The baseline is the accuracy the package would reach without quantization, on the same hpatches-val pairs and
inputs. The ONNX's raw heads are decoded in NumPy with Magic Leap's demo settings, the decode Neat's MagicLeapDemoV1
profile runs on the board. model.yaml publishes the result as the reference_metrics entry with origin fp32_baseline.
Run it on a host with numpy, opencv-python-headless and onnxruntime, and tools/model_report on PYTHONPATH:

    PYTHONPATH=tools python models/superpoint/scripts/fp32_baseline_accuracy.py \\
        --onnx model.onnx --dataset hpatches-val=<directory> --output fp32-baseline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import accuracy
import numpy as np
from keypoints import CELL

# Magic Leap's demo decode (SuperPointPretrainedNetwork's SuperPointFrontend).
THRESHOLD = 0.015
NMS_RADIUS = 4
BORDER = 4


def score_map(logits: np.ndarray) -> np.ndarray:
    """The demo's [H, W] keypoint scores: exp over each cell's 65 logits, divided by their sum plus 1e-5."""
    dense = np.exp(logits.astype(np.float64))
    dense = dense / (dense.sum(0) + 1e-5)
    cells_y, cells_x = logits.shape[1:]
    scores = dense[:-1].transpose(1, 2, 0).reshape(cells_y, cells_x, CELL, CELL)
    return scores.transpose(0, 2, 1, 3).reshape(cells_y * CELL, cells_x * CELL).astype(np.float32)


def grid_nms(xs: np.ndarray, ys: np.ndarray, scores: np.ndarray, height: int, width: int) -> np.ndarray:
    """The demo's greedy NMS: keep the best remaining point and suppress its (2r+1)-square neighbourhood.

    Returns the indices of the kept points, best first.
    """
    order = np.argsort(-scores, kind="stable")
    occupied = np.zeros((height + 2 * NMS_RADIUS, width + 2 * NMS_RADIUS), bool)
    occupied[ys + NMS_RADIUS, xs + NMS_RADIUS] = True
    kept = []
    for index in order:
        x, y = xs[index] + NMS_RADIUS, ys[index] + NMS_RADIUS
        if occupied[y, x]:
            kept.append(index)
            occupied[y - NMS_RADIUS:y + NMS_RADIUS + 1, x - NMS_RADIUS:x + NMS_RADIUS + 1] = False
    return np.array(kept, int)


def sample_descriptors(keypoints: np.ndarray, raw_descriptors: np.ndarray) -> np.ndarray:
    """The demo's descriptors: bilinear samples of the L2-normalized coarse map at each keypoint, normalized again."""
    dense = raw_descriptors / np.maximum(np.linalg.norm(raw_descriptors, axis=0, keepdims=True), 1e-12)
    channels, cells_y, cells_x = dense.shape
    u = keypoints[:, 0] / (cells_x * CELL) * (cells_x - 1)
    v = keypoints[:, 1] / (cells_y * CELL) * (cells_y - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    sampled = np.zeros((len(keypoints), channels), np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            ui, vi = np.clip(u0 + du, 0, cells_x - 1), np.clip(v0 + dv, 0, cells_y - 1)
            weight = (1 - np.abs(u - (u0 + du))) * (1 - np.abs(v - (v0 + dv)))
            sampled += weight[:, None] * dense[:, vi, ui].T
    return sampled / np.maximum(np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12)


def decode(logits: np.ndarray, raw_descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One image's channels-first heads to (x, y) keypoints, scores and unit descriptors, best first."""
    scores = score_map(logits)
    height, width = scores.shape
    ys, xs = np.nonzero(scores >= THRESHOLD)
    kept = grid_nms(xs, ys, scores[ys, xs], height, width)
    xs, ys = xs[kept], ys[kept]
    inside = (xs >= BORDER) & (xs < width - BORDER) & (ys >= BORDER) & (ys < height - BORDER)
    xs, ys = xs[inside][: accuracy.TOP_K], ys[inside][: accuracy.TOP_K]
    keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
    return keypoints, scores[ys, xs], sample_descriptors(keypoints, raw_descriptors.astype(np.float32))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, required=True, help="the converted model.onnx")
    parser.add_argument("--dataset", required=True, help="hpatches-val=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the metrics")
    parser.add_argument("--limit", type=int, help="score the first sequences only, for a quick check")
    args = parser.parse_args(argv)
    dataset_id, directory = args.dataset.split("=", 1)
    sequences = accuracy.sequence_dirs(Path(directory))[: args.limit]

    import onnxruntime

    session = onnxruntime.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    (input_name,) = [value.name for value in session.get_inputs()]

    def features_of(image: np.ndarray) -> tuple:
        logits, descriptors = session.run(None, {input_name: image[None, None]})
        return decode(logits[0], descriptors[0])

    metrics, keypoints = accuracy.evaluate(map(accuracy.load_sequence, sequences), len(sequences), features_of)
    rounded = {name: round(value, 4) for name, value in metrics.items()}
    result = {"dataset": dataset_id, "pairs": len(sequences) * len(accuracy.TARGETS), "metrics": rounded,
              "keypoints_per_image": keypoints}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
