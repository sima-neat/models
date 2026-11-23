"""Measure FAN 2dfan2_alt's NME on the AFW test faces with its MPK on a Modalix board.

Each AFW test image is a face crop, so its bounds are the face box. The CPU crops the face as the model takes it
(landmarks.py), the MPK returns 68 heatmaps, and the CPU decodes them to landmark coordinates. NME is each face's mean
point error over its inter-ocular distance (the outer eye corners, points 36 and 45), averaged over the faces.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import model_report
import numpy as np
import pyneat
from landmarks import DATASET_ID, HEATMAP_SIZE, LANDMARKS, TEST_SPLIT, dataset_path, evaluate, test_faces


def tensor_options() -> pyneat.ModelOptions:
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Tensor
    # The script already cropped and scaled the face; Neat still quantizes and tessellates.
    options.preprocess.resize.enable = pyneat.AutoFlag.Off
    options.preprocess.color_convert.enable = pyneat.AutoFlag.Off
    options.preprocess.normalize.enable = pyneat.AutoFlag.Off
    return options


def parse_mpks(values: list[str]) -> Path:
    if len(values) != 1 or not values[0].startswith("model=") or not values[0][6:].strip():
        raise SystemExit("--mpk must name exactly model=<path>")
    return Path(values[0][6:])


def heatmaps_of(tensor: pyneat.Tensor, *, copy: bool = True) -> np.ndarray:
    """The MPK's one face as [68, 64, 64]; a borrowed view retains its tensor owner until released."""
    array = tensor.to_numpy(copy=copy)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one face heatmap batch, got {array.shape}")
        array = array[0]
    if array.shape != (HEATMAP_SIZE, HEATMAP_SIZE, LANDMARKS) or not np.isfinite(array).all():
        raise ValueError(f"unexpected heatmaps: shape {array.shape}")
    return array.transpose(2, 0, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True, help="model=<path>")
    parser.add_argument("--dataset", required=True, help="afw-faces-default=<extracted directory>")
    parser.add_argument("--output", type=Path, required=True, help="directory for model-report.json")
    args = parser.parse_args(argv)
    faces = test_faces(dataset_path(args.dataset))
    model = pyneat.Model(str(parse_mpks(args.mpk)), tensor_options())
    runner = None
    started = time.monotonic()

    def infer(patch):
        nonlocal runner
        tensor = pyneat.Tensor.from_numpy(patch, copy=True, layout=pyneat.TensorLayout.HWC)
        if runner is None:
            runner = model.build([tensor])
        outputs = runner.run([tensor], timeout_ms=120000)
        if len(outputs) != 1:
            raise ValueError(f"expected one heatmap output, got {len(outputs)}")
        return heatmaps_of(outputs[0])

    try:
        metrics, details = evaluate(faces, infer)
    finally:
        if runner is not None:
            runner.close()
    print(f"{len(faces)} faces in {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)

    report = model_report.from_accuracy(
        {
            DATASET_ID: {
                "metrics": metrics,
                "split": TEST_SPLIT,
                "evaluated_samples": len(faces),
                "available_samples": len(faces),
                "details": {"normalization": "inter-ocular (outer eye corners)", **details},
            },
        }
    )
    report.save(args.output)
    measured = report.document["datasets"][DATASET_ID]["metrics"]
    print(json.dumps({name: round(value, 5) for name, value in measured.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
