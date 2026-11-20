"""RF-DETR's captured INT8 handoff and parameter-free CPU output contract.

No learned layer, observer, or quantization grid is changed here. The helpers
also provide a CPU oracle for qualifying the native TopK/Gather/BoxDecode path;
their own timings are not native-kernel performance evidence.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

QUERY_COUNT = 300
PROPOSAL_COUNT = 576
CLASS_COUNT = 91
COCO_CATEGORY_IDS = frozenset(
    (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    )
)


@dataclass(frozen=True)
class Int8Grid:
    step_bits: int
    zero_point: int

    def __post_init__(self) -> None:
        if type(self.step_bits) is not int or not 0 <= self.step_bits < 2**32:
            raise ValueError("step_bits must be an unsigned 32-bit integer")
        if type(self.zero_point) is not int or not -128 <= self.zero_point <= 127:
            raise ValueError("zero_point must be an INT8 integer")
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("the FLOAT32 step must be finite and positive")

    @property
    def step(self) -> np.float32:
        return np.float32(struct.unpack("<f", struct.pack("<I", self.step_bits))[0])

    def quantize(self, values: np.ndarray) -> np.ndarray:
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("quantization requires finite FLOAT32 input")
        # Do not replace division with a reciprocal multiply, move zero point
        # before RNE, or allow signed INT8 wrapping before saturation.
        with np.errstate(over="ignore"):
            scaled = values / self.step
        return np.clip(np.rint(scaled) + np.float32(self.zero_point), -128, 127).astype(np.int8)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        if codes.dtype != np.int8:
            raise ValueError("dequantization requires INT8 codes, not re-quantized FLOAT32 outputs")
        return (codes.astype(np.float32) - np.float32(self.zero_point)) * self.step


def top_proposals(scores: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Captured lower-index-first INT8 TopK, including MaskMinRepeatZero.

    After all scores greater than INT8_MIN have been masked to INT8_MIN,
    the original repeated-max program returns index zero for every remaining
    slot. A unique stable argsort is NOT equivalent in this exhaustion case.
    """
    if scores.dtype != np.int8 or scores.size != PROPOSAL_COUNT:
        raise ValueError("proposal scores must contain exactly 576 INT8 codes")
    if boxes.dtype != np.int8 or boxes.shape != (1, PROPOSAL_COUNT, 4):
        raise ValueError("proposal boxes must be INT8 [1,576,4]")
    flat = scores.reshape(PROPOSAL_COUNT)
    # Widen before negation: -INT8_MIN cannot be represented by INT8.
    order = np.argsort(-flat.astype(np.int16), kind="stable")
    live = order[flat[order] > np.iinfo(np.int8).min][:QUERY_COUNT]
    indices = np.zeros(QUERY_COUNT, dtype=np.int64)
    indices[: live.size] = live
    return np.ascontiguousarray(boxes[:, indices, :]), indices


def decode_boxes(deltas: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Approved external_fp32_v1 CPU decode; no learned FP32 model layer."""
    if (
        deltas.dtype != np.float32
        or references.dtype != np.float32
        or deltas.shape != (1, QUERY_COUNT, 4)
        or references.shape != deltas.shape
        or not np.isfinite(deltas).all()
        or not np.isfinite(references).all()
    ):
        raise ValueError("box decode requires finite FLOAT32 [1,300,4] deltas and references")
    with np.errstate(over="raise", invalid="raise"):
        centers = deltas[..., :2] * references[..., 2:] + references[..., :2]
        sizes = np.exp(deltas[..., 2:]) * references[..., 2:]
    result = np.concatenate((centers, sizes), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("box decode produced non-finite coordinates")
    return result


def detections(
    outputs: list[np.ndarray],
    grids: tuple[Int8Grid, ...],
    *,
    width: int,
    height: int,
    image_id: int,
    score_threshold: float = 0.0,
) -> np.ndarray:
    """Return COCO rows: image id, x, y, w, h, score, raw category id.

    Rank all 300x91 pairs before filtering unused COCO slots. There is no NMS.
    Caller chooses whether to include category filtering in a timed workload.
    """
    if len(outputs) != 3 or len(grids) != 3 or width < 1 or height < 1:
        raise ValueError("three output grids and positive original image dimensions are required")
    expected = ((1, QUERY_COUNT, 4), (1, QUERY_COUNT, CLASS_COUNT), (1, QUERY_COUNT, 4))
    if tuple(value.shape for value in outputs) != expected:
        raise ValueError("decoder output shape/order differs from deltas, logits, references")
    if not np.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must lie in [0,1]")
    deltas, logits, references = [grid.dequantize(value) for value, grid in zip(outputs, grids)]
    boxes = decode_boxes(deltas, references)[0]
    with np.errstate(over="ignore"):
        scores = (np.float32(1) / (np.float32(1) + np.exp(-logits))).reshape(-1)
    order = np.argsort(-scores, kind="stable")[:QUERY_COUNT]
    categories = order % CLASS_COUNT
    selected = np.array([value in COCO_CATEGORY_IDS for value in categories]) & (scores[order] >= score_threshold)
    order, categories = order[selected], categories[selected]
    chosen = boxes[order // CLASS_COUNT]
    lower = chosen[:, :2] - chosen[:, 2:] / np.float32(2)
    upper = chosen[:, :2] + chosen[:, 2:] / np.float32(2)
    dimensions = np.array([width, height], dtype=np.float32)
    lower = np.clip(lower * dimensions, 0, dimensions)
    upper = np.clip(upper * dimensions, 0, dimensions)
    result = np.column_stack((np.full(order.size, image_id), lower, upper - lower, scores[order], categories))
    if not np.isfinite(result).all():
        raise ValueError("decoded detections must be finite")
    return result
