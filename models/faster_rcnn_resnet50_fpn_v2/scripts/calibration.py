"""Torchvision's Faster R-CNN steps around the two MLA components, in NumPy and OpenCV, and calibration images.

These follow torchvision's fasterrcnn_resnet50_fpn_v2 at inference with its defaults: GeneralizedRCNNTransform
(normalize, then resize the short side to 800 with the long side capped at 1333), the RPN's anchors, top 1,000
proposals per level, NMS at 0.7 and top 1,000 overall, MultiScaleRoIAlign (7x7, sampling ratio 2, aligned=False)
and RoIHeads' decoding, score threshold 0.05, per-class NMS at 0.5 and 100 detections. The one departure is the
input: the resized image sits in the top-left corner of a fixed 1344x1344 input, padded with zeros as torchvision
pads each image to a multiple of 32.
"""

from __future__ import annotations

import math
import random
import tarfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

INPUT_SIZE = 1344
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_SIZE, MAX_SIZE = 800, 1333
ANCHOR_SIZES = (32, 64, 128, 256, 512)
ASPECT_RATIOS = (0.5, 1.0, 2.0)
PRE_NMS_TOP_N = 1000
POST_NMS_TOP_N = 1000
RPN_NMS_THRESHOLD = 0.7
RPN_MIN_SIZE = 1e-3
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BOX_CLIP = math.log(1000.0 / 16)
SCORE_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
BOX_MIN_SIZE = 1e-2
DETECTIONS = 100
CHANNELS = 256
CLASSES = 91
POOL = 7
SAMPLING_RATIO = 2
# The box head's input: the proposals' 7x7 maps in a grid of 25 rows at an 8-pixel pitch, zero between them. A build
# profile sets how many proposals the box head takes (box_head's proposals parameter): 1,000, torchvision's default,
# fill a 25x40 grid.
GRID_ROWS, PITCH = 25, 8
# The backbone's one output, (1, 271, 1, 150381) NCHW: each FPN level's cells as rows, P2-P6 in order, with
# P2-P5's 256 features (zero on P6's rows) then the RPN head's 3 objectness logits and 12 box deltas per cell.
# Under INT8 the output shares one scale, so the graph scales the RPN head's outputs into the features' range.
LEVEL_SIZES = tuple(INPUT_SIZE // stride for stride in (4, 8, 16, 32, 64))
LEVEL_ROWS = tuple(int(np.cumsum([0] + [size * size for size in LEVEL_SIZES])[i]) for i in range(6))
ANCHORS = 3
RPN_CHANNELS = ANCHORS + 4 * ANCHORS
BACKBONE_CHANNELS = CHANNELS + RPN_CHANNELS
OBJECTNESS_GAIN, DELTA_GAIN = 1 / 16, 1 / 2
# RoIAlign and the RPN's per-level NMS run on worker threads; OpenCV and NumPy release the GIL.
WORKERS = 8
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261102
_POOL = ThreadPoolExecutor(WORKERS)


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resized_size(height: int, width: int) -> tuple[int, int]:
    """The size GeneralizedRCNNTransform resizes an image to, as eager torchvision computes it."""
    scale = min(MIN_SIZE / min(height, width), MAX_SIZE / max(height, width))
    return math.floor(height * scale), math.floor(width * scale)


def preprocess(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB uint8 HWC -> the backbone's (1344, 1344, 3) float32 input and the resized image's (height, width).

    Torchvision normalizes, then resizes bilinearly without antialiasing (align_corners=False), as OpenCV's
    INTER_LINEAR does, then pads with zeros on the bottom and right.
    """
    resized = resized_size(*image.shape[:2])
    normalized = (image.astype(np.float32) * np.float32(1 / 255) - MEAN) / STD
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.float32)
    canvas[: resized[0], : resized[1]] = cv2.resize(normalized, resized[::-1], interpolation=cv2.INTER_LINEAR)
    return canvas, resized


def _base_anchors(size: int) -> np.ndarray:
    """AnchorGenerator's zero-centred anchors for one level, rounded as torchvision rounds them."""
    h_ratios = np.sqrt(np.array(ASPECT_RATIOS, np.float32))
    widths, heights = size / h_ratios, size * h_ratios
    return np.round(np.stack([-widths, -heights, widths, heights], axis=1) / 2).astype(np.float32)


def decode_boxes(deltas: np.ndarray, boxes: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """BoxCoder.decode_single: (N, 4) deltas applied to (N, 4) boxes."""
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    center_x, center_y = boxes[:, 0] + 0.5 * widths, boxes[:, 1] + 0.5 * heights
    dx, dy = deltas[:, 0] / weights[0], deltas[:, 1] / weights[1]
    dw = np.minimum(deltas[:, 2] / weights[2], BOX_CLIP)
    dh = np.minimum(deltas[:, 3] / weights[3], BOX_CLIP)
    x, y = dx * widths + center_x, dy * heights + center_y
    half_w, half_h = np.float32(0.5) * np.exp(dw) * widths, np.float32(0.5) * np.exp(dh) * heights
    return np.stack([x - half_w, y - half_h, x + half_w, y + half_h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack(
        [
            np.clip(boxes[:, 0], 0, size[1]),
            np.clip(boxes[:, 1], 0, size[0]),
            np.clip(boxes[:, 2], 0, size[1]),
            np.clip(boxes[:, 3], 0, size[0]),
        ],
        axis=1,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold: float) -> np.ndarray:
    """torchvision.ops.batched_nms through OpenCV: the indices kept, by decreasing score."""
    if not len(boxes):
        return np.zeros(0, np.int64)
    rects = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1).astype(np.float64)
    keep = np.asarray(
        cv2.dnn.NMSBoxesBatched(rects, scores.astype(np.float32), groups.astype(np.int32), -1.0, threshold),
        np.int64,
    ).reshape(-1)
    return keep[np.argsort(-scores[keep], kind="stable")]


def unpack_backbone(output: np.ndarray) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """The backbone's output, NCHW (1, 271, 1, R) or NHWC (1, 1, R, 271) -> P2-P5 as contiguous (H, W, 256) maps,
    and per level P2-P6 the objectness logits (H*W*3,) and box deltas (H*W*3, 4) in torchvision's anchor order."""
    rows = (
        output.reshape(BACKBONE_CHANNELS, -1).T
        if output.shape[1] == BACKBONE_CHANNELS
        else output.reshape(-1, BACKBONE_CHANNELS)
    )
    if len(rows) != LEVEL_ROWS[-1]:
        raise ValueError(f"unexpected backbone output: {output.shape}")
    features = [
        np.ascontiguousarray(rows[LEVEL_ROWS[i] : LEVEL_ROWS[i + 1], :CHANNELS]).reshape(size, size, CHANNELS)
        for i, size in enumerate(LEVEL_SIZES[:4])
    ]
    rpn = []
    for i in range(len(LEVEL_SIZES)):
        level = rows[LEVEL_ROWS[i] : LEVEL_ROWS[i + 1], CHANNELS:]
        logits = level[:, :ANCHORS].reshape(-1) * np.float32(1 / OBJECTNESS_GAIN)
        deltas = level[:, ANCHORS:].reshape(-1, 4) * np.float32(1 / DELTA_GAIN)
        rpn.append((logits, deltas))
    return features, rpn


def _level_proposals(logits: np.ndarray, deltas: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """One FPN level's top PRE_NMS_TOP_N anchors, decoded: (boxes, logits), by decreasing logit."""
    width = LEVEL_SIZES[level]
    count = min(PRE_NMS_TOP_N, logits.size)
    top = np.argpartition(-logits, count - 1)[:count] if count < logits.size else np.arange(count)
    top = top[np.argsort(-logits[top], kind="stable")]
    anchor, cell = top % ANCHORS, top // ANCHORS
    row, column = cell // width, cell % width
    shifts = np.stack([column, row, column, row], axis=1).astype(np.float32) * (INPUT_SIZE // width)
    boxes = _base_anchors(ANCHOR_SIZES[level])[anchor] + shifts
    return decode_boxes(deltas[top], boxes, (1.0, 1.0, 1.0, 1.0)), logits[top]


def grid_cols(regions: int) -> int:
    """The box head grid's columns for a number of proposals, a multiple of its 25 rows."""
    if regions <= 0 or regions % GRID_ROWS:
        raise ValueError(f"the box head takes a positive multiple of {GRID_ROWS} proposals, not {regions}")
    return regions // GRID_ROWS


def grid_shape(regions: int) -> tuple[int, int, int]:
    """The box head's (height, width, channels) input for a number of proposals."""
    return GRID_ROWS * PITCH, grid_cols(regions) * PITCH, CHANNELS


def proposals(
    rpn: Sequence[tuple[np.ndarray, np.ndarray]], size: tuple[int, int], regions: int = POST_NMS_TOP_N
) -> np.ndarray:
    """RegionProposalNetwork at inference: unpack_backbone's RPN rows of P2-P6 -> up to regions (x1, y1, x2, y2)
    boxes, 1,000 by default as in torchvision.

    Batched NMS by level equals NMS within each level, so the levels run on worker threads.
    """

    def level_nms(level: int) -> tuple[np.ndarray, np.ndarray]:
        boxes, logits = _level_proposals(*rpn[level], level)
        boxes = clip_boxes(boxes, size)
        scores = 1 / (1 + np.exp(-logits))
        keep = np.where((boxes[:, 2] - boxes[:, 0] >= RPN_MIN_SIZE) & (boxes[:, 3] - boxes[:, 1] >= RPN_MIN_SIZE))[0]
        keep = keep[nms(boxes[keep], scores[keep], np.zeros(len(keep)), RPN_NMS_THRESHOLD)]
        return boxes[keep], scores[keep]

    kept = list(_POOL.map(level_nms, range(len(rpn))))
    boxes = np.concatenate([item[0] for item in kept])
    scores = np.concatenate([item[1] for item in kept])
    return boxes[np.argsort(-scores, kind="stable")[:regions]]


def roi_levels(boxes: np.ndarray) -> np.ndarray:
    """LevelMapper: the FPN map, 0-3 for P2-P5, each box pools from."""
    scale = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    level = np.floor(4 + np.log2(scale / 224 + np.float32(1e-6)))
    return (np.clip(level, 2, 5) - 2).astype(np.int64)


def _sample_coordinates(start: np.ndarray, end: np.ndarray, size: int) -> np.ndarray:
    """roi_align's sample coordinates along one axis, (R, 14), with its edge rules applied.

    A sample beyond one pixel outside the map is zero, and one within that pixel reads the edge row; moving
    the former far outside and clamping the latter makes OpenCV's zero border give the same values.
    """
    length = np.maximum(end - start, np.float32(1.0))
    bin_size = length / np.float32(POOL)
    bins = np.arange(POOL, dtype=np.float32).repeat(SAMPLING_RATIO)
    offsets = np.tile(np.arange(SAMPLING_RATIO, dtype=np.float32) + np.float32(0.5), POOL) / np.float32(SAMPLING_RATIO)
    coordinates = start[:, None] + (bins + offsets)[None, :] * bin_size[:, None]
    inside = (coordinates >= -1.0) & (coordinates <= size)
    return np.where(inside, np.clip(coordinates, 0, size - 1), np.float32(-4.0)).astype(np.float32)


def _roi_align(feature: np.ndarray, boxes: np.ndarray, scale: float) -> np.ndarray:
    """roi_align on one (H, W, 256) map -> (R, 7, 7, 256), sampling with cv2.remap.

    OpenCV places bilinear samples on a 1/32-pixel grid, a small departure from torchvision's exact positions.
    """
    height, width = feature.shape[:2]
    scaled = boxes * np.float32(scale)
    rows = _sample_coordinates(scaled[:, 1], scaled[:, 3], height)
    columns = _sample_coordinates(scaled[:, 0], scaled[:, 2], width)
    samples = POOL * SAMPLING_RATIO
    map_y = np.ascontiguousarray(np.broadcast_to(rows[:, :, None], (len(boxes), samples, samples)).reshape(-1, samples))
    map_x = np.ascontiguousarray(
        np.broadcast_to(columns[:, None, :], (len(boxes), samples, samples)).reshape(-1, samples)
    )
    sampled = cv2.remap(feature, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sampled = sampled.reshape(len(boxes), POOL, SAMPLING_RATIO, POOL, SAMPLING_RATIO, -1)
    return (sampled[:, :, 0, :, 0] + sampled[:, :, 0, :, 1] + sampled[:, :, 1, :, 0] + sampled[:, :, 1, :, 1]) * 0.25


def roi_align(features: Sequence[np.ndarray], boxes: np.ndarray) -> np.ndarray:
    """MultiScaleRoIAlign over (H, W, 256) maps P2-P5 -> (R, 7, 7, 256), in chunks on worker threads."""
    levels = roi_levels(boxes)
    pooled = np.empty((len(boxes), POOL, POOL, CHANNELS), np.float32)
    jobs = []
    for level, feature in enumerate(features):
        indices = np.where(levels == level)[0]
        scale = 2.0 ** round(math.log2(feature.shape[0] / INPUT_SIZE))
        for chunk in np.array_split(indices, max(1, math.ceil(len(indices) / 64))):
            if len(chunk):
                jobs.append((chunk, _POOL.submit(_roi_align, feature, boxes[chunk], scale)))
    for chunk, job in jobs:
        pooled[chunk] = job.result()
    return pooled


def tile_regions(pooled: np.ndarray, regions: int = POST_NMS_TOP_N) -> np.ndarray:
    """(R <= regions, 7, 7, 256) pooled regions -> the box head's input, (200, 320, 256) for 1,000, in grid order."""
    cols = grid_cols(regions)
    if pooled.shape[1:] != (POOL, POOL, CHANNELS) or len(pooled) > regions:
        raise ValueError(f"unexpected pooled regions: {pooled.shape}")
    grid = np.zeros((GRID_ROWS, cols, PITCH, PITCH, CHANNELS), np.float32)
    grid.reshape(regions, PITCH, PITCH, CHANNELS)[: len(pooled), :POOL, :POOL] = pooled
    return grid.transpose(0, 2, 1, 3, 4).reshape(GRID_ROWS * PITCH, cols * PITCH, CHANNELS)


def region_rows(output: np.ndarray) -> np.ndarray:
    """A box head output, NCHW (1, C, 25, cols) or NHWC (1, 25, cols, C) -> (25 * cols, C), one row per region."""
    if output.ndim == 4 and output.shape[1] == GRID_ROWS:
        return output.reshape(GRID_ROWS * output.shape[2], -1)
    if output.ndim == 4 and output.shape[2] == GRID_ROWS:
        return output.reshape(output.shape[1], GRID_ROWS * output.shape[3]).T
    raise ValueError(f"unexpected box head output: {output.shape}")


def detections(
    class_logits: np.ndarray, box_regression: np.ndarray, boxes: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RoIHeads.postprocess_detections: (R, 91) logits and (R, 364) deltas of R proposals -> boxes in the
    resized image, scores and COCO category IDs, at most 100, by decreasing score."""
    logits = class_logits - class_logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores = scores[:, 1:] / scores.sum(axis=1, keepdims=True)
    region, column = np.nonzero(scores > SCORE_THRESHOLD)
    labels = column + 1
    deltas = box_regression.reshape(-1, CLASSES, 4)[region, labels]
    decoded = clip_boxes(decode_boxes(deltas, boxes[region], BOX_WEIGHTS), size)
    scores = scores[region, column]
    keep = np.where((decoded[:, 2] - decoded[:, 0] >= BOX_MIN_SIZE) & (decoded[:, 3] - decoded[:, 1] >= BOX_MIN_SIZE))[
        0
    ]
    decoded, scores, labels = decoded[keep], scores[keep], labels[keep]
    keep = nms(decoded, scores, labels, NMS_THRESHOLD)[:DETECTIONS]
    return decoded[keep], scores[keep], labels[keep]


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def _image_names(dataset: Path) -> list[str]:
    if dataset.is_dir():
        return sorted(
            path.relative_to(dataset).as_posix()
            for path in dataset.rglob("*")
            if path.is_file() and _is_image(path.name)
        )
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            return sorted(member.name for member in archive.getmembers() if member.isfile() and _is_image(member.name))
    raise ValueError(f"unsupported image dataset input: {dataset}")


def _image_bytes(dataset: Path, names: Sequence[str]) -> Iterator[bytes]:
    if dataset.is_dir():
        for name in names:
            yield (dataset / name).read_bytes()
        return
    with tarfile.open(dataset) as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in names:
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"could not read image from dataset: {name}")
            yield stream.read()


def calibration_names(dataset: Path, count: int) -> list[str]:
    """A seeded sample of the dataset's cal2017 images, in sampling order."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    """Each selected calibration image as the backbone's input and its resized (height, width)."""
    for content in _image_bytes(dataset, calibration_names(dataset, count)):
        yield preprocess(decode_image(content))
