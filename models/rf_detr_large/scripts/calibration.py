"""Select and preprocess RF-DETR images, and select the decoder's top proposals."""

from __future__ import annotations

import io
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from model_inputs.images import image_names, read_images
from PIL import Image

IMAGE_SIZE = 704
QUERY_COUNT = 300
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def select_images(dataset: Path, seed: int, start: int, count: int) -> tuple[str, ...]:
    """Return images start to start + count of the dataset's seeded shuffle."""
    names = image_names(dataset, IMAGE_SUFFIXES)
    if start < 0 or count < 1:
        raise ValueError("an image selection needs a non-negative start and a positive count")
    if len(names) < start + count:
        raise ValueError(f"dataset contains {len(names)} images, but {start + count} are required")
    random.Random(seed).shuffle(names)
    return tuple(names[start : start + count])


def preprocess_image(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - MEAN) / STD


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    if not names:
        raise ValueError("at least one image must be selected")
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])


def to_afe_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def select_top_proposals(scores: np.ndarray, boxes: np.ndarray, count: int = QUERY_COUNT) -> np.ndarray:
    """Return the boxes of the highest-scoring proposals, keeping equal scores in their original order."""
    if scores.ndim != 2 or boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError(f"unexpected proposal shapes: scores={scores.shape}, boxes={boxes.shape}")
    if scores.shape != boxes.shape[:2]:
        raise ValueError(f"proposal count mismatch: scores={scores.shape}, boxes={boxes.shape}")
    if count < 1 or scores.shape[1] < count:
        raise ValueError(f"cannot select {count} proposals from {scores.shape[1]}")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("proposal scores and boxes must be finite")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :count]
    return np.take_along_axis(boxes, indices[..., None], axis=1).astype(np.float32)
