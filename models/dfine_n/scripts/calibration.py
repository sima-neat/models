"""Select and preprocess D-FINE images, and select the decoder's top proposals."""

from __future__ import annotations

import io
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from model_inputs.images import image_names, read_images
from PIL import Image

IMAGE_SIZE = 640
QUERY_COUNT = 300
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# D-FINE takes RGB scaled to [0, 1] with no ImageNet mean or std.
MEAN = np.zeros(3, np.float32)
STD = np.ones(3, np.float32)


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
    return (array - MEAN[:, None, None]) / STD[:, None, None]


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    if not names:
        raise ValueError("at least one image must be selected")
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])


def to_afe_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def select_top_queries(scores, content, reference_logits):
    """Gather content and raw references in stable Top300 order, never attention memory."""
    if scores.shape != (1, 2000) or content.shape != (1, 2000, 128) or reference_logits.shape != (1, 2000, 4):
        raise ValueError("Unexpected Nano proposal boundary")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (scores, content, reference_logits)):
        raise ValueError("Expected finite float32 proposal tensors")
    order = np.argsort(-scores, axis=1, kind="stable")[0, :QUERY_COUNT]
    return tuple(x[0, order, :].reshape(1, QUERY_COUNT, x.shape[-1]) for x in (content, reference_logits))


def decoder_inputs(outputs):
    memory, content, scores, references = outputs
    return memory, *select_top_queries(scores, content, references)
