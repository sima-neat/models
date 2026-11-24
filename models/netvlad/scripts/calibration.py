"""Resize and normalize images the way Patch-NetVLAD extracts features, and select the compile-time images."""

from __future__ import annotations

import io
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from model_inputs.images import image_names, read_images
from PIL import Image

# Patch-NetVLAD's input_transform: Resize((480, 640)), ToTensor, then ImageNet normalization (HWC order here).
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# pitts30k/cal holds 1,000 Pitts30k-train database JPEGs under pitts30k/images/, beside the split and README.
IMAGE_ROOT = "pitts30k/images/"


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode one image and return the model input: 480x640 RGB, ImageNet-normalized, HWC float32."""
    with Image.open(io.BytesIO(content)) as source:
        # torchvision's Resize on a PIL image is PIL's antialiased bilinear resize to (width, height).
        image = source.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / np.float32(255) - MEAN) / STD


def select_images(dataset: Path, seed: int, start: int, count: int) -> tuple[str, ...]:
    """Return images start to start + count of a seeded shuffle of the dataset's Pittsburgh images."""
    names = [name for name in image_names(dataset, (".jpg",)) if name.startswith(IMAGE_ROOT)]
    if start < 0 or count < 1:
        raise ValueError("an image selection needs a non-negative start and a positive count")
    if len(names) < start + count:
        raise ValueError(f"{dataset} has {len(names)} Pittsburgh images, but {start + count} are required")
    random.Random(seed).shuffle(names)
    return tuple(names[start : start + count])


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    """Return the selected images, preprocessed, as one NHWC float32 batch."""
    if not names:
        raise ValueError("at least one image must be selected")
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
