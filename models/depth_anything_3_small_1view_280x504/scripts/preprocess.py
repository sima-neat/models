"""DA3 RGB cover-resize, center crop and ImageNet normalization."""

from __future__ import annotations

import numpy as np

HEIGHT, WIDTH = 280, 504
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def cover_geometry(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int]]:
    """Return the scale, resized (width, height) and (left, top) crop offset that cover HEIGHT x WIDTH."""
    scale = max(HEIGHT / height, WIDTH / width)
    resized = (max(WIDTH, round(width * scale)), max(HEIGHT, round(height * scale)))
    return scale, resized, ((resized[0] - WIDTH) // 2, (resized[1] - HEIGHT) // 2)


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the depth ground truth does."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def cover_crop(image: np.ndarray) -> np.ndarray:
    """Resize an RGB image to cover HEIGHT x WIDTH and crop its center, as upstream's InputProcessor does.

    Upstream resizes with cubic interpolation when it enlarges and area interpolation when it shrinks.
    """
    import cv2

    height, width = image.shape[:2]
    scale, resized, (left, top) = cover_geometry(width, height)
    if resized != (width, height):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, resized, interpolation=interpolation)
    return image[top : top + HEIGHT, left : left + WIDTH]


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: ImageNet-normalized RGB, HWC float32."""
    image = cover_crop(decode_image(content)).astype(np.float32) / np.float32(255)
    return np.ascontiguousarray((image - MEAN) / STD, dtype=np.float32)
