"""Letterbox images the way Ultralytics YOLO11 inference does, and select the INT8 calibration images."""

from __future__ import annotations

import random
from pathlib import Path, PurePosixPath

import numpy as np
from model_inputs.images import image_names, read_images

IMAGE_SIZE = 640
FILL = 114
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20260930


def letterbox_geometry(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Return Ultralytics' LetterBox gain, resized (width, height) and (left, top, right, bottom) padding."""
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized = (round(width * gain), round(height * gain))
    pad_x, pad_y = (IMAGE_SIZE - resized[0]) / 2, (IMAGE_SIZE - resized[1]) / 2
    # The 0.1 offsets put an odd padding pixel on the right or bottom, as Ultralytics does.
    return gain, resized, (round(pad_x - 0.1), round(pad_y - 0.1), round(pad_x + 0.1), round(pad_y + 0.1))


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(image: np.ndarray) -> np.ndarray:
    """Scale an RGB image to fit IMAGE_SIZE, up or down, and pad it to a centered square."""
    import cv2

    height, width = image.shape[:2]
    _, resized, (left, top, right, bottom) = letterbox_geometry(width, height)
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(FILL, FILL, FILL))


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: letterboxed RGB scaled to [0, 1], HWC float32."""
    return letterbox(decode_image(content)).astype(np.float32) / np.float32(255)


def calibration_names(dataset: Path, count: int) -> list[str]:
    """Return a seeded sample of the dataset's cal2017 images, in sampling order."""
    names = image_names(dataset, IMAGE_SUFFIXES)
    names = [name for name in names if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return the selected calibration images as one NHWC float32 batch."""
    names = calibration_names(dataset, count)
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
