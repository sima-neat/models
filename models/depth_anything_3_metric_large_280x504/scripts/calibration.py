"""Resize and crop images the way Depth Anything 3 inference does, and select the calibration images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

HEIGHT, WIDTH = 280, 504
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGE_SUFFIXES = (".jpg",)
# In nyu-depth-v2/default, val, cal and test together are the 654-frame Eigen test set that accuracy.py
# scores. Calibration draws only from train, so no scored frame calibrates the model.
CALIBRATION_SPLIT = "train"
CALIBRATION_PREFIX = "rgb_"
CALIBRATION_SEED = 20261011


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
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            members = {member.name: member for member in archive.getmembers()}
            for name in names:
                stream = archive.extractfile(members[name])
                if stream is None:
                    raise ValueError(f"could not read image from dataset: {name}")
                yield stream.read()
        return
    raise ValueError(f"unsupported image dataset input: {dataset}")


def calibration_names(dataset: Path, count: int) -> list[str]:
    """Return a seeded sample of the dataset's train RGB frames, in sampling order."""
    paths = [PurePosixPath(name) for name in _image_names(dataset)]
    names = [
        path.as_posix()
        for path in paths
        if path.parent.name == CALIBRATION_SPLIT and path.name.startswith(CALIBRATION_PREFIX)
    ]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return the selected calibration images as one NHWC float32 batch."""
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, calibration_names(dataset, count))])
