"""Letterbox images as MMDetection's RTMDet test pipeline does, and select the INT8 calibration images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 640
FILL = 114
# ImageNet's mean and std, in RGB order: MMDetection's BGR constants for a graph that surgery.py makes take RGB.
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261113


def letterbox_geometry(width: int, height: int) -> tuple[int, int, float, float]:
    """Return the resized width and height and the per-axis scale factors MMDetection's keep-ratio Resize uses.

    mmcv rescales to fit IMAGE_SIZE and rounds each side half up; MMDetection keeps the two resulting factors.
    """
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width, resized_height = int(width * scale + 0.5), int(height * scale + 0.5)
    return resized_width, resized_height, resized_width / width, resized_height / height


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(image: np.ndarray) -> np.ndarray:
    """Resize an RGB image to fit IMAGE_SIZE, keeping its aspect ratio, and pad it on the right and bottom."""
    import cv2

    height, width = image.shape[:2]
    resized_width, resized_height, _, _ = letterbox_geometry(width, height)
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        image, 0, IMAGE_SIZE - resized_height, 0, IMAGE_SIZE - resized_width, cv2.BORDER_CONSTANT,
        value=(FILL, FILL, FILL),
    )


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: letterboxed, normalized RGB, HWC float32."""
    return (letterbox(decode_image(content)).astype(np.float32) - MEAN) / STD


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
    """Return a seeded sample of the dataset's cal2017 images, in sampling order."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return the selected calibration images as one NHWC float32 batch."""
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, calibration_names(dataset, count))])
