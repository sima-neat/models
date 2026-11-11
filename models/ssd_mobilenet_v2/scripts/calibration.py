"""Stretch images to the SSD-MobileNet input as TensorFlow's fixed_shape_resizer does, and select calibration images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 300
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261028


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to RGB, ignoring EXIF orientation as the COCO annotations do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def stretch(image: np.ndarray) -> np.ndarray:
    """Resize an HWC image to IMAGE_SIZE x IMAGE_SIZE with TF1's bilinear resize, as the source graph does.

    TF1 samples source row y * height / IMAGE_SIZE, without OpenCV's half-pixel offset; OpenCV's
    INTER_LINEAR differs from the source graph by up to 0.98 on the [-1, 1] input scale.
    """
    height, width = image.shape[:2]
    rows, columns = np.arange(IMAGE_SIZE) * (height / IMAGE_SIZE), np.arange(IMAGE_SIZE) * (width / IMAGE_SIZE)
    top, left = rows.astype(np.int64), columns.astype(np.int64)
    bottom, right = np.minimum(top + 1, height - 1), np.minimum(left + 1, width - 1)
    down = (rows - top).astype(np.float32)[:, None, None]
    across = (columns - left).astype(np.float32)[None, :, None]
    pixels = image.astype(np.float32)
    upper = pixels[top][:, left] * (1 - across) + pixels[top][:, right] * across
    lower = pixels[bottom][:, left] * (1 - across) + pixels[bottom][:, right] * across
    return upper * (1 - down) + lower * down


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: RGB stretched to 300x300 and scaled to [-1, 1], HWC float32."""
    return stretch(decode_image(content)) * np.float32(2 / 255) - np.float32(1)


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def _members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Admit regular files/directories with unambiguous, dataset-local names."""
    members = archive.getmembers()
    files = set()
    for member in members:
        name = PurePosixPath(member.name)
        if not member.name or name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe dataset member: {member.name}")
        if member.isfile():
            normalized = name.as_posix()
            if normalized == "." or normalized in files:
                raise ValueError(f"duplicate dataset member: {member.name}")
            files.add(normalized)
    return members


def _image_names(dataset: Path) -> list[str]:
    if dataset.is_dir():
        root = dataset.resolve()
        paths = [path for path in dataset.rglob("*") if path.is_file() and _is_image(path.name)]
        if any(not path.resolve().is_relative_to(root) for path in paths):
            raise ValueError("dataset image resolves outside dataset")
        return sorted(path.relative_to(dataset).as_posix() for path in paths)
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            return sorted(member.name for member in _members(archive) if member.isfile() and _is_image(member.name))
    raise ValueError(f"unsupported image dataset input: {dataset}")


def _image_bytes(dataset: Path, names: Sequence[str]) -> Iterator[bytes]:
    if dataset.is_dir():
        root = dataset.resolve()
        for name in names:
            path = (dataset / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("dataset image resolves outside dataset")
            yield path.read_bytes()
        return
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            members = {member.name: member for member in _members(archive)}
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
