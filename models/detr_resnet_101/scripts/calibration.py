"""Stretch and normalize images the way Neat does for DETR, and select the INT8 calibration images."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# A square 640x640 input runs about three times as fast as a stretched 1280x720 input and scores about as well
# in FP32 on COCO val2017. Images of any shape are stretched to it.
INPUT_HEIGHT, INPUT_WIDTH = 640, 640
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20260930


def stretch(image: Image.Image) -> Image.Image:
    """Resize an RGB image to the input size, as Neat's Stretch resize does.

    DETR trained with its padding masked out, and the export treats every input pixel as image, so padding
    would read as scene content and cost accuracy.
    """
    return image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: RGB stretched, ImageNet-normalized, HWC float32."""
    with Image.open(io.BytesIO(content)) as image:
        array = np.asarray(stretch(image.convert("RGB")), dtype=np.float32) / np.float32(255)
    return (array - MEAN) / STD


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
