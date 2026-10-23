"""Resize and normalize images the way Patch-NetVLAD extracts features, and select the compile-time images."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# Patch-NetVLAD's input_transform: Resize((480, 640)), ToTensor, then ImageNet normalization (HWC order here).
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# coco-2017/cal holds coco/cal2017/<image> beside annotation PNGs, which are not inputs.
CALIBRATION_SPLIT = "cal2017"


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode one image and return the model input: 480x640 RGB, ImageNet-normalized, HWC float32."""
    with Image.open(io.BytesIO(content)) as source:
        # torchvision's Resize on a PIL image is PIL's antialiased bilinear resize to (width, height).
        image = source.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / np.float32(255) - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    elif tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
    else:
        raise ValueError(f"unsupported image dataset input: {dataset}")
    return sorted(
        name
        for name in names
        if PurePosixPath(name).parent.name == CALIBRATION_SPLIT and name.lower().endswith(IMAGE_SUFFIXES)
    )


def select_images(dataset: Path, seed: int, start: int, count: int) -> tuple[str, ...]:
    """Return images start to start + count of a seeded shuffle of the dataset's cal2017 images."""
    names = _image_names(dataset)
    if start < 0 or count < 1:
        raise ValueError("an image selection needs a non-negative start and a positive count")
    if len(names) < start + count:
        raise ValueError(f"{dataset} has {len(names)} {CALIBRATION_SPLIT} images, but {start + count} are required")
    random.Random(seed).shuffle(names)
    return tuple(names[start : start + count])


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


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    """Return the selected images, preprocessed, as one NHWC float32 batch."""
    if not names:
        raise ValueError("at least one image must be selected")
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, names)])
