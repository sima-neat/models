"""FCN-HRNet preprocessing and ADE20K image selection, shared by compile.py and its fidelity check."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

# The checkpoint's own pipeline, at the package's fixed input: OpenCV's bilinear resize (here a stretch to
# 512x512), then RGB order and MMSegmentation's mean and std in 0-255 units (the ImageNet statistics times 255), in HWC
# order.
IMAGE_SIZE = 512
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# The ade20k cal archive holds 500 ADE20K training images, so no calibration image is a scored validation image.
CALIBRATION_SPLIT = "training"
CALIBRATION_SEED = 42


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, stretch and normalize one image into the model's float32 HWC input."""
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode the image")
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    """Image paths relative to the dataset root, from an archive or an extracted directory."""
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    else:
        with tarfile.open(dataset, "r:*") as archive:
            names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if name.lower().endswith(IMAGE_SUFFIXES))


def split_names(dataset: Path, split: str) -> list[str]:
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == split]
    if not names:
        raise ValueError(f"{dataset} has no images under {split}/")
    return names


def read_images(dataset: Path, names: Iterable[str]) -> dict[str, bytes]:
    """Read the requested images; an archive is scanned once rather than seeked per image."""
    wanted = set(names)
    if dataset.is_dir():
        return {name: (dataset / name).read_bytes() for name in wanted}
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|*") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name in wanted and member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    found[name] = stream.read()
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested images, such as {sorted(missing)[0]}")
    return found


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """A seeded sample of the archive's training images, preprocessed and stacked in NHWC order."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return np.stack([preprocess_image(images[name]) for name in selected])


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Training images for the fidelity comparison, selected as model.yaml declares."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
