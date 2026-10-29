"""Depth Anything V2 preprocessing and NYU Depth V2 sample selection, shared by compile.py and accuracy.py."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np

# The authors' inference transform for a 640x480 frame: keep the aspect ratio with 518 on the short side, round
# both sides to multiples of 14 (518x686), resize bicubically, and normalize with ImageNet statistics.
HEIGHT, WIDTH = 518, 686
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# nyu-depth-v2 cal holds cal/rgb_NNNNN.jpg; val holds test/rgb_NNNNN.jpg, NYU's official test split.
CALIBRATION_SPLIT = "cal"
VALIDATION_SPLIT = "test"
IMAGE_PREFIX = "rgb_"


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize and normalize one image into a float32 HWC tensor of the model's input size."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    return (cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    """Image paths below the archive's top directory, from an archive or an extracted directory."""
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob(f"{IMAGE_PREFIX}*.jpg")]
    else:
        with tarfile.open(dataset, "r:gz") as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if PurePosixPath(name).name.startswith(IMAGE_PREFIX) and name.endswith(".jpg"))


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
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            if member.name in wanted and member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    found[member.name] = stream.read()
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested images, such as {sorted(missing)[0]}")
    return found


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """The first count calibration images, in the archive's sorted order, preprocessed and stacked NHWC."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count < 1 or count > len(names):
        raise ValueError(f"cannot take {count} calibration images from {len(names)}")
    names = names[:count]
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Validation images for the fidelity comparison, selected as declared in model.yaml."""
    names = split_names(dataset, VALIDATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
