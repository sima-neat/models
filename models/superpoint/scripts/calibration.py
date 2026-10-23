"""Select SuperPoint's INT8 calibration images from the coco-2017 cal variant and prepare them as model inputs."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
from keypoints import decode_image, preprocess

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261023


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
    """Return the selected calibration images as one NHWC float32 batch of grayscale model inputs."""
    images = [preprocess(decode_image(content)) for content in _image_bytes(dataset, calibration_names(dataset, count))]
    return np.stack(images)[..., None]
