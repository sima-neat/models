"""TorchVision SSDLite320 preprocessing and deterministic COCO calibration."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 320
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20261028


def decode_image(content: bytes) -> np.ndarray:
    """Decode RGB with PIL, without applying EXIF orientation, as the source audit does."""
    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        return np.array(image.convert("RGB"), copy=True)


def preprocess_image(content: bytes) -> np.ndarray:
    """Return HWC float32: normalize first, then TorchVision's half-pixel resize.

    Torch is imported lazily, so archive selection and reference decoding need only NumPy.
    A different resizer must be validated independently before replacing this source path.
    """
    import torch
    from torch.nn.functional import interpolate

    pixels = decode_image(content)
    image = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    image = (image - 0.5) / 0.5
    resized = interpolate(
        image[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=False
    )
    return resized[0].permute(1, 2, 0).numpy()


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
    """Select only cal2017, from sorted names, in deterministic sampling order."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return one NHWC float32 batch from the full Registry archive or an equivalent directory."""
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, calibration_names(dataset, count))])
