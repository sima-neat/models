"""COCO 2017 image selection, and the clips compile.py calibrates and measures fidelity with.

The registry has no video dataset, so each clip is a slow pan across one COCO image: FRAMES crops of
FRAME_SIZE x FRAME_SIZE, moving evenly along the image's longer side after its shorter side is resized to PAN_SIZE.
"""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from pipeline import FRAME_SIZE, FRAMES, MEAN, STD, stack_clip

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017 and fidelity only from val2017, so no image does both.
CALIBRATION_SPLIT = "cal2017"
VALIDATION_SPLIT = "val2017"
CALIBRATION_SEED = 20261029
# The pan covers 320 - 256 = 64 pixels across the shorter side, more along the longer one.
PAN_SIZE = 320


def decode(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as source:
        return np.asarray(source.convert("RGB"))


def pan_clip(image: np.ndarray) -> np.ndarray:
    """An RGB uint8 image as the encoder's input for a FRAMES-frame pan across it, from its top-left corner to its
    bottom-right one."""
    height, width = image.shape[:2]
    scale = PAN_SIZE / min(height, width)
    size = (max(PAN_SIZE, round(width * scale)), max(PAN_SIZE, round(height * scale)))
    resized = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BILINEAR), np.float32) / 255
    normalized = (resized - MEAN) / STD
    steps = np.linspace(0, 1, FRAMES)
    tops = np.round(steps * (size[1] - FRAME_SIZE)).astype(int)
    lefts = np.round(steps * (size[0] - FRAME_SIZE)).astype(int)
    frames = [normalized[top : top + FRAME_SIZE, left : left + FRAME_SIZE] for top, left in zip(tops, lefts)]
    return stack_clip(np.stack(frames))


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


def calibration_clips(dataset: Path, count: int) -> list[np.ndarray]:
    """Pan clips over a seeded sample of cal2017 images, each as the encoder's NCHW input."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return [pan_clip(decode(images[name])) for name in selected]


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """val2017 images for the fidelity comparison, selected as model.yaml declares."""
    names = split_names(dataset, VALIDATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
