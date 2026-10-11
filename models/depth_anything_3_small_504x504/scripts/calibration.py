"""Resize and crop images the way Depth Anything 3 inference does, and select the calibration view pairs."""

from __future__ import annotations

import random
import re
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

HEIGHT, WIDTH = 504, 504
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# kitti/default holds the RGB frames as test/images/<drive>_<10-digit frame>.png. Its cal folder holds
# grayscale frames of another size without a drive, so it cannot give RGB view pairs.
FRAME_NAME = re.compile(r"(.+_drive_\d{4}_sync)_(\d{10})\.png")
# Only these drives have frames at most MAX_PAIR_GAP indices apart, the short baseline the pairs need.
# Calibration draws from them; accuracy.py scores only the other drives.
CALIBRATION_DRIVES = (
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0013_sync",
    "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0046_sync",
    "2011_09_26_drive_0048_sync",
    "2011_09_26_drive_0052_sync",
    "2011_09_28_drive_0002_sync",
)
MAX_PAIR_GAP = 5
CALIBRATION_SEED = 0


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


def frame_key(name: str) -> tuple[str, int]:
    """Return the (drive, frame index) of a KITTI RGB frame path."""
    match = FRAME_NAME.fullmatch(PurePosixPath(name).name)
    if match is None:
        raise ValueError(f"unexpected KITTI frame name: {name}")
    return match[1], int(match[2])


def _is_frame(name: str) -> bool:
    path = PurePosixPath(name)
    return path.parent.name == "images" and path.parent.parent.name == "test" and path.suffix == ".png"


def _frame_names(dataset: Path) -> list[str]:
    if dataset.is_dir():
        return [path.relative_to(dataset).as_posix() for path in dataset.rglob("*.png") if _is_frame(str(path))]
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            return [member.name for member in archive.getmembers() if member.isfile() and _is_frame(member.name)]
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


def calibration_pairs(dataset: Path, count: int) -> list[tuple[str, str]]:
    """Return a seeded sample of (view 0, view 1) frame pairs, in sampling order.

    Within a drive, consecutive frames at most MAX_PAIR_GAP indices apart form a pair, the earlier frame
    being view 0. Pairs are listed by drive and frame before sampling.
    """
    frames = sorted((frame_key(name), name) for name in _frame_names(dataset))
    pairs = [
        (first, second)
        for ((drive, index), first), ((next_drive, next_index), second) in zip(frames, frames[1:])
        if drive == next_drive and next_index - index <= MAX_PAIR_GAP
    ]
    drives = tuple(sorted({frame_key(first)[0] for first, _ in pairs}))
    if drives != CALIBRATION_DRIVES:
        raise ValueError(f"expected view pairs only in {CALIBRATION_DRIVES}, found them in {drives}")
    if count < 1 or len(pairs) < count:
        raise ValueError(f"cannot select {count} calibration pairs from {len(pairs)} pairs")
    return random.Random(CALIBRATION_SEED).sample(pairs, count)


def calibration_images(dataset: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected view 0 and view 1 images as two NHWC float32 batches."""
    pairs = calibration_pairs(dataset, count)
    names = [name for pair in pairs for name in pair]
    images = [preprocess_image(content) for content in _image_bytes(dataset, names)]
    return np.stack(images[0::2]), np.stack(images[1::2])
