"""Prepare YuNet's centered, zero-padded BGR pixel inputs, and select the INT8 calibration and fidelity images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

import numpy as np

IMAGE_SIZE = 640
# The MPK's 12 head maps, in output order: each head at strides 8, 16 and 32 with its channels (face and object
# scores, box offsets, five landmarks as (x, y) offsets). surgery.py cuts the graph to these outputs and accuracy.py
# reads them back by name, so both take them from here.
STRIDES = (8, 16, 32)
HEADS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
OUTPUTS = [f"{head}_{stride}" for head in HEADS for stride in STRIDES]
# YuNet takes raw BGR pixel values; OpenCV's FaceDetectorYN pads its input with zeros.
FILL = 0
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# The wider-face cal archive holds training images only, so no scored validation image calibrates the model.
CALIBRATION_SPLIT = "WIDER_train"
CALIBRATION_SEED = 42


def letterbox_geometry(width: int, height: int) -> tuple[float, tuple[int, int], tuple[int, int, int, int]]:
    """Return the centered letterbox gain, resized (width, height) and (left, top, right, bottom) padding."""
    gain = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
    resized = (round(width * gain), round(height * gain))
    pad_x, pad_y = (IMAGE_SIZE - resized[0]) / 2, (IMAGE_SIZE - resized[1]) / 2
    # The 0.1 offsets put an odd padding pixel on the right or bottom, with a deterministic centered origin.
    return gain, resized, (round(pad_x - 0.1), round(pad_y - 0.1), round(pad_x + 0.1), round(pad_y + 0.1))


def decode_image(content: bytes) -> np.ndarray:
    """Decode an encoded image to BGR, ignoring EXIF orientation as the WIDER FACE boxes do."""
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    return image


def letterbox(image: np.ndarray) -> np.ndarray:
    """Scale a BGR image to fit IMAGE_SIZE, up or down, and zero-pad it to a centered square."""
    import cv2

    height, width = image.shape[:2]
    _, resized, (left, top, right, bottom) = letterbox_geometry(width, height)
    if resized != (width, height):
        image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(FILL, FILL, FILL))


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the model input for one encoded image: letterboxed BGR values in [0, 255], HWC float32."""
    return letterbox(decode_image(content)).astype(np.float32)


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


def read_images(dataset: Path, names: Sequence[str]) -> dict[str, bytes]:
    """The requested images' bytes by name; an archive is read in one pass."""
    wanted = set(names)
    if dataset.is_dir():
        return {name: (dataset / name).read_bytes() for name in wanted}
    if not tarfile.is_tarfile(dataset):
        raise ValueError(f"unsupported image dataset input: {dataset}")
    found: dict[str, bytes] = {}
    with tarfile.open(dataset) as archive:
        for member in archive:
            if member.name in wanted and member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    found[member.name] = stream.read()
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested images, such as {sorted(missing)[0]}")
    return found


def shuffled_training_images(dataset: Path, seed: int) -> list[str]:
    """The archive's training images in a seeded random order."""
    names = [name for name in _image_names(dataset) if CALIBRATION_SPLIT in PurePosixPath(name).parts]
    if not names:
        raise ValueError(f"{dataset} has no {CALIBRATION_SPLIT} images")
    random.Random(seed).shuffle(names)
    return names


def calibration_names(dataset: Path, count: int) -> list[str]:
    """The first count images of the seeded order; the fidelity check reads images after them."""
    names = shuffled_training_images(dataset, CALIBRATION_SEED)
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return names[:count]


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Training images for the fidelity comparison, selected as model.yaml declares."""
    names = shuffled_training_images(dataset, seed)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    return names[offset : offset + samples]


def _in_order(dataset: Path, names: Sequence[str]) -> Iterator[bytes]:
    images = read_images(dataset, names)
    for name in names:
        yield images[name]


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return the selected calibration images as one NHWC float32 batch."""
    return np.stack([preprocess_image(content) for content in _in_order(dataset, calibration_names(dataset, count))])
