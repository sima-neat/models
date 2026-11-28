"""MNIST reading and preprocessing, shared by compile.py and accuracy.py.

The Dataset Registry's mnist-default archive holds the four MNIST IDX files, flat: train-images-idx3-ubyte and
train-labels-idx1-ubyte (60,000 digits), t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte (the 10,000 test digits).
Each image is 28x28 uint8, white digit on black. The model takes it scaled to [0, 1], as a [28, 28, 1] tensor.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

SIZE = 28
CLASSES = 10
SPLITS = {"train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 60000),
          "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 10000)}
IMAGE_MAGIC, LABEL_MAGIC = 0x00000803, 0x00000801


def _read(dataset: Path, name: str) -> bytes:
    """One IDX file from an extracted directory or the archive itself."""
    if dataset.is_dir():
        return (dataset / name).read_bytes()
    with tarfile.open(dataset, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.removeprefix("./") == name:
                stream = archive.extractfile(member)
                if stream is not None:
                    return stream.read()
    raise ValueError(f"{dataset} has no {name}")


def _header(content: bytes, words: int) -> np.ndarray:
    return np.frombuffer(content[: 4 * words], dtype=">u4")


def read_split(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """A split's images [N, 28, 28] uint8 and labels [N] uint8, with their IDX headers checked."""
    images_name, labels_name, count = SPLITS[split]
    images, labels = _read(dataset, images_name), _read(dataset, labels_name)
    magic, number, rows, columns = _header(images, 4)
    if (magic, number, rows, columns) != (IMAGE_MAGIC, count, SIZE, SIZE) or len(images) != 16 + count * SIZE * SIZE:
        raise ValueError(f"{images_name} is not {count} {SIZE}x{SIZE} MNIST images")
    magic, number = _header(labels, 2)
    if (magic, number) != (LABEL_MAGIC, count) or len(labels) != 8 + count:
        raise ValueError(f"{labels_name} is not {count} MNIST labels")
    label_array = np.frombuffer(labels, dtype=np.uint8, offset=8)
    if label_array.max() >= CLASSES:
        raise ValueError(f"{labels_name} holds a label outside 0-9")
    return np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, SIZE, SIZE), label_array


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 [N, 28, 28] images as the model's float32 [N, 28, 28, 1] input in [0, 1]."""
    return (images.astype(np.float32) / np.float32(255))[..., None]


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """The first count training images, preprocessed: MNIST's training digits are already in random class order."""
    images, _ = read_split(dataset, "train")
    if count > len(images):
        raise ValueError(f"calibration requires {count} images, but the training split has {len(images)}")
    return preprocess(images[:count])
