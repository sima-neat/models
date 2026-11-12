"""DINOv3 ViT-B/16 preprocessing and the seeded selection of imagenet-2012 cal images, used by compile.py."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# Meta's transform for the LVD-1689M weights, make_transform in the README of facebookresearch/dinov3 at a044468d:
# an antialiased bilinear resize straight to the input size, then ImageNet's mean and std. PIL's bilinear resize
# antialiases when it shrinks. The channel constants are in HWC order.
SIZE = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# imagenet-2012/cal holds cal2012/<image>.
IMAGE_SUFFIXES = (".jpeg",)
CALIBRATION_SPLIT = "cal2012"


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize and normalize one image into a float32 HWC tensor."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) * RESCALE - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    """Image paths relative to the root of the dataset archive."""
    with tarfile.open(dataset, "r:gz") as archive:
        names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if name.lower().endswith(IMAGE_SUFFIXES))


def read_images(dataset: Path, names: Iterable[str]) -> dict[str, bytes]:
    """Read the requested images, scanning the archive once rather than seeking each image."""
    wanted = set(names)
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|gz") as archive:
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


def select_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """samples imagenet-2012 cal images from offset in a seeded order, as model.yaml's numerical_fidelity selects."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if offset + samples > len(names):
        raise ValueError(f"the selection needs {offset + samples} images, but {dataset} has {len(names)} under "
                         f"{CALIBRATION_SPLIT}/")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]


def load_images(dataset: Path, names: list[str]) -> np.ndarray:
    """The named images, preprocessed and stacked in NHWC order."""
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
