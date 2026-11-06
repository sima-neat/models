"""ResNet-50 preprocessing and ImageNet mini sample selection, shared by compile.py and accuracy.py."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# ImageNet input preprocessing, with channel constants in HWC order.
SHORTEST_EDGE = 256
CROP = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# imagenet-2012/mini holds cal/<image> and val/<synset>/<image>; one listed image ends in .webp.bak.
IMAGE_SUFFIXES = (".webp", ".webp.bak")
CALIBRATION_SPLIT = "cal"
VALIDATION_SPLIT = "val"
CLASSES = 1000


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize, crop and normalize one image into a float32 HWC tensor."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (SHORTEST_EDGE, int(SHORTEST_EDGE * height / width))
    else:
        size = (int(SHORTEST_EDGE * width / height), SHORTEST_EDGE)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left, top = (image.width - CROP) // 2, (image.height - CROP) // 2
    image = image.crop((left, top, left + CROP, top + CROP))
    return (np.asarray(image, dtype=np.float32) * RESCALE - MEAN) / STD


def _image_names(dataset: Path) -> list[str]:
    """Image paths relative to the dataset root, from an archive or an extracted directory."""
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    else:
        with tarfile.open(dataset, "r:gz") as archive:
            names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if name.lower().endswith(IMAGE_SUFFIXES))


def split_names(dataset: Path, split: str) -> list[str]:
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parts[0] == split]
    if not names:
        raise ValueError(f"{dataset} has no images under {split}/")
    return names


def validation_labels(names: Iterable[str]) -> dict[str, int]:
    """Label each validation image by its synset directory; ImageNet class indices follow sorted synsets."""
    synsets = {name: PurePosixPath(name).parts[-2] for name in names}
    indices = {synset: index for index, synset in enumerate(sorted(set(synsets.values())))}
    if len(indices) != CLASSES:
        raise ValueError(f"expected {CLASSES} validation classes, found {len(indices)}")
    return {name: indices[synset] for name, synset in synsets.items()}


def read_images(dataset: Path, names: Iterable[str]) -> dict[str, bytes]:
    """Read the requested images; an archive is scanned once rather than seeked per image."""
    wanted = set(names)
    if dataset.is_dir():
        return {name: (dataset / name).read_bytes() for name in wanted}
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


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """Every calibration image, preprocessed and stacked in NHWC order."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if len(names) != count:
        raise ValueError(f"calibration requires {count} images, but {dataset} has {len(names)}")
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Validation images for the fidelity comparison, selected as declared in model.yaml."""
    names = split_names(dataset, VALIDATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
