"""SigLIP 2 ViT-B/16 image preprocessing and ImageNet sample selection, shared by compile.py and accuracy.py."""

from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# The checkpoint's declared preprocessing (its preprocessor_config.json): a bilinear resize straight to 224x224,
# then mean and std 0.5, with channel constants in HWC order.
SIZE = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
# imagenet-2012/cal holds cal2012/<image> and imagenet-2012/val holds val2012/<image>, each with
# annotations/labels_<split>.json, whose category IDs are the ImageNet class indices.
IMAGE_SUFFIXES = (".jpeg",)
CALIBRATION_SPLIT = "cal2012"
VALIDATION_SPLIT = "val2012"
CLASSES = 1000


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize and normalize one image into a float32 HWC tensor."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
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
    """The split's images; an archive holds them under its top-level directory."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == split]
    if not names:
        raise ValueError(f"{dataset} has no images under {split}/")
    return names


def validation_labels(dataset: Path) -> dict[str, int]:
    """Each validation image's class index, from the extracted dataset's labels file."""
    labels = json.loads((dataset / "annotations" / f"labels_{VALIDATION_SPLIT}.json").read_text(encoding="utf-8"))
    if len(labels["categories"]) != CLASSES:
        raise ValueError(f"expected {CLASSES} validation classes, found {len(labels['categories'])}")
    names = {image["id"]: f"{VALIDATION_SPLIT}/{image['file_name']}" for image in labels["images"]}
    return {names[annotation["image_id"]]: annotation["category_id"] for annotation in labels["annotations"]}


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


def select_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Calibration images selected as model.yaml declares: a seeded shuffle, then a slice."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"the selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]


def load_images(dataset: Path, names: list[str]) -> np.ndarray:
    """The named images, preprocessed and stacked in NHWC order."""
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
