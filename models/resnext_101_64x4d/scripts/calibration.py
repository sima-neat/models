"""ResNeXt-101 64x4d preprocessing and ImageNet sample selection, shared by compile.py and accuracy.py."""

from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# The checkpoint's declared preprocessing (pretrained_cfg in its config.json): a bilinear resize of the shorter
# side to 256 (224 / crop_pct 0.875, rounded down), a centre crop to 224x224, then the ImageNet mean
# and std, with channel constants in HWC order.
SHORTEST_EDGE = 256
CROP = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# imagenet-2012/cal holds cal2012/<image> and imagenet-2012/val holds val2012/<image>, each with
# annotations/labels_<split>.json, whose category IDs are the ImageNet class indices.
IMAGE_SUFFIXES = (".jpeg",)
CALIBRATION_SPLIT = "cal2012"
CALIBRATION_SEED = 42
VALIDATION_SPLIT = "val2012"
CLASSES = 1000


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize, crop and normalize one image into a float32 HWC tensor, as torchvision's Resize and
    CenterCrop do: the longer side truncates to an integer and the crop offset rounds to the nearest pixel."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (SHORTEST_EDGE, int(SHORTEST_EDGE * height / width))
    else:
        size = (int(SHORTEST_EDGE * width / height), SHORTEST_EDGE)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = int(round((image.width - CROP) / 2.0)), int(round((image.height - CROP) / 2.0))
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


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """A seeded sample of the calibration images, preprocessed and stacked in NHWC order."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count > len(names):
        raise ValueError(f"calibration requires {count} images, but {dataset} has {len(names)}")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return np.stack([preprocess_image(images[name]) for name in selected])


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """Calibration images for the fidelity comparison, selected as declared in model.yaml."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
