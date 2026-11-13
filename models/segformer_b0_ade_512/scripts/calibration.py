"""SegFormer preprocessing and COCO image selection, shared by compile.py and its fidelity check."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np

# Match Neat's centered letterbox and ImageNet normalization, in HWC order.
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
PAD_VALUE = 114
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017 and fidelity only from val2017, so no image does both.
CALIBRATION_SPLIT = "cal2017"
VALIDATION_SPLIT = "val2017"
CALIBRATION_SEED = 20261029


def letterbox_geometry(width: int, height: int) -> tuple[int, int, int, int]:
    """Return left/top padding and resized width/height, rounded up as Neat does."""
    if height * IMAGE_WIDTH <= width * IMAGE_HEIGHT:
        scaled_width = IMAGE_WIDTH
        scaled_height = (height * IMAGE_WIDTH + width - 1) // width
    else:
        scaled_height = IMAGE_HEIGHT
        scaled_width = (width * IMAGE_HEIGHT + height - 1) // height
    return (IMAGE_WIDTH - scaled_width) // 2, (IMAGE_HEIGHT - scaled_height) // 2, scaled_width, scaled_height


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, letterbox and normalize one image into the model's float32 HWC input."""
    from PIL import Image

    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB")
    left, top, width, height = letterbox_geometry(*image.size)
    canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (PAD_VALUE,) * 3)
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    return (np.asarray(canvas, dtype=np.float32) * RESCALE - MEAN) / STD


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


def calibration_samples(dataset: Path, count: int) -> np.ndarray:
    """A seeded sample of cal2017 images, preprocessed and stacked in NHWC order."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return np.stack([preprocess_image(images[name]) for name in selected])


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """val2017 images for the fidelity comparison, selected as model.yaml declares."""
    names = split_names(dataset, VALIDATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]
