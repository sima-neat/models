"""COCO 2017 image selection and the prompts compile.py calibrates and measures fidelity with."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from pipeline import preprocess, prompt_inputs

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# Calibration draws only from cal2017 and fidelity only from val2017, so no image does both.
CALIBRATION_SPLIT = "cal2017"
VALIDATION_SPLIT = "val2017"
CALIBRATION_SEED = 20261029


def decode(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as source:
        return np.asarray(source.convert("RGB"))


def image_prompts(original_size: tuple[int, int]) -> list[dict[str, np.ndarray]]:
    """Five patterns cover token counts and the image-coordinate extrema used by Fourier lookup tables.

    Clicks include pixel-centre corners. Boxes also admit the outer image edges (width, height), one beyond the
    last pixel index; the mixed prompt covers both off-diagonal edge combinations, not just the box diagonal.
    """
    height, width = original_size
    centre = (width / 2, height / 2)
    clicks = [centre, (0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    kinds = ["foreground", "background", "foreground", "background", "foreground"]
    box = [(0, 0), (width, height)]
    box_kinds = ["box_top_left", "box_bottom_right"]
    return [
        *(prompt_inputs(clicks[:count], kinds[:count], original_size) for count in (1, 2, 5)),
        prompt_inputs(box, box_kinds, original_size),
        prompt_inputs([*box, (width, 0), (0, height), centre], [*box_kinds, *kinds[:3]], original_size),
    ]


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


def calibration_images(dataset: Path, count: int) -> list[np.ndarray]:
    """A seeded sample of cal2017 images, decoded to RGB."""
    names = split_names(dataset, CALIBRATION_SPLIT)
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return [decode(images[name]) for name in selected]


def fidelity_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """val2017 images for the fidelity comparison, selected as model.yaml declares."""
    names = split_names(dataset, VALIDATION_SPLIT)
    if offset + samples > len(names):
        raise ValueError(f"fidelity selection needs {offset + samples} images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]


def encoder_inputs(images: list[np.ndarray]) -> np.ndarray:
    """The encoder's NHWC float32 inputs for the images."""
    return np.stack([preprocess(image) for image in images])
