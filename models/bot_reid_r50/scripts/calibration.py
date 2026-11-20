"""BoT ReID R50 preprocessing and Market-1501 image lists, shared by compile.py and accuracy.py."""

from __future__ import annotations

import io
import random
import re
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

# fast-reid's test transform: RGB, resized to 256x128 with bicubic interpolation, pixel values kept in [0, 255].
# The network subtracts the ImageNet mean and divides by the standard deviation itself, so Neat's preprocessing
# passes pixels through unchanged: x / 255 normalized with mean 0 and standard deviation 1/255.
HEIGHT, WIDTH = 256, 128
MEAN = np.zeros(3, dtype=np.float32)
STD = np.full(3, 1 / 255, dtype=np.float32)
# Market-1501 folders: training identities for calibration, then the query and gallery images scored.
TRAIN, QUERY, GALLERY = "bounding_box_train", "query", "bounding_box_test"
CALIBRATION_SEED = 42
# <person id>_c<camera>..., where person -1 marks a junk detection that the evaluation leaves out.
NAME = re.compile(r"^(-?\d+)_c(\d)")


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode one image and resize it into a float32 HWC tensor of RGB pixel values."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32)


def identity(name: str) -> tuple[int, int] | None:
    """An image's (person, camera) from its file name, cameras counted from 0; None for a junk image."""
    match = NAME.match(PurePosixPath(name).name)
    if match is None:
        raise ValueError(f"{name} does not follow Market-1501's <person>_c<camera> naming")
    person, camera = int(match[1]), int(match[2]) - 1
    return None if person == -1 else (person, camera)


def _image_names(dataset: Path) -> list[str]:
    """Image paths relative to the dataset root, from an archive or an extracted directory."""
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*.jpg") if path.is_file()]
    else:
        with tarfile.open(dataset, "r:gz") as archive:
            names = [member.name.removeprefix("./") for member in archive.getmembers() if member.isfile()]
    return sorted(name for name in names if name.endswith(".jpg"))


def split_names(dataset: Path, split: str) -> list[str]:
    """The split's scored images, in name order, without junk images."""
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == split]
    names = [name for name in names if identity(name) is not None]
    if not names:
        raise ValueError(f"{dataset} has no images under {split}/")
    return names


def read_images(dataset: Path, names: list[str]) -> dict[str, bytes]:
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
    """A seeded sample of the training images, preprocessed and stacked in NHWC order."""
    names = split_names(dataset, TRAIN)
    if count > len(names):
        raise ValueError(f"calibration requires {count} images, but {dataset} has {len(names)}")
    selected = random.Random(CALIBRATION_SEED).sample(names, count)
    images = read_images(dataset, selected)
    return np.stack([preprocess_image(images[name]) for name in selected])
