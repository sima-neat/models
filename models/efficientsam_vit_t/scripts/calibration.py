"""EfficientSAM ViT-T inputs for compile.py: coco-2017 cal images and box prompts from their annotations.

The encoder takes the image stretched to 1024x1024 with values in [0, 1]; it subtracts ImageNet's mean and divides by
its std itself. Each image's prompt is the box of its largest annotated object, as an application would draw it.
"""

from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

SIZE = 1024
MAX_POINTS = 6
# Point types in the one-hot order of the prompt decoder's points rows.
POINT_TYPES = ("padding", "negative", "positive", "box_top_left", "box_bottom_right")
# A points row: x, y, the one-hot point type and a 0. The MLA reads each input in whole 16-byte words, and Neat refuses
# an input whose tensor does not fill them; 6 rows of 8 values do at INT8 and BF16 alike.
POINT_COLUMNS = 8
IMAGE_SPLIT = "cal2017"
ANNOTATIONS = "annotations/instances_cal2017.json"


def encoder_input(content: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """The image stretched to 1024x1024 as float32 CHW in [0, 1], and its original (height, width)."""
    with Image.open(io.BytesIO(content)) as source:
        rgb = source.convert("RGB")
        size = (rgb.height, rgb.width)
        stretched = rgb.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return np.asarray(stretched, np.float32).transpose(2, 0, 1) / 255.0, size


def prompt_inputs(points: list[tuple[float, float]], types: list[str], size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The prompt decoder's points [1,6,8] for points in the image's pixels: per row x, y, the one-hot type and a 0.

    A point (x, y) of a height x width image becomes ((x * 1024 / width + 0.5) / 1024,
    (y * 1024 / height + 0.5) / 1024);
    unused rows are padding at coordinate -1, as EfficientSAM pads its prompts.
    """
    if not 0 < len(points) == len(types) <= MAX_POINTS:
        raise ValueError(f"a prompt has 1 to {MAX_POINTS} points, each with a type")
    height, width = size
    rows = np.zeros((1, MAX_POINTS, POINT_COLUMNS), np.float32)
    rows[0, :, :2] = -0.5 / SIZE
    rows[0, :, 2] = 1.0
    for row, ((x, y), kind) in enumerate(zip(points, types)):
        rows[0, row, :2] = ((x * SIZE / width + 0.5) / SIZE, (y * SIZE / height + 0.5) / SIZE)
        rows[0, row, 2] = 0.0
        rows[0, row, 2 + POINT_TYPES.index(kind)] = 1.0
    return {"points": rows}


def box_prompt(box: list[float]) -> tuple[list[tuple[float, float]], list[str]]:
    """A COCO box [x, y, width, height] as EfficientSAM's two corner points."""
    x, y, width, height = box
    return [(x, y), (x + width, y + height)], ["box_top_left", "box_bottom_right"]


def _member_name(name: str) -> str:
    """A path relative to the dataset root; an archive holds the root under one top-level directory."""
    return str(PurePosixPath(*PurePosixPath(name.removeprefix("./")).parts[1:]))


def read_files(dataset: Path, wanted: set[str]) -> dict[str, bytes]:
    """Read files by their path relative to the dataset root, from an archive (scanned once) or a directory."""
    if dataset.is_dir():
        return {name: (dataset / name).read_bytes() for name in wanted}
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            name = _member_name(member.name)
            if name in wanted and member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    found[name] = stream.read()
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{dataset} is missing {len(missing)} requested files, such as {sorted(missing)[0]}")
    return found


def largest_boxes(dataset: Path) -> dict[str, list[float]]:
    """Each cal2017 image with an annotated object, mapped to the box of its largest non-crowd object."""
    annotations = json.loads(read_files(dataset, {ANNOTATIONS})[ANNOTATIONS])
    names = {image["id"]: f"{IMAGE_SPLIT}/{image['file_name']}" for image in annotations["images"]}
    largest: dict[str, tuple[float, list[float]]] = {}
    for item in annotations["annotations"]:
        if item.get("iscrowd"):
            continue
        name = names[item["image_id"]]
        if name not in largest or item["area"] > largest[name][0]:
            largest[name] = (item["area"], item["bbox"])
    return {name: box for name, (_, box) in sorted(largest.items())}


def select_samples(dataset: Path, seed: int, offset: int, samples: int) -> list[tuple[str, list[float]]]:
    """Annotated images and their largest boxes in a seeded order; calibration takes the first, fidelity the next."""
    boxes = largest_boxes(dataset)
    names = sorted(boxes)
    if offset + samples > len(names):
        raise ValueError(f"the selection needs {offset + samples} annotated images, but {dataset} has {len(names)}")
    random.Random(seed).shuffle(names)
    return [(name, boxes[name]) for name in names[offset : offset + samples]]


def load_samples(
    dataset: Path, selected: list[tuple[str, list[float]]]
) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Each selected image's encoder input and the prompt decoder's inputs for its box."""
    images = read_files(dataset, {name for name, _ in selected})
    samples = []
    for name, box in selected:
        image, size = encoder_input(images[name])
        points, types = box_prompt(box)
        samples.append((image, prompt_inputs(points, types, size)))
    return samples
