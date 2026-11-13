"""Prepare EfficientSAM3 inputs: COCO category prompts, packed text and stretched images."""

from __future__ import annotations

import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

IMAGE_SIZE = 1008
# Pixels in [0, 1] are normalized to [-1, 1], as SAM 3 does.
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)
CONTEXT = 16
VOCABULARY = 49408
FEATURES = 256
START, END = 49406, 49407
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
# The 80 COCO 2017 category names in category-id order, with their BPE token IDs between the start
# and end tokens. convert.py checks this table against the pinned CLIP tokenizer.
COCO_TOKENS = {
    "person": (2533,),
    "bicycle": (11652,),
    "car": (1615,),
    "motorcycle": (10297,),
    "airplane": (16451,),
    "bus": (2840,),
    "train": (3231,),
    "truck": (4629,),
    "boat": (4440,),
    "traffic light": (3399, 1395),
    "fire hydrant": (1769, 8031, 773),
    "stop sign": (1691, 2292),
    "parking meter": (5984, 10104),
    "bench": (8771,),
    "bird": (3329,),
    "cat": (2368,),
    "dog": (1929,),
    "horse": (4558,),
    "sheep": (9629,),
    "cow": (9706,),
    "elephant": (10299,),
    "bear": (4298,),
    "zebra": (22548,),
    "giraffe": (22826,),
    "backpack": (14894,),
    "umbrella": (17143,),
    "handbag": (22654,),
    "tie": (3422,),
    "suitcase": (27792,),
    "frisbee": (45768,),
    "skis": (32843,),
    "snowboard": (33403,),
    "sports ball": (2054, 1069),
    "kite": (19867,),
    "baseball bat": (3470, 6575),
    "baseball glove": (3470, 16078),
    "skateboard": (31777,),
    "surfboard": (10176, 1972),
    "tennis racket": (5298, 37691),
    "bottle": (5392,),
    "wine glass": (2604, 3313),
    "cup": (1937,),
    "fork": (13700,),
    "knife": (8960,),
    "spoon": (14024,),
    "bowl": (3814,),
    "banana": (8922,),
    "apple": (3055,),
    "sandwich": (8687,),
    "orange": (4287,),
    "broccoli": (19094,),
    "carrot": (15435,),
    "hot dog": (2069, 1929),
    "pizza": (4474,),
    "donut": (18471,),
    "cake": (2972,),
    "chair": (4269,),
    "couch": (12724,),
    "potted plant": (48581, 3912),
    "bed": (2722,),
    "dining table": (8658, 2175),
    "toilet": (11071,),
    "tv": (1583,),
    "laptop": (10464,),
    "mouse": (9301,),
    "remote": (9687,),
    "keyboard": (13017,),
    "cell phone": (5533, 1951),
    "microwave": (24240,),
    "oven": (12579,),
    "toaster": (39061,),
    "sink": (14551,),
    "refrigerator": (36662,),
    "book": (1116,),
    "clock": (6716,),
    "vase": (20431,),
    "scissors": (30867,),
    "teddy bear": (11798, 4298),
    "hair drier": (2225, 39877),
    "toothbrush": (36841,),
}


def tokens(prompt: str) -> list[int]:
    """Return the prompt's token IDs, start and end tokens included, without padding."""
    return [START, *COCO_TOKENS[prompt], END]


def one_hot(prompt_tokens: Sequence[int]) -> np.ndarray:
    """Return the text encoder input: one row per context position, padding rows all zero."""
    value = np.zeros((1, CONTEXT, VOCABULARY), np.float32)
    value[0, np.arange(len(prompt_tokens)), prompt_tokens] = 1
    return value


def packed(features: np.ndarray, prompt_tokens: Sequence[int]) -> np.ndarray:
    """Return the image model's text input: rows 0-15 hold the features, row 16 the token validity mask."""
    text = np.zeros((1, CONTEXT + 1, FEATURES), np.float32)
    text[0, :CONTEXT] = np.asarray(features, np.float32).reshape(CONTEXT, FEATURES)
    text[0, CONTEXT, : len(prompt_tokens)] = 1
    return text


def prompts(count: int) -> list[str]:
    """Spread count prompts evenly over the category list."""
    names = list(COCO_TOKENS)
    return [names[index * len(names) // count] for index in range(count)]


def preprocess_image(content: bytes) -> np.ndarray:
    """Return the image model input for one encoded image: RGB stretched to 1008x1008, scaled to [-1, 1], HWC.

    EXIF orientation is ignored, as the COCO annotations do. SAM 3 resizes with bilinear filtering, antialiased
    only where an axis shrinks; COCO images are at most 640 pixels a side, so every axis enlarges.
    """
    # convert.py imports this module in an environment without OpenCV.
    import cv2

    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("could not decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return (image.astype(np.float32) / np.float32(255) - MEAN) / STD


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


def select_images(dataset: Path, seed: int, start: int, count: int) -> tuple[str, ...]:
    """Return images start to start + count of the dataset's seeded shuffle."""
    names = _image_names(dataset)
    if start < 0 or count < 1:
        raise ValueError("an image selection needs a non-negative start and a positive count")
    if len(names) < start + count:
        raise ValueError(f"dataset contains {len(names)} images, but {start + count} are required")
    random.Random(seed).shuffle(names)
    return tuple(names[start : start + count])


def _image_bytes(dataset: Path, names: Sequence[str]) -> Iterator[bytes]:
    if dataset.is_dir():
        for name in names:
            yield (dataset / name).read_bytes()
        return
    with tarfile.open(dataset) as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in names:
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError(f"could not read image from dataset: {name}")
            yield stream.read()


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    """Return the selected images as one NHWC batch of image model inputs."""
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, names)])
