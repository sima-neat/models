"""Read Scene Flow cal pairs as RAFT-Stereo's side-by-side input, select the calibration and fidelity pairs, and score
disparities as upstream's Scene Flow evaluation does."""

from __future__ import annotations

import io
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

# Scene Flow views are 960x540, the size the model takes; the graph pads the rows itself.
HEIGHT, WIDTH = 540, 960
# Neat's ImageNet preset: RGB scaled to [0, 1], minus the mean, divided by the standard deviation.
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# The model takes one side-by-side frame: the left view in the first WIDTH columns, the right view after it.
INPUT = "stereo"
# The scene-flow cal archive holds scene_flow/cal.txt (left, right, disparity per line) and the listed files.
ROOT = "scene_flow"
CAL_LIST = "cal.txt"
# Upstream's Scene Flow evaluation scores the pixels whose true disparity is under 192.
MAX_DISPARITY = 192

Pair = tuple[str, str, str]


def preprocess_image(content: bytes) -> np.ndarray:
    """Return one view as model input: RGB scaled to [0, 1] and ImageNet-normalized, HWC float32."""
    with Image.open(io.BytesIO(content)) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float32)
    if image.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"expected a {WIDTH}x{HEIGHT} view, found {image.shape[1]}x{image.shape[0]}")
    return (image / np.float32(255) - MEAN) / STD


def read_pfm(content: bytes) -> np.ndarray:
    """Return a one-channel PFM, such as a Scene Flow disparity map, as float32 rows from the top."""
    stream = io.BytesIO(content)
    if stream.readline().strip() != b"Pf":
        raise ValueError("expected a one-channel PFM")
    width, height = (int(value) for value in stream.readline().split())
    scale = float(stream.readline())
    data = np.frombuffer(stream.read(4 * width * height), "<f4" if scale < 0 else ">f4")
    return np.flipud(data.reshape(height, width)).astype(np.float32)


def _lines(dataset: Path) -> list[Pair]:
    """The (left, right, disparity) paths cal.txt lists, in list order."""
    with tarfile.open(dataset, "r:gz") as archive:
        stream = archive.extractfile(f"{ROOT}/{CAL_LIST}")
        if stream is None:
            raise ValueError(f"{dataset} has no {ROOT}/{CAL_LIST}")
        lines = [tuple(line.split()) for line in stream.read().decode().splitlines() if line.strip()]
    if not lines or any(len(line) != 3 for line in lines):
        raise ValueError(f"{dataset} must list a left image, a right image and a disparity per line of {CAL_LIST}")
    return lines


def selected_pairs(dataset: Path, seed: int, offset: int, samples: int) -> list[Pair]:
    """Return a span of the cal.txt pairs in one seeded order: calibration takes the first, fidelity a later span."""
    pairs = _lines(dataset)
    if offset + samples > len(pairs):
        raise ValueError(f"the selection needs {offset + samples} pairs, but {dataset} has {len(pairs)}")
    random.Random(seed).shuffle(pairs)
    return pairs[offset : offset + samples]


def read_pairs(dataset: Path, pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs as one NHWC batch of side-by-side frames, [left | right] views preprocessed, and their left
    views' true disparities, in order."""
    wanted = {path for pair in pairs for path in pair}
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix(f"{ROOT}/")
            if name in wanted and member.isfile():
                found[name] = archive.extractfile(member).read()
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{dataset} lacks {len(missing)} listed files, e.g. {missing[:2]}")
    frames = np.stack(
        [
            np.concatenate([preprocess_image(found[left]), preprocess_image(found[right])], axis=1)
            for left, right, _ in pairs
        ]
    )
    return frames, np.stack([read_pfm(found[disparity]) for _, _, disparity in pairs])


def end_point_error(disparity: np.ndarray, truth: np.ndarray) -> float:
    """Upstream's Scene Flow end-point error: the mean absolute error over pixels whose true disparity is under 192."""
    valid = np.abs(truth) < MAX_DISPARITY
    return float(np.mean(np.abs(disparity.reshape(truth.shape) - truth)[valid]))
