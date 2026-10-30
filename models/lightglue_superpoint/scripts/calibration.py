"""LightGlue's inputs from the yfcc100m-keypoints dataset, for compile.py, performance.py and accuracy.py.

The dataset holds each image's SuperPoint keypoints, best first, in pixels of SuperPoint's 640x480 input, which is the
image stretched to that size, and their unit-length 256-d descriptors. An image has 512 keypoints unless it yields
fewer. The matcher takes keypoints normalized as LightGlue normalizes them, ((x, y) - (320, 240)) / 320. A pair's
inputs are padded to 512 keypoints per image with keypoints at the image centre and zero descriptors, and match
extraction ignores them.
"""

from __future__ import annotations

import random
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

KEYPOINTS = 512
DESCRIPTOR_DIM = 256
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
INPUT_SHAPES = {
    "keypoints0": (1, KEYPOINTS, 2),
    "descriptors0": (1, KEYPOINTS, DESCRIPTOR_DIM),
    "keypoints1": (1, KEYPOINTS, 2),
    "descriptors1": (1, KEYPOINTS, DESCRIPTOR_DIM),
}
OUTPUT_SHAPE = (1, KEYPOINTS, KEYPOINTS)
# LightGlue's filter_threshold: a mutual best pair matches when its assignment exceeds it.
MATCH_THRESHOLD = 0.1
# A pairs line: both image names, both rotations, both 3x3 intrinsics and the 4x4 pose from image 0 to image 1.
PAIR_FIELDS = 38


@dataclass(frozen=True)
class Pair:
    """Two images and their ground truth: each image's intrinsics, at its original size, and the relative pose."""

    name0: str
    name1: str
    intrinsics0: np.ndarray
    intrinsics1: np.ndarray
    pose: np.ndarray


@dataclass(frozen=True)
class Image:
    """An image's original size, the quarter turns its SuperPoint input was rotated by, and its keypoint count."""

    width: int
    height: int
    rotation: int
    keypoints: int


@contextmanager
def opened(path: Path) -> Iterator[Path]:
    """The dataset's root, holding annotations/, from an extracted directory or a Dataset Registry archive."""
    if path.is_dir():
        yield dataset_root(path)
        return
    with tempfile.TemporaryDirectory(prefix="yfcc100m-keypoints-") as directory:
        with tarfile.open(path, "r:*") as archive:
            # The data filter refuses absolute paths, links out of the directory and special files.
            archive.extractall(directory, filter="data")
        yield dataset_root(Path(directory))


def dataset_root(path: Path) -> Path:
    if (path / "annotations").is_dir():
        return path
    roots = [child for child in sorted(path.iterdir()) if (child / "annotations").is_dir()]
    if len(roots) != 1:
        raise ValueError(f"{path} holds no single yfcc100m-keypoints directory with annotations/")
    return roots[0]


def split_of(root: Path) -> str:
    """The variant's split, cal or val, which names its pairs file and its features directory."""
    splits = sorted(path.stem.removeprefix("pairs_") for path in (root / "annotations").glob("pairs_*.txt"))
    if len(splits) != 1:
        raise ValueError(f"expected one annotations/pairs_<split>.txt in {root}, found {splits}")
    return splits[0]


def read_images(root: Path) -> dict[str, Image]:
    images = {}
    for line in (root / "annotations" / "images.txt").read_text(encoding="utf-8").splitlines():
        name, width, height, rotation, keypoints = line.split()
        images[name] = Image(int(width), int(height), int(rotation) % 4, int(keypoints))
    return images


def read_pairs(root: Path) -> list[Pair]:
    pairs = []
    for line in (root / "annotations" / f"pairs_{split_of(root)}.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != PAIR_FIELDS:
            raise ValueError(f"a pairs line has {len(fields)} fields, expected {PAIR_FIELDS}: {line[:80]}")
        # images.txt holds each image's rotation; the ground truth is the unrotated images'.
        values = np.asarray(fields[4:], np.float64)
        pairs.append(Pair(fields[0], fields[1], values[:9].reshape(3, 3), values[9:18].reshape(3, 3),
                          values[18:].reshape(4, 4)))
    return pairs


def seeded_pairs(root: Path, seed: int) -> list[Pair]:
    """Every pair in one seeded order: calibration takes the first ones, numerical fidelity a later span."""
    pairs = sorted(read_pairs(root), key=lambda pair: (pair.name0, pair.name1))
    random.Random(seed).shuffle(pairs)
    return pairs


def features(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """An image's keypoints [K, 2] in pixels of the 640x480 input and its descriptors [K, 256]."""
    directory = root / split_of(root) / PurePosixPath(name).with_suffix("")
    keypoints = np.load(directory / "keypoints.npy").astype(np.float32)
    descriptors = np.load(directory / "descriptors.npy").astype(np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2 or descriptors.shape != (len(keypoints), DESCRIPTOR_DIM):
        raise ValueError(f"{name}: unexpected keypoints {keypoints.shape} or descriptors {descriptors.shape}")
    if not 0 < len(keypoints) <= KEYPOINTS:
        raise ValueError(f"{name} has {len(keypoints)} keypoints; the matcher takes 1 to {KEYPOINTS}")
    return keypoints, descriptors


def normalized(keypoints: np.ndarray) -> np.ndarray:
    scale = max(IMAGE_WIDTH, IMAGE_HEIGHT) / 2
    return ((keypoints - [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2]) / scale).astype(np.float32)


def image_inputs(keypoints: np.ndarray, descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One image's matcher inputs [1, 512, 2] and [1, 512, 256], padded with centred keypoints and zero descriptors."""
    padded_keypoints = np.zeros((1, KEYPOINTS, 2), np.float32)
    padded_descriptors = np.zeros((1, KEYPOINTS, DESCRIPTOR_DIM), np.float32)
    padded_keypoints[0, : len(keypoints)] = normalized(keypoints)
    padded_descriptors[0, : len(descriptors)] = descriptors
    return padded_keypoints, padded_descriptors


def matcher_inputs(image0: tuple[np.ndarray, np.ndarray], image1: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """A pair's matcher inputs from each image's keypoints and descriptors."""
    return dict(zip(INPUT_SHAPES, (*image_inputs(*image0), *image_inputs(*image1))))


def pair_inputs(root: Path, pair: Pair) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """A pair's matcher inputs and each image's number of real keypoints."""
    image0, image1 = features(root, pair.name0), features(root, pair.name1)
    return matcher_inputs(image0, image1), (len(image0[0]), len(image1[0]))


def matches(assignment: np.ndarray, counts: tuple[int, int], threshold: float = MATCH_THRESHOLD) -> np.ndarray:
    """LightGlue's match extraction on the real keypoints: mutual best pairs whose assignment exceeds the threshold,
    as rows (i, j) of keypoint indices in image 0 and image 1."""
    assignment = np.asarray(assignment, np.float32).reshape(KEYPOINTS, KEYPOINTS)[: counts[0], : counts[1]]
    best0, best1 = assignment.argmax(1), assignment.argmax(0)
    rows = np.arange(len(best0))
    keep = (best1[best0] == rows) & (assignment[rows, best0] > threshold)
    return np.stack([rows[keep], best0[keep]], 1)


def calibration_inputs(dataset: Path, samples: int, seed: int = 42) -> dict[str, np.ndarray]:
    """The first pairs of the seeded order, each input stacked along the batch axis."""
    with opened(dataset) as root:
        pairs = seeded_pairs(root, seed)
        if len(pairs) < samples:
            raise ValueError(f"calibration requires {samples} pairs, the dataset has {len(pairs)}")
        stacked = [pair_inputs(root, pair)[0] for pair in pairs[:samples]]
    return {name: np.concatenate([inputs[name] for inputs in stacked]) for name in INPUT_SHAPES}


def fidelity_inputs(dataset: Path, seed: int, offset: int,
                    samples: int) -> list[tuple[dict[str, np.ndarray], tuple[int, int]]]:
    """The declared span of the seeded order: each pair's inputs and its images' numbers of real keypoints."""
    with opened(dataset) as root:
        pairs = seeded_pairs(root, seed)
        if len(pairs) < offset + samples:
            raise ValueError(f"numerical fidelity needs {offset + samples} pairs, the dataset has {len(pairs)}")
        return [pair_inputs(root, pair) for pair in pairs[offset : offset + samples]]
