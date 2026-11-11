"""SuperGlue Outdoor's calibration inputs from the yfcc100m-keypoints cal pairs, for compile.py.

The dataset holds each image's SuperPoint keypoints, best first, in pixels of SuperPoint's 640x480 input, their scores
and their unit-length 256-d descriptors. The matcher takes exactly 512 real keypoints per image, so calibration uses
only the pairs whose images both have 512. Keypoints are normalized as accuracy.py normalizes them: centred on the
image and divided by 0.7 times its longer side.
"""

from __future__ import annotations

import random
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

KEYPOINTS = 512
DESCRIPTOR_SIZE = 256
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
# The matcher's inputs as the compiler takes them, channels-last: each image's keypoints as the width of a one-row
# image. surgery.py's ONNX takes them channels-first.
INPUT_SHAPES = {
    "keypoints0": (1, 1, KEYPOINTS, 2),
    "scores0": (1, 1, KEYPOINTS, 1),
    "descriptors0": (1, 1, KEYPOINTS, DESCRIPTOR_SIZE),
    "keypoints1": (1, 1, KEYPOINTS, 2),
    "scores1": (1, 1, KEYPOINTS, 1),
    "descriptors1": (1, 1, KEYPOINTS, DESCRIPTOR_SIZE),
}
# Calibration takes the first pairs of this seeded order.
SEED = 42


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


def seeded_pairs(root: Path) -> list[tuple[str, str]]:
    """The image names of every pair whose images both have 512 keypoints, in one seeded order."""
    image_lines = (root / "annotations" / "images.txt").read_text(encoding="utf-8").splitlines()
    counts = {fields[0]: int(fields[-1]) for fields in map(str.split, image_lines)}
    pair_lines = (root / "annotations" / "pairs_cal.txt").read_text(encoding="utf-8").splitlines()
    pairs = sorted({tuple(line.split()[:2]) for line in pair_lines})
    pairs = [pair for pair in pairs if counts[pair[0]] == counts[pair[1]] == KEYPOINTS]
    random.Random(SEED).shuffle(pairs)
    return pairs


def normalized(keypoints: np.ndarray) -> np.ndarray:
    size = np.array([IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float64)
    return ((keypoints.astype(np.float64) - size / 2) / (size.max() * 0.7)).astype(np.float32)


def image_inputs(root: Path, name: str) -> list[np.ndarray]:
    """One image's keypoints [1, 1, 512, 2], scores [1, 1, 512, 1] and descriptors [1, 1, 512, 256]."""
    directory = root / "cal" / PurePosixPath(name).with_suffix("")
    keypoints = np.load(directory / "keypoints.npy", allow_pickle=False)
    scores = np.load(directory / "scores.npy", allow_pickle=False)
    descriptors = np.load(directory / "descriptors.npy", allow_pickle=False)
    if keypoints.shape != (KEYPOINTS, 2) or scores.shape != (KEYPOINTS,) or descriptors.shape != (
            KEYPOINTS, DESCRIPTOR_SIZE):
        raise ValueError(f"{name}: unexpected keypoints {keypoints.shape}, scores {scores.shape} or descriptors "
                         f"{descriptors.shape}")
    values = (normalized(keypoints), scores[:, None], descriptors)
    return [np.ascontiguousarray(value, dtype=np.float32)[None, None] for value in values]


def calibration_inputs(dataset: Path, samples: int) -> dict[str, np.ndarray]:
    """The first pairs of the seeded order, each input stacked along the batch axis."""
    with opened(dataset) as root:
        pairs = seeded_pairs(root)
        if len(pairs) < samples:
            raise ValueError(f"calibration requires {samples} pairs, the dataset has {len(pairs)}")
        stacked = [[*image_inputs(root, name0), *image_inputs(root, name1)] for name0, name1 in pairs[:samples]]
    return {name: np.concatenate([inputs[index] for inputs in stacked]) for index, name in enumerate(INPUT_SHAPES)}
