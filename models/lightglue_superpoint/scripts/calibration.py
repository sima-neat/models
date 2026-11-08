"""LightGlue's inputs from the yfcc100m-keypoints dataset, for compile.py, performance.py and accuracy.py.

The dataset holds each image's SuperPoint keypoints, best first, in pixels of SuperPoint's 640x480 input, which is the
image stretched to that size, and their unit-length 256-d descriptors. The matcher takes exactly 512 real keypoints per
image and has no padding mask, so the scripts use only the pairs whose images both have 512: each archive holds one
image with fewer, which leaves out 22 of the 4,000 val pairs. The matcher takes each keypoint's rotary position
encoding, which the CPU computes in float32: the keypoint normalized as LightGlue normalizes it,
((x, y) - (320, 240)) / 320, times LightGlue's learned 2-to-32 projection, then the cosines and the sines, each repeated
twice. Float32 differs from float64 by at most 5e-7, far below the rounding of the BF16 and INT8 inputs. Computed in the
graph at BF16 instead, where the phases themselves round, the encoding kept 91% of the ONNX's matches at w16a16,
against 97% from the CPU.
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
# LightGlue's learned keypoint projection (positional_encoder.projector.weight, [32, 2]) in the pinned checkpoint;
# convert.py checks it against the checkpoint. Each row turns a normalized (x, y) into one phase.
KEYPOINT_PROJECTION = np.asarray([
    (0.07869736105203629, 0.6576962471008301),
    (0.30121079087257385, -1.3067657947540283),
    (-0.17319072782993317, 0.3640412390232086),
    (-1.8221570253372192, 0.07125379890203476),
    (4.831599235534668, 0.024766402319073677),
    (-0.6115880608558655, -0.5418711304664612),
    (0.5497006177902222, -0.7080969214439392),
    (0.9728097915649414, 0.8288111686706543),
    (-0.18647703528404236, 0.2699703872203827),
    (1.3076542615890503, -1.4809051752090454),
    (-0.19871585071086884, 2.5542619228363037),
    (-1.342088222503662, -1.5769871473312378),
    (-0.023420853540301323, -0.2613159120082855),
    (-0.5988006591796875, -4.0408616065979),
    (0.4600417912006378, -0.07377243041992188),
    (-2.12711763381958, -1.0835118293762207),
    (-0.6925885677337646, 1.3105987310409546),
    (0.25808796286582947, -0.06581223011016846),
    (0.09479046612977982, -0.44248539209365845),
    (-1.2112597227096558, -1.857155680656433),
    (-0.24696263670921326, -0.0016804899787530303),
    (-2.5080809593200684, 3.7625091075897217),
    (-1.390500783920288, 0.7762705087661743),
    (0.3077039122581482, -1.0518392324447632),
    (-1.6019573211669922, -0.010330487042665482),
    (0.5500544309616089, 2.5014572143554688),
    (-1.0771361589431763, 0.19658993184566498),
    (0.25206997990608215, -0.5227173566818237),
    (-0.5797532200813293, 0.07100092619657516),
    (-0.36304882168769836, 1.7311886548995972),
    (0.554409921169281, 1.0818008184432983),
    (0.4458959400653839, 0.6291354298591614),
], np.float32)
# Cosines then sines of the 32 phases, each repeated twice: the rotary encoding of 64-wide attention heads.
ENCODING_DIM = 4 * len(KEYPOINT_PROJECTION)
INPUT_SHAPES = {
    "encoding0": (1, KEYPOINTS, ENCODING_DIM),
    "descriptors0": (1, KEYPOINTS, DESCRIPTOR_DIM),
    "encoding1": (1, KEYPOINTS, ENCODING_DIM),
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


def usable_pairs(root: Path) -> list[Pair]:
    """The pairs whose images both have the 512 keypoints the matcher takes."""
    counts = {name: image.keypoints for name, image in read_images(root).items()}
    return [pair for pair in read_pairs(root) if counts[pair.name0] == counts[pair.name1] == KEYPOINTS]


def seeded_pairs(root: Path, seed: int) -> list[Pair]:
    """Every usable pair in one seeded order: calibration takes the first ones, numerical fidelity a later span."""
    pairs = sorted(usable_pairs(root), key=lambda pair: (pair.name0, pair.name1))
    random.Random(seed).shuffle(pairs)
    return pairs


def features(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """An image's 512 keypoints [512, 2] in pixels of the 640x480 input and its descriptors [512, 256]."""
    directory = root / split_of(root) / PurePosixPath(name).with_suffix("")
    keypoints = np.load(directory / "keypoints.npy").astype(np.float32)
    descriptors = np.load(directory / "descriptors.npy").astype(np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2 or descriptors.shape != (len(keypoints), DESCRIPTOR_DIM):
        raise ValueError(f"{name}: unexpected keypoints {keypoints.shape} or descriptors {descriptors.shape}")
    if len(keypoints) != KEYPOINTS:
        raise ValueError(f"{name} has {len(keypoints)} keypoints; the matcher takes {KEYPOINTS} real keypoints")
    return keypoints, descriptors


def normalized(keypoints: np.ndarray) -> np.ndarray:
    scale = max(IMAGE_WIDTH, IMAGE_HEIGHT) / 2
    return (keypoints - [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2]) / scale


def encoding(normalized_keypoints: np.ndarray) -> np.ndarray:
    """Normalized keypoints [..., 2] to their rotary encoding [..., 128]: the cosines, then the sines, of the projected
    phases, each repeated twice, as LightGlue's positional encoder computes them. The cosines and sines of the 32 phases
    are taken before the repeat, which halves the CPU's work."""
    phases = np.asarray(normalized_keypoints, np.float32) @ KEYPOINT_PROJECTION.T
    return np.repeat(np.concatenate([np.cos(phases), np.sin(phases)], -1), 2, -1)


def image_inputs(keypoints: np.ndarray, descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One image's matcher inputs [1, 512, 128] and [1, 512, 256] from its 512 keypoints and descriptors."""
    if keypoints.shape != (KEYPOINTS, 2) or descriptors.shape != (KEYPOINTS, DESCRIPTOR_DIM):
        raise ValueError(
            f"the matcher takes {KEYPOINTS} keypoints and descriptors per image, not {keypoints.shape} and "
            f"{descriptors.shape}"
        )
    return encoding(normalized(keypoints))[None], np.asarray(descriptors, np.float32)[None]


def matcher_inputs(
    image0: tuple[np.ndarray, np.ndarray], image1: tuple[np.ndarray, np.ndarray]
) -> dict[str, np.ndarray]:
    """A pair's matcher inputs from each image's keypoints and descriptors."""
    return dict(zip(INPUT_SHAPES, (*image_inputs(*image0), *image_inputs(*image1))))


def pair_inputs(root: Path, pair: Pair) -> dict[str, np.ndarray]:
    """A pair's matcher inputs."""
    return matcher_inputs(features(root, pair.name0), features(root, pair.name1))


def matches(assignment: np.ndarray, threshold: float = MATCH_THRESHOLD) -> np.ndarray:
    """LightGlue's match extraction: mutual best pairs whose assignment exceeds the threshold, as rows (i, j) of
    keypoint indices in image 0 and image 1."""
    assignment = np.asarray(assignment, np.float32).reshape(KEYPOINTS, KEYPOINTS)
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
        stacked = [pair_inputs(root, pair) for pair in pairs[:samples]]
    return {name: np.concatenate([inputs[name] for inputs in stacked]) for name in INPUT_SHAPES}


def fidelity_inputs(dataset: Path, seed: int, offset: int, samples: int) -> list[dict[str, np.ndarray]]:
    """The declared span of the seeded order: each pair's matcher inputs."""
    with opened(dataset) as root:
        pairs = seeded_pairs(root, seed)
        if len(pairs) < offset + samples:
            raise ValueError(f"numerical fidelity needs {offset + samples} pairs, the dataset has {len(pairs)}")
        return [pair_inputs(root, pair) for pair in pairs[offset : offset + samples]]
