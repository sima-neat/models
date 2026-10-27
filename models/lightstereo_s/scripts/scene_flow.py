"""Pad and normalize Scene Flow pairs the way OpenStereo evaluates LightStereo, and select fidelity pairs."""

from __future__ import annotations

import io
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

# cfgs/lightstereo/lightstereo_s_sceneflow.yaml: RightTopPad to 544x960, then ImageNet normalization.
HEIGHT, WIDTH = 544, 960
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
INPUTS = ("left", "right")
# The scene-flow cal archive holds scene_flow/cal.txt (left, right, disparity per line) and the listed files.
ROOT = "scene_flow"
CAL_LIST = "cal.txt"


def pad_geometry(height: int, width: int) -> tuple[int, int]:
    """Return the rows added above and the columns added to the right of a height x width image."""
    if height > HEIGHT or width > WIDTH:
        raise ValueError(f"a {height}x{width} image exceeds the {HEIGHT}x{WIDTH} input")
    return HEIGHT - height, WIDTH - width


def preprocess_image(content: bytes) -> np.ndarray:
    """Return one view as model input: RGB, edge-padded at the top and right, normalized, HWC float32."""
    with Image.open(io.BytesIO(content)) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float32)
    top, right = pad_geometry(*image.shape[:2])
    padded = np.pad(image, ((top, 0), (0, right), (0, 0)), mode="edge")
    return (padded / np.float32(255) - MEAN) / STD


def _lines(dataset: Path) -> list[tuple[str, str]]:
    """The (left, right) image paths cal.txt lists, in list order."""
    with tarfile.open(dataset, "r:gz") as archive:
        stream = archive.extractfile(f"{ROOT}/{CAL_LIST}")
        if stream is None:
            raise ValueError(f"{dataset} has no {ROOT}/{CAL_LIST}")
        lines = [tuple(line.split()[:2]) for line in stream.read().decode().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{dataset} lists no stereo pairs in {CAL_LIST}")
    return lines


def fidelity_pairs(dataset: Path, seed: int, offset: int, samples: int) -> list[tuple[str, str]]:
    """Return the cal.txt pairs the fidelity comparison uses, selected as model.yaml declares."""
    pairs = _lines(dataset)
    if offset + samples > len(pairs):
        raise ValueError(f"fidelity selection needs {offset + samples} pairs, but {dataset} has {len(pairs)}")
    random.Random(seed).shuffle(pairs)
    return pairs[offset : offset + samples]


def read_pairs(dataset: Path, pairs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Return the preprocessed left and right views of the pairs as two NHWC batches, in the given order."""
    wanted = {path for pair in pairs for path in pair}
    found: dict[str, bytes] = {}
    with tarfile.open(dataset, "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix(f"{ROOT}/")
            if name in wanted and member.isfile():
                found[name] = archive.extractfile(member).read()
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{dataset} lacks {len(missing)} listed images, e.g. {missing[:2]}")
    return {
        name: np.stack([preprocess_image(found[pair[index]]) for pair in pairs]) for index, name in enumerate(INPUTS)
    }
