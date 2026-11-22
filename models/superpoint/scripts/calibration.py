"""Select SuperPoint's INT8 calibration images from the hpatches cal variant and prepare them as model inputs."""

from __future__ import annotations

import random
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from keypoints import decode_image, preprocess

# Calibration draws only from the cal sequences, so no scored hpatches-val sequence calibrates the model.
CALIBRATION_SPLIT = "cal"
CALIBRATION_SEED = 20261023


def _calibration_member(name: str) -> bool:
    """Whether an archive member is an image of a cal sequence: hpatches/cal/<sequence>/<k>.ppm."""
    path = PurePosixPath(name)
    # macOS tar writes a "._<name>" companion beside a file with extended attributes; it holds no image.
    is_image = path.suffix == ".ppm" and not path.name.startswith("._")
    return is_image and path.parent.parent.name == CALIBRATION_SPLIT


def calibration_images(dataset: Path, count: int) -> np.ndarray:
    """Return a seeded sample of the cal images as one NHWC float32 batch of grayscale model inputs."""
    with tarfile.open(dataset) as archive:
        members = sorted((member for member in archive.getmembers() if member.isfile()
                          and _calibration_member(member.name)), key=lambda member: member.name)
        if count < 1 or len(members) < count:
            raise ValueError(f"cannot select {count} calibration images from {len(members)} {CALIBRATION_SPLIT} images")
        selected = random.Random(CALIBRATION_SEED).sample(members, count)
        images = [preprocess(decode_image(archive.extractfile(member).read())) for member in selected]
    return np.stack(images)[..., None]
