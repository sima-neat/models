"""DINOv3 ViT-H+/16 preprocessing and the seeded selection of imagenet-2012 cal images, used by compile.py."""

from __future__ import annotations

import io
import random
from pathlib import Path, PurePosixPath

import numpy as np
from model_inputs.images import image_names, read_images
from PIL import Image

# Meta's transform for the LVD-1689M weights, make_transform in the README of facebookresearch/dinov3 at a044468d:
# an antialiased bilinear resize straight to the input size, then ImageNet's mean and std. PIL's bilinear resize
# antialiases when it shrinks. The channel constants are in HWC order.
SIZE = 224
RESCALE = np.float32(1 / 255)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# imagenet-2012/cal holds cal2012/<image>.
IMAGE_SUFFIXES = (".jpeg",)
CALIBRATION_SPLIT = "cal2012"


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode, resize and normalize one image into a float32 HWC tensor."""
    with Image.open(io.BytesIO(content)) as source:
        image = source.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) * RESCALE - MEAN) / STD


def select_names(dataset: Path, seed: int, offset: int, samples: int) -> list[str]:
    """samples imagenet-2012 cal images from offset in a seeded order, as model.yaml's numerical_fidelity selects."""
    names = image_names(dataset, IMAGE_SUFFIXES)
    names = [name for name in names if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if offset + samples > len(names):
        raise ValueError(f"the selection needs {offset + samples} images, but {dataset} has {len(names)} under "
                         f"{CALIBRATION_SPLIT}/")
    random.Random(seed).shuffle(names)
    return names[offset : offset + samples]


def load_images(dataset: Path, names: list[str]) -> np.ndarray:
    """The named images, preprocessed and stacked in NHWC order."""
    images = read_images(dataset, names)
    return np.stack([preprocess_image(images[name]) for name in names])
