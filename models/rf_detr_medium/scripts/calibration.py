"""Prepare representative images and RF-DETR transformer boundary tensors."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 576
FEATURE_SIZE = IMAGE_SIZE // 16
QUERY_COUNT = 300
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


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
    if tarfile.is_tarfile(dataset):
        with tarfile.open(dataset) as archive:
            members = {member.name: member for member in archive.getmembers()}
            for name in names:
                stream = archive.extractfile(members[name])
                if stream is None:
                    raise ValueError(f"could not read image from dataset: {name}")
                yield stream.read()
        return
    raise ValueError(f"unsupported image dataset input: {dataset}")


def preprocess_image(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - MEAN) / STD


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    if not names:
        raise ValueError("at least one image must be selected")
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, names)])


def to_afe_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def stable_topk_gather(scores: np.ndarray, boxes: np.ndarray, count: int = QUERY_COUNT) -> np.ndarray:
    if scores.ndim != 2 or boxes.ndim != 3 or boxes.shape[2] != 4:
        raise ValueError(f"unexpected proposal shapes: scores={scores.shape}, boxes={boxes.shape}")
    if scores.shape != boxes.shape[:2]:
        raise ValueError(f"proposal count mismatch: scores={scores.shape}, boxes={boxes.shape}")
    if count < 1 or scores.shape[1] < count:
        raise ValueError(f"cannot select {count} proposals from {scores.shape[1]}")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("proposal scores and boxes must be finite")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :count]
    return np.take_along_axis(boxes, indices[..., None], axis=1).astype(np.float32)


def normalize_features(feature: np.ndarray) -> np.ndarray:
    if feature.shape == (1, FEATURE_SIZE, FEATURE_SIZE, 256):
        return feature.transpose(0, 3, 1, 2)
    if feature.shape != (1, 256, FEATURE_SIZE, FEATURE_SIZE):
        raise RuntimeError(f"unexpected backbone feature shape: {feature.shape}")
    return feature


def execute_backbone(backbone: object, image: np.ndarray, input_name: object) -> tuple[np.ndarray, ...]:
    outputs = backbone.execute(inputs={input_name("input"): to_afe_layout(image[None])}, use_jax=True)
    feature, scores, boxes = (np.asarray(output) for output in outputs)
    return normalize_features(feature), scores, boxes


def transformer_inputs(
    backbone: object,
    images: np.ndarray,
    input_name: object,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    proposals = []
    for image in images:
        feature, scores, boxes = execute_backbone(backbone, image, input_name)
        features.append(feature[0].astype(np.float32))
        proposals.append(stable_topk_gather(scores, boxes)[0])
    return np.stack(features), np.stack(proposals)
