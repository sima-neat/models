"""Select and preprocess D-FINE images, and select the decoder's top proposals."""

from __future__ import annotations

import io
import random
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 640
QUERY_COUNT = 300
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".webp")


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
    return array


def load_images(dataset: Path, names: Sequence[str]) -> np.ndarray:
    if not names:
        raise ValueError("at least one image must be selected")
    return np.stack([preprocess_image(content) for content in _image_bytes(dataset, names)])


def to_afe_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4:
        array = array.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(array)


def select_top_queries(scores, content, reference_logits):
    """Gather content and raw references in stable Top300 order, never attention memory."""
    if scores.shape != (1, 8400) or content.shape != (1, 8400, 256) or reference_logits.shape != (1, 8400, 4):
        raise ValueError("Unexpected Large proposal boundary")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (scores, content, reference_logits)):
        raise ValueError("Expected finite float32 proposal tensors")
    order = np.argsort(-scores, axis=1, kind="stable")[0, :QUERY_COUNT]
    return tuple(x[0, order, :].reshape(1, QUERY_COUNT, x.shape[-1]) for x in (content, reference_logits))


def decoder_inputs(outputs):
    memory, content, scores, references = outputs
    return memory, *select_top_queries(scores, content, references)
