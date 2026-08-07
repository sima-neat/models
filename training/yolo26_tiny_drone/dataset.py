"""Sequence-safe manifest validation and QAT dataset loading."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from contract import DEFAULT_MODEL_SIZE
from paths import ultralytics_label_path
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset

VALID_SPLITS = ("train", "val", "test")
LABEL_BOUNDARY_EPSILON = 1e-6


@dataclass(frozen=True)
class FrameRecord:
    image: Path
    label: Path
    sequence: str
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_sha256(manifest: Path, records: list[FrameRecord]) -> str:
    """Bind manifest metadata to the bytes of every referenced image and label."""
    root = manifest.parent.resolve()
    digest = hashlib.sha256()

    def update_field(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    update_field("yolo26-sequence-safe-dataset-v1")
    for record in records:
        update_field(record.sequence)
        update_field(record.split)
        for kind, path in (("image", record.image), ("label", record.label)):
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise RuntimeError(f"dataset {kind} is missing or outside the manifest root: {path}")
            update_field(kind)
            update_field(resolved.relative_to(root).as_posix())
            with resolved.open("rb") as stream:
                size = resolved.stat().st_size
                digest.update(size.to_bytes(8, "big"))
                read = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    read += len(chunk)
                    digest.update(chunk)
            if read != size or resolved.stat().st_size != size:
                raise RuntimeError(f"dataset {kind} changed while hashing: {path}")
    return digest.hexdigest()


def load_manifest(path: Path, *, require_files: bool = True) -> list[FrameRecord]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("frames"), list):
        raise ValueError("dataset manifest must have schema_version 1 and a frames list")
    root = path.parent
    resolved_root = root.resolve()
    records = []
    seen_images: set[Path] = set()
    seen_labels: set[Path] = set()
    sequence_splits: dict[str, str] = {}
    for index, raw in enumerate(document["frames"]):
        if not isinstance(raw, dict):
            raise ValueError(f"frames[{index}] must be an object")
        split = raw.get("split")
        sequence = raw.get("sequence")
        if split not in VALID_SPLITS:
            raise ValueError(f"frames[{index}].split must be one of {VALID_SPLITS}")
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(f"frames[{index}].sequence must be a non-empty string")
        if sequence != sequence.strip():
            raise ValueError(f"frames[{index}].sequence must not have surrounding whitespace")
        image_value = raw.get("image")
        label_value = raw.get("label")
        if not isinstance(image_value, str) or not image_value or not isinstance(label_value, str) or not label_value:
            raise ValueError(f"frames[{index}] image and label must be strings")
        image_path = Path(image_value)
        label_path = Path(label_value)
        if image_path.is_absolute() or label_path.is_absolute() or ".." in image_path.parts or ".." in label_path.parts:
            raise ValueError(f"frames[{index}] image and label must stay within the manifest directory")
        image = (resolved_root / image_path).resolve()
        label = (resolved_root / label_path).resolve()
        if not image.is_relative_to(resolved_root) or not label.is_relative_to(resolved_root):
            raise ValueError(f"frames[{index}] image or label resolves outside the manifest directory")
        if image in seen_images:
            raise ValueError(f"duplicate image in manifest: {image_value}")
        if label in seen_labels:
            raise ValueError(f"duplicate label in manifest: {label_value}")
        seen_images.add(image)
        seen_labels.add(label)
        previous_split = sequence_splits.setdefault(sequence, split)
        if previous_split != split:
            raise ValueError(f"sequence {sequence!r} crosses {previous_split!r} and {split!r} splits")
        if require_files and (not image.is_file() or not label.is_file()):
            raise ValueError(f"missing image or label for frames[{index}]: {image_value}")
        records.append(FrameRecord(image=image, label=label, sequence=sequence, split=split))
    if not records:
        raise ValueError("dataset manifest contains no frames")
    for split in ("train", "val"):
        if not any(record.split == split for record in records):
            raise ValueError(f"dataset manifest has no {split} frames")
    return records


def write_ultralytics_dataset(manifest: Path, output_dir: Path) -> Path:
    records = load_manifest(manifest)
    for record in records:
        derived_label = ultralytics_label_path(record.image)
        if derived_label != record.label:
            raise ValueError(
                "manifest label does not match the path Ultralytics derives: "
                f"image={record.image}, manifest={record.label}, derived={derived_label}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in VALID_SPLITS:
        images = [str(record.image) for record in records if record.split == split]
        if images:
            (output_dir / f"{split}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    data_yaml = output_dir / "dataset.yaml"
    lines = [f"path: {manifest.parent.resolve()}"]
    for split in VALID_SPLITS:
        if (output_dir / f"{split}.txt").exists():
            lines.append(f"{split}: {str((output_dir / f'{split}.txt').resolve())}")
    lines.extend(["names:", "  0: drone"])
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def read_yolo_label(path: Path) -> Tensor:
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected class and xywh")
        class_id, x, y, width, height = (float(value) for value in parts)
        if class_id != 0.0:
            raise ValueError(f"{path}:{line_number}: one-class detector requires class id 0")
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            raise ValueError(f"{path}:{line_number}: normalized xywh must be in [0, 1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: width and height must be positive")
        if (
            x - width * 0.5 < -LABEL_BOUNDARY_EPSILON
            or x + width * 0.5 > 1.0 + LABEL_BOUNDARY_EPSILON
            or y - height * 0.5 < -LABEL_BOUNDARY_EPSILON
            or y + height * 0.5 > 1.0 + LABEL_BOUNDARY_EPSILON
        ):
            raise ValueError(f"{path}:{line_number}: normalized box must remain inside the image")
        # Six-decimal dataset exports can exceed an edge by half a quantization
        # unit. Canonicalize only that rounding noise before augmentation.
        x = min(max(x, width * 0.5), 1.0 - width * 0.5)
        y = min(max(y, height * 0.5), 1.0 - height * 0.5)
        labels.append([class_id, x, y, width, height])
    return torch.tensor(labels, dtype=torch.float32).reshape(-1, 5)


def letterbox(image: Image.Image, labels: Tensor, size: int = DEFAULT_MODEL_SIZE) -> tuple[Tensor, Tensor]:
    source_width, source_height = image.size
    scale = min(size / source_width, size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0

    adjusted = labels.clone()
    if adjusted.numel():
        adjusted[:, 1] = (labels[:, 1] * source_width * scale + left) / size
        adjusted[:, 2] = (labels[:, 2] * source_height * scale + top) / size
        adjusted[:, 3] = labels[:, 3] * source_width * scale / size
        adjusted[:, 4] = labels[:, 4] * source_height * scale / size
    return torch.from_numpy(np.ascontiguousarray(pixels)), adjusted


class DroneDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[FrameRecord],
        split: str,
        *,
        augment: bool,
        seed: int,
        model_size: int = DEFAULT_MODEL_SIZE,
    ) -> None:
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"no records for split {split!r}")
        self.augment = augment
        self.seed = seed
        self.model_size = model_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image) as source:
            image = source.convert("RGB")
        labels = read_yolo_label(record.label)
        if self.augment:
            generator = random.Random((self.seed << 32) ^ index ^ random.getrandbits(32))
            image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.85, 1.15))
            if generator.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if labels.numel():
                    labels[:, 1] = 1.0 - labels[:, 1]
        tensor, labels = letterbox(image, labels, self.model_size)
        return {
            "img": tensor,
            "cls": labels[:, :1],
            "bboxes": labels[:, 1:],
            "image": str(record.image),
            "sequence": record.sequence,
        }


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch_indices = []
    classes = []
    boxes = []
    for batch_index, sample in enumerate(samples):
        count = sample["cls"].shape[0]
        if count:
            batch_indices.append(torch.full((count,), batch_index, dtype=torch.float32))
            classes.append(sample["cls"])
            boxes.append(sample["bboxes"])
    return {
        "img": torch.stack([sample["img"] for sample in samples]),
        "batch_idx": torch.cat(batch_indices) if batch_indices else torch.empty(0),
        "cls": torch.cat(classes) if classes else torch.empty((0, 1)),
        "bboxes": torch.cat(boxes) if boxes else torch.empty((0, 4)),
        "images": [sample["image"] for sample in samples],
        "sequences": [sample["sequence"] for sample in samples],
        "valid_count": len(samples),
    }


@dataclass(frozen=True)
class FixedBatchCollator:
    """Pad a final batch without inventing labels; PT2E QAT has a static batch."""

    batch_size: int

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        valid_count = len(samples)
        if valid_count == 0:
            raise ValueError("cannot collate an empty batch")
        padded = list(samples)
        while len(padded) < self.batch_size:
            source = samples[(len(padded) - valid_count) % valid_count]
            padded.append(
                {
                    **source,
                    "cls": torch.empty((0, 1), dtype=torch.float32),
                    "bboxes": torch.empty((0, 4), dtype=torch.float32),
                }
            )
        batch = collate_batch(padded)
        batch["valid_count"] = valid_count
        return batch
