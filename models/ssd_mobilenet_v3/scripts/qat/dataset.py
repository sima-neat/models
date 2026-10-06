"""COCO input pipeline for SSDLite QAT and evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor


class DetectionCoco(Dataset[tuple[Tensor, dict[str, Tensor]]]):
    """Convert pycocotools annotations into TorchVision detection tensors."""

    def __init__(self, images: Path, annotations: Path, *, horizontal_flip: bool) -> None:
        self.dataset = CocoDetection(str(images), str(annotations))
        self.coco = self.dataset.coco
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, annotations = self.dataset[index]
        width, height = image.size
        boxes = []
        labels = []
        areas = []
        crowds = []
        for annotation in annotations:
            x, y, box_width, box_height = annotation["bbox"]
            x1 = min(max(float(x), 0.0), float(width))
            y1 = min(max(float(y), 0.0), float(height))
            x2 = min(max(float(x + box_width), 0.0), float(width))
            y2 = min(max(float(y + box_height), 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(int(annotation["category_id"]))
            areas.append(float(annotation.get("area", box_width * box_height)))
            crowds.append(int(annotation.get("iscrowd", 0)))

        image_tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.horizontal_flip and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            if boxes_tensor.numel():
                left = width - boxes_tensor[:, 2]
                right = width - boxes_tensor[:, 0]
                boxes_tensor[:, 0] = left
                boxes_tensor[:, 2] = right

        image_id = int(self.dataset.ids[index])
        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        }
        return image_tensor, target


def build_coco_dataset(root: Path, split: str, *, training: bool) -> DetectionCoco:
    if split not in {"train2017", "val2017"}:
        raise ValueError(f"unsupported COCO split: {split}")
    images = root / split
    annotations = root / "annotations" / f"instances_{split}.json"
    if not images.is_dir() or not annotations.is_file():
        raise FileNotFoundError(f"COCO {split} not found; expected images in {images} and annotations at {annotations}")
    return DetectionCoco(images, annotations, horizontal_flip=training)


def deterministic_subset(dataset: Dataset[Any], limit: int | None, seed: int) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def collate_detection(
    batch: Sequence[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
