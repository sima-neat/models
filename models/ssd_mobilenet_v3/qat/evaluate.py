"""COCO accuracy evaluation for raw TorchVision SSDLite heads."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from typing import Any

import torch
from model import decode_detections, resize_batch, validate_raw_outputs
from pycocotools.cocoeval import COCOeval
from torch import Tensor, nn


def _xyxy_to_xywh(boxes: Tensor) -> Tensor:
    converted = boxes.clone()
    converted[:, 2] -= converted[:, 0]
    converted[:, 3] -= converted[:, 1]
    return converted


@torch.inference_mode()
def evaluate_coco(
    qat_model: nn.Module,
    detector: nn.Module,
    loader: Iterable[tuple[list[Tensor], list[dict[str, Tensor]]]],
    coco_ground_truth: Any,
    device: torch.device,
) -> dict[str, float]:
    qat_model.eval()
    results = []
    image_ids = []
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        image_list, _resized_targets, original_sizes = resize_batch(detector, images, None)
        outputs = qat_model(image_list.tensors)
        validate_raw_outputs(outputs, len(images))
        detections = decode_detections(detector, image_list, original_sizes, outputs)
        for target, detection in zip(targets, detections):
            image_id = int(target["image_id"])
            image_ids.append(image_id)
            boxes = _xyxy_to_xywh(detection["boxes"].detach().cpu())
            scores = detection["scores"].detach().cpu()
            labels = detection["labels"].detach().cpu()
            results.extend(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                }
                for box, score, label in zip(boxes, scores, labels)
            )

    if not results:
        raise RuntimeError("COCO evaluation produced no detections")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_detections = coco_ground_truth.loadRes(results)
        evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
        evaluator.params.imgIds = sorted(set(image_ids))
        evaluator.evaluate()
        evaluator.accumulate()
    evaluator.summarize()
    names = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100")
    return {name: float(value) for name, value in zip(names, evaluator.stats[:9])}
