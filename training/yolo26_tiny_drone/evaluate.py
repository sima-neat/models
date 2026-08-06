"""Accuracy evaluation for the exact raw-head deployment graph."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from contract import MODEL_SIZE
from torch import Tensor, nn
from torchvision.ops import box_iou, nms


def _targets_for_image(batch: dict[str, Any], image_index: int) -> Tensor:
    mask = batch["batch_idx"].long() == image_index
    normalized = batch["bboxes"][mask]
    if not normalized.numel():
        return torch.empty((0, 4), device=normalized.device)
    center_x, center_y, width, height = normalized.unbind(1)
    return torch.stack(
        (
            (center_x - width * 0.5) * MODEL_SIZE,
            (center_y - height * 0.5) * MODEL_SIZE,
            (center_x + width * 0.5) * MODEL_SIZE,
            (center_y + height * 0.5) * MODEL_SIZE,
        ),
        dim=1,
    )


def decode(
    predictions: dict[str, Any], head: nn.Module, score_threshold: float, nms_iou: float
) -> list[tuple[Tensor, Tensor]]:
    one2one = predictions["one2one"]
    # The Detect anchor cache is a plain tensor attribute. PT2E preparation can
    # restore that cache to CPU even when the QAT graph remains on CUDA.
    device = one2one["boxes"].device
    head.anchors = head.anchors.to(device)
    head.strides = head.strides.to(device)
    boxes = head._get_decode_boxes(one2one).permute(0, 2, 1)
    scores = one2one["scores"].sigmoid().squeeze(1)
    decoded = []
    for image_boxes, image_scores in zip(boxes, scores):
        keep = image_scores >= score_threshold
        image_boxes = image_boxes[keep]
        image_scores = image_scores[keep]
        if image_scores.numel():
            selected = nms(image_boxes, image_scores, nms_iou)
            image_boxes = image_boxes[selected]
            image_scores = image_scores[selected]
        decoded.append((image_boxes, image_scores))
    return decoded


def _match_image(predicted: Tensor, truth: Tensor, threshold: float) -> list[bool]:
    if predicted.shape[0] == 0:
        return []
    if truth.shape[0] == 0:
        return [False] * predicted.shape[0]
    overlaps = box_iou(predicted, truth)
    matched_truth: set[int] = set()
    matches = []
    for prediction_index in range(predicted.shape[0]):
        values, indices = overlaps[prediction_index].sort(descending=True)
        matched = False
        for value, truth_index in zip(values.tolist(), indices.tolist()):
            if value < threshold:
                break
            if truth_index not in matched_truth:
                matched_truth.add(truth_index)
                matched = True
                break
        matches.append(matched)
    return matches


def _average_precision(scores: list[float], matches: list[bool], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    true_positive = np.asarray(matches, dtype=np.float64)[order]
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(1.0 - true_positive)
    recall = cumulative_tp / ground_truth_count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(
        np.mean([np.max(precision[recall >= threshold], initial=0.0) for threshold in np.linspace(0.0, 1.0, 101)])
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    head: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    score_threshold: float = 0.05,
    nms_iou: float = 0.60,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    matches_50: list[bool] = []
    matches_30: list[bool] = []
    ground_truth_count = 0
    tiny_ground_truth = 0
    tiny_matched_50 = 0
    for batch in loader:
        images = batch["img"].to(device, non_blocking=True)
        for key in ("batch_idx", "cls", "bboxes"):
            batch[key] = batch[key].to(device, non_blocking=True)
        outputs = model(images)
        decoded = decode(outputs["predictions"], head, score_threshold, nms_iou)
        for image_index, (predicted_boxes, predicted_scores) in enumerate(
            decoded[: int(batch.get("valid_count", len(decoded)))]
        ):
            truth = _targets_for_image(batch, image_index)
            order = predicted_scores.argsort(descending=True)
            predicted_boxes = predicted_boxes[order]
            predicted_scores = predicted_scores[order]
            image_matches_50 = _match_image(predicted_boxes, truth, 0.50)
            image_matches_30 = _match_image(predicted_boxes, truth, 0.30)
            scores.extend(predicted_scores.cpu().tolist())
            matches_50.extend(image_matches_50)
            matches_30.extend(image_matches_30)
            ground_truth_count += truth.shape[0]

            if truth.numel():
                area = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
                tiny_indices = set(torch.where(area < 16.0**2)[0].tolist())
                tiny_ground_truth += len(tiny_indices)
                if tiny_indices and predicted_boxes.numel():
                    overlaps = box_iou(predicted_boxes, truth)
                    matched_truth: set[int] = set()
                    for prediction_index in range(predicted_boxes.shape[0]):
                        value, truth_index = overlaps[prediction_index].max(0)
                        index = int(truth_index)
                        if float(value) >= 0.50 and index not in matched_truth:
                            matched_truth.add(index)
                    tiny_matched_50 += len(matched_truth & tiny_indices)

    if ground_truth_count == 0:
        raise RuntimeError("validation set contains no drone annotations")
    true_positive_50 = sum(matches_50)
    true_positive_30 = sum(matches_30)
    return {
        "ap50": _average_precision(scores, matches_50, ground_truth_count),
        "precision50": true_positive_50 / len(matches_50) if matches_50 else 0.0,
        "recall50": true_positive_50 / ground_truth_count,
        "recall30": true_positive_30 / ground_truth_count,
        "tiny_recall50": tiny_matched_50 / tiny_ground_truth if tiny_ground_truth else 0.0,
        "ground_truth_count": float(ground_truth_count),
        "tiny_ground_truth_count": float(tiny_ground_truth),
    }
