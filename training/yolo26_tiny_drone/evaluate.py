"""Accuracy evaluation for the exact raw-head deployment graph."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from contract import DEFAULT_MODEL_SIZE
from torch import Tensor, nn
from torchvision.ops import box_iou, nms

AP_CANDIDATE_SCORE_THRESHOLD = 0.001
MAX_NMS_CANDIDATES = 30_000
MAX_DETECTIONS_PER_IMAGE = 300


def _targets_for_image(batch: dict[str, Any], image_index: int, model_size: int) -> Tensor:
    mask = batch["batch_idx"].long() == image_index
    normalized = batch["bboxes"][mask]
    if not normalized.numel():
        return torch.empty((0, 4), device=normalized.device)
    center_x, center_y, width, height = normalized.unbind(1)
    return torch.stack(
        (
            (center_x - width * 0.5) * model_size,
            (center_y - height * 0.5) * model_size,
            (center_x + width * 0.5) * model_size,
            (center_y + height * 0.5) * model_size,
        ),
        dim=1,
    )


def decode(
    predictions: dict[str, Any],
    head: nn.Module,
    score_threshold: float,
    nms_iou: float,
    max_nms_candidates: int,
    max_detections: int,
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
            if image_scores.numel() > max_nms_candidates:
                candidates = image_scores.topk(max_nms_candidates).indices
                image_boxes = image_boxes[candidates]
                image_scores = image_scores[candidates]
            selected = nms(image_boxes, image_scores, nms_iou)
            image_boxes = image_boxes[selected[:max_detections]]
            image_scores = image_scores[selected[:max_detections]]
        decoded.append((image_boxes, image_scores))
    return decoded


def _match_truth_indices(predicted: Tensor, truth: Tensor, threshold: float) -> list[int | None]:
    if predicted.shape[0] == 0:
        return []
    if truth.shape[0] == 0:
        return [None] * predicted.shape[0]
    overlaps = box_iou(predicted, truth)
    matched_truth: set[int] = set()
    assignments = []
    for prediction_index in range(predicted.shape[0]):
        values, indices = overlaps[prediction_index].sort(descending=True)
        assignment = None
        for value, truth_index in zip(values.tolist(), indices.tolist()):
            if value < threshold:
                break
            if truth_index not in matched_truth:
                matched_truth.add(truth_index)
                assignment = truth_index
                break
        assignments.append(assignment)
    return assignments


def _match_image(predicted: Tensor, truth: Tensor, threshold: float) -> list[bool]:
    return [truth_index is not None for truth_index in _match_truth_indices(predicted, truth, threshold)]


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
    score_threshold: float = AP_CANDIDATE_SCORE_THRESHOLD,
    nms_iou: float = 0.60,
    max_nms_candidates: int = MAX_NMS_CANDIDATES,
    max_detections: int = MAX_DETECTIONS_PER_IMAGE,
    model_size: int = DEFAULT_MODEL_SIZE,
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
        decoded = decode(
            outputs["predictions"],
            head,
            score_threshold,
            nms_iou,
            max_nms_candidates,
            max_detections,
        )
        for image_index, (predicted_boxes, predicted_scores) in enumerate(
            decoded[: int(batch.get("valid_count", len(decoded)))]
        ):
            truth = _targets_for_image(batch, image_index, model_size)
            order = predicted_scores.argsort(descending=True)
            predicted_boxes = predicted_boxes[order]
            predicted_scores = predicted_scores[order]
            assignments_50 = _match_truth_indices(predicted_boxes, truth, 0.50)
            image_matches_50 = [truth_index is not None for truth_index in assignments_50]
            image_matches_30 = _match_image(predicted_boxes, truth, 0.30)
            scores.extend(predicted_scores.cpu().tolist())
            matches_50.extend(image_matches_50)
            matches_30.extend(image_matches_30)
            ground_truth_count += truth.shape[0]

            if truth.numel():
                area = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
                tiny_indices = set(torch.where(area < 16.0**2)[0].tolist())
                tiny_ground_truth += len(tiny_indices)
                tiny_matched_50 += len({index for index in assignments_50 if index in tiny_indices})

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
