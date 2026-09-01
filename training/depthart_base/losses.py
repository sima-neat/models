"""Scale/shift-invariant teacher and metric-depth supervision for QAT."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def nyu_eval_crop_mask(valid: torch.Tensor) -> torch.Tensor:
    """Restrict a resized NYUv2 mask to the official Eigen evaluation crop.

    The labeled NYUv2 frames are 480x640, while DepthART QAT supervises the
    fixed 448x576 deployment tensor.  Scale the canonical [45:471, 41:601]
    crop onto the current tensor with rounded pixel boundaries.  Returning a
    new mask keeps the dataset adapter generic and makes metric-aligned
    supervision an explicit training policy.
    """
    if valid.ndim < 2:
        raise ValueError(f"NYUv2 crop expects an image mask, found {valid.shape}")
    height, width = int(valid.shape[-2]), int(valid.shape[-1])
    top = round(45 * height / 480)
    bottom = round(471 * height / 480)
    left = round(41 * width / 640)
    right = round(601 * width / 640)
    crop = torch.zeros_like(valid, dtype=torch.bool)
    crop[..., top:bottom, left:right] = True
    return valid.bool() & crop


def standardize(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    std = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return (value - mean) / std


def gradient_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    sx, sy = student[..., :, 1:] - student[..., :, :-1], student[..., 1:, :] - student[..., :-1, :]
    tx, ty = teacher[..., :, 1:] - teacher[..., :, :-1], teacher[..., 1:, :] - teacher[..., :-1, :]
    return F.smooth_l1_loss(sx, tx) + F.smooth_l1_loss(sy, ty)


def feature_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match an intermediate feature without letting large channels dominate.

    DepthART's strict graph requantizes every selective-scan arithmetic edge.
    Output-only distillation consequently sends its useful gradient through
    thousands of STE boundaries.  Stage feature matching provides shorter
    supervision paths while retaining both feature magnitude and spatial
    structure.  The teacher RMS only normalizes the loss scale; it is detached
    and therefore cannot become a student escape hatch.
    """
    if isinstance(student, (tuple, list)):
        student = student[0]
    if isinstance(teacher, (tuple, list)):
        teacher = teacher[0]
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError("feature distillation expects tensor stage outputs")
    if student.shape != teacher.shape:
        raise RuntimeError(
            f"student/teacher stage shape mismatch: {tuple(student.shape)} vs "
            f"{tuple(teacher.shape)}"
        )
    teacher = teacher.detach()
    # One normalization value per sample keeps the released cross-channel
    # ratios observable to the loss, unlike per-channel standardization.
    dims = tuple(range(1, teacher.ndim))
    rms = teacher.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1.0e-4)
    value = F.smooth_l1_loss(student / rms, teacher / rms)
    if student.ndim >= 4:
        value = value + 0.10 * gradient_loss(student / rms, teacher / rms)
    return value


def teacher_distillation(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_n, teacher_n = standardize(student), standardize(teacher.detach())
    return F.smooth_l1_loss(student_n, teacher_n) + 0.25 * gradient_loss(student_n, teacher_n)


def affine_inverse_depth_loss(
    prediction: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Align relative prediction to inverse metric depth, then apply robust L1."""
    losses = []
    for pred_i, depth_i, mask_i in zip(prediction, depth, valid):
        mask = mask_i.bool() & torch.isfinite(depth_i) & (depth_i > 0.01) & (depth_i < 10.0)
        if mask.sum() < 32:
            continue
        x = pred_i[mask]
        y = depth_i[mask].reciprocal()
        # Solve [scale, shift] in closed form. Detach the fit so gradients tune
        # the predicted field rather than exploiting the small linear solve.
        xm, ym = x.mean(), y.mean()
        scale = ((x - xm) * (y - ym)).sum() / ((x - xm).square().sum() + 1e-6)
        shift = ym - scale * xm
        aligned = scale.detach() * x + shift.detach()
        scale_y = y.std(unbiased=False).clamp_min(1e-4)
        losses.append(F.smooth_l1_loss(aligned / scale_y, y / scale_y))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def nyu_metric_surrogate_loss(
    prediction: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    delta1_temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable surrogate for the aligned NYUv2 acceptance metrics.

    Evaluation first fits an affine map from relative prediction to inverse
    metric depth, then reports depth-domain AbsRel, log-RMSE, delta1, and edge
    fidelity.  Optimizing only inverse-depth SmoothL1 is not equivalent to
    optimizing those gates.  This loss uses the same detached affine fit and
    directly penalizes the depth-domain quantities, including a smooth hinge
    at the exact ``log(1.25)`` delta1 boundary.
    """

    if not 0.0 < delta1_temperature <= 1.0:
        raise ValueError("delta1_temperature must be in (0, 1]")
    losses = []
    threshold = prediction.new_tensor(1.25).log()
    for pred_i, depth_i, mask_i in zip(prediction, depth, valid):
        mask = (
            mask_i.bool()
            & torch.isfinite(depth_i)
            & (depth_i > 0.01)
            & (depth_i < 10.0)
        )
        if mask.sum() < 32:
            continue
        x = pred_i[mask]
        target_depth = depth_i[mask]
        target_disparity = target_depth.reciprocal()
        xm, ym = x.mean(), target_disparity.mean()
        scale = ((x - xm) * (target_disparity - ym)).sum() / (
            (x - xm).square().sum() + 1.0e-6
        )
        shift = ym - scale * xm
        aligned_disparity = (
            scale.detach() * x + shift.detach()
        ).clamp_min(0.1)
        predicted_depth = aligned_disparity.reciprocal()
        log_ratio = predicted_depth.log() - target_depth.log()
        abs_rel = ((predicted_depth - target_depth).abs() / target_depth).mean()
        log_rmse = (log_ratio.square().mean() + 1.0e-8).sqrt()
        delta1_hinge = (
            F.softplus(
                (log_ratio.abs() - threshold) / delta1_temperature
            ).mean()
            * delta1_temperature
        )

        # Match local log-depth structure only across pairs with two valid
        # pixels.  This targets depth boundaries without creating gradients at
        # invalid-mask edges.
        predicted_map = torch.zeros_like(pred_i)
        target_map = torch.zeros_like(depth_i)
        predicted_map[mask] = predicted_depth.log()
        target_map[mask] = target_depth.log()
        valid_x = mask[..., :, 1:] & mask[..., :, :-1]
        valid_y = mask[..., 1:, :] & mask[..., :-1, :]
        predicted_x = predicted_map[..., :, 1:] - predicted_map[..., :, :-1]
        target_x = target_map[..., :, 1:] - target_map[..., :, :-1]
        predicted_y = predicted_map[..., 1:, :] - predicted_map[..., :-1, :]
        target_y = target_map[..., 1:, :] - target_map[..., :-1, :]
        edge_terms = []
        if valid_x.any():
            edge_terms.append(F.smooth_l1_loss(predicted_x[valid_x], target_x[valid_x]))
        if valid_y.any():
            edge_terms.append(F.smooth_l1_loss(predicted_y[valid_y], target_y[valid_y]))
        edge = (
            torch.stack(edge_terms).mean()
            if edge_terms else prediction.sum() * 0.0
        )
        losses.append(abs_rel + log_rmse + 0.5 * delta1_hinge + 0.1 * edge)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()
