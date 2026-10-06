"""TorchVision SSDLite raw heads and the training-only detector adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import torch
from contract import ANCHORS_PER_LOCATION, NUM_CLASSES, OUTPUTS, output_shapes
from torch import Tensor, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops

WEIGHTS = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
WEIGHTS_URL = WEIGHTS.url
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_NORMALIZATION_COUNT = 70


def validate_cached_weights() -> None:
    filename = Path(urlparse(WEIGHTS_URL).path).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.is_file():
        raise RuntimeError(f"TorchVision did not cache the expected SSDLite weights: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError(f"TorchVision SSDLite weights SHA-256 mismatch: {path}")


class RawSSDLite(nn.Module):
    """The deployable SSDLite graph: backbone and 12 raw heads."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.regression_heads = detector.head.regression_head.module_list
        self.classification_heads = detector.head.classification_head.module_list

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        features = list(self.backbone(images).values())
        locations = [head(feature) for head, feature in zip(self.regression_heads, features)]
        confidences = [head(feature) for head, feature in zip(self.classification_heads, features)]
        return tuple(locations + confidences)


def build_detector(*, pretrained: bool = True) -> nn.Module:
    weights = WEIGHTS if pretrained else None
    detector = ssdlite320_mobilenet_v3_large(
        weights=weights,
        weights_backbone=None,
    )
    if pretrained:
        validate_cached_weights()
    anchors = tuple(detector.anchor_generator.num_anchors_per_location())
    if anchors != ANCHORS_PER_LOCATION:
        raise RuntimeError(f"TorchVision SSDLite anchor contract changed: {anchors}")
    return detector


def fold_batch_normalization(module: nn.Module) -> int:
    """Fold every adjacent eval Conv2d-BatchNorm2d pair in-place."""

    module.eval()
    folded = sum(fold_batch_normalization(child) for child in module.children())
    if not isinstance(module, nn.Sequential):
        return folded
    names = list(module._modules)
    for convolution_name, batch_norm_name in zip(names, names[1:]):
        convolution = module._modules[convolution_name]
        batch_norm = module._modules[batch_norm_name]
        if isinstance(convolution, nn.Conv2d) and isinstance(batch_norm, nn.BatchNorm2d):
            module._modules[convolution_name] = fuse_conv_bn_eval(convolution, batch_norm)
            module._modules[batch_norm_name] = nn.Identity()
            folded += 1
    return folded


def build_raw_model(*, pretrained: bool = True) -> tuple[RawSSDLite, nn.Module]:
    detector = build_detector(pretrained=pretrained)
    folded = fold_batch_normalization(detector)
    remaining = sum(isinstance(module, nn.BatchNorm2d) for module in detector.modules())
    if folded != BATCH_NORMALIZATION_COUNT or remaining:
        raise RuntimeError(f"TorchVision BatchNorm folding contract changed: folded={folded}, remaining={remaining}")
    raw_model = RawSSDLite(detector)
    return raw_model, detector


def validate_raw_outputs(outputs: Sequence[Tensor], batch_size: int) -> None:
    observed = [tuple(output.shape) for output in outputs]
    expected = output_shapes(batch_size)
    if observed != expected:
        raise RuntimeError(f"SSDLite raw output mismatch: {observed}; expected {expected}")


def flatten_heads(outputs: Sequence[Tensor], columns: int) -> Tensor:
    flattened = []
    for output in outputs:
        batch, _channels, height, width = output.shape
        flattened.append(
            output.view(batch, -1, columns, height, width).permute(0, 3, 4, 1, 2).reshape(batch, -1, columns)
        )
    return torch.cat(flattened, dim=1)


def as_head_outputs(outputs: Sequence[Tensor]) -> dict[str, Tensor]:
    if len(outputs) != len(OUTPUTS):
        raise RuntimeError(f"expected {len(OUTPUTS)} raw heads, received {len(outputs)}")
    return {
        "bbox_regression": flatten_heads(outputs[:6], 4),
        "cls_logits": flatten_heads(outputs[6:], NUM_CLASSES),
    }


def resize_batch(
    detector: nn.Module,
    images: Sequence[Tensor],
    targets: Sequence[dict[str, Tensor]] | None,
) -> tuple[ImageList, list[dict[str, Tensor]] | None, list[tuple[int, int]]]:
    original_sizes = [tuple(image.shape[-2:]) for image in images]
    resized_images = []
    resized_targets = [] if targets is not None else None
    for index, image in enumerate(images):
        target = None if targets is None else targets[index]
        normalized = detector.transform.normalize(image)
        resized, resized_target = detector.transform.resize(normalized, target)
        resized_images.append(resized)
        if resized_targets is not None:
            if resized_target is None:
                raise RuntimeError("detector transform discarded a training target")
            resized_targets.append(resized_target)
    image_sizes = [tuple(image.shape[-2:]) for image in resized_images]
    tensors = detector.transform.batch_images(resized_images)
    return ImageList(tensors, image_sizes), resized_targets, original_sizes


def compute_losses(
    detector: nn.Module,
    image_list: ImageList,
    targets: Sequence[dict[str, Tensor]],
    outputs: Sequence[Tensor],
) -> dict[str, Tensor]:
    head_outputs = as_head_outputs(outputs)
    feature_proxies = list(outputs[:6])
    anchors = detector.anchor_generator(image_list, feature_proxies)
    matched_indices = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_indices.append(
                torch.full(
                    (anchors_per_image.size(0),),
                    -1,
                    dtype=torch.int64,
                    device=anchors_per_image.device,
                )
            )
        else:
            quality = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_indices.append(detector.proposal_matcher(quality))
    return detector.compute_loss(list(targets), head_outputs, anchors, matched_indices)


def decode_detections(
    detector: nn.Module,
    image_list: ImageList,
    original_sizes: Sequence[tuple[int, int]],
    outputs: Sequence[Tensor],
) -> list[dict[str, Tensor]]:
    anchors = detector.anchor_generator(image_list, list(outputs[:6]))
    detections = detector.postprocess_detections(as_head_outputs(outputs), anchors, image_list.image_sizes)
    return detector.transform.postprocess(detections, image_list.image_sizes, list(original_sizes))
