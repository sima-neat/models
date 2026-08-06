"""YOLO26-P2 raw-head wrapper shared by QAT training and deployment export."""

from __future__ import annotations

import copy
from typing import Any

import torch
from contract import DEFAULT_CONTRACT, DeploymentContract
from torch import Tensor, nn


def fuse_batch_norm_for_qat(detector: nn.Module) -> nn.Module:
    """Fold mature BatchNorm layers without discarding YOLO26's training head.

    Ultralytics ``DetectionModel.fuse()`` intentionally deletes the one-to-many
    head used by the loss. QAT needs both heads, so this applies the same module
    folding while deliberately leaving ``Detect.fuse()`` untouched.
    """
    from ultralytics.nn.modules import Conv, Conv2, ConvTranspose, DWConv, RepConv, RepVGGDW
    from ultralytics.utils.torch_utils import fuse_conv_and_bn, fuse_deconv_and_bn

    detector.eval()
    for module in detector.model.modules():
        if isinstance(module, Conv | Conv2 | DWConv) and hasattr(module, "bn"):
            if isinstance(module, Conv2):
                module.fuse_convs()
            module.conv = fuse_conv_and_bn(module.conv, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
        elif isinstance(module, ConvTranspose) and hasattr(module, "bn"):
            module.conv_transpose = fuse_deconv_and_bn(module.conv_transpose, module.bn)
            delattr(module, "bn")
            module.forward = module.forward_fuse
        elif isinstance(module, RepConv):
            module.fuse_convs()
            module.forward = module.forward_fuse
        elif isinstance(module, RepVGGDW):
            module.fuse()
            module.forward = module.forward_fuse
    if any(isinstance(module, nn.BatchNorm2d) for module in detector.modules()):
        raise RuntimeError("BatchNorm remains after QAT fusion")
    return detector


class Yolo26TrainingDeployWrapper(nn.Module):
    """Expose loss inputs plus grouped one-to-one raw heads in one traceable graph."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector
        head = detector.model[-1]
        if not getattr(head, "end2end", False) or head.nl != 4 or head.reg_max != 1 or head.nc != 1:
            raise ValueError("expected a one-class end-to-end YOLO26 P2 head with reg_max=1")

    def _features(self, images: Tensor) -> list[Tensor]:
        outputs: list[Tensor | None] = []
        value: Any = images
        for module in self.detector.model[:-1]:
            if module.f != -1:
                value = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [value if index == -1 else outputs[index] for index in module.f]
                )
            value = module(value)
            outputs.append(value if module.i in self.detector.save else None)
        head = self.detector.model[-1]
        features = [outputs[index] for index in head.f]
        if any(feature is None for feature in features):
            raise RuntimeError("YOLO26 head input was not retained by the model save list")
        return features  # type: ignore[return-value]

    @staticmethod
    def _branch(
        features: list[Tensor], box_heads: nn.ModuleList, class_heads: nn.ModuleList
    ) -> tuple[dict[str, Any], tuple[Tensor, ...], tuple[Tensor, ...]]:
        # Keep this indexing explicit: torch.export in the SDK-pinned PyTorch
        # 2.3 stack cannot always specialize zip(ModuleList, list) after fusion.
        raw_boxes = tuple(box_heads[index](features[index]) for index in range(4))
        raw_classes = tuple(class_heads[index](features[index]) for index in range(4))
        batch_size = features[0].shape[0]
        predictions = {
            "boxes": torch.cat([value.view(batch_size, 4, -1) for value in raw_boxes], dim=-1),
            "scores": torch.cat([value.view(batch_size, 1, -1) for value in raw_classes], dim=-1),
            "feats": features,
        }
        return predictions, raw_boxes, raw_classes

    def forward(self, images: Tensor) -> dict[str, Any]:
        features = self._features(images)
        head = self.detector.model[-1]
        one2many, _unused_boxes, _unused_classes = self._branch(features, head.cv2, head.cv3)
        detached = [feature.detach() for feature in features]
        one2one, raw_boxes, raw_classes = self._branch(detached, head.one2one_cv2, head.one2one_cv3)
        return {
            "predictions": {"one2many": one2many, "one2one": one2one},
            "deploy": raw_boxes + raw_classes,
        }


class DeployOutputs(nn.Module):
    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        return self.wrapped(images)["deploy"]


def validate_deploy_outputs(
    outputs: tuple[Tensor, ...],
    batch_size: int,
    contract: DeploymentContract = DEFAULT_CONTRACT,
) -> None:
    expected = contract.output_shapes(batch_size)
    actual = [tuple(value.shape) for value in outputs]
    if actual != expected:
        raise RuntimeError(f"raw-head output contract mismatch: expected {expected}, got {actual}")
    spatial = tuple(shape[-1] for shape in actual[:4])
    if spatial != contract.feature_sizes:
        raise RuntimeError(f"expected P2-P5 feature sizes {contract.feature_sizes}, got {spatial}")


def select_deploy_graph(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Prune loss-only outputs while retaining the finalized QDQ graph."""
    selected = copy.deepcopy(graph_module)
    output_nodes = [node for node in selected.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise RuntimeError(f"expected one FX output node, found {len(output_nodes)}")
    flattened = output_nodes[0].args[0]
    if not isinstance(flattened, list | tuple) or len(flattened) != 20:
        raise RuntimeError(f"unexpected finalized YOLO26 output tree: {flattened!r}")
    output_nodes[0].args = (tuple(flattened[-8:]),)
    # capture_pre_autograd_graph installs a pytree code generator for the
    # training dictionary. The deployment graph has a plain eight-tensor ABI.
    selected.graph.set_codegen(torch.fx.graph.CodeGen())
    selected.graph.eliminate_dead_code()
    selected.recompile()
    return selected
