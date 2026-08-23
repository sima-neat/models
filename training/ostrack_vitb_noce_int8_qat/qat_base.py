#!/usr/bin/env python3
"""Deterministic OSTrack Conv/Linear W8A8 fake-quantization primitives."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

QMIN, QMAX = -128, 127


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_determinism(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def inverse_softplus(value: float) -> float:
    value = float(max(value, 1e-6))
    return value if value > 20 else math.log(math.expm1(value))


def ste_round_clip(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, QMIN, QMAX)
    return clipped + (torch.round(clipped) - clipped).detach()


def fake_quantize_weight_per_output(weight: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, weight.ndim))
    maximum = weight.detach().abs().amax(dim=dimensions, keepdim=True).clamp_min(1e-8)
    scale = maximum / 127.0
    return ste_round_clip(weight / scale) * scale


class QATBase:
    quant_mode = "float"

    def _init_qat(self) -> None:
        self.log_clip = nn.Parameter(torch.tensor(inverse_softplus(1.0), dtype=torch.float32))
        self.register_buffer("observed_absmax", torch.tensor(0.0, dtype=torch.float32))

    def set_mode(self, mode: str) -> None:
        self.quant_mode = mode

    def activation_scale(self) -> float:
        return float(F.softplus(self.log_clip.detach()).item() / 127.0)

    def fake_quantize_activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.quant_mode == "float":
            return value
        scale = F.softplus(self.log_clip).clamp_min(1e-6) / 127.0
        return ste_round_clip(value / scale) * scale


class QATLinear(nn.Linear, QATBase):
    @classmethod
    def from_float(cls, module: nn.Linear) -> QATLinear:
        quantized = cls(module.in_features, module.out_features, module.bias is not None)
        quantized.weight = nn.Parameter(module.weight.detach().clone())
        quantized.bias = nn.Parameter(module.bias.detach().clone()) if module.bias is not None else None
        quantized._init_qat()
        return quantized

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_weight_per_output(self.weight) if self.quant_mode == "qat" else self.weight
        return F.linear(self.fake_quantize_activation(value), weight, self.bias)


class QATConv2d(nn.Conv2d, QATBase):
    @classmethod
    def from_float(cls, module: nn.Conv2d) -> QATConv2d:
        quantized = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
        )
        quantized.weight = nn.Parameter(module.weight.detach().clone())
        quantized.bias = nn.Parameter(module.bias.detach().clone()) if module.bias is not None else None
        quantized._init_qat()
        return quantized

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize_weight_per_output(self.weight) if self.quant_mode == "qat" else self.weight
        return self._conv_forward(self.fake_quantize_activation(value), weight, self.bias)


def replace_qat(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QATLinear.from_float(child))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QATConv2d.from_float(child))
        else:
            replace_qat(child)
    return module


def qat_modules(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, (QATLinear, QATConv2d))]


def set_quant_mode(model: nn.Module, mode: str) -> None:
    for _, module in qat_modules(model):
        module.set_mode(mode)


def build_float_model(bundle: str | Path):
    bundle = Path(bundle).resolve()
    sys.path.insert(0, str(bundle / "source"))
    from lib.config.ostrack.config import cfg, update_config_from_file

    update_config_from_file(str(bundle / "source/experiments/ostrack/vitb_256_mae_32x4_ep300.yaml"))
    if cfg.MODEL.BACKBONE.TYPE != "vit_base_patch16_224" or list(cfg.MODEL.BACKBONE.CE_LOC):
        raise RuntimeError("the deployment contract requires ViT-B with CE disabled")
    from lib.models.ostrack import build_ostrack

    model = build_ostrack(cfg, training=False).cpu().eval()
    checkpoint = bundle / "artifacts/OSTrack_ep0300.pth.tar"
    expected = "8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41"
    if checkpoint.stat().st_size != 370_179_249 or sha256_file(checkpoint) != expected:
        raise RuntimeError("OSTrack source checkpoint identity mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["net"], strict=True)
    if str(result) != "<All keys matched successfully>":
        raise RuntimeError(f"strict OSTrack checkpoint load failed: {result}")
    return model
