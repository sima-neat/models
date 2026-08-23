#!/usr/bin/env python3
"""OSTrack target-facing attention, normalization, GELU and residual QAT."""

from __future__ import annotations

import math
import types

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    return value if value > 20 else math.log(math.expm1(max(float(value), 1e-8)))


def ste_quantize(value: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    code = torch.clamp(value / scale, -127, 127)
    return (code + (torch.round(code) - code).detach()) * scale


class TrainableA8(nn.Module):
    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.log_clip = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.register_buffer("observed", torch.tensor(0.0))
        self.mode = "bypass"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode == "bypass":
            return value
        scale = F.softplus(self.log_clip).clamp_min(1e-7) / 127.0
        return ste_quantize(value, scale)

    def scale(self) -> float:
        return float(F.softplus(self.log_clip.detach()) / 127.0)


class FixedProbabilityQ7(nn.Module):
    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.register_buffer("scale", torch.tensor(1 / 127, dtype=torch.float32))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

    def forward(self, probability: torch.Tensor) -> torch.Tensor:
        return ste_quantize(probability, self.scale)


class QATLayerNorm(nn.Module):
    def __init__(self, module: nn.LayerNorm, name: str):
        super().__init__()
        self.normalized_shape = module.normalized_shape
        self.eps = module.eps
        self.elementwise_affine = module.elementwise_affine
        if module.elementwise_affine:
            self.weight = nn.Parameter(module.weight.detach().clone())
            self.bias = nn.Parameter(module.bias.detach().clone())
        self.out_quant = TrainableA8(name + ".output")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_quant(F.layer_norm(value, self.normalized_shape, self.weight, self.bias, self.eps))


class QATGELU(nn.Module):
    def __init__(self, module: nn.GELU, name: str):
        super().__init__()
        self.approximate = module.approximate
        self.input_quant = TrainableA8(name + ".input")
        self.output_quant = TrainableA8(name + ".output")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_quant(F.gelu(self.input_quant(value), approximate=self.approximate))


def replace_ln_gelu(module: nn.Module, prefix: str = ""):
    layer_norms, gelus = [], []
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, QATLayerNorm(child, full))
            layer_norms.append(full)
        elif isinstance(child, nn.GELU):
            setattr(module, name, QATGELU(child, full))
            gelus.append(full)
        else:
            nested_ln, nested_gelu = replace_ln_gelu(child, full)
            layer_norms.extend(nested_ln)
            gelus.extend(nested_gelu)
    return layer_norms, gelus


def _physical_qk(value: torch.Tensor, base: float) -> torch.Tensor:
    float_value = value.float()
    rounded = torch.sign(float_value) * torch.floor(torch.abs(float_value / base) + 0.5)
    rounded = torch.clamp(rounded, -32767, 32767)
    row = torch.clamp(rounded.abs().amax(-1, keepdim=True), 128, 32767)
    index = torch.clamp(torch.floor(row / 128), 1, 255)
    power = torch.pow(
        torch.tensor(2.0, device=value.device),
        torch.clamp(torch.ceil(torch.log2(index * 128)) - 1, min=0),
    )
    code = torch.clamp(torch.floor(rounded * 64 / power + 0.5), -127, 127)
    dequantized = (code * (power * base / 64)).to(value.dtype)
    return value + (dequantized - value).detach()


def _quantize_logits_pow2(value: torch.Tensor) -> torch.Tensor:
    maximum = value.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8)
    scale = torch.pow(2.0, torch.ceil(torch.log2(maximum / 127.0)))
    return ste_quantize(value, scale)


def attach_attention_residual(model: nn.Module) -> None:
    for index, block in enumerate(model.backbone.blocks):
        attention = block.attn
        attention.v_quant = TrainableA8(f"block{index}.V")
        attention.prob_q7 = FixedProbabilityQ7(f"block{index}.softmax_P")
        attention.av_out_quant = TrainableA8(f"block{index}.AV_output")
        attention.register_buffer("q_base", torch.tensor(0.0))
        attention.register_buffer("k_base", torch.tensor(0.0))
        attention.register_buffer("q_absmax", torch.tensor(0.0))
        attention.register_buffer("k_absmax", torch.tensor(0.0))
        attention.quant_mode = "qat"

        def attention_forward(self, value, return_attention=False):
            batch, tokens, channels = value.shape
            qkv = (
                self.qkv(value)
                .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            query = _physical_qk(qkv[0], float(self.q_base))
            key = _physical_qk(qkv[1], float(self.k_base))
            val = self.v_quant(qkv[2])
            logits = _quantize_logits_pow2((query @ key.transpose(-2, -1)) * self.scale)
            probability = self.prob_q7(logits.softmax(-1))
            output = self.av_out_quant(probability @ val).transpose(1, 2).reshape(batch, tokens, channels)
            output = self.proj_drop(self.proj(output))
            return (output, probability) if return_attention else output

        attention.forward = types.MethodType(attention_forward, attention)
        block.res1_quant = TrainableA8(f"block{index}.residual1")
        block.res2_quant = TrainableA8(f"block{index}.residual2")

        def block_forward(self, value, return_attention=False):
            if return_attention:
                update, probability = self.attn(self.norm1(value), True)
                value = self.res1_quant(value + self.drop_path(update))
                value = self.res2_quant(value + self.drop_path(self.mlp(self.norm2(value))))
                return value, probability
            value = self.res1_quant(value + self.drop_path(self.attn(self.norm1(value))))
            return self.res2_quant(value + self.drop_path(self.mlp(self.norm2(value))))

        block.forward = types.MethodType(block_forward, block)


def extra_qmods(model: nn.Module):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, TrainableA8)]


def set_qat_mode(base, model: nn.Module) -> None:
    base.set_quant_mode(model, "qat")
    for _, module in extra_qmods(model):
        module.mode = "qat"
    for block in model.backbone.blocks:
        block.attn.quant_mode = "qat"
