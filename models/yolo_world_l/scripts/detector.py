"""Inference-only YOLO-World v2, loaded strictly from its original checkpoint.

The layer order follows Tencent's GPL-3.0 YOLO-World and MMYOLO implementations.
Keeping this adapter in PyTorch avoids MMCV's unrelated compiled training ops.
Surgery fuses evaluation batch norms and exposes the six YOLOv8 decoder heads.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

OUTPUTS = [f"bbox_dfl_s{s}" for s in (8, 16, 32)] + [f"class_prob_s{s}" for s in (8, 16, 32)]


class Weights:
    def __init__(self, state):
        self.state = state
        self.used = set()

    def take(self, name):
        self.used.add(name)
        return self.state[name]

    def load(self, module, prefix):
        module.load_state_dict({name: self.take(prefix + "." + name) for name in module.state_dict()}, strict=True)
        return module.eval()

    def conv(self, prefix, stride=1):
        shape = self.state[prefix + ".weight"].shape
        if len(shape) != 4 or shape[2] != shape[3]:
            raise ValueError(f"unexpected convolution: {prefix} {shape}")
        return self.load(nn.Conv2d(shape[1], shape[0], shape[2], stride, shape[2] // 2,
                                   bias=prefix + ".bias" in self.state), prefix)

    def batch_norm(self, prefix, channels):
        return self.load(nn.BatchNorm2d(channels, eps=0.001, momentum=0.03), prefix)

    def linear(self, prefix):
        out_channels, in_channels = self.state[prefix + ".weight"].shape
        return self.load(nn.Linear(in_channels, out_channels), prefix)

    def check_detector(self):
        unused = set(self.state) - self.used
        unused = {name for name in unused if not name.startswith("backbone.text_model.")}
        # These non-learned DFL bins are replaced by the decoder's fixed 0..15 bins.
        unused.discard("bbox_head.head_module.proj")
        if unused:
            raise ValueError(f"unconsumed detector checkpoint tensors: {sorted(unused)}")


class Conv(nn.Module):
    def __init__(self, weights, prefix, stride=1, activate=True):
        super().__init__()
        self.conv = weights.conv(prefix + ".conv", stride)
        bn = weights.batch_norm(prefix + ".bn", self.conv.out_channels)
        self.conv = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv, bn)
        self.activate = nn.SiLU() if activate else nn.Identity()

    def forward(self, x):
        return self.activate(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, weights, prefix, shortcut):
        super().__init__()
        self.conv1 = Conv(weights, prefix + ".conv1")
        self.conv2 = Conv(weights, prefix + ".conv2")
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.shortcut else out


class Attention(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.guide_fc = weights.linear(prefix + ".guide_fc")
        self.bias = nn.Parameter(weights.take(prefix + ".bias"))
        self.num_heads = self.bias.numel()
        self.head_channels = self.guide_fc.out_features // self.num_heads
        self.embed_conv = (Conv(weights, prefix + ".embed_conv", activate=False)
                           if prefix + ".embed_conv.conv.weight" in weights.state else None)
        self.project_conv = Conv(weights, prefix + ".project_conv", activate=False)

    def forward(self, x, guide_and_mask):
        guide, mask = guide_and_mask
        b, _, h, w = x.shape
        projected = self.guide_fc(guide).reshape(b, -1, self.num_heads, self.head_channels).permute(0, 2, 3, 1)
        image = self.embed_conv(x) if self.embed_conv is not None else x
        image = image.reshape(b, self.num_heads, self.head_channels, h * w)
        similarity = torch.matmul(image.transpose(-1, -2), projected)
        # For a nonempty binary mask, sigmoid(max_valid(z)) == max(sigmoid(z) * mask).
        weight = torch.sigmoid(similarity / self.head_channels**0.5 + self.bias[None, :, None, None])
        weight = (weight * mask[:, None, None, :]).amax(dim=-1).reshape(b, self.num_heads, h, w)
        output = self.project_conv(x).reshape(b, self.num_heads, -1, h, w)
        return (output * weight.unsqueeze(2)).reshape(b, -1, h, w)


class CSP(nn.Module):
    def __init__(self, weights, prefix, shortcut=True, attention=False):
        super().__init__()
        self.main_conv = Conv(weights, prefix + ".main_conv")
        self.final_conv = Conv(weights, prefix + ".final_conv")
        self.mid_channels = self.main_conv.conv.out_channels // 2
        blocks = {name[len(prefix + ".blocks."):].split(".", 1)[0] for name in weights.state
                  if name.startswith(prefix + ".blocks.")}
        self.blocks = nn.ModuleList(Bottleneck(weights, prefix + f".blocks.{i}", shortcut)
                                    for i in range(len(blocks)))
        self.attn_block = Attention(weights, prefix + ".attn_block") if attention else None

    def forward(self, x, guide=None):
        parts = list(self.main_conv(x).split((self.mid_channels, self.mid_channels), 1))
        parts.extend(block(parts[-1]) for block in self.blocks)
        if self.attn_block is not None:
            parts.append(self.attn_block(parts[-1], guide))
        return self.final_conv(torch.cat(parts, 1))


class SPPF(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.conv1 = Conv(weights, prefix + ".conv1")
        self.conv2 = Conv(weights, prefix + ".conv2")
        self.poolings = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        a = self.poolings(x)
        b = self.poolings(a)
        return self.conv2(torch.cat((x, a, b, self.poolings(b)), 1))


class Backbone(nn.Module):
    def __init__(self, weights):
        super().__init__()
        prefix = "backbone.image_model"
        self.stem = Conv(weights, prefix + ".stem", stride=2)
        for i in range(1, 5):
            p = prefix + f".stage{i}"
            modules = [Conv(weights, p + ".0", stride=2), CSP(weights, p + ".1")]
            if i == 4:
                modules.append(SPPF(weights, p + ".2"))
            setattr(self, f"stage{i}", nn.Sequential(*modules))

    def forward(self, x):
        x = self.stage1(self.stem(x))
        a = self.stage2(x)
        b = self.stage3(a)
        return a, b, self.stage4(b)


class Neck(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.top_down_layers = nn.ModuleList(CSP(weights, f"neck.top_down_layers.{i}", False, True) for i in range(2))
        self.bottom_up_layers = nn.ModuleList(CSP(weights, f"neck.bottom_up_layers.{i}", False, True) for i in range(2))
        self.downsample_layers = nn.ModuleList(Conv(weights, f"neck.downsample_layers.{i}", stride=2) for i in range(2))
        self.upsample_layers = nn.ModuleList(nn.Upsample(scale_factor=2, mode="nearest") for _ in range(2))

    def forward(self, features, guide):
        a, b, c = features
        b = self.top_down_layers[0](torch.cat((self.upsample_layers[0](c), b), 1), guide)
        a = self.top_down_layers[1](torch.cat((self.upsample_layers[1](b), a), 1), guide)
        b = self.bottom_up_layers[0](torch.cat((self.downsample_layers[0](a), b), 1), guide)
        c = self.bottom_up_layers[1](torch.cat((self.downsample_layers[1](b), c), 1), guide)
        return a, b, c


class Detector(nn.Module):
    def __init__(self, state):
        super().__init__()
        weights = Weights(state)
        self.image_model = Backbone(weights)
        self.neck = Neck(weights)
        self.reg_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.logit_scales = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(3):
            prefix = "bbox_head.head_module"
            for name, target in (("reg_preds", self.reg_preds), ("cls_preds", self.cls_preds)):
                p = prefix + f".{name}.{i}"
                layers = [Conv(weights, p + ".0"), Conv(weights, p + ".1"), weights.conv(p + ".2")]
                if name == "cls_preds":
                    bn = weights.batch_norm(prefix + f".cls_contrasts.{i}.norm", layers[-1].out_channels)
                    layers[-1] = torch.nn.utils.fusion.fuse_conv_bn_eval(layers[-1], bn)
                target.append(nn.Sequential(*layers))
            self.logit_scales.append(nn.Parameter(weights.take(prefix + f".cls_contrasts.{i}.logit_scale")))
            self.biases.append(nn.Parameter(weights.take(prefix + f".cls_contrasts.{i}.bias")))
        weights.check_detector()
        self.requires_grad_(False)
        self.eval()

    def forward(self, images, text_embeddings, prompt_mask):
        features = self.neck(self.image_model(images), (text_embeddings, prompt_mask))
        text = F.normalize(text_embeddings, p=2, dim=-1).transpose(1, 2)
        boxes, probabilities = [], []
        for i, feature in enumerate(features):
            boxes.append(self.reg_preds[i](feature))
            image = self.cls_preds[i](feature)
            b, _, h, w = image.shape
            score = torch.matmul(image.flatten(2).transpose(1, 2), text).transpose(1, 2).reshape(b, -1, h, w)
            score = score * self.logit_scales[i].exp() + self.biases[i]
            probabilities.append(torch.sigmoid(score) * prompt_mask[:, :, None, None])
        return tuple(boxes + probabilities)
