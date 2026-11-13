"""Build EfficientSAM3 for the compiler from its upstream weights, export ONNX and split it for BF16 compilation.

build_graph re-expresses the upstream computation with four-dimensional token tensors, finite masks and explicit
constants. The token lookup runs as a one-hot MatMul in vocabulary chunks, because the MLA cannot gather from a
49,408-row table. The neck's leading 1x1 and 2x2 transposed convs run on the trunk's 32x32 map before the resize, as
1x1 convs and PixelShuffle; the instance head's 1x1 conv is folded into the mask queries; group norms are computed
with 1x1 convs; BatchNorm and RepViT's branches are fused into their convs. Before reducing the upstream 72x72 token
grid to 48x48, export_model checks the reconstructed graph against the upstream PyTorch reference cases. It then
checks the exported ONNX against the reconstructed PyTorch model at the deployment grid. The grid change's accuracy
impact is measured separately from numerical fidelity at a fixed grid.

The compiler lowers Sin, Cos and Log through BF16 lookup tables. They occur only in the decoder:
Sin and Cos build the sine query embedding from reference boxes (angles in [0, 2*pi]); Log builds the box
relative-position bias from log(1 + relu(t)) with |t| <= 12. Each becomes a polynomial with a stated float32
error bound over its clipped domain, and Log adds Newton steps through Exp. The text encoder is cut out
unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from calibration import CONTEXT, FEATURES, IMAGE_SIZE, VOCABULARY, one_hot, packed, tokens
from onnx import helper, numpy_helper

QUERIES = 200
GRID = 48
MASK_SIZE = 4 * GRID
RUNTIME_FILES = ("vocab.json", "merges.txt")

TEXT_INPUT, TEXT_OUTPUT = "token_one_hot", "text_features"
IMAGE_INPUTS, IMAGE_OUTPUTS = ["pixel_values", "text"], ["detections", "mask_logits"]
INTERFACE = {
    "token_one_hot": [1, CONTEXT, VOCABULARY],
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "text": [1, CONTEXT + 1, FEATURES],
    "text_features": [1, CONTEXT, FEATURES],
    "detections": [1, QUERIES, 6],
    "mask_logits": [1, QUERIES, MASK_SIZE, MASK_SIZE],
}
EXPORT_INPUTS = ("token_one_hot", "pixel_values", "text")
EXPORT_OUTPUTS = ("text_features", "detections", "mask_logits")
# The MLA cannot gather from a 49,408-row table, so the token lookup runs as a one-hot MatMul in this many
# vocabulary chunks.
VOCABULARY_CHUNKS = 16
# Fused BatchNorm, folded attention scales and reordered float32 sums change rounding only; a wrong rewrite moves
# values by O(1). Twelve refinement layers amplify rounding in queries the model rejects, so boxes and masks are
# bounded relative to their scale over all queries and absolutely over confident queries (upstream score >=
# CONFIDENT).
UPSTREAM_TOLERANCE = {
    "text_features": 1e-3,
    "scores": 1e-3,
    "boxes_relative": 1e-3,
    "masks_relative": 1e-3,
    "boxes_px_confident": 0.1,
    "mask_logits_confident": 0.05,
}
# Data-dependent uses in the folded export; onnxsim folds the first decoder layer's.
EXPECTED = {"Sin": 5, "Cos": 5, "Log": 20}

# Near-minimax float64 fits on the domains below, evaluated by Horner's rule.
# sin(theta) = t * A(t^2) and cos(theta) = B(t^2) with t = clip(theta, 0, 2*pi) - pi.
SIN = [
    -0.9999999992796773, 0.16666665931858382, -0.008333321350174647, 0.00019840536491520168,
    -2.7535897092468157e-06, 2.4729237242536458e-08, -1.361453851749336e-10,
]  # fmt: skip
COS = [
    -0.9999999963716656, 0.499999998694325, -0.04166666312537827, 0.0013888859079996924,
    -2.4800407827067865e-05, 2.753229457456087e-07, -2.0583100355451864e-09, 9.658306828471712e-12,
]  # fmt: skip
# log(w) for w = clip(w, 1, 16): y = C(2w/15 - 17/15), then y <- y - 1 + w * exp(-y).
LOG = [
    2.1492067526874274, 0.9237985730862641, -0.587449189721419, -0.047279953972127316,
    0.5108554787988003, 0.498664101231361, -0.6863134993307177,
]  # fmt: skip
NEWTON_STEPS = 3
DOMAINS = {"Sin": (0.0, 2 * np.pi), "Cos": (0.0, 2 * np.pi), "Log": (1.0, 16.0)}

# The replacements' largest float32 error over their domains, which tests/build/test_compile.py checks on a dense
# grid. The text encoder is a pure cut. The replacements sit below float32 rounding, which the refinement layers
# amplify in rejected queries, so boxes and masks are bounded relative to their scale over all queries and
# absolutely over confident ones.
FUNCTION_BOUND = 2e-6
TEXT_TOLERANCE = 1e-5
IMAGE_TOLERANCE = {
    "scores": 1e-4,
    "boxes_relative": 1e-3,
    "masks_relative": 1e-3,
    "boxes_px_confident": 0.1,
    "mask_logits_confident": 0.05,
}
CONFIDENT = 0.05


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported EfficientSAM3 graph: {message}")


def _interface(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = (*model.graph.input, *model.graph.output)
    return {value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim] for value in values}


class _Builder:
    """Emit replacement nodes with unique names and scalar float32 constants."""

    def __init__(self, model: onnx.ModelProto):
        self.model, self.nodes, self.count = model, [], 0

    def name(self, stem):
        self.count += 1
        return f"surgery/{stem}_{self.count}"

    def constant(self, value):
        name = self.name("constant")
        self.model.graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), name))
        return name

    def op(self, op_type, *inputs, output=None):
        output = output or self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, list(inputs), [output], name=self.name(op_type)))
        return output

    def horner(self, coefficients, x, output=None):
        value = self.op("Add", self.op("Mul", x, self.constant(coefficients[-1])), self.constant(coefficients[-2]))
        for index, coefficient in enumerate(reversed(coefficients[:-2])):
            last = index == len(coefficients) - 3
            value = self.op(
                "Add", self.op("Mul", value, x), self.constant(coefficient), output=output if last else None
            )
        return value

    def reduced_angle(self, theta):
        low, high = DOMAINS["Sin"]
        t = self.op("Sub", self.op("Clip", theta, self.constant(low), self.constant(high)), self.constant(np.pi))
        return t, self.op("Mul", t, t)

    def log(self, w, output):
        low, high = DOMAINS["Log"]
        w = self.op("Clip", w, self.constant(low), self.constant(high))
        z = self.op("Add", self.op("Mul", w, self.constant(2 / 15)), self.constant(-17 / 15))
        y = self.horner(LOG, z)
        for step in range(NEWTON_STEPS):
            correction = self.op("Mul", w, self.op("Exp", self.op("Neg", y)))
            y = self.op(
                "Add",
                self.op("Add", y, self.constant(-1.0)),
                correction,
                output=output if step == NEWTON_STEPS - 1 else None,
            )
        return y


def function_inputs(model: onnx.ModelProto) -> dict[str, list[str]]:
    """Return the input names of the Sin, Cos and Log nodes, by operator."""
    return {op: [node.input[0] for node in model.graph.node if node.op_type == op] for op in EXPECTED}


def replace_functions(model: onnx.ModelProto, expected=EXPECTED) -> None:
    """Replace Sin, Cos and Log in place by their polynomial forms."""
    counts = {op: sum(node.op_type == op for node in model.graph.node) for op in expected}
    _require(counts == expected, f"expected {expected} data-dependent functions, found {counts}")
    builder, nodes, angles = _Builder(model), [], {}
    for node in model.graph.node:
        if node.op_type not in expected:
            nodes.append(node)
            continue
        (source,), (target,) = node.input, node.output
        builder.nodes = []
        if node.op_type == "Log":
            builder.log(source, target)
        else:
            # Sin and Cos of one angle tensor share the reduction.
            if source not in angles:
                angles[source] = builder.reduced_angle(source)
            t, u = angles[source]
            if node.op_type == "Sin":
                builder.op("Mul", t, builder.horner(SIN, u), output=target)
            else:
                builder.horner(COS, u, output=target)
        nodes.extend(builder.nodes)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def build_attention(weight, bias, output_weight, output_bias, heads, *, fuse_qkv=False):
    """Build attention on (1, 1, tokens, channels), preserving head order, scale and additive masks.

    The model's seven measured attention shapes use channel-first internals.
    Only text self-attention fuses Q/K/V: its forward takes one shared input instead of three distinct inputs.
    """
    import torch
    from torch import nn

    width = weight.shape[1]
    _require(heads > 0 and width % heads == 0, f"attention width {width} and heads {heads}")
    head_size = width // heads

    def projection(w, b, scale=1.0):
        layer = nn.Conv2d(width, w.shape[0], 1)
        layer.weight.data.copy_(w[:, :, None, None] * scale)
        layer.bias.data.copy_(b * scale)
        return layer.requires_grad_(False)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            if fuse_qkv:
                self.qkv = projection(
                    torch.cat([weight[:width] * head_size**-0.5, weight[width:]]),
                    torch.cat([bias[:width] * head_size**-0.5, bias[width:]]),
                )
            else:
                self.q = projection(weight[:width], bias[:width], head_size**-0.5)
                self.k = projection(weight[width : 2 * width], bias[width : 2 * width])
                self.v = projection(weight[2 * width :], bias[2 * width :])
            self.o = projection(output_weight, output_bias)

        def heads_first(self, x):
            # Channel h*D+d maps to [d, h, token]; concatenating heads reverses this packing.
            return torch.cat(torch.split(x, head_size, dim=1), dim=2)

        def forward(self, query, key=None, value=None, bias=None):
            if fuse_qkv:
                _require(key is None and value is None, "fused self-attention takes one shared Q/K/V input")
                q, k, v = self.qkv(query.permute(0, 3, 1, 2)).split(width, dim=1)
            else:
                _require(key is not None and value is not None, "attention requires key and value inputs")
                query, key, value = (x.permute(0, 3, 1, 2) for x in (query, key, value))
                q, k, v = self.q(query), self.k(key), self.v(value)
            q, k, v = (self.heads_first(x) for x in (q, k, v))
            scores = torch.einsum("nchw,nchq->nqhw", q, k)
            if bias is not None:
                scores = scores + bias.permute(0, 3, 1, 2)
            out = torch.einsum("nchw,nqhc->nqhw", scores.softmax(1), v)
            merged = torch.cat(torch.split(out, 1, dim=2), dim=1)
            return self.o(merged).permute(0, 2, 3, 1)

    return Attention().eval()


def position_encoding(grid):
    """Recompute SAM 3's normalized sine encoding at the deployment grid, without resampling its values.

    convert.py checks the upstream encoding's width, temperature, normalization and scale before bundling it.
    """
    import torch

    coordinates = torch.arange(1, grid + 1, dtype=torch.float32)
    coordinates = coordinates / (coordinates[-1] + 1e-6) * (2 * math.pi)
    dimension = torch.arange(128, dtype=torch.float32)
    frequency = 10000 ** (2 * (dimension // 2) / 128)
    phase = coordinates[:, None] / frequency
    encoded = torch.stack((phase[:, 0::2].sin(), phase[:, 1::2].cos()), dim=2).flatten(1)
    y = encoded[:, None, :].expand(grid, grid, 128)
    x = encoded[None, :, :].expand(grid, grid, 128)
    return torch.cat((y, x), dim=2).permute(2, 0, 1)[None].contiguous()


def build_graph(weights, *, grid=None):
    """Reconstruct the model at its upstream grid, or at an explicit deployment grid."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    state, settings = weights["state_dict"], weights["settings"]
    eps, source_grid = settings["norm_eps"], settings["grid"]
    grid = source_grid if grid is None else grid
    _require(grid > 0, f"token grid {grid}")
    _require(tuple(weights["position_encoding"].shape) == (1, 256, source_grid, source_grid), "position encoding shape")
    grid_tokens = grid * grid

    def constant(value):
        return nn.Parameter(value.detach().clone().float(), requires_grad=False)

    def count(prefix):
        """Return how many numbered modules the upstream model has under a prefix."""
        return len({key[len(prefix) :].split(".")[0] for key in state if key.startswith(prefix)})

    def gelu(x):
        # Exact erf GELU written out, so the compiler does not lower it through its GELU composite.
        return x * (torch.erf(x * 0.7071067811865476) * 0.5 + 0.5)

    class Gelu(nn.Module):
        def forward(self, x):
            return gelu(x)

    class Linear(nn.Module):
        def __init__(self, weight, bias, scale=1.0):
            super().__init__()
            self.weight, self.bias = constant(weight * scale), constant(bias * scale)

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    def linear(name, scale=1.0):
        return Linear(state[f"{name}.weight"], state[f"{name}.bias"], scale)

    def conv(weight, bias, stride=1, groups=1):
        """Return a conv with these weights, padded to keep the size at stride 1."""
        module = nn.Conv2d(
            weight.shape[1] * groups, weight.shape[0], weight.shape[2:], stride, weight.shape[2] // 2, groups=groups
        )
        module.weight.data.copy_(weight)
        module.bias.data.copy_(bias)
        return module

    def plain_conv(name):
        return conv(state[f"{name}.weight"], state[f"{name}.bias"])

    class Norm(nn.Module):
        """LayerNorm over the channel axis of a (1, 1, tokens, channels) tensor."""

        def __init__(self, name):
            super().__init__()
            self.weight, self.bias, self.eps = constant(state[f"{name}.weight"]), constant(state[f"{name}.bias"]), eps

        def forward(self, x):
            return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)

    class Mlp(nn.Module):
        def __init__(self, name, first=None):
            super().__init__()
            self.layers = nn.ModuleList(linear(f"{name}.layers.{index}") for index in range(count(f"{name}.layers.")))
            if first is not None:
                self.layers[0].weight = constant(first)

        def forward(self, x):
            for index, layer in enumerate(self.layers):
                x = layer(x) if index == len(self.layers) - 1 else F.relu(layer(x))
            return x

    def attention(name, heads=settings["heads"], *, fuse_qkv=False):
        return build_attention(
            state[f"{name}.in_proj_weight"],
            state[f"{name}.in_proj_bias"],
            state[f"{name}.out_proj.weight"],
            state[f"{name}.out_proj.bias"],
            heads,
            fuse_qkv=fuse_qkv,
        )

    class TextBlock(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.ln_1, self.ln_2 = Norm(f"{name}.ln_1"), Norm(f"{name}.ln_2")
            self.attn = attention(f"{name}.attn", settings["text_heads"], fuse_qkv=True)
            self.fc, self.proj = linear(f"{name}.mlp.c_fc"), linear(f"{name}.mlp.c_proj")

        def forward(self, x, causal):
            h = self.ln_1(x)
            x = x + self.attn(h, bias=causal)
            return x + self.proj(gelu(self.fc(self.ln_2(x))))

    class TextEncoder(nn.Module):
        def __init__(self, name):
            super().__init__()
            encoder = f"{name}.encoder"
            # Each one-hot row has at most one nonzero, so all chunks but one add exact zeros and the
            # sum equals the embedding lookup.
            self.chunk = VOCABULARY // VOCABULARY_CHUNKS
            self.embedding = nn.ParameterList(
                constant(part) for part in state[f"{encoder}.token_embedding.weight"].split(self.chunk)
            )
            self.position = constant(state[f"{encoder}.positional_embedding"][:CONTEXT])
            blocks = f"{encoder}.transformer.resblocks."
            self.blocks = nn.ModuleList(TextBlock(f"{blocks}{index}") for index in range(count(blocks)))
            self.ln_final, self.resizer = Norm(f"{encoder}.ln_final"), linear(f"{name}.resizer")
            # Finite causal mask: exp(-1e4) is exactly zero in float32, like the upstream -inf.
            self.register_buffer("causal", torch.full((1, 1, CONTEXT, CONTEXT), -1e4).triu(1))

        def forward(self, one_hot):
            one_hot = one_hot.reshape(1, 1, CONTEXT, VOCABULARY)
            x = self.position
            for index, part in enumerate(self.embedding):
                x = x + torch.matmul(one_hot[..., index * self.chunk : (index + 1) * self.chunk], part)
            for block in self.blocks:
                x = block(x, self.causal)
            return self.resizer(self.ln_final(x)).reshape(1, CONTEXT, FEATURES)

    def fused(weight, bias, norm):
        """Return a conv's weight and bias with the BatchNorm behind it folded in, in timm's order of operations."""
        gain, deviation = state[f"{norm}.weight"], (state[f"{norm}.running_var"] + eps) ** 0.5
        shifted = (bias - state[f"{norm}.running_mean"]) * gain / deviation
        return weight * (gain / deviation)[:, None, None, None], state[f"{norm}.bias"] + shifted

    def conv_norm(name, stride=1, groups=1):
        """Return RepViT's bias-free conv and its BatchNorm as one conv, as timm's ConvNorm.fuse does."""
        gain, deviation = state[f"{name}.bn.weight"], (state[f"{name}.bn.running_var"] + eps) ** 0.5
        weight = state[f"{name}.c.weight"] * (gain / deviation)[:, None, None, None]
        bias = state[f"{name}.bn.bias"] - state[f"{name}.bn.running_mean"] * gain / deviation
        return conv(weight, bias, stride, groups)

    def token_mixer(name):
        """Return RepViT's 3x3 depthwise branch, 1x1 depthwise branch, identity and BatchNorm as one 3x3 depthwise
        conv, as timm's RepVggDw.fuse does."""
        branch = conv_norm(f"{name}.conv", groups=state[f"{name}.conv.c.weight"].shape[0])
        pointwise = F.pad(state[f"{name}.conv1.weight"], [1, 1, 1, 1])
        identity = F.pad(torch.ones(pointwise.shape[0], pointwise.shape[1], 1, 1), [1, 1, 1, 1])
        weight, bias = fused(
            branch.weight.data + pointwise + identity, branch.bias.data + state[f"{name}.conv1.bias"], f"{name}.bn"
        )
        branch.weight.data.copy_(weight)
        branch.bias.data.copy_(bias)
        return branch

    class SqueezeExcite(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.fc1, self.act = plain_conv(f"{name}.fc1"), nn.ReLU()
            self.fc2, self.gate = plain_conv(f"{name}.fc2"), nn.Sigmoid()

        def forward(self, x):
            return x * self.gate(self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True)))))

    class ChannelMixer(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.conv1, self.act, self.conv2 = conv_norm(f"{name}.conv1"), Gelu(), conv_norm(f"{name}.conv2")

        def forward(self, x):
            return self.conv2(self.act(self.conv1(x)))

    class TrunkBlock(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.token_mixer = token_mixer(f"{name}.token_mixer")
            self.se = SqueezeExcite(f"{name}.se") if f"{name}.se.fc1.weight" in state else nn.Identity()
            self.channel_mixer = ChannelMixer(f"{name}.channel_mixer")

        def forward(self, x):
            x = self.se(self.token_mixer(x))
            return x + self.channel_mixer(x)

    class Downsample(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.pre_block = TrunkBlock(f"{name}.pre_block")
            channels = state[f"{name}.spatial_downsample.c.weight"].shape[0]
            self.spatial_downsample = conv_norm(f"{name}.spatial_downsample", stride=2, groups=channels)
            self.channel_downsample = conv_norm(f"{name}.channel_downsample")
            self.ffn = ChannelMixer(f"{name}.ffn")

        def forward(self, x):
            x = self.channel_downsample(self.spatial_downsample(self.pre_block(x)))
            return self.ffn(x) + x

    class Stage(nn.Module):
        def __init__(self, name):
            super().__init__()
            downsample = f"{name}.downsample.pre_block.token_mixer.conv.c.weight" in state
            self.downsample = Downsample(f"{name}.downsample") if downsample else nn.Identity()
            blocks = f"{name}.blocks."
            self.blocks = nn.Sequential(*(TrunkBlock(f"{blocks}{index}") for index in range(count(blocks))))

        def forward(self, x):
            return self.blocks(self.downsample(x))

    class Stem(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.conv1, self.act1 = conv_norm(f"{name}.conv1", stride=2), Gelu()
            self.conv2 = conv_norm(f"{name}.conv2", stride=2)

        def forward(self, x):
            return self.conv2(self.act1(self.conv1(x)))

    class ImageTrunk(nn.Module):
        """RepViT-M1.1 with the student's head, each BatchNorm fused into its conv."""

        def __init__(self, name):
            super().__init__()
            body, stages = f"{name}.backbone.model", f"{name}.backbone.model.stages."
            self.body = nn.Module()
            self.body.stem = Stem(f"{body}.stem")
            self.body.stages = nn.Sequential(*(Stage(f"{stages}{index}") for index in range(count(stages))))
            scale = state[f"{name}.head.1.weight"] / torch.sqrt(state[f"{name}.head.1.running_var"] + eps)
            shift = state[f"{name}.head.1.bias"] + (0 - state[f"{name}.head.1.running_mean"]) * scale
            self.reduce = conv(state[f"{name}.head.0.weight"] * scale[:, None, None, None], shift)
            self.mix = plain_conv(f"{name}.head.3")

        def forward(self, x):
            # The neck resamples to the token grid after its leading per-pixel layers.
            x = self.body.stages(self.body.stem(x))
            return self.mix(gelu(self.reduce(x)))

    def shuffle_conv(transposed, after=None):
        """Return a 2x2 stride-2 transposed conv, and an optional 1x1 conv after it, as one 1x1 conv to four times
        the channels, whose PixelShuffle(2) equals the original output."""
        weight, bias = state[f"{transposed}.weight"], state[f"{transposed}.bias"]
        if after is not None:
            mix = state[f"{after}.weight"][:, :, 0, 0]
            weight, bias = torch.einsum("oc,icab->ioab", mix, weight), mix @ bias + state[f"{after}.bias"]
        shape = (4 * weight.shape[1], weight.shape[0], 1, 1)
        return conv(weight.permute(1, 2, 3, 0).reshape(shape), bias.repeat_interleave(4))

    class NeckLevel(nn.Module):
        """One neck level whose leading per-pixel affine layers run on the trunk's 32x32 map. Bilinear weights sum
        to 1, so resampling commutes with a per-pixel affine map; resampling a transposed conv's 4C channels before
        PixelShuffle equals resampling its input."""

        def __init__(self, name):
            super().__init__()
            if f"{name}.dconv_2x2_0.weight" in state:
                self.pre = shuffle_conv(f"{name}.dconv_2x2_0")
                second = shuffle_conv(f"{name}.dconv_2x2_1", f"{name}.conv_1x1")
                tail = [nn.PixelShuffle(2), Gelu(), second, nn.PixelShuffle(2)]
            elif f"{name}.dconv_2x2.weight" in state:
                self.pre, tail = shuffle_conv(f"{name}.dconv_2x2", f"{name}.conv_1x1"), [nn.PixelShuffle(2)]
            else:
                self.pre, tail = plain_conv(f"{name}.conv_1x1"), []
            self.tail = nn.Sequential(*tail, plain_conv(f"{name}.conv_3x3"))

        def forward(self, x):
            return self.tail(F.interpolate(self.pre(x), size=(grid, grid), mode="bilinear", align_corners=False))

    class Neck(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.scale4, self.scale2, self.scale1 = (NeckLevel(f"{name}.convs.{index}") for index in range(3))

        def forward(self, x):
            return self.scale4(x), self.scale2(x), self.scale1(x)

    class EncoderLayer(nn.Module):
        """Pre-norm encoder layer; geometry layers use one query and keys with positions."""

        def __init__(self, name, geometry):
            super().__init__()
            self.geometry = geometry
            self.norm1, self.norm2, self.norm3 = (Norm(f"{name}.norm{index}") for index in (1, 2, 3))
            if geometry:
                # Self-attention over one token returns its value projection exactly.
                weight, bias = state[f"{name}.self_attn.in_proj_weight"], state[f"{name}.self_attn.in_proj_bias"]
                width = weight.shape[1]
                self.value = Linear(weight[2 * width :], bias[2 * width :])
                self.out = linear(f"{name}.self_attn.out_proj")
            else:
                self.self_attn = attention(f"{name}.self_attn")
            self.cross_attn = attention(f"{name}.cross_attn_image")
            self.linear1, self.linear2 = linear(f"{name}.linear1"), linear(f"{name}.linear2")

        def forward(self, x, pos, key, value, bias=None):
            h = self.norm1(x)
            x = x + (self.out(self.value(h)) if self.geometry else self.self_attn(h + pos, h + pos, h))
            x = x + self.cross_attn(self.norm2(x), key, value, bias)
            return x + self.linear2(F.relu(self.linear1(self.norm3(x))))

    class DecoderLayer(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.self_attn, self.norm2 = attention(f"{name}.self_attn"), Norm(f"{name}.norm2")
            self.ca_text, self.catext_norm = attention(f"{name}.ca_text"), Norm(f"{name}.catext_norm")
            self.cross_attn, self.norm1 = attention(f"{name}.cross_attn"), Norm(f"{name}.norm1")
            self.linear1, self.linear2 = linear(f"{name}.linear1"), linear(f"{name}.linear2")
            self.norm3 = Norm(f"{name}.norm3")

        def forward(self, t, pos, prompt, prompt_bias, key, memory, box_bias):
            t = self.norm2(t + self.self_attn(t + pos, t + pos, t))
            t = self.catext_norm(t + self.ca_text(t + pos, prompt, prompt, prompt_bias))
            t = self.norm1(t + self.cross_attn(t + pos, key, memory, box_bias))
            return self.norm3(t + self.linear2(F.relu(self.linear1(t))))

    def log_scale(t):
        # sign(t) * log2(|t| + 1) / 3 without Sign or Abs.
        return (torch.log(1 + F.relu(t)) - torch.log(1 + F.relu(-t))) * (1 / (3 * math.log(2)))

    class Decoder(nn.Module):
        def __init__(self, name):
            super().__init__()
            layers = f"{name}.layers."
            self.layers = nn.ModuleList(DecoderLayer(f"{layers}{index}") for index in range(count(layers)))
            self.norm, self.box_head = Norm(f"{name}.norm"), Mlp(f"{name}.bbox_embed")
            self.rpb_x, self.rpb_y = Mlp(f"{name}.boxRPB_embed_x"), Mlp(f"{name}.boxRPB_embed_y")
            self.presence_norm = Norm(f"{name}.presence_token_out_norm")
            self.presence_head = Mlp(f"{name}.presence_token_head")
            heads = settings["heads"]
            self.register_buffer("query", state[f"{name}.query_embed.weight"].reshape(1, 1, QUERIES, 256).clone())
            self.register_buffer("presence", state[f"{name}.presence_token.weight"].reshape(1, 1, 1, 256).clone())
            reference = state[f"{name}.reference_points.weight"].sigmoid()
            self.register_buffer("reference", reference.reshape(1, 1, QUERIES, 4))
            # Sine embedding: angles for (y, x, w, h) blocks of 64 frequencies. The upstream embedding
            # interleaves sin/cos pairs; the first ref_point_head columns are permuted to match.
            steps = torch.arange(128, dtype=torch.float32)
            frequency = (2 * math.pi) / (10000 ** (2 * torch.div(steps, 2, rounding_mode="floor") / 128))
            angles = torch.zeros(4, 256)
            upstream_columns = []
            for block, coordinate in enumerate((1, 0, 2, 3)):
                angles[coordinate, block * 64 : (block + 1) * 64] = frequency[0::2]
                upstream_columns.append([block * 128 + 2 * k for k in range(64)])
            sin_columns = [column for block in upstream_columns for column in block]
            permutation = sin_columns + [column + 1 for column in sin_columns]
            self.register_buffer("angles", angles)
            first = state[f"{name}.ref_point_head.layers.0.weight"]
            self.ref_head = Mlp(f"{name}.ref_point_head", first[:, permutation])
            # 8 * (grid coordinate - box edge) for every (grid line, edge) pair, as one affine map of the
            # reference box instead of a coordinate column broadcast against the edges. Each edge sums at
            # most two exact products and 8 is a power of two, so the map is exact.
            edges = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]])
            coordinates = torch.arange(grid, dtype=torch.float32) / grid
            for axis, columns in (("x", edges[:, 0::2]), ("y", edges[:, 1::2])):
                self.register_buffer(f"{axis}_offset_weight", (-8 * columns).repeat(1, grid))
                self.register_buffer(f"{axis}_offset_bias", (8 * coordinates).repeat_interleave(2))
            expand = torch.zeros(2, grid, grid_tokens)
            for row in range(grid):
                for column in range(grid):
                    expand[0, row, row * grid + column] = 1
                    expand[1, column, row * grid + column] = 1
            self.register_buffer("expand_y", expand[0].clone())
            self.register_buffer("expand_x", expand[1].clone())
            self.register_buffer("presence_bias", torch.zeros(1, heads, 1, grid_tokens))
            self.register_buffer("presence_pos", torch.zeros(1, 1, 1, 256))

        def box_bias(self, reference):
            offsets_y = torch.matmul(reference, self.y_offset_weight) + self.y_offset_bias
            offsets_x = torch.matmul(reference, self.x_offset_weight) + self.x_offset_bias
            bias_y = self.rpb_y(log_scale(offsets_y.reshape(1, QUERIES, grid, 2))).permute(0, 3, 1, 2)
            bias_x = self.rpb_x(log_scale(offsets_x.reshape(1, QUERIES, grid, 2))).permute(0, 3, 1, 2)
            bias = torch.matmul(bias_y, self.expand_y) + torch.matmul(bias_x, self.expand_x)
            return torch.cat([self.presence_bias, bias], 2)

        def forward(self, memory, pos, prompt, prompt_bias):
            key, target, presence, reference = memory + pos, self.query, self.presence, self.reference
            for layer in self.layers:
                angles = torch.matmul(reference, self.angles)
                query_pos = self.ref_head(torch.cat([torch.sin(angles), torch.cos(angles)], -1))
                t = layer(
                    torch.cat([presence, target], 2),
                    torch.cat([self.presence_pos, query_pos], 2),
                    prompt,
                    prompt_bias,
                    key,
                    memory,
                    self.box_bias(reference),
                )
                presence, target = t[:, :, :1], t[:, :, 1:]
                states = self.norm(target)
                # sigmoid(delta + inverse_sigmoid(reference)) with the upstream eps, without Log.
                delta = self.box_head(states)
                inside = reference.clamp(1e-3, 1) * torch.sigmoid(delta)
                reference = inside / (inside + (1 - reference).clamp(1e-3, 1) * torch.sigmoid(-delta))
            return states, reference, self.presence_head(self.presence_norm(presence))

    class Scoring(nn.Module):
        def __init__(self, name):
            super().__init__()
            mlp = f"{name}.prompt_mlp"
            self.fc1, self.fc2 = linear(f"{mlp}.layers.0"), linear(f"{mlp}.layers.1")
            self.norm = Norm(f"{mlp}.out_norm")
            # The logit scale is a power of two, so folding it into the projection is exact.
            self.prompt_proj = linear(f"{name}.prompt_proj", settings["score_scale"])
            self.hs_proj, self.limit = linear(f"{name}.hs_proj"), settings["score_limit"]

        def forward(self, states, prompt, valid):
            prompt = self.norm(prompt + self.fc2(F.relu(self.fc1(prompt))))
            count = valid.sum(-1, keepdim=True).clamp(1, valid.shape[-1])
            pooled = self.prompt_proj(torch.einsum("bnqk,bnkc->bnqc", valid, prompt) / count)
            logits = torch.einsum("bnqc,bnkc->bnqk", self.hs_proj(states), pooled)
            return logits.clamp(-self.limit, self.limit)

    class GroupNorm(nn.Module):
        """GroupNorm from 1x1 convs and global pooling: one conv averages each group's channels, another copies a
        group's value back to its channels, so no tensor is split or reshaped per group."""

        def __init__(self, name):
            super().__init__()
            groups, channels = settings["groups"], state[f"{name}.weight"].shape[0]
            size = channels // groups
            members = (torch.arange(channels) // size == torch.arange(groups)[:, None]).float()
            self.register_buffer("average", (members / size).reshape(groups, channels, 1, 1))
            self.register_buffer("expand", members.t().reshape(channels, groups, 1, 1))
            self.weight = constant(state[f"{name}.weight"].reshape(1, -1, 1, 1))
            self.bias, self.eps = constant(state[f"{name}.bias"].reshape(1, -1, 1, 1)), eps

        def forward(self, x):
            centered = x - F.conv2d(F.adaptive_avg_pool2d(F.conv2d(x, self.average), 1), self.expand)
            variance = F.adaptive_avg_pool2d(F.conv2d(centered * centered, self.average), 1)
            return centered * (F.conv2d(torch.rsqrt(variance + self.eps), self.expand) * self.weight) + self.bias

    class Segmentation(nn.Module):
        def __init__(self, name):
            super().__init__()
            decoder = f"{name}.pixel_decoder"
            self.norm, self.attn = Norm(f"{name}.cross_attn_norm"), attention(f"{name}.cross_attend_prompt")
            self.conv = nn.ModuleList(plain_conv(f"{decoder}.conv_layers.{index}") for index in range(2))
            self.group = nn.ModuleList(GroupNorm(f"{decoder}.norms.{index}") for index in range(2))
            # (W p + b) . m = p . (W^T m) + b . m: the instance head's 1x1 conv moves from the 36,864 pixels to the
            # 200 queries, whose last column carries b . m.
            self.mask_embed = Mlp(f"{name}.mask_predictor.mask_embed")
            last = self.mask_embed.layers[-1]
            instance = f"{name}.instance_seg_head"
            weight, bias = state[f"{instance}.weight"][:, :, 0, 0], state[f"{instance}.bias"]
            last.weight = constant(torch.cat([weight.t() @ last.weight, (bias @ last.weight)[None]]))
            last.bias = constant(torch.cat([weight.t() @ last.bias, (bias @ last.bias)[None]]))

        def forward(self, memory, prompt, prompt_bias, features, states):
            x = memory + self.attn(self.norm(memory), prompt, prompt, prompt_bias)
            x = x.reshape(1, grid, grid, 256).permute(0, 3, 1, 2)
            for conv_layer, norm, feature in zip(self.conv, self.group, features):
                x = feature + F.interpolate(x, size=feature.shape[-2:], mode="nearest")
                x = F.relu(norm(conv_layer(x)))
            queries = self.mask_embed(states)
            # Channels first on both sides, so the product writes the (queries, height, width) output directly.
            masks = torch.einsum("ndhw,ndhq->nqhw", x, queries[..., :256].permute(0, 3, 1, 2))
            return masks + queries[..., 256:].reshape(1, QUERIES, 1, 1)

    class Graph(nn.Module):
        def __init__(self):
            super().__init__()
            vision, geometry = "backbone.vision_backbone", "geometry_encoder"
            self.text = TextEncoder("backbone.language_backbone")
            self.trunk, self.neck = ImageTrunk(f"{vision}.trunk.model"), Neck(vision)
            pos = weights["position_encoding"] if grid == source_grid else position_encoding(grid)
            self.register_buffer("pos", pos.permute(0, 2, 3, 1).reshape(1, 1, grid_tokens, 256).clone())
            cls = F.linear(
                state[f"{geometry}.cls_embed.weight"],
                state[f"{geometry}.final_proj.weight"],
                state[f"{geometry}.final_proj.bias"],
            )
            cls = F.layer_norm(cls, (256,), state[f"{geometry}.norm.weight"], state[f"{geometry}.norm.bias"], eps)
            self.register_buffer("cls", cls.reshape(1, 1, 1, 256).clone())
            layers = f"{geometry}.encode."
            self.geometry = nn.ModuleList(EncoderLayer(f"{layers}{index}", True) for index in range(count(layers)))
            self.geometry_norm = Norm(f"{geometry}.encode_norm")
            layers = "transformer.encoder.layers."
            self.encoder = nn.ModuleList(EncoderLayer(f"{layers}{index}", False) for index in range(count(layers)))
            self.decoder, self.scoring = Decoder("transformer.decoder"), Scoring("dot_prod_scoring")
            self.segmentation = Segmentation("segmentation_head")
            corners = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]) * IMAGE_SIZE
            self.register_buffer("corners", corners)
            self.register_buffer("one", torch.ones(1, 1, 1, 1))

        def image(self, pixel_values, text):
            scale4, scale2, scale1 = self.neck(self.trunk(pixel_values))
            memory = scale1.permute(0, 2, 3, 1).reshape(1, 1, grid_tokens, 256)
            key, g = memory + self.pos, self.cls
            for layer in self.geometry:
                g = layer(g, None, key, memory)
            prompt = torch.cat([text[:, :CONTEXT].reshape(1, 1, CONTEXT, 256), self.geometry_norm(g)], 2)
            valid = torch.cat([text[:, CONTEXT:, :CONTEXT].reshape(1, 1, 1, CONTEXT), self.one], -1)
            # Padding keys get -1e4, whose softmax weight is exactly zero in float32.
            prompt_bias = (valid - 1) * 1e4
            x = memory
            for layer in self.encoder:
                x = layer(x, self.pos, prompt, prompt, prompt_bias)
            states, reference, presence = self.decoder(x, self.pos, prompt, prompt_bias)
            score = torch.sigmoid(self.scoring(states, prompt, valid)) * torch.sigmoid(presence)
            boxes = torch.matmul(reference, self.corners)
            detections = torch.cat([boxes, score, torch.zeros_like(score)], -1).reshape(1, QUERIES, 6)
            return detections, self.segmentation(x, prompt, prompt_bias, (scale2, scale4), states)

        def forward(self, token_one_hot, pixel_values, text):
            return (self.text(token_one_hot), *self.image(pixel_values, text))

    return Graph().eval()


def fold_resize_sizes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Make each Resize's output size a constant and drop the constants no node reads.

    The exporter joins the batch and channel dimensions of the Resize's own input (Shape, Slice) with the constant
    target size (Concat), which the compiler environment's onnxsim leaves in the graph. Every shape is static, so
    shape inference gives those dimensions; behind another Resize it gives them once that one has its constant.
    """
    while any(node.op_type == "Shape" for node in model.graph.node):
        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(model)
        shapes = {
            value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
            for value in (*model.graph.input, *model.graph.value_info)
        }
        producers = {node.output[0]: node for node in model.graph.node}
        targets = [tensor for tensor in model.graph.initializer if tensor.dims == [2]]
        constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in targets}
        folded = 0
        for resize in [node for node in model.graph.node if node.op_type == "Resize"]:
            chain = [producers.get(resize.input[3])]
            for _ in range(2):
                chain.append(producers.get(chain[-1].input[0]) if chain[-1] is not None else None)
            if [node.op_type if node is not None else None for node in chain] != ["Concat", "Slice", "Shape"]:
                continue
            _require(chain[2].input[0] == resize.input[0], f"{resize.name} sized by its own input")
            leading = shapes.get(resize.input[0], [])[:2]
            if len(leading) != 2 or not all(leading):
                continue
            size = np.array([*leading, *constants[chain[0].input[1]]], np.int64)
            model.graph.initializer.append(numpy_helper.from_array(size, resize.input[3]))
            for node in chain:
                model.graph.node.remove(node)
            folded += 1
        _require(folded, "a Resize whose input has a known shape")
    read = {name for node in model.graph.node for name in node.input}
    kept = [tensor for tensor in model.graph.initializer if tensor.name in read]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    del model.graph.value_info[:]
    return model


def validate_export(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("EfficientSAM3 export requires opset 17")
    if [value.name for value in model.graph.input] != list(EXPORT_INPUTS) or [
        value.name for value in model.graph.output
    ] != list(EXPORT_OUTPUTS):
        raise ValueError(f"Expected inputs {EXPORT_INPUTS} and outputs {EXPORT_OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("The export must be one self-contained ONNX file")


def compare_outputs(expected, actual, valid) -> dict[str, float]:
    """Return one case's largest errors against its FP32 reference, failing beyond UPSTREAM_TOLERANCE.

    Text features are compared at the prompt's tokens only; the image model masks the padding positions.
    """
    for name in expected:
        if expected[name].shape != actual[name].shape:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(expected[name]).all() or not np.isfinite(actual[name]).all():
            raise ValueError(f"Non-finite output: {name}")

    def batched(outputs):
        detections = np.concatenate([outputs["boxes_px"], outputs["scores"][:, None]], 1)
        return detections[None], outputs["mask_logits"][None]

    text = np.abs(expected["text_features"].astype(np.float64) - actual["text_features"])[:valid].max()
    errors = {"text_features": float(text), **image_differences(batched(expected), batched(actual))}
    failed = {name: value for name, value in errors.items() if not value <= UPSTREAM_TOLERANCE[name]}
    if failed:
        raise ValueError(f"Model differs from its FP32 reference beyond tolerance: {failed}")
    return errors


def pytorch_outputs(graph, case):
    """Run one recorded image and prompt through the reconstructed FP32 model."""
    import torch

    prompt_tokens = tokens(case["prompt"])
    with torch.no_grad():
        features = graph.text(torch.from_numpy(one_hot(prompt_tokens)))
        detections, masks = graph.image(case["pixel_values"], torch.from_numpy(packed(features.numpy(), prompt_tokens)))
    return {
        "text_features": features[0].numpy(),
        "scores": detections[0, :, 4].numpy(),
        "boxes_px": detections[0, :, :4].numpy(),
        "mask_logits": masks[0].numpy(),
    }


def check_against_upstream(cases, graph) -> list[dict[str, float]]:
    """Check the reconstruction at the untouched upstream grid before deployment adaptations."""
    errors = []
    for case in cases:
        expected = {name: case[name].numpy() for name in ("text_features", "scores", "boxes_px", "mask_logits")}
        errors.append(compare_outputs(expected, pytorch_outputs(graph, case), len(tokens(case["prompt"]))))
    return errors


def check_export(cases, path: Path) -> list[dict[str, float]]:
    """Check ONNX against the reconstructed PyTorch model at the same deployment grid."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    empty_text = np.zeros(INTERFACE["text"], np.float32)
    errors = []
    for case in cases:
        prompt_tokens, pixels = tokens(case["prompt"]), case["pixel_values"].numpy()
        expected = {name: case[name].numpy() for name in ("text_features", "scores", "boxes_px", "mask_logits")}
        # Feed the ONNX text features back as an application does.
        feeds = {"token_one_hot": one_hot(prompt_tokens), "pixel_values": pixels}
        features = session.run(["text_features"], {**feeds, "text": empty_text})[0]
        text_features, detections, masks = session.run(None, {**feeds, "text": packed(features, prompt_tokens)})
        if np.abs(detections[0, :, 5]).max() != 0:
            raise ValueError("Detection column 5 must be zero")
        actual = {
            "text_features": text_features[0],
            "scores": detections[0, :, 4],
            "boxes_px": detections[0, :, :4],
            "mask_logits": masks[0],
        }
        errors.append(compare_outputs(expected, actual, len(prompt_tokens)))
    return errors


def write_runtime_files(assets: dict[str, bytes], directory: Path) -> None:
    """Preserve the conversion's tokenizer beside the graphs for the compiled package."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        (directory / name).write_bytes(assets[name])


def export_model(weights_path: Path, output: Path) -> Path:
    """Check the upstream reconstruction, reduce the token grid, and export the deployment ONNX."""
    import torch
    from onnxsim import simplify

    torch.set_grad_enabled(False)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    write_runtime_files(weights.pop("assets"), output.parent / "tokenizer")
    cases = weights.pop("cases")
    _require(weights["settings"]["grid"] == 72, "conversion must preserve the upstream 72x72 token grid")
    torch.testing.assert_close(weights["position_encoding"], position_encoding(72), rtol=0, atol=0)
    upstream = build_graph(weights)
    for case, errors in zip(cases, check_against_upstream(cases, upstream), strict=True):
        print(f"Upstream grid 72, prompt {case['prompt']!r}: {errors}")
    del upstream
    graph = build_graph(weights, grid=GRID)
    del weights
    deployment_cases = [
        {**case, **{name: torch.from_numpy(value) for name, value in pytorch_outputs(graph, case).items()}}
        for case in cases
    ]
    prompt_tokens = tokens(cases[0]["prompt"])
    example = (
        torch.from_numpy(one_hot(prompt_tokens)),
        cases[0]["pixel_values"],
        torch.from_numpy(packed(np.zeros((CONTEXT, FEATURES), np.float32), prompt_tokens)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph,
        example,
        str(output),
        input_names=list(EXPORT_INPUTS),
        output_names=list(EXPORT_OUTPUTS),
        opset_version=17,
        dynamo=False,
    )
    del graph
    model, valid = simplify(
        onnx.load(output),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in EXPORT_INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded EfficientSAM3 export")
    onnx.save(fold_resize_sizes(model), output)
    del model
    validate_export(output)
    for case, errors in zip(deployment_cases, check_export(deployment_cases, output), strict=True):
        print(
            f"Deployment grid {GRID}, prompt {case['prompt']!r}: "
            + ", ".join(f"{name} {value:.2e}" for name, value in errors.items())
        )
    return output


def graph_paths(output_dir: Path) -> tuple[Path, Path]:
    """Return where prepare_model writes the text encoder and image model graphs."""
    return tuple(output_dir / f"efficientsam3_repvit_m_{part}.onnx" for part in ("text_encoder", "image_model"))


def prepare_model(source: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full = onnx.load(source)
    onnx.checker.check_model(full)
    _require(_interface(full) == INTERFACE, f"interface {_interface(full)}")
    # export_model folds the export with onnxsim; the function counts assume a folded graph.
    unfolded = sorted({node.op_type for node in full.graph.node} & {"Constant", "Shape"})
    _require(not unfolded, f"source still has {unfolded} nodes; fold it with onnxsim first")
    replace_functions(full)
    del full.graph.value_info[:]
    full = onnx.shape_inference.infer_shapes(full, strict_mode=True)
    full.ir_version = 8
    onnx.checker.check_model(full)
    text_path, image_path = graph_paths(output_dir)
    full_path = output_dir / "efficientsam3_repvit_m_full.onnx"
    onnx.save(full, full_path)
    del full
    try:
        onnx.utils.extract_model(str(full_path), str(text_path), [TEXT_INPUT], [TEXT_OUTPUT])
        onnx.utils.extract_model(str(full_path), str(image_path), IMAGE_INPUTS, IMAGE_OUTPUTS)
    finally:
        full_path.unlink()
    for path in (text_path, image_path):
        model = onnx.load(path)
        model.ir_version = 8
        left = {node.op_type for node in model.graph.node} & set(EXPECTED)
        _require(not left, f"{path.name} still has {sorted(left)}")
        onnx.checker.check_model(model)
        onnx.save(model, path)
    return text_path, image_path


def image_differences(expected, actual) -> dict[str, float]:
    (detections, masks), (actual_detections, actual_masks) = expected, actual
    scores = detections[0, :, 4].astype(np.float64)
    boxes = np.abs(detections[0, :, :4].astype(np.float64) - actual_detections[0, :, :4]).max(1)
    mask = np.abs(masks[0].astype(np.float64) - actual_masks[0]).reshape(QUERIES, -1).max(1)
    confident = scores >= CONFIDENT
    return {
        "scores": float(np.abs(scores - actual_detections[0, :, 4]).max()),
        "boxes_relative": float(boxes.max() / IMAGE_SIZE),
        "masks_relative": float(mask.max() / np.abs(masks).max()),
        "boxes_px_confident": float(boxes[confident].max(initial=0.0)),
        "mask_logits_confident": float(mask[confident].max(initial=0.0)),
    }


def validate_float_split(
    source: Path, text_path: Path, image_path: Path, images: np.ndarray, prompts
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare both components with the exported graph on real images and prompts.

    Return the largest errors and the exported graph's image outputs, stacked over the images.
    """
    import onnxruntime as ort

    failures = []
    options = {"providers": ["CPUExecutionProvider"]}
    reference = ort.InferenceSession(str(source), **options)
    replaced = function_inputs(onnx.load(source, load_external_data=False))
    text_session = ort.InferenceSession(str(text_path), **options)
    # Also return the replaced functions' inputs to check that real values stay inside their domains.
    probes = sorted({name for names in replaced.values() for name in names})
    image = onnx.load(image_path)
    image.graph.output.extend(helper.make_empty_tensor_value_info(name) for name in probes)
    image_session = ort.InferenceSession(image.SerializeToString(), **options)
    del image
    ranges = {op: [np.inf, -np.inf] for op in replaced}
    largest = {"text_features": 0.0, **dict.fromkeys(IMAGE_TOLERANCE, 0.0)}
    empty = np.zeros(INTERFACE["text"], np.float32)
    outputs = {name: [] for name in IMAGE_OUTPUTS}
    for index, (pixels, prompt) in enumerate(zip(images, prompts, strict=True)):
        prompt_tokens = tokens(prompt)
        encoded = one_hot(prompt_tokens)
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        feeds = {TEXT_INPUT: encoded, "pixel_values": pixels, "text": empty}
        expected_text = reference.run([TEXT_OUTPUT], feeds)[0]
        (features,) = text_session.run(None, {TEXT_INPUT: encoded})
        text = packed(expected_text, prompt_tokens)
        expected = reference.run(IMAGE_OUTPUTS, {**feeds, "text": text})
        actual = image_session.run(None, {"pixel_values": pixels, "text": text})
        for name, value in zip(IMAGE_OUTPUTS, expected, strict=True):
            outputs[name].append(value[0])
        values = dict(zip(probes, actual[2:], strict=True))
        for op, names in replaced.items():
            ranges[op][0] = min(ranges[op][0], *(float(values[name].min()) for name in names))
            ranges[op][1] = max(ranges[op][1], *(float(values[name].max()) for name in names))
        case = {
            "text_features": float(np.abs(features.astype(np.float64) - expected_text).max()),
            **image_differences(expected, actual[:2]),
        }
        largest = {name: max(largest[name], value) for name, value in case.items()}
        limits = {"text_features": TEXT_TOLERANCE, **IMAGE_TOLERANCE}
        failures += [
            f"{index}/{prompt}/{name} {case[name]:.2e}" for name, limit in limits.items() if not case[name] <= limit
        ]
        if np.abs(actual[0][0, :, 5]).max() != 0:
            failures.append(f"{index}/{prompt}/detection column 5 is not zero")
    for op, (low, high) in ranges.items():
        if not (DOMAINS[op][0] <= low and high <= DOMAINS[op][1]):
            failures.append(f"{op} inputs [{low:.4g}, {high:.4g}] leave the domain {DOMAINS[op]}")
    if failures:
        raise RuntimeError(f"EfficientSAM3 split differs from the exported graph: {failures}")
    report = {"observed_inputs": ranges, **largest}
    return report, {name: np.stack(values) for name, values in outputs.items()}
