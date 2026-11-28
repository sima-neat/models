"""Mask2Former's three parts in the layout the Model Compiler maps to the MLA: every tensor 4D with batch 1.

Each module takes transformers' Mask2Former modules and holds the same weights, rearranged; convert.py checks each
against transformers before export. Linear layers become 1x1 convolutions, and LayerNorm normalizes over the
channel axis. Attention is RF-DETR's einsum form: q and k [1, head_dim, heads (x windows), tokens], scores
[1, keys, heads (x windows), queries] with the softmax on axis 1. Query, key and value convolutions' output
channels and the output convolution's input channels are permuted so that channel d * heads + h is head h's d-th
channel, as the 4D reshape groups them.

Swin4D, the backbone, features [1, C, H, W]. transformers batches the windows (batch = windows x heads), which the
MLA cannot run. Here a window partition is a Reshape, a Transpose of the last two axes and a Reshape, to
[1, C, windows, 49], each window's tokens column-major, and heads and windows share one axis. The relative-position
bias and the shifted-window mask are one constant added to the scores; the cyclic shift is Slice and Concat.

PixelDecoder4D: the three scales' tokens as [1, 256, 1, tokens]. Sampling offsets come from two 1x1 convolutions (x
and y), pre-scaled by 2 / scale size, so that a grid coordinate is one convolution plus the token's constant
reference coordinate 2 * ref - 1. The attention weights' convolution emits channel point * heads + head, so that a
channel-axis softmax over [1, 12, 8, tokens] normalizes each head's 12 points. Each head samples each scale with
one GridSample [1, 32, S, S] over a grid [1, tokens, 4, 2]; the compiler handles S up to 25, so a larger scale is
sampled as two by two overlapping 25 x 25 tiles (see `sample`). The sine position embedding, level embeddings and
reference points are constants of the input size. GroupNorm is written out over [1, groups, channels per group x
H, W].

TransformerDecoder4D: 100 queries as [1, 256, 1, 100]. transformers bilinearly resizes each query's mask logits to
the scale and masks keys where sigmoid(logit) < 0.5, unmasking a query whose every key is masked. Resizing is
linear, so the logits are the einsum of the mask embedding with the mask features resized once per scale, already
in the scores' [1, keys, 1, queries] layout. The mask is a bias of -BIAS where logit < 0, through a steep clip ramp,
and a reduce-max over keys zeroes it for a fully masked query. The final masks come from the same einsum with its
operands swapped, [1, 100, 1, H x W], reshaped to [1, 100, H, W].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

TILE = 25
RAMP, BIAS = 1e4, 1e4
# The backbone compiles as two components, split before this stage (0-based), and the pixel decoder as two components
# per deformable layer (four heads each) plus one for its 1/4 level: compiled whole, the backbone needs more memory
# than a CI runner has, and so does a single deformable layer once the compiler's check simulation is counted.
SPLIT_STAGE = 2


def channel_layer_norm(x, weight, bias, eps):
    mean = x.mean(dim=1, keepdim=True)
    centred = x - mean
    variance = (centred * centred).mean(dim=1, keepdim=True)
    return centred / torch.sqrt(variance + eps) * weight.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)


def conv_from_linear(linear, out_order=None, in_order=None):
    weight = linear.weight.detach()
    bias = linear.bias.detach() if linear.bias is not None else None
    if out_order is not None:
        weight, bias = weight[out_order], (bias[out_order] if bias is not None else None)
    if in_order is not None:
        weight = weight[:, in_order]
    conv = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    conv.weight.data = weight[:, :, None, None].clone()
    if bias is not None:
        conv.bias.data = bias.clone()
    return conv


def head_major_to_dim_major(channels, heads):
    """new channel d * heads + h takes transformers' channel h * head_dim + d."""
    head_dim = channels // heads
    return torch.tensor([h * head_dim + d for d in range(head_dim) for h in range(heads)])


class Norm(torch.nn.Module):
    def __init__(self, layer_norm):
        super().__init__()
        self.weight = torch.nn.Parameter(layer_norm.weight.detach().clone())
        self.bias = torch.nn.Parameter(layer_norm.bias.detach().clone())
        self.eps = layer_norm.eps

    def forward(self, x):
        return channel_layer_norm(x, self.weight, self.bias, self.eps)


def window_tokens(size):
    """Each token's (row, column) within a window, in this layout's column-major order."""
    return [(r, c) for c in range(size) for r in range(size)]


class Block(torch.nn.Module):
    def __init__(self, layer, resolution: int):
        super().__init__()
        attention = layer.attention
        channels = attention.q_proj.in_features
        heads = attention.num_attention_heads
        # SwinBackbone runs with always_partition: the window and shift are never clamped to the feature size.
        window, shift = layer.window_size, layer.shift_size
        padded = -(-resolution // window) * window
        self.channels, self.heads, self.window, self.shift = channels, heads, window, shift
        self.resolution, self.padded = resolution, padded
        self.windows = (padded // window) ** 2
        order = head_major_to_dim_major(channels, heads)
        self.norm1, self.norm2 = Norm(layer.layernorm_before), Norm(layer.layernorm_after)
        self.q = conv_from_linear(attention.q_proj, out_order=order)
        self.q.weight.data *= attention.scaling
        if self.q.bias is not None:
            self.q.bias.data *= attention.scaling
        self.k = conv_from_linear(attention.k_proj, out_order=order)
        self.v = conv_from_linear(attention.v_proj, out_order=order)
        self.o = conv_from_linear(attention.o_proj, in_order=order)
        self.fc1, self.fc2 = conv_from_linear(layer.mlp.fc1), conv_from_linear(layer.mlp.fc2)
        self.register_buffer("bias", self._bias(attention, layer))

    def _bias(self, attention, layer):
        """[1, keys, heads x windows, queries]: the relative-position bias plus the shift mask, -100 across regions."""
        w, n, heads, windows = self.window, self.window * self.window, self.heads, self.windows
        table = attention.relative_position_bias.relative_position_bias_table.detach().numpy()  # [(2w-1)^2, heads]
        tokens = window_tokens(w)
        index = np.array(
            [[(qr - kr + w - 1) * (2 * w - 1) + (qc - kc + w - 1) for (kr, kc) in tokens] for (qr, qc) in tokens]
        )  # [q, k]
        relative = table[index]  # [q, k, heads]
        mask = np.zeros((windows, n, n), np.float32)
        if self.shift:
            p, s = self.padded, self.shift
            region = lambda i: int(i >= p - w) + int(i >= p - s)  # noqa: E731
            per_axis = p // w
            for index_w in range(windows):
                wr, wc = divmod(index_w, per_axis)
                labels = np.array([region(wr * w + r) * 3 + region(wc * w + c) for (r, c) in tokens])
                mask[index_w] = np.where(labels[:, None] != labels[None, :], -100.0, 0.0)  # [q, k]
        combined = relative.transpose(2, 0, 1)[:, None] + mask[None]  # [heads, windows, q, k]
        combined = combined.reshape(heads * windows, n, n).transpose(2, 0, 1)  # [k, h*W, q]
        return torch.from_numpy(np.ascontiguousarray(combined[None], np.float32))

    def partition(self, x):
        c, p, w = self.channels, self.padded, self.window
        x = x.reshape(1, c * (p // w), w, p).transpose(2, 3)  # [1, C*rows, P, w_r]
        return x.reshape(1, c, self.windows, w * w)  # token = column * w + row

    def reverse(self, x):
        c, p, w = self.channels, self.padded, self.window
        x = x.reshape(1, c * (p // w), p, w).transpose(2, 3)
        return x.reshape(1, c, p, p)

    def roll(self, x, amount):
        """torch.roll by -amount on both spatial axes (amount < 0 rolls back), as Slice and Concat."""
        if not amount:
            return x
        amount %= self.padded
        x = torch.cat([x[:, :, amount:], x[:, :, :amount]], 2)
        return torch.cat([x[:, :, :, amount:], x[:, :, :, :amount]], 3)

    def forward(self, x):
        shortcut = x
        y = self.norm1(x)
        pad = self.padded - self.resolution
        if pad:
            y = F.pad(y, (0, pad, 0, pad))
        y = self.partition(self.roll(y, self.shift))
        head = self.channels // self.heads
        q = self.q(y).reshape(1, head, self.heads * self.windows, -1)
        k = self.k(y).reshape(1, head, self.heads * self.windows, -1)
        v = self.v(y).reshape(1, head, self.heads * self.windows, -1)
        scores = torch.einsum("nchw,nchq->nqhw", q, k) + self.bias
        out = torch.einsum("nchw,nqhc->nqhw", scores.softmax(dim=1), v)
        y = self.o(out.reshape(1, self.channels, self.windows, -1))
        y = self.roll(self.reverse(y), -self.shift)
        if pad:
            y = y[:, :, : self.resolution, : self.resolution]
        x = shortcut + y
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Merge(torch.nn.Module):
    """transformers' patch merging: the 2x2 neighbours as channel blocks (row 0 col 0, row 1 col 0, row 0 col 1,
    row 1 col 1), LayerNorm, then the reduction."""

    def __init__(self, merging):
        super().__init__()
        self.norm = Norm(merging.norm)
        self.reduction = conv_from_linear(merging.reduction)

    def forward(self, x):
        parts = [x[:, :, row::2, col::2] for col in range(2) for row in range(2)]
        return self.reduction(self.norm(torch.cat(parts, 1)))


class Swin4D(torch.nn.Module):
    def __init__(self, backbone, image_size: int):
        super().__init__()
        swin = backbone.swin
        self.patch = swin.embeddings.patch_embeddings.projection
        self.patch_norm = Norm(swin.embeddings.norm)
        resolution = image_size // self.patch.stride[0]
        self.stages, self.merges, self.out_norms = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        for index, stage in enumerate(swin.encoder.layers):
            self.stages.append(torch.nn.Sequential(*(Block(layer, resolution) for layer in stage.blocks)))
            self.merges.append(Merge(stage.downsample) if stage.downsample is not None else torch.nn.Identity())
            self.out_norms.append(Norm(backbone.hidden_states_norms[backbone.out_features[index]]))
            resolution //= 2

    def _stages(self, x, indices):
        features = []
        for index in indices:
            x = self.stages[index](x)
            features.append(self.out_norms[index](x))
            x = self.merges[index](x)
        return features, x

    def start(self, pixel_values):
        """Patch embedding and stages 1 and 2: their feature maps, and stage 3's input."""
        features, x = self._stages(self.patch_norm(self.patch(pixel_values)), range(SPLIT_STAGE))
        return (*features, x)

    def finish(self, x):
        """Stages 3 and 4: their feature maps."""
        return tuple(self._stages(x, range(SPLIT_STAGE, len(self.stages)))[0])

    def forward(self, pixel_values):
        *features, x = self.start(pixel_values)
        return (*features, *self.finish(x))


def group_norm(x, norm):
    _, c, h, w = x.shape
    g = x.reshape(1, norm.num_groups, c // norm.num_groups * h, w)
    mean = g.mean(dim=(2, 3), keepdim=True)
    centred = g - mean
    variance = (centred * centred).mean(dim=(2, 3), keepdim=True)
    g = centred / torch.sqrt(variance + norm.eps)
    return g.reshape(1, c, h, w) * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)


def sample(value, x, y):
    """grid_sample(value, [x, y]) with align_corners=False and zero padding, for value [1, C, S, S] and x, y
    [1, tokens, points, 1]. Up to TILE, one GridSample. Larger (up to 2 * TILE - 2), two tiles per axis covering
    pixels [0, TILE - 1] and [S - TILE, S - 1]: a sample with pixel coordinate px reads pixels floor(px) and
    floor(px) + 1, so tile 0 is exact for px < TILE - 1 and tile 1 for px >= S - TILE, and blending them with a
    weight that ramps from 1 to 0 across [S - TILE, TILE - 1] is exact everywhere. Zero padding is kept because each
    tile's outer edge is the map's edge."""
    size = value.shape[-1]
    if size <= TILE:
        return F.grid_sample(value, torch.cat([x, y], 3), mode="bilinear", padding_mode="zeros", align_corners=False)
    starts = (0, size - TILE)
    low, high = size - TILE, TILE - 1

    def axis(coordinate):
        pixel = coordinate * (size / 2) + (size - 1) / 2
        w0 = torch.clamp((high - pixel) / (high - low), 0, 1)
        # Tile coordinate of the same point: (2 * (pixel - start) + 1) / TILE - 1.
        coords = [coordinate * (size / TILE) + ((size - 1 - 2 * start) + 1) / TILE - 1 for start in starts]
        return coords, (w0, 1 - w0)

    (xs, wx), (ys, wy) = axis(x), axis(y)
    out = 0
    for ty, sy in enumerate(starts):
        for tx, sx in enumerate(starts):
            tile = value[:, :, sy : sy + TILE, sx : sx + TILE]
            sampled = F.grid_sample(
                tile, torch.cat([xs[tx], ys[ty]], 3), mode="bilinear", padding_mode="zeros", align_corners=False
            )
            out = out + sampled * (wx[tx] * wy[ty]).permute(0, 3, 1, 2)
    return out


class DeformableLayer(torch.nn.Module):
    def __init__(self, layer, shapes, reference):
        super().__init__()
        attention = layer.self_attn
        heads, levels, points = attention.n_heads, attention.n_levels, attention.n_points
        self.heads, self.levels, self.points, self.shapes = heads, levels, points, shapes
        # transformers' offsets channel ((head * levels + level) * points + point) * 2 + xy; ours per axis:
        # (head * levels + level) * points + point, pre-scaled by 2 / that level's size along the axis.
        offsets = attention.sampling_offsets
        for xy, name in ((0, "offset_x"), (1, "offset_y")):
            order = torch.arange(heads * levels * points) * 2 + xy
            conv = conv_from_linear(offsets, out_order=order)
            scale = torch.tensor([2.0 / shapes[(c // points) % levels][1 - xy] for c in range(heads * levels * points)])
            conv.weight.data *= scale.view(-1, 1, 1, 1)
            conv.bias.data *= scale
            setattr(self, name, conv)
        # transformers' weight channel head * (levels * points) + level * points + point; ours lp * heads + head.
        lp = levels * points
        self.weights = conv_from_linear(
            attention.attention_weights, out_order=torch.tensor([h * lp + j for j in range(lp) for h in range(heads)])
        )
        self.value = conv_from_linear(attention.value_proj)
        self.output = conv_from_linear(attention.output_proj)
        self.norm1, self.norm2 = Norm(layer.self_attn_layer_norm), Norm(layer.final_layer_norm)
        self.fc1, self.fc2 = conv_from_linear(layer.fc1), conv_from_linear(layer.fc2)
        self.register_buffer("ref_x", (2 * reference[:, 0] - 1).view(1, 1, 1, -1))
        self.register_buffer("ref_y", (2 * reference[:, 1] - 1).view(1, 1, 1, -1))

    def attend(self, h, positioned, heads=None):
        """The attention output for all heads, or, given heads, only those heads' share of the output projection
        (without its bias): the projection is linear over the concatenated heads, so the shares add up to it."""
        selected = range(self.heads) if heads is None else heads
        levels, points = self.levels, self.points
        heads = self.heads
        tokens = h.shape[-1]
        value = self.value(h)  # [1, 256, 1, N]
        gx = self.offset_x(positioned) + self.ref_x  # [1, H*L*P, 1, N]
        gy = self.offset_y(positioned) + self.ref_y
        weights = self.weights(positioned).reshape(1, levels * points, heads, tokens).softmax(dim=1)
        head_dim = value.shape[1] // heads
        sizes = [hh * ww for hh, ww in self.shapes]
        starts = [sum(sizes[:i]) for i in range(levels)]
        outputs = []
        for head in selected:
            total = 0
            for level, (hh, ww) in enumerate(self.shapes):
                v = value[:, head * head_dim : (head + 1) * head_dim, :, starts[level] : starts[level] + sizes[level]]
                v = v.reshape(1, head_dim, hh, ww)
                first = (head * levels + level) * points
                x = gx[:, first : first + points].permute(0, 3, 1, 2)  # [1, N, P, 1]
                y = gy[:, first : first + points].permute(0, 3, 1, 2)
                sampled = sample(v, x, y)  # [1, d, N, P]
                w = weights[:, level * points : (level + 1) * points, head : head + 1].permute(0, 2, 3, 1)  # [1,1,N,P]
                total = total + (sampled * w).sum(dim=3, keepdim=True)  # [1, d, N, 1]
            outputs.append(total)
        concatenated = torch.cat(outputs, 1).reshape(1, len(selected) * head_dim, 1, tokens)
        if len(selected) == heads:
            return self.output(concatenated)
        columns = torch.cat([torch.arange(head * head_dim, (head + 1) * head_dim) for head in selected])
        return F.conv2d(concatenated, self.output.weight[:, columns])

    def first_half(self, h, position):
        """The first half of the heads' share of the attention output projection."""
        return self.attend(h, h + position, range(self.heads // 2))

    def second_half(self, h, partial, position):
        """The second half of the heads, added to the first half's share, then the residual, LayerNorm and FFN."""
        attended = partial + self.attend(h, h + position, range(self.heads // 2, self.heads))
        h = self.norm1(h + attended + self.output.bias.view(1, -1, 1, 1))
        return self.norm2(h + self.fc2(F.relu(self.fc1(h))))

    def forward(self, h, position):
        h = self.norm1(h + self.attend(h, h + position))
        return self.norm2(h + self.fc2(F.relu(self.fc1(h))))


class PixelDecoder4D(torch.nn.Module):
    def __init__(self, decoder, feature_shapes):
        """feature_shapes: the backbone's four (channels, size) from high to low resolution."""
        super().__init__()
        self.decoder = decoder
        levels = decoder.num_feature_levels
        low_to_high = feature_shapes[::-1][:levels]
        self.shapes = [(s, s) for _, s in low_to_high]
        with torch.no_grad():
            position = [
                decoder.position_embedding((1, c, s, s), "cpu", torch.float32)
                + decoder.level_embed[i].view(1, -1, 1, 1)
                for i, (c, s) in enumerate(low_to_high)
            ]
            position = torch.cat([p.reshape(1, p.shape[1], 1, -1) for p in position], 3)
            valid = torch.ones(1, levels, 2)
            reference = decoder.encoder.get_reference_points(self.shapes, valid, "cpu")[0, :, 0]  # [N, 2]
        self.register_buffer("position", position)
        self.layers = torch.nn.ModuleList(
            DeformableLayer(layer, self.shapes, reference) for layer in decoder.encoder.layers
        )

    def embed(self, f1, f2, f3):
        """The input projections: the three scales' tokens [1, 256, 1, tokens], 1/32 to 1/8."""
        d = self.decoder
        embeds = []
        for level, x in enumerate((f3, f2, f1)):
            conv, norm = d.input_projections[level]
            embeds.append(group_norm(conv(x), norm))
        return torch.cat([e.reshape(1, e.shape[1], 1, -1) for e in embeds], 3)

    def start(self, f1, f2, f3):
        """The input projections' tokens, and deformable layer 0's first half on them."""
        h = self.embed(f1, f2, f3)
        return h, self.layers[0].first_half(h, self.position)

    def first_half(self, index, h):
        """Deformable layer index's first half of the heads on the tokens."""
        return self.layers[index].first_half(h, self.position)

    def second_half(self, index, h, partial):
        """Deformable layer index's second half of the heads, completing the layer: the next tokens."""
        return self.layers[index].second_half(h, partial, self.position)

    def scales(self, h):
        """The tokens as the three multi-scale feature maps, 1/32 to 1/8."""
        maps, start = [], 0
        for hh, ww in self.shapes:
            maps.append(h[:, :, :, start : start + hh * ww].reshape(1, h.shape[1], hh, ww))
            start += hh * ww
        return maps

    def finish(self, h, f0):
        """The mask features: the FPN level from the 1/8 map and features0, then the mask projection. It returns only
        them, because the compiler gives each independent output branch its own MLA segment; the transformer decoder
        takes the tokens themselves for the multi-scale features."""
        d = self.decoder
        lateral, output = d.lateral_convolutions[0], d.output_convolutions[0]
        current = group_norm(lateral[0](f0), lateral[1])
        out = current + F.interpolate(self.scales(h)[-1], size=current.shape[-2:], mode="bilinear", align_corners=False)
        return d.mask_projection(F.relu(group_norm(output[0](out), output[1])))

    def forward(self, f0, f1, f2, f3):
        """transformers' pixel decoder outputs: the mask features and the three multi-scale feature maps."""
        h, partial = self.start(f1, f2, f3)
        h = self.second_half(0, h, partial)
        for index in range(1, len(self.layers)):
            h = self.second_half(index, h, self.first_half(index, h))
        return (self.finish(h, f0), *self.scales(h))


def packed_projection(attention, index, order):
    """Query, key or value (index 0, 1, 2) of nn.MultiheadAttention's packed projection, as a 1x1 convolution."""
    dim = attention.embed_dim
    linear = torch.nn.Linear(dim, dim)
    linear.weight.data = attention.in_proj_weight.detach()[index * dim : (index + 1) * dim].clone()
    linear.bias.data = attention.in_proj_bias.detach()[index * dim : (index + 1) * dim].clone()
    return conv_from_linear(linear, out_order=order)


class Attention4D(torch.nn.Module):
    def __init__(self, q, k, v, o, heads, scaling):
        super().__init__()
        self.q, self.k, self.v, self.o, self.heads = q, k, v, o, heads
        self.q.weight.data *= scaling
        self.q.bias.data *= scaling

    def forward(self, query, key, value, bias=None):
        dim = query.shape[1]
        head = dim // self.heads
        q = self.q(query).reshape(1, head, self.heads, -1)
        k = self.k(key).reshape(1, head, self.heads, -1)
        v = self.v(value).reshape(1, head, self.heads, -1)
        scores = torch.einsum("nchw,nchq->nqhw", q, k)
        if bias is not None:
            scores = scores + bias
        out = torch.einsum("nchw,nqhc->nqhw", scores.softmax(dim=1), v)
        return self.o(out.reshape(1, dim, 1, -1))


def cross_attention(mha):
    dim, heads = mha.embed_dim, mha.num_heads
    order = head_major_to_dim_major(dim, heads)
    projections = [packed_projection(mha, i, order) for i in range(3)]
    return Attention4D(*projections, conv_from_linear(mha.out_proj, in_order=order), heads, (dim // heads) ** -0.5)


def self_attention(attention):
    dim, heads = attention.embed_dim, attention.num_heads
    order = head_major_to_dim_major(dim, heads)
    return Attention4D(
        conv_from_linear(attention.q_proj, out_order=order),
        conv_from_linear(attention.k_proj, out_order=order),
        conv_from_linear(attention.v_proj, out_order=order),
        conv_from_linear(attention.out_proj, in_order=order),
        heads,
        attention.scaling,
    )


class MaskEmbedder(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.layers = torch.nn.ModuleList(conv_from_linear(block.layers[0]) for block in head.layers)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x


def attention_bias(logits):
    masked = torch.clamp(-logits * RAMP, 0, 1)
    any_open = torch.clamp(logits.amax(dim=1, keepdim=True) * RAMP, 0, 1)
    return masked * any_open * -BIAS


class TransformerDecoder4D(torch.nn.Module):
    def __init__(self, module, class_predictor, scale_sizes):
        super().__init__()
        decoder = module.decoder
        self.levels = module.num_feature_levels
        self.sizes = scale_sizes
        for projection in module.input_projections:
            if len(list(projection.parameters())):
                raise ValueError("expected identity input projections (enforce_input_projection: false)")
        with torch.no_grad():
            positions = [
                module.position_embedder((1, 256, s, s), "cpu", torch.float32).reshape(1, 256, 1, -1)
                for s in scale_sizes
            ]
        self.register_buffer("positions", torch.cat(positions, 3))
        self.register_buffer("level_embed", module.level_embed.weight.detach().clone())
        self.register_buffer(
            "query_position",
            module.queries_embedder.weight.detach().T.reshape(1, -1, 1, module.queries_embedder.num_embeddings).clone(),
        )
        self.register_buffer(
            "query_features",
            module.queries_features.weight.detach().T.reshape(1, -1, 1, module.queries_features.num_embeddings).clone(),
        )
        self.norm = Norm(decoder.layernorm)
        self.mask_embedder = MaskEmbedder(decoder.mask_predictor.mask_embedder)
        self.cross = torch.nn.ModuleList(cross_attention(layer.cross_attn) for layer in decoder.layers)
        self.self_attn = torch.nn.ModuleList(self_attention(layer.self_attn) for layer in decoder.layers)
        self.cross_norms = torch.nn.ModuleList(Norm(layer.cross_attn_layer_norm) for layer in decoder.layers)
        self.self_norms = torch.nn.ModuleList(Norm(layer.self_attn_layer_norm) for layer in decoder.layers)
        self.final_norms = torch.nn.ModuleList(Norm(layer.final_layer_norm) for layer in decoder.layers)
        self.fc1 = torch.nn.ModuleList(conv_from_linear(layer.fc1) for layer in decoder.layers)
        self.fc2 = torch.nn.ModuleList(conv_from_linear(layer.fc2) for layer in decoder.layers)
        self.classifier = conv_from_linear(class_predictor)

    def forward(self, mask_features, tokens):
        """tokens: the three multi-scale feature maps as one [1, 256, 1, tokens] sequence, 1/32 to 1/8, as the pixel
        decoder's deformable layers leave them."""
        dim = mask_features.shape[1]
        scales, keys, resized = [], [], []
        starts = [0]
        for s in self.sizes:
            starts.append(starts[-1] + s * s)
        for level, size in enumerate(self.sizes):
            n = size * size
            features = tokens[:, :, :, starts[level] : starts[level + 1]] + self.level_embed[level].view(1, -1, 1, 1)
            scales.append(features)
            keys.append(features + self.positions[:, :, :, starts[level] : starts[level + 1]])
            small = F.interpolate(mask_features, size=(size, size), mode="bilinear", align_corners=False)
            resized.append(small.reshape(1, dim, 1, n))
        h = self.query_features
        embedding = self.mask_embedder(self.norm(h))
        for index in range(len(self.cross)):
            level = index % self.levels
            bias = attention_bias(torch.einsum("nchw,nchq->nqhw", embedding, resized[level]))
            h = self.cross_norms[index](
                h + self.cross[index](h + self.query_position, keys[level], scales[level], bias)
            )
            q = h + self.query_position
            h = self.self_norms[index](h + self.self_attn[index](q, q, h))
            h = self.final_norms[index](h + self.fc2[index](F.relu(self.fc1[index](h))))
            normed = self.norm(h)
            embedding = self.mask_embedder(normed)
        height, width = mask_features.shape[-2:]
        masks = torch.einsum("nchw,nchq->nqhw", mask_features.reshape(1, dim, 1, height * width), embedding)
        return self.classifier(normed), masks.reshape(1, -1, height, width)
