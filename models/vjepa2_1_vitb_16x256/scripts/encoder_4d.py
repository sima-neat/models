"""The V-JEPA 2.1 ViT-B encoder for one fixed clip, rewritten in the 4D layout the compiler handles and built from the
upstream encoder's weights. It computes the same features as upstream's VisionTransformer.

Tokens are [1, C, 1, N]; every Linear is a 1x1 convolution; heads sit on the third axis, so attention is two Einsums
over query chunks. For a fixed clip the 3D RoPE angles are constants, and RoPE's pair rotation is a constant 1x1
convolution. The Conv3d patch embedding is a Conv2d over the clip's tubelets stacked down the height
(stack_tubelets), whose channels are each tubelet's frames and colours.
"""

import torch


def pointwise(linear, scale=1.0):
    """A Linear as a 1x1 convolution, its output scaled."""
    layer = torch.nn.Conv2d(linear.in_features, linear.out_features, 1)
    layer.weight.data = linear.weight.data[:, :, None, None] * scale
    layer.bias.data = linear.bias.data * scale
    return layer


class ChannelNorm(torch.nn.Module):
    """A LayerNorm over the channel axis of [1, C, 1, N]."""

    def __init__(self, norm):
        super().__init__()
        self.eps = norm.eps
        self.weight = torch.nn.Parameter(norm.weight.data.reshape(1, -1, 1, 1).clone())
        self.bias = torch.nn.Parameter(norm.bias.data.reshape(1, -1, 1, 1).clone())

    def forward(self, x):
        centred = x - x.mean(dim=1, keepdim=True)
        variance = (centred * centred).mean(dim=1, keepdim=True)
        return centred / torch.sqrt(variance + self.eps) * self.weight + self.bias


def rope_tables(attention, frames, height, width, head_dim):
    """Upstream RoPEAttention's cos and sin [1, head_dim, 1, N] for a frames x height x width token grid, and its pair
    rotation [head_dim, head_dim]: the depth, height and width dimensions each rotate in pairs."""
    ids = torch.arange(frames * height * width)
    frame_ids, height_ids, width_ids = attention.separate_positions(ids, height, width)
    if attention.interpolate_rope:
        height_ids = height_ids * (attention.pretrained_grid_size - 1) / (height - 1)
        width_ids = width_ids * (attention.pretrained_grid_size - 1) / (width - 1)
    cos = torch.ones(head_dim, ids.numel())
    sin = torch.zeros(head_dim, ids.numel())
    rotation = torch.zeros(head_dim, head_dim)
    start = 0
    for positions, size in ((frame_ids, attention.d_dim), (height_ids, attention.h_dim), (width_ids, attention.w_dim)):
        omega = 1.0 / 10000 ** (torch.arange(size // 2, dtype=torch.float32) / (size / 2.0))
        angles = positions[:, None].float() * omega[None]
        cos[start : start + size] = angles.cos().repeat_interleave(2, dim=-1).T
        sin[start : start + size] = angles.sin().repeat_interleave(2, dim=-1).T
        for pair in range(size // 2):
            even, odd = start + 2 * pair, start + 2 * pair + 1
            rotation[even, odd] = -1.0  # rotated[even] = -x[odd]
            rotation[odd, even] = 1.0  # rotated[odd] = x[even]
        start += size
    return cos[None, :, None, :], sin[None, :, None, :], rotation


class Block(torch.nn.Module):
    """An upstream transformer block with RoPE attention on [1, C, 1, N]."""

    def __init__(self, block, cos, sin, rotation, query_chunks):
        super().__init__()
        attention = block.attn
        self.heads, self.head_dim = attention.num_heads, attention.head_dim
        dim = self.heads * self.head_dim
        self.query_chunks = query_chunks
        self.norm1, self.norm2 = ChannelNorm(block.norm1), ChannelNorm(block.norm2)
        projections = []
        for rows in (slice(0, dim), slice(dim, 2 * dim), slice(2 * dim, 3 * dim)):
            linear = torch.nn.Linear(dim, dim)
            linear.weight.data = attention.qkv.weight.data[rows].clone()
            linear.bias.data = attention.qkv.bias.data[rows].clone()
            projections.append(linear)
        # The softmax scale is folded into the query; RoPE commutes with it.
        self.query = pointwise(projections[0], attention.scale)
        self.key, self.value = pointwise(projections[1]), pointwise(projections[2])
        self.rotate = torch.nn.Conv2d(self.head_dim, self.head_dim, 1, bias=False)
        self.rotate.weight.data = rotation[:, :, None, None].clone()
        self.register_buffer("cos", cos.clone())
        self.register_buffer("sin", sin.clone())
        self.output = pointwise(attention.proj)
        self.fc1, self.fc2 = pointwise(block.mlp.fc1), pointwise(block.mlp.fc2)

    def heads_last(self, x):
        """[1, heads * head_dim, 1, N] as [1, head_dim, heads, N]."""
        return x.reshape(1, self.heads, self.head_dim, -1).transpose(1, 2)

    def rope(self, x):
        return x * self.cos + self.rotate(x) * self.sin

    def forward(self, x):
        tokens = self.norm1(x)
        query = self.rope(self.heads_last(self.query(tokens)))
        key = self.rope(self.heads_last(self.key(tokens)))
        value = self.heads_last(self.value(tokens))
        contexts = []
        for chunk in torch.chunk(query, self.query_chunks, dim=3):
            # [1, keys, heads, queries]: softmax over the keys.
            probabilities = torch.softmax(torch.einsum("nchw,nchq->nqhw", chunk, key), dim=1)
            contexts.append(torch.einsum("nchw,nqhc->nqhw", probabilities, value))
        context = torch.cat(contexts, dim=3).transpose(1, 2).reshape(1, self.heads * self.head_dim, 1, -1)
        x = x + self.output(context)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x))))


class Encoder(torch.nn.Module):
    """Upstream's encoder for a frames x height x width clip. Input: the clip as stack_tubelets gives it,
    [1, 3 * tubelet, (frames / tubelet) * height, width]. Output: [1, 768, 1, N], tokens in upstream's
    (tubelet, row, column) order, after the last norms_block LayerNorm."""

    def __init__(self, encoder, frames, height, width, query_chunks=8):
        super().__init__()
        projection = encoder.patch_embed.proj  # Conv3d [dim, 3, tubelet, patch, patch]
        tubelet, patch = encoder.tubelet_size, encoder.patch_size
        self.dim = dim = projection.out_channels
        self.patch = torch.nn.Conv2d(3 * tubelet, dim, patch, stride=patch)
        weight = projection.weight.data.permute(0, 2, 1, 3, 4).reshape(dim, tubelet * 3, patch, patch)
        self.patch.weight.data = weight.clone()
        self.patch.bias.data = projection.bias.data.clone()
        self.register_buffer("modality", encoder.video_mod_embed.data.reshape(1, dim, 1, 1).clone())
        grid = (frames // tubelet, height // patch, width // patch)
        cos, sin, rotation = rope_tables(encoder.blocks[0].attn, *grid, dim // encoder.num_heads)
        self.blocks = torch.nn.ModuleList(Block(block, cos, sin, rotation, query_chunks) for block in encoder.blocks)
        self.norm = ChannelNorm(encoder.norms_block[-1])

    def part(self, index, x, parts):
        """Part index of parts, each running len(blocks) / parts blocks: the first takes the video and embeds its
        patches, the last applies the final norm, and the others take and return tokens."""
        if index == 0:
            x = self.patch(x).reshape(1, self.dim, 1, -1) + self.modality
        size = len(self.blocks) // parts
        for block in self.blocks[index * size : (index + 1) * size]:
            x = block(x)
        return self.norm(x) if index == parts - 1 else x

    def forward(self, video):
        x = self.patch(video).reshape(1, self.dim, 1, -1) + self.modality
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def stack_tubelets(video, tubelet=2):
    """A [1, 3, frames, height, width] clip as the encoder's [1, 3 * tubelet, (frames / tubelet) * height, width]."""
    batch, colours, frames, height, width = video.shape
    x = video.reshape(batch, colours, frames // tubelet, tubelet, height, width)
    return x.permute(0, 3, 1, 2, 4, 5).reshape(batch, tubelet * colours, frames // tubelet * height, width)
