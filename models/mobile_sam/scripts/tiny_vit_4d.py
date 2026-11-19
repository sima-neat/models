"""TinyViT's image encoder rewritten in the 4D layout the compiler handles, built from the upstream encoder's
weights. It computes the same embedding as tiny_vit_sam.TinyViT.

Upstream carries each attention stage as tokens [1, H * W, C] and its windows as a batch of [windows, tokens, C]
sets. Here the feature map stays [1, C, H, W] from the patch embedding to the neck: window padding and partition use
only 4D reshapes and transposes, giving [1, C, windows, tokens]; every Linear is a 1x1 convolution; heads join the
window axis, so attention is two Einsums with the attention bias as a per-block constant. Within a window the
partition orders tokens column-major, and the bias is permuted to match. PatchMerging's convolutions run on the
feature map directly.
"""

import torch
import torch.nn.functional as F


def pointwise(linear, rows=None):
    """A Linear, or some of its output rows, as a 1x1 convolution."""
    rows = slice(None) if rows is None else rows
    layer = torch.nn.Conv2d(linear.in_features, linear.weight.data[rows].shape[0], 1)
    layer.weight.data = linear.weight.data[rows][:, :, None, None].clone()
    layer.bias.data = linear.bias.data[rows].clone()
    return layer


class ChannelNorm(torch.nn.Module):
    """A LayerNorm over the channel axis of [1, C, H, W]."""

    def __init__(self, norm):
        super().__init__()
        self.eps = norm.eps
        self.weight = torch.nn.Parameter(norm.weight.data.reshape(1, -1, 1, 1).clone())
        self.bias = torch.nn.Parameter(norm.bias.data.reshape(1, -1, 1, 1).clone())

    def forward(self, x):
        centred = x - x.mean(dim=1, keepdim=True)
        variance = (centred * centred).mean(dim=1, keepdim=True)
        return centred / torch.sqrt(variance + self.eps) * self.weight + self.bias


class WindowBlock(torch.nn.Module):
    """An upstream TinyViTBlock on [1, C, H, W]."""

    def __init__(self, block):
        super().__init__()
        attention = block.attn
        self.height, self.width = block.input_resolution
        self.window = window = block.window_size
        self.pad_bottom = (window - self.height % window) % window
        self.pad_right = (window - self.width % window) % window
        self.rows = (self.height + self.pad_bottom) // window
        self.columns = (self.width + self.pad_right) // window
        heads, key_dim, value_dim = attention.num_heads, attention.key_dim, attention.d
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.norm = ChannelNorm(attention.norm)
        # qkv's output is per head (query key_dim, key key_dim, value value_dim); split it into three convolutions.
        per_head = 2 * key_dim + value_dim
        index = torch.arange(heads * per_head).reshape(heads, per_head)
        self.query = pointwise(attention.qkv, index[:, :key_dim].reshape(-1))
        self.query.weight.data *= attention.scale
        self.query.bias.data *= attention.scale
        self.key = pointwise(attention.qkv, index[:, key_dim : 2 * key_dim].reshape(-1))
        self.value = pointwise(attention.qkv, index[:, 2 * key_dim :].reshape(-1))
        self.output = pointwise(attention.proj)
        # Upstream's bias[head, query, key] over row-major window tokens, reordered to the partition's column-major
        # tokens, as [1, keys, heads * windows, queries].
        tokens = window * window
        order = torch.arange(tokens).reshape(window, window).T.reshape(-1)
        bias = attention.attention_biases.data[:, attention.attention_bias_idxs][:, order][:, :, order]
        windows = self.rows * self.columns
        bias = bias.permute(2, 0, 1)[:, :, None, :].expand(tokens, heads, windows, tokens)
        self.register_buffer("bias", bias.reshape(1, tokens, heads * windows, tokens).clone())
        self.local_conv = block.local_conv
        self.mlp_norm = ChannelNorm(block.mlp.norm)
        self.fc1, self.fc2 = pointwise(block.mlp.fc1), pointwise(block.mlp.fc2)

    def partition(self, x):
        """[1, C, rows * w, columns * w] as [1, C, rows * columns, w * w], tokens column-major in each window."""
        channels = x.shape[1]
        x = x.reshape(1, channels * self.rows, self.window, self.columns * self.window).transpose(2, 3)
        return x.reshape(1, channels, self.rows * self.columns, self.window * self.window)

    def reverse(self, x):
        channels = x.shape[1]
        x = x.reshape(1, channels * self.rows, self.columns * self.window, self.window).transpose(2, 3)
        return x.reshape(1, channels, self.rows * self.window, self.columns * self.window)

    def heads_on_windows(self, x, size):
        """[1, heads * size, windows, tokens] as [1, size, heads * windows, tokens]."""
        windows, tokens = x.shape[2], x.shape[3]
        x = x.reshape(1, self.heads, size, windows * tokens).transpose(1, 2)
        return x.reshape(1, size, self.heads * windows, tokens)

    def attend(self, x):
        windows, tokens = x.shape[2], x.shape[3]
        x = self.norm(x)
        query = self.heads_on_windows(self.query(x), self.key_dim)
        key = self.heads_on_windows(self.key(x), self.key_dim)
        value = self.heads_on_windows(self.value(x), self.value_dim)
        scores = torch.einsum("nchw,nchq->nqhw", query, key) + self.bias
        context = torch.einsum("nchw,nqhc->nqhw", torch.softmax(scores, dim=1), value)
        context = context.reshape(1, self.value_dim, self.heads, windows * tokens).transpose(1, 2)
        return self.output(context.reshape(1, self.heads * self.value_dim, windows, tokens))

    def forward(self, x):
        padded = F.pad(x, (0, self.pad_right, 0, self.pad_bottom)) if self.pad_bottom or self.pad_right else x
        attended = self.reverse(self.attend(self.partition(padded)))
        x = x + attended[:, :, : self.height, : self.width]
        x = self.local_conv(x)
        return x + self.fc2(F.gelu(self.fc1(self.mlp_norm(x))))


def merge(downsample, x):
    """An upstream PatchMerging on [1, C, H, W], returning [1, C', H', W'] instead of tokens."""
    x = downsample.act(downsample.conv1(x))
    x = downsample.act(downsample.conv2(x))
    return downsample.conv3(x)


class ImageEncoder(torch.nn.Module):
    """A whole upstream TinyViT encoder, [1,3,1024,1024] -> [1,256,64,64], in the 4D compiler layout."""

    def __init__(self, encoder):
        super().__init__()
        self.patch_embed = encoder.patch_embed
        self.convolutions = encoder.layers[0].blocks
        self.downsamples = torch.nn.ModuleList(layer.downsample for layer in encoder.layers[:-1])
        self.stages = torch.nn.ModuleList(
            torch.nn.ModuleList(WindowBlock(block) for block in layer.blocks) for layer in encoder.layers[1:]
        )
        self.neck = encoder.neck

    def forward(self, image):
        x = self.patch_embed(image)
        for block in self.convolutions:
            x = block(x)
        x = merge(self.downsamples[0], x)
        for block in self.stages[0]:
            x = block(x)
        x = merge(self.downsamples[1], x)
        for block in self.stages[1]:
            x = block(x)
        x = merge(self.downsamples[2], x)
        for block in self.stages[2]:
            x = block(x)
        return self.neck(x)
