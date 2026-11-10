"""The checkpoint's CLIP ViT-B/32 text tower, with CPU token lookup.

All twelve transformer blocks, EOS pooling, projection and L2 normalization
remain in the compiled graph. The context is always 77 tokens; the one-hot
selector has 96 storage entries, of which only the first 77 are model tokens.
"""
import torch
from torch import nn
from torch.nn import functional as F
from detector import Weights

TOKENS, SELECTOR_WIDTH, WIDTH, HEADS = 77, 96, 512, 8


def linear_conv(weights, prefix):
    linear = weights.linear(prefix)
    module = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
    with torch.no_grad():
        module.weight.copy_(linear.weight[:, :, None, None])
        if linear.bias is not None:
            module.bias.copy_(linear.bias)
    return module


class ChannelNorm(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.weight = nn.Parameter(weights.take(prefix + ".weight"))
        self.bias = nn.Parameter(weights.take(prefix + ".bias"))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        return F.layer_norm(x, (WIDTH,), self.weight, self.bias, 1e-5).permute(0, 3, 1, 2)


def pack_heads(x):
    return torch.cat(torch.split(x, WIDTH // HEADS, dim=1), dim=2)


def merge_heads(x):
    # Each head retains every independent prompt row. The deployed graph has one row.
    return torch.cat(torch.split(x, x.shape[2] // HEADS, dim=2), dim=1)


class Block(nn.Module):
    def __init__(self, weights, prefix):
        super().__init__()
        self.ln1 = ChannelNorm(weights, prefix + ".layer_norm1")
        self.ln2 = ChannelNorm(weights, prefix + ".layer_norm2")
        projections = [weights.linear(prefix + ".self_attn." + name + "_proj") for name in ("q", "k", "v")]
        self.qkv = nn.Conv2d(WIDTH, 3 * WIDTH, 1)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat([p.weight for p in projections])[:, :, None, None])
            self.qkv.bias.copy_(torch.cat([p.bias for p in projections]))
        self.out = linear_conv(weights, prefix + ".self_attn.out_proj")
        self.fc1 = linear_conv(weights, prefix + ".mlp.fc1")
        self.fc2 = linear_conv(weights, prefix + ".mlp.fc2")

    def forward(self, x, causal_mask):
        q, k, v = self.qkv(self.ln1(x)).split(WIDTH, dim=1)
        q, k, v = pack_heads(q) * (WIDTH // HEADS)**-0.5, pack_heads(k), pack_heads(v)
        scores = torch.einsum("nchw,nchq->nqhw", q, k) + causal_mask
        attended = torch.einsum("nchw,nqhc->nqhw", scores.softmax(dim=1), v)
        x = x + self.out(merge_heads(attended))
        hidden = self.fc1(self.ln2(x))
        return x + self.fc2(hidden * torch.sigmoid(1.702 * hidden))


class TextEncoder(nn.Module):
    def __init__(self, state):
        super().__init__()
        weights = Weights(state)
        prefix = "backbone.text_model.model"
        position = weights.take(prefix + ".text_model.embeddings.position_embedding.weight")
        self.register_buffer("position", position.T[None, :, None, :].clone())
        mask = torch.triu(torch.ones(TOKENS, TOKENS), diagonal=1) * -10000.0
        self.register_buffer("causal_mask", mask.T[None, :, None, :])
        self.layers = nn.ModuleList(Block(weights, prefix + f".text_model.encoder.layers.{i}") for i in range(12))
        self.final_norm = ChannelNorm(weights, prefix + ".text_model.final_layer_norm")
        # CLIP's text projection has no bias.
        projection = weights.take(prefix + ".text_projection.weight")
        self.projection = nn.Conv2d(WIDTH, WIDTH, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(projection[:, :, None, None])
        ignored = {prefix + ".text_model.embeddings." + name for name in ("token_embedding.weight", "position_ids")}
        unused = {name for name in state if name.startswith("backbone.text_model.")} - weights.used - ignored
        if unused:
            raise ValueError(f"unconsumed text checkpoint tensors: {sorted(unused)}")
        self.requires_grad_(False)
        self.eval()

    def forward(self, token_embeddings, eos_selector):
        x = token_embeddings + self.position
        for layer in self.layers:
            x = layer(x, self.causal_mask)
        x = self.final_norm(x)
        x = (x * eos_selector[:, :, :, :TOKENS]).sum(dim=3, keepdim=True)
        return F.normalize(self.projection(x), p=2, dim=1, eps=1e-12)


def token_table(state):
    return state["backbone.text_model.model.text_model.embeddings.token_embedding.weight"].detach().numpy().copy()
