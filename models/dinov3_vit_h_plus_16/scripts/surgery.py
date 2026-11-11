"""Rebuild DINOv3 ViT-H+/16 for the MLA from the converted weights, export it and check it against reference.onnx.

The rebuilt encoder runs timm's DINOv3 on the 5 prefix tokens (the class token and 4 registers) and the 196 patch
tokens of a 224x224 image as 4D, batch-one operations on a [1, HIDDEN_SIZE, 1, 201] tensor, channels on C and tokens
on W, and returns the same class_embedding and patch_embeddings. Each rewrite is exact:

- each Linear runs as a 1x1 convolution, the query, key and value projections as three;
- the rotary encoding q * cos + rotate_half(q) * sin runs on each head's channels, rotate_half as two channel slices
  and a negation; the prefix tokens, which timm leaves unrotated, take cosine 1 and sine 0. Folding rotate_half into
  the projections as extra output channels is exact too, but the wider convolutions made the ViT-S/16 w16a16 MLA
  stage 10% slower (3327 against 3017 us), and one fused query-key-value convolution split three ways 13 us slower;
- the attention scale 1/8, a power of two, folds into the query weights, and each LayerScale into the convolution
  before it. Each LayerNorm keeps its own scale and shift: folded into the INT8 weights of the convolution after it,
  they would scale its input channels, which per-output-channel weight scales cannot absorb (the ViT-S/16 w8a16 class
  embedding keeps cosine 0.976 with reference.onnx folded, 0.997 kept);
- attention runs for all heads at once as two Einsums and one Softmax over keys on axis 1 of the scores, a
  [1, 201 keys, HEADS, 201 queries] tensor;
- the MLP is timm's SwiGLU, SiLU(gate projection) * value projection, as two convolutions and a SiLU.

compile.py calls export_encoder. To get the FP32 ONNX on a host, run:

    python models/dinov3_vit_h_plus_16/scripts/surgery.py --weights weights.pth --reference reference.onnx \
        --output model.onnx
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx

IMAGE_SIZE = 224
PATCH_SIZE = 16
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
HIDDEN_SIZE = 1280
HEADS = 20
HEAD_SIZE = HIDDEN_SIZE // HEADS
MLP_SIZE = 4 * HIDDEN_SIZE
DEPTH = 32
REGISTERS = 4
PREFIX = 1 + REGISTERS
EPSILON = 1e-5
# convert.py's weights.pth entry with timm's rotary embedding of the 14x14 grid: [196, 128], sines then cosines.
ROTARY_EMBEDDING = "rope.embedding"
INPUTS = {"pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE]}
OUTPUTS = {"class_embedding": [1, HIDDEN_SIZE], "patch_embeddings": [1, PATCHES, HIDDEN_SIZE]}
INTERFACE = {**INPUTS, **OUTPUTS}
OPSET = 17
SEEDS = range(123, 127)
# The export's least cosine with reference.onnx per output on the seeded inputs; float32 rounding alone separates
# them, and the least output measured 0.9999998.
MIN_COSINE = 0.99999


def weight_shapes():
    """The shape of every weights.pth entry, as convert.py writes timm's DINOv3 ViT-H+/16."""
    shapes = {
        "cls_token": [1, 1, HIDDEN_SIZE],
        "reg_token": [1, REGISTERS, HIDDEN_SIZE],
        "patch_embed.proj.weight": [HIDDEN_SIZE, 3, PATCH_SIZE, PATCH_SIZE],
        "patch_embed.proj.bias": [HIDDEN_SIZE],
        "norm.weight": [HIDDEN_SIZE],
        "norm.bias": [HIDDEN_SIZE],
        ROTARY_EMBEDDING: [PATCHES, 2 * HEAD_SIZE],
    }
    for index in range(DEPTH):
        block = {
            "norm1.weight": [HIDDEN_SIZE], "norm1.bias": [HIDDEN_SIZE],
            "attn.qkv.weight": [3 * HIDDEN_SIZE, HIDDEN_SIZE],
            "attn.proj.weight": [HIDDEN_SIZE, HIDDEN_SIZE], "attn.proj.bias": [HIDDEN_SIZE],
            "gamma_1": [HIDDEN_SIZE], "gamma_2": [HIDDEN_SIZE],
            "norm2.weight": [HIDDEN_SIZE], "norm2.bias": [HIDDEN_SIZE],
            "mlp.fc1_g.weight": [MLP_SIZE, HIDDEN_SIZE], "mlp.fc1_g.bias": [MLP_SIZE],
            "mlp.fc1_x.weight": [MLP_SIZE, HIDDEN_SIZE], "mlp.fc1_x.bias": [MLP_SIZE],
            "mlp.fc2.weight": [HIDDEN_SIZE, MLP_SIZE], "mlp.fc2.bias": [HIDDEN_SIZE],
        }
        shapes.update({f"blocks.{index}.{name}": shape for name, shape in block.items()})
    return shapes


def load_weights(path):
    import torch

    weights = torch.load(path, weights_only=True)
    expected = weight_shapes()
    differing = sorted(name for name in {*weights, *expected}
                       if name not in weights or list(weights[name].shape) != expected.get(name))
    if differing:
        raise ValueError(f"weights.pth is not DINOv3 ViT-H+/16 as convert.py writes it; it differs at {differing[:4]}")
    return {name: value.to(torch.float64) for name, value in weights.items()}


def rotate_half(heads):
    """timm's DINOv3 rotate_half, [-x2, x1] on the two halves of each head's channels, on [1, HEAD_SIZE, heads, T]."""
    import torch

    half = HEAD_SIZE // 2
    return torch.cat([-heads[:, half:], heads[:, :half]], dim=1)


def build_encoder(weights):
    """DINOv3 ViT-H+/16 with every layer as a 4D, batch-one operation on [1, C, 1, tokens] tensors."""
    import torch
    from torch import nn

    def conv(weight, bias=None):
        layer = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
        layer.weight.data = weight[:, :, None, None].float().clone()
        if bias is not None:
            layer.bias.data = bias.float().clone()
        return layer

    class Norm(nn.Module):
        """timm's LayerNorm over each token's channels, with its own scale and shift."""

        def __init__(self, prefix):
            super().__init__()
            self.register_buffer("scale", weights[f"{prefix}.weight"].float().clone())
            self.register_buffer("shift", weights[f"{prefix}.bias"].float().clone())

        def forward(self, hidden):
            channels_last = hidden.permute(0, 2, 3, 1)
            normalized = nn.functional.layer_norm(channels_last, (HIDDEN_SIZE,), self.scale, self.shift, EPSILON)
            return normalized.permute(0, 3, 1, 2)

    def split_heads(tensor):
        # [1, heads * 64, 1, T] -> [1, 64, heads, T]; channel h * 64 + d is head h, as timm splits heads.
        return torch.cat(torch.split(tensor, HEAD_SIZE, dim=1), dim=2)

    sines, cosines = weights[ROTARY_EMBEDDING].T.tensor_split(2)

    class Block(nn.Module):
        def __init__(self, index):
            super().__init__()
            weight = {name.removeprefix(f"blocks.{index}."): value for name, value in weights.items()
                      if name.startswith(f"blocks.{index}.")}
            query, key, value = weight["attn.qkv.weight"].chunk(3)
            query = query * HEAD_SIZE**-0.5
            self.norm1, self.norm2 = Norm(f"blocks.{index}.norm1"), Norm(f"blocks.{index}.norm2")
            self.query, self.key, self.value = conv(query), conv(key), conv(value)
            tokens = PREFIX + PATCHES
            cosine, sine = torch.ones(1, HEAD_SIZE, 1, tokens), torch.zeros(1, HEAD_SIZE, 1, tokens)
            cosine[0, :, 0, PREFIX:], sine[0, :, 0, PREFIX:] = cosines, sines
            self.register_buffer("cosine", cosine)
            self.register_buffer("sine", sine)
            # LayerScale gamma * (W x + b) folds into the convolution before it.
            layer_scale = weight["gamma_1"][:, None]
            self.output = conv(weight["attn.proj.weight"] * layer_scale, weight["attn.proj.bias"] * layer_scale[:, 0])
            self.gate = conv(weight["mlp.fc1_g.weight"], weight["mlp.fc1_g.bias"])
            self.fc1 = conv(weight["mlp.fc1_x.weight"], weight["mlp.fc1_x.bias"])
            layer_scale = weight["gamma_2"][:, None]
            self.fc2 = conv(weight["mlp.fc2.weight"] * layer_scale, weight["mlp.fc2.bias"] * layer_scale[:, 0])

        def attention(self, hidden):
            query, key, value = (split_heads(projection(hidden)) for projection in (self.query, self.key, self.value))
            query = query * self.cosine + rotate_half(query) * self.sine
            key = key * self.cosine + rotate_half(key) * self.sine
            # n batch, h head, w query token; scores: c channel, q key token; context: c key token, q value channel.
            probabilities = torch.softmax(torch.einsum("nchw,nchq->nqhw", query, key), dim=1)
            context = torch.einsum("nchw,nqhc->nqhw", probabilities, value)
            return self.output(torch.cat(torch.split(context, 1, dim=2), dim=1))

        def forward(self, hidden):
            hidden = hidden + self.attention(self.norm1(hidden))
            normalized = self.norm2(hidden)
            return hidden + self.fc2(nn.functional.silu(self.gate(normalized)) * self.fc1(normalized))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patches = nn.Conv2d(3, HIDDEN_SIZE, PATCH_SIZE, PATCH_SIZE)
            self.patches.weight.data = weights["patch_embed.proj.weight"].float().clone()
            self.patches.bias.data = weights["patch_embed.proj.bias"].float().clone()
            prefix = torch.cat([weights["cls_token"], weights["reg_token"]], dim=1)
            self.register_buffer("prefix", prefix.permute(0, 2, 1)[:, :, None].float().clone())
            self.blocks = nn.ModuleList(Block(index) for index in range(DEPTH))
            self.norm = Norm("norm")

        def forward(self, pixel_values):
            hidden = torch.cat([self.prefix, self.patches(pixel_values).reshape(1, HIDDEN_SIZE, 1, PATCHES)], dim=3)
            for block in self.blocks:
                hidden = block(hidden)
            hidden = self.norm(hidden)
            return (hidden[..., 0].reshape(1, HIDDEN_SIZE),
                    hidden[..., PREFIX:].reshape(1, HIDDEN_SIZE, PATCHES).transpose(1, 2))

    return Encoder().eval()


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    # From the path, the checker finds the external data beside the file.
    onnx.checker.check_model(path)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The DINOv3 export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected inputs and outputs {list(INTERFACE)}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    dynamic = sorted({node.op_type for node in model.graph.node if node.op_type in {"Shape", "Size", "If", "Loop"}})
    if dynamic:
        raise ValueError(f"The folded export still computes shapes or control flow: {dynamic}")


def save_export(exported, path):
    """Save exported to path, with the weights, above protobuf's 2 GB limit, in one external data file beside it.

    torch has folded the constants. onnxsim 0.4 in the Model Compiler cannot read a model above 2 GB; on ViT-S+/16 it
    only turns torch's Constant nodes into initializers and drops its Identity nodes, and changes no output.
    """
    onnx.save(onnx.load(exported), path, save_as_external_data=True, all_tensors_to_one_file=True,
              location=f"{path.name}.data")


def cosine(expected, actual):
    expected, actual = np.asarray(expected, np.float64).ravel(), np.asarray(actual, np.float64).ravel()
    return float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))


def compare_embeddings(expected, actual):
    """Each output must have reference.onnx's shape, be finite and keep at least MIN_COSINE with it."""
    for name, want, got in zip(OUTPUTS, expected, actual, strict=True):
        if got.shape != want.shape or not np.isfinite(got).all():
            raise ValueError(f"{name} has shape {got.shape} or non-finite values; reference.onnx gives {want.shape}")
        if cosine(want, got) < MIN_COSINE:
            raise ValueError(f"{name} keeps cosine {cosine(want, got):.7f} with reference.onnx, below {MIN_COSINE}")


def synthetic_pixels(seed):
    return np.random.RandomState(seed).standard_normal(INPUTS["pixel_values"]).astype(np.float32)


def export_encoder(weights, reference, output):
    """Export the rebuilt encoder to output and require its embeddings to follow reference.onnx."""
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    encoder = build_encoder(load_weights(weights))
    # Above 2 GB torch writes each weight to its own file beside the export, so it exports into a directory of its own.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary, torch.inference_mode():
        exported = Path(temporary) / output.name
        torch.onnx.export(
            encoder, (torch.from_numpy(synthetic_pixels(SEEDS.start)),), str(exported), input_names=list(INPUTS),
            output_names=list(OUTPUTS), opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
        save_export(exported, output)
    validate_export(output)
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (reference, output)]
    for seed in SEEDS:
        pixels = synthetic_pixels(seed)
        expected, actual = (session.run(list(OUTPUTS), {"pixel_values": pixels}) for session in sessions)
        compare_embeddings(expected, actual)
    print(f"Seeded-input reference.onnx/export check passed at cosine {MIN_COSINE} with seeds "
          f"{SEEDS.start}-{SEEDS.stop - 1}: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="the converted weights.pth")
    parser.add_argument("--reference", type=Path, required=True, help="the converted reference.onnx")
    parser.add_argument("--output", type=Path, required=True, help="the ONNX to write")
    args = parser.parse_args()
    export_encoder(args.weights.resolve(), args.reference.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
