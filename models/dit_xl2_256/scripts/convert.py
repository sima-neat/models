#!/usr/bin/env python3
"""Export DiT-XL/2 256's four transformer parts and its VAE decoder side by side in one ONNX, check the parts against
the whole transformer, and the ONNX against PyTorch.

The source is the pinned diffusers checkpoint. Each transformer block embeds the timestep and the class itself;
the export feeds those embedders the CPU's timestep features and a one-hot class (dit.py), which equals
diffusers' own embedding. At float32 the two components' weights exceed ONNX's 2 GB single-file limit, so
weights are stored as float16 behind a Cast; the w16a16 build keeps fewer mantissa bits than float16 has. The
check compares ONNX Runtime with PyTorch running the same float16-rounded weights.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import onnx
from dit import (
    COMPONENTS,
    HIDDEN,
    INPUT_SHAPES,
    LATENT_CHANNELS,
    OPSET,
    OUTPUT_SHAPES,
    PATCH_GRID,
    TRANSFORMER,
    TRANSFORMER_PARTS,
    VAE_SCALING,
    class_one_hot,
    part_blocks,
    tensor_name,
    timestep_features,
)
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from onnx import TensorProto, helper, numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[1]
FILES = ("model_index.json", "transformer/config.json", "transformer/diffusion_pytorch_model.bin",
         "vae/config.json", "vae/diffusion_pytorch_model.bin")
# Initializers this large are stored as float16; smaller ones, such as norms' and shapes', stay as they are.
FLOAT16_MIN_ELEMENTS = 4096
MIN_COSINE = 0.9999


def validate_source(source: dict) -> None:
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "model_index.json" or files != set(FILES):
        raise ValueError(f"the source must be the DiT pipeline's {', '.join(FILES)}, opened at model_index.json")


def validate_config(root: Path) -> None:
    transformer = json.loads((root / "transformer/config.json").read_text(encoding="utf-8"))
    expected = {"num_layers": 28, "num_attention_heads": 16, "attention_head_dim": 72, "patch_size": 2,
                "sample_size": 32, "in_channels": 4, "out_channels": 8, "norm_type": "ada_norm_zero",
                "num_embeds_ada_norm": 1000}
    actual = {key: transformer.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Unsupported DiT transformer config: {actual}")


def build(root: Path):
    """The diffusers transformer, its embedders fed the CPU's features, and the VAE with its attention in 4D, in
    float32 eval mode; also diffusers' own embedding and VAE attention, for the checks."""
    import torch
    from diffusers import AutoencoderKL, Transformer2DModel

    class FeatureEmbedding(torch.nn.Module):
        """CombinedTimestepLabelEmbeddings with the sinusoid on the CPU and the label lookup as a matrix product."""

        def __init__(self, embedding):
            super().__init__()
            self.timestep_embedder = embedding.timestep_embedder
            self.table = embedding.class_embedder.embedding_table

        def forward(self, features, one_hot, hidden_dtype=None):
            batch = features.shape[0]
            return self.timestep_embedder(features.reshape(batch, -1)) + one_hot.reshape(batch, -1) @ self.table.weight

    transformer = Transformer2DModel.from_pretrained(root / "transformer", use_safetensors=False).eval()
    original = transformer.transformer_blocks[0].norm1.emb
    for block in transformer.transformer_blocks:
        block.norm1.emb = FeatureEmbedding(block.norm1.emb)
    vae = AutoencoderKL.from_pretrained(root / "vae", use_safetensors=False).eval()

    def conv(linear, scale=1.0):
        layer = torch.nn.Conv2d(linear.in_features, linear.out_features, 1)
        layer.weight.data = linear.weight.data[:, :, None, None] * scale
        layer.bias.data = linear.bias.data * scale
        return layer

    class GroupNorm4D(torch.nn.Module):
        """GroupNorm with every tensor 4D: per-channel means over H and W, averaged within each group by a constant
        1x1 convolution, with the variance taken about the group mean. Exported as is, GroupNorm reshapes the
        decoder's 128x256x256 activations to [1, 32, 4 x 256 x 256], which kept compiler 3.0.0 in its large-tensor
        segmentation for over 4 hours; this form leaves it there for minutes."""

        def __init__(self, norm):
            super().__init__()
            channels, groups = norm.num_channels, norm.num_groups
            size = channels // groups
            average = torch.zeros(channels, channels, 1, 1)
            for group in range(groups):
                average[group * size:(group + 1) * size, group * size:(group + 1) * size] = 1.0 / size
            self.average = torch.nn.Conv2d(channels, channels, 1, bias=False)
            self.average.weight.data = average
            self.eps = norm.eps
            self.weight = torch.nn.Parameter(norm.weight.detach().clone().view(1, -1, 1, 1))
            self.bias = torch.nn.Parameter(norm.bias.detach().clone().view(1, -1, 1, 1))

        def forward(self, x):
            centred = x - self.average(x.mean(dim=(2, 3), keepdim=True))
            variance = self.average((centred * centred).mean(dim=(2, 3), keepdim=True))
            return centred / torch.sqrt(variance + self.eps) * self.weight + self.bias

    def replace_norms(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GroupNorm):
                setattr(module, name, GroupNorm4D(child))
            else:
                replace_norms(child)

    class Attention4D(torch.nn.Module):
        """The decoder's mid-block attention over its 1024 latent pixels as 4D Einsums on [1, 512, 1, 1024], the
        softmax over keys on axis 1, for QUERY_CHUNKS groups of queries at a time: one [1, 1024, 1, 1024] score
        tensor needs the compiler's large-tensor segmentation, while four of [1, 1024, 1, 256] compile as one MLA
        segment in about two minutes."""

        QUERY_CHUNKS = 4

        def __init__(self, attention):
            super().__init__()
            if attention.heads != 1 or not attention.residual_connection or attention.rescale_output_factor != 1:
                raise ValueError("unsupported VAE attention")
            self.norm = attention.group_norm
            self.query = conv(attention.to_q, attention.to_q.in_features ** -0.5)
            self.key, self.value, self.output = conv(attention.to_k), conv(attention.to_v), conv(attention.to_out[0])

        def forward(self, hidden, *args, **kwargs):
            batch, channels, height, width = hidden.shape
            tokens = self.norm(hidden).reshape(batch, channels, 1, height * width)
            query, key, value = self.query(tokens), self.key(tokens), self.value(tokens)
            contexts = []
            for chunk in torch.chunk(query, self.QUERY_CHUNKS, dim=3):
                probabilities = torch.softmax(torch.einsum("nchw,nchq->nqhw", chunk, key), dim=1)
                contexts.append(torch.einsum("nchw,nqhc->nqhw", probabilities, value))
            return self.output(torch.cat(contexts, dim=3)).reshape(batch, channels, height, width) + hidden

    vae_attention = vae.decoder.mid_block.attentions[0]
    vae.decoder.mid_block.attentions[0] = Attention4D(vae_attention)
    reference_decoder = copy.deepcopy(vae.decoder)
    replace_norms(vae.decoder)

    class Part(torch.nn.Module):
        """Blocks part_blocks(part) of diffusers' Transformer2DModel.forward on patched input: the first part also
        embeds the patches, the last also projects and unpatchifies; parts pass tokens on as a [1, 1152, 16, 16]
        grid."""

        def __init__(self, part):
            super().__init__()
            self.part, self.transformer = part, transformer

        def forward(self, value, features, one_hot):
            batch = value.shape[0]
            if self.part == 0:
                hidden = self.transformer.pos_embed(value)
            else:
                hidden = value.reshape(batch, HIDDEN, PATCH_GRID * PATCH_GRID).transpose(1, 2)
            for index in part_blocks(self.part):
                hidden = self.transformer.transformer_blocks[index](hidden, timestep=features, class_labels=one_hot)
            if self.part < TRANSFORMER_PARTS - 1:
                return hidden.transpose(1, 2).reshape(batch, HIDDEN, PATCH_GRID, PATCH_GRID)
            # DiTTransformer2DModel.forward's output: block 0's conditioning modulates the final norm, then unpatchify.
            model = self.transformer
            conditioning = model.transformer_blocks[0].norm1.emb(features, one_hot, hidden_dtype=hidden.dtype)
            shift, scale = model.proj_out_1(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
            hidden = model.proj_out_2(model.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None])
            patch, channels = model.patch_size, model.out_channels
            hidden = hidden.reshape(batch, PATCH_GRID, PATCH_GRID, patch, patch, channels)
            size = PATCH_GRID * patch
            return torch.einsum("nhwpqc->nchpwq", hidden).reshape(batch, channels, size, size)

    class SideBySide(torch.nn.Module):
        """Every component on its own inputs, in COMPONENTS order."""

        def __init__(self):
            super().__init__()
            self.parts = torch.nn.ModuleList(Part(part) for part in range(TRANSFORMER_PARTS))
            self.vae = vae

        def forward(self, *inputs):
            outputs = [part(*inputs[3 * index:3 * index + 3]) for index, part in enumerate(self.parts)]
            return (*outputs, self.vae.decode(inputs[-1] / VAE_SCALING).sample)

    return SideBySide().eval(), transformer, original, vae_attention, reference_decoder


def check_embedding(transformer, original) -> None:
    """The feature embedding equals diffusers' own, `original`, on its timestep and label inputs."""
    import torch

    with torch.inference_mode():
        expected = original(torch.tensor([500.0, 500.0]), torch.tensor([207, 1000]), hidden_dtype=torch.float32)
        actual = transformer.transformer_blocks[0].norm1.emb(
            torch.from_numpy(timestep_features(500.0)), torch.from_numpy(class_one_hot(207)))
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4, err_msg="embedding")


def check_vae_norms(model, reference_decoder) -> None:
    """The decoder with 4D GroupNorms equals it with torch's GroupNorm, on a seeded latent."""
    import torch

    latent = torch.from_numpy(np.random.RandomState(1).standard_normal((1, LATENT_CHANNELS, 32, 32)).astype(np.float32))
    with torch.inference_mode():
        expected, actual = reference_decoder(latent), model.vae.decoder(latent)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4, err_msg="VAE GroupNorm")


def check_vae_attention(model, original) -> None:
    """The decoder's 4D attention equals diffusers' own, `original`, on a seeded mid-block input."""
    import torch

    hidden = torch.from_numpy(np.random.RandomState(0).standard_normal((1, 512, 32, 32)).astype(np.float32))
    with torch.inference_mode():
        expected, actual = original(hidden), model.vae.decoder.mid_block.attentions[0](hidden)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4, err_msg="VAE attention")


def store_float16(path: Path, output: Path) -> None:
    """Rewrite large float32 initializers as float16 consumed through a Cast, in one file. The export keeps its
    weights in external files; each is read from disk on its own, so only the float16 model is held at once."""
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    def internalize(tensor: onnx.TensorProto) -> None:
        if tensor.data_location != TensorProto.EXTERNAL:
            return
        info = {entry.key: entry.value for entry in tensor.external_data}
        with open(path.parent / info["location"], "rb") as stream:
            stream.seek(int(info.get("offset", 0)))
            raw = stream.read(int(info["length"])) if "length" in info else stream.read()
        del tensor.external_data[:]
        tensor.data_location = TensorProto.DEFAULT
        tensor.raw_data = raw

    # The export also stores some Constant nodes' values externally.
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                internalize(attribute.t)
    casts, kept = [], []
    for initializer in graph.initializer:
        internalize(initializer)
        if initializer.data_type != TensorProto.FLOAT or np.prod(initializer.dims) < FLOAT16_MIN_ELEMENTS:
            kept.append(initializer)
            continue
        stored = numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float16),
                                         f"{initializer.name}_float16")
        initializer.ClearField("raw_data")
        kept.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [initializer.name], to=TensorProto.FLOAT,
                                      name=f"{initializer.name}_cast"))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    # The transformer's output comes out of the export without a static shape; declare it.
    declared = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    for value in graph.output:
        dims = value.type.tensor_type.shape.dim
        del dims[:]
        for size in declared[value.name]:
            dims.add().dim_value = size
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, str(output))


def round_to_float16(module) -> None:
    """Round the PyTorch weights as store_float16 does, so the check isolates the export."""
    import torch

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.numel() >= FLOAT16_MIN_ELEMENTS:
                parameter.copy_(parameter.half().float())


def validate_export(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    shape = lambda value: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)  # noqa: E731
    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in INPUT_SHAPES[c].items()}
    expected_outputs = {tensor_name(c, n): s for c in COMPONENTS for n, s in OUTPUT_SHAPES[c].items()}
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"Unexpected DiT export interface: inputs {inputs}, outputs {outputs}")
    if not any(opset.domain == "" and opset.version == OPSET for opset in model.opset_import):
        raise ValueError(f"the export must use opset {OPSET}")


def cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected output: shape {actual.shape}")
    expected, actual = expected.ravel().astype(np.float64), actual.ravel().astype(np.float64)
    return float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))


def sample_inputs(model) -> dict[str, np.ndarray]:
    """The class slot of one guidance step, each part fed the previous part's output, and a decoder latent."""
    import torch

    rng = np.random.RandomState(0)
    latent = rng.standard_normal(INPUT_SHAPES["transformer_0"]["latent"]).astype(np.float32)
    features, one_hot = timestep_features(500.0)[:1], class_one_hot(207)[:1]
    inputs, value = {}, latent
    for index, part in enumerate(TRANSFORMER):
        inputs[tensor_name(part, "latent" if index == 0 else "previous_hidden")] = value
        inputs[tensor_name(part, "timestep_features")] = features
        inputs[tensor_name(part, "class_one_hot")] = one_hot
        with torch.inference_mode():
            value = model.parts[index](*(torch.from_numpy(item) for item in (value, features, one_hot))).numpy()
    inputs[tensor_name("vae_decoder", "latent")] = (latent * VAE_SCALING).astype(np.float32)
    return inputs


def check_parts(model, transformer, inputs: dict[str, np.ndarray]) -> None:
    """The parts in turn equal the whole transformer."""
    import torch

    names = [tensor_name("transformer_0", name) for name in INPUT_SHAPES["transformer_0"]]
    with torch.inference_mode():
        whole = transformer(*(torch.from_numpy(inputs[name]) for name in names[:1]),
                            timestep=torch.from_numpy(inputs[names[1]]),
                            class_labels=torch.from_numpy(inputs[names[2]])).sample.numpy()
        last = TRANSFORMER[-1]
        chained = model.parts[-1](*(torch.from_numpy(inputs[tensor_name(last, name)])
                                    for name in INPUT_SHAPES[last])).numpy()
    if cosine(whole, chained) < MIN_COSINE:
        raise ValueError(f"the transformer parts disagree with the whole transformer: cosine {cosine(whole, chained)}")


def convert(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    root = source.parent
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(root)
    model, transformer, original, vae_attention, reference_decoder = build(root)
    check_embedding(transformer, original)
    check_vae_attention(model, vae_attention)
    check_vae_norms(model, reference_decoder)
    del original, vae_attention, reference_decoder
    round_to_float16(model)
    inputs = sample_inputs(model)
    check_parts(model, transformer, inputs)
    names = [tensor_name(c, n) for c in COMPONENTS for n in INPUT_SHAPES[c]]
    float32_dir = output_dir / "float32"
    float32_dir.mkdir()
    with torch.inference_mode():
        expected = [value.numpy() for value in model(*(torch.from_numpy(inputs[name]) for name in names))]
        torch.onnx.export(
            model, tuple(torch.from_numpy(inputs[name]) for name in names), str(float32_dir / "model.onnx"),
            input_names=names,
            output_names=[tensor_name(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]],
            opset_version=OPSET, dynamo=False,
        )
    del model, transformer
    gc.collect()
    store_float16(float32_dir / "model.onnx", output)
    for item in float32_dir.iterdir():
        item.unlink()
    float32_dir.rmdir()
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {name: inputs[name] for name in names})
    scores = {tensor_name(c, n): cosine(want, got) for (c, n), want, got in zip(
        [(c, n) for c in COMPONENTS for n in OUTPUT_SHAPES[c]], expected, actual)}
    if min(scores.values()) < MIN_COSINE:
        raise ValueError(f"ONNX Runtime disagrees with PyTorch: cosine {scores}")
    noise = actual[len(TRANSFORMER) - 1][:, :LATENT_CHANNELS]
    print(f"PyTorch/ONNX check passed (cosine {scores}; noise std {noise.std():.3f}): {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
