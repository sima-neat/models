"""Adapt CLIP ViT-B/16 in PyTorch, export ONNX and split its image and text encoders.

The image encoder returns its raw embedding. ImageNet scoring stays in accuracy.py.
The text encoder starts after token lookup and returns each token's projected row; the CPU selects end-of-text.
Its static causal mask uses -1e4, which gives zero masked softmax probabilities while remaining finite in BF16.
A separate prompts.npz holds 64 fixed prompts for numerical fidelity; neither encoder needs calibration.

Run with --weights weights.pth --output-dir graphs. To regenerate scripts/text_probe.npz, pass --text-probe
<graphs/model.onnx> and a fresh --output-dir, then update the checksum in accuracy.py and performance.py.
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from calibration import CLASS_NAMES, TEMPLATES

IMAGE_SIZE = 224
TEXT_LENGTH = 77
EMBED_DIM = 512
CLASS_COUNT = 1000
FIDELITY_PROMPTS = 64
TOKENIZER_FILE = "tokenizer.json"
TOKEN_EMBEDDINGS_FILE = "token_embeddings.npy"
RUNTIME_FILES = (TOKENIZER_FILE, TOKEN_EMBEDDINGS_FILE)
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "token_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "image_embeds": [1, EMBED_DIM],
    "text_embeds": [1, TEXT_LENGTH, EMBED_DIM],
}
INPUTS = ("pixel_values", "token_embeds")
OUTPUTS = ("image_embeds", "text_embeds")
TEXT_PROBE_PROMPTS = tuple(f"a photo of a {CLASS_NAMES[index]}." for index in range(0, CLASS_COUNT, 125))


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The CLIP export requires opset 17")
    if [value.name for value in model.graph.input] != list(INPUTS) or [
        value.name for value in model.graph.output
    ] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        dtype = onnx.TensorProto.FLOAT
        if tensor.elem_type != dtype or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded CLIP export")
    onnx.save(model, path)


def build_image_encoder(clip):
    """Use 4D channel-first transformer blocks and fused Q/K/V projections for the image encoder."""
    import torch
    from torch import nn

    def projection(linear):
        conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=linear.bias is not None)
        conv.weight.data.copy_(linear.weight.detach()[:, :, None, None])
        if linear.bias is not None:
            conv.bias.data.copy_(linear.bias.detach())
        return conv

    class ChannelNorm(nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.register_buffer("weight", norm.weight.detach()[None, :, None, None])
            self.register_buffer("bias", norm.bias.detach()[None, :, None, None])
            self.eps = norm.eps

        def forward(self, x):
            centered = x - x.mean(dim=1, keepdim=True)
            return (
                centered / torch.sqrt((centered * centered).mean(dim=1, keepdim=True) + self.eps) * self.weight
                + self.bias
            )

    class Attention(nn.Module):
        def __init__(self, attention):
            super().__init__()
            channels, self.heads = attention.embed_dim, attention.num_heads
            self.head_size = channels // self.heads
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.qkv.weight.data.copy_(
                torch.cat(
                    [
                        attention.q_proj.weight * self.head_size**-0.5,
                        attention.k_proj.weight,
                        attention.v_proj.weight,
                    ]
                )[:, :, None, None]
            )
            self.qkv.bias.data.copy_(
                torch.cat(
                    [
                        attention.q_proj.bias * self.head_size**-0.5,
                        attention.k_proj.bias,
                        attention.v_proj.bias,
                    ]
                )
            )
            self.output = projection(attention.out_proj)

        def split_heads(self, x):
            # Channel h*D+d belongs to head h; split/concat preserves that exact mapping.
            return torch.cat(torch.split(x, self.head_size, dim=1), dim=2)

        def forward(self, x):
            query, key, value = [self.split_heads(v) for v in self.qkv(x).chunk(3, dim=1)]
            scores = torch.einsum("nchw,nchq->nqhw", query, key)
            probs = torch.softmax(scores, dim=1)
            context = torch.einsum("nchw,nqhc->nqhw", probs, value)
            merged = torch.cat(torch.split(context, 1, dim=2), dim=1)
            return self.output(merged)

    class Block(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.norm1, self.norm2 = ChannelNorm(block.layer_norm1), ChannelNorm(block.layer_norm2)
            self.attention = Attention(block.self_attn)
            self.fc1, self.fc2 = projection(block.mlp.fc1), projection(block.mlp.fc2)

        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            hidden = self.fc1(self.norm2(x))
            # The pinned CLIP checkpoint uses QuickGELU in both encoders.
            hidden = hidden * torch.sigmoid(1.702 * hidden)
            return x + self.fc2(hidden)

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            vision = clip.vision_model
            assert clip.config.vision_config.hidden_act == "quick_gelu"
            self.patches = vision.embeddings.patch_embedding
            self.register_buffer("class_embedding", vision.embeddings.class_embedding.detach()[None, :, None, None])
            self.register_buffer("positions", vision.embeddings.position_embedding.weight.detach().T[None, :, None, :])
            self.pre_norm = ChannelNorm(vision.pre_layrnorm)
            self.post_norm = ChannelNorm(vision.post_layernorm)
            self.blocks = nn.ModuleList(Block(block) for block in vision.encoder.layers)
            self.projection = projection(clip.visual_projection)

        def forward(self, pixels):
            patches = self.patches(pixels).reshape(1, 768, 1, 196)
            image = self.pre_norm(torch.cat([self.class_embedding, patches], dim=3) + self.positions)
            for block in self.blocks:
                image = block(image)
            image = self.projection(self.post_norm(image[:, :, :, :1]))
            embedding = image.reshape(1, EMBED_DIM)
            return embedding

    return ImageEncoder().float().eval()


def load_clip(source):
    """Restore the upstream model, processor and original assets from the conversion bundle."""
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPProcessor

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    assets = checkpoint["assets"]
    with tempfile.TemporaryDirectory(prefix="clip-processor-") as directory:
        for name, content in assets.items():
            (Path(directory) / name).write_text(content, encoding="utf-8")
        processor = CLIPProcessor.from_pretrained(directory, local_files_only=True)
    config = CLIPConfig.from_dict(json.loads(assets["config.json"]))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    clip = CLIPModel(config).float().eval()
    state = checkpoint["state_dict"]
    # Older checkpoints persist these deterministic buffers; current Transformers does not.
    for name, length in (("text_model", TEXT_LENGTH), ("vision_model", 197)):
        positions = state.pop(f"{name}.embeddings.position_ids")
        if not torch.equal(positions, torch.arange(length)[None]):
            raise ValueError(f"unexpected {name} position IDs")
    clip.load_state_dict(state, strict=True)
    return clip, processor, assets


def write_runtime_files(clip, assets, directory):
    """Write the original tokenizer and FP32 lookup weights beside the exported text graph."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = assets[TOKENIZER_FILE]
    (directory / TOKENIZER_FILE).write_bytes(tokenizer.encode("utf-8") if isinstance(tokenizer, str) else tokenizer)
    table = clip.text_model.embeddings.token_embedding.weight.detach().numpy()
    np.save(directory / TOKEN_EMBEDDINGS_FILE, table, allow_pickle=False)


def tokenize(tokenizer, prompts, padding="max_length"):
    return tokenizer(list(prompts), padding=padding, max_length=TEXT_LENGTH, truncation=True, return_tensors="pt")[
        "input_ids"
    ]


def text_features(clip, input_ids):
    features = clip.get_text_features(input_ids=input_ids)
    return getattr(features, "pooler_output", features)


def prepare_prompts(clip, tokenizer):
    """Prepare the fixed token IDs and embeddings used for export and fidelity checks."""
    import torch

    with torch.inference_mode():
        prompt_ids = tokenize(
            tokenizer,
            (
                TEMPLATES[index % len(TEMPLATES)].format(CLASS_NAMES[index * CLASS_COUNT // FIDELITY_PROMPTS])
                for index in range(FIDELITY_PROMPTS)
            ),
        )
        prompt_embeds = clip.text_model.embeddings.token_embedding(prompt_ids)
    return prompt_ids, prompt_embeds


def build_model(clip):
    """Compose the optimized image encoder and the static text transformer."""
    import torch

    class Clip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text = clip.text_model
            self.text_projection = clip.text_projection
            self.image_encoder = build_image_encoder(clip)
            self.register_buffer("mask", torch.full((TEXT_LENGTH, TEXT_LENGTH), -1e4).triu(1)[None, None])

        def forward(self, pixel_values, token_embeds):
            image_embeds = self.image_encoder(pixel_values)
            text = self.text
            positions = text.embeddings.position_embedding.weight[:TEXT_LENGTH][None]
            hidden = text.encoder(inputs_embeds=token_embeds + positions, attention_mask=self.mask).last_hidden_state
            text_embeds = self.text_projection(text.final_layer_norm(hidden))
            return image_embeds, text_embeds

    return Clip().eval()


def export_onnx(model, pixel_values, prompt_embeds, output):
    """Export the fixed interface and fold constant ONNX expressions."""
    import torch

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values, prompt_embeds[:1]),
            str(output),
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)


def compare_checkpoint(output, clip, pixel_values, prompt_ids, prompt_embeds):
    """Compare the exported image and end-of-text outputs with upstream CLIP."""
    import onnxruntime as ort
    import torch

    # The end-of-text token has the highest id in CLIP's vocabulary.
    prompt_ends = prompt_ids.argmax(-1)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected_image = clip.get_image_features(pixel_values=pixel_values)
        expected_image = getattr(expected_image, "pooler_output", expected_image).numpy()
        expected_text = text_features(clip, prompt_ids[:4]).numpy()
    for index, image in enumerate(pixel_values):
        (image_embeds,) = session.run(
            ["image_embeds"], {"pixel_values": image[None].numpy(), "token_embeds": prompt_embeds[:1].numpy()}
        )
        np.testing.assert_allclose(image_embeds[0], expected_image[index], rtol=1e-3, atol=1e-3)
    for index in range(4):
        (text_embeds,) = session.run(
            ["text_embeds"],
            {"pixel_values": pixel_values[:1].numpy(), "token_embeds": prompt_embeds[index : index + 1].numpy()},
        )
        np.testing.assert_allclose(text_embeds[0, int(prompt_ends[index])], expected_text[index], rtol=1e-3, atol=1e-3)


def export_model(source, output_dir):
    """Restore CLIP, prepare its static inputs, export the graph and verify checkpoint parity."""
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    clip, processor, assets = load_clip(source)
    write_runtime_files(clip, assets, output_dir / "text_encoder")
    prompt_ids, prompt_embeds = prepare_prompts(clip, processor.tokenizer)
    model = build_model(clip)
    rng = np.random.RandomState(0)
    images = [
        Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in ((300, 400), (400, 300), (224, 224))
    ]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    export_onnx(model, pixel_values[:1], prompt_embeds, output)
    compare_checkpoint(output, clip, pixel_values, prompt_ids, prompt_embeds)
    np.savez(output_dir / "prompts.npz", prompts=prompt_embeds.numpy(), ends=prompt_ids.argmax(-1).numpy())
    print(f"PyTorch/ONNX check passed: {output}")
    return output


def write_text_probe(source, converted, output_dir):
    import onnxruntime as ort
    import torch
    from transformers import CLIPProcessor

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    validate_export(converted)
    with tempfile.TemporaryDirectory(prefix="clip-tokenizer-") as directory:
        for name, content in checkpoint["assets"].items():
            (Path(directory) / name).write_text(content, encoding="utf-8")
        tokenizer = CLIPProcessor.from_pretrained(directory, local_files_only=True).tokenizer
    token_ids = tokenizer(
        list(TEXT_PROBE_PROMPTS), padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="np"
    )["input_ids"]
    table = checkpoint["state_dict"]["text_model.embeddings.token_embedding.weight"].numpy()
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros(INTERFACE["pixel_values"], dtype=np.float32)
    text_embeds = []
    for ids in token_ids:
        (rows,) = session.run(["text_embeds"], {"pixel_values": pixel_values, "token_embeds": table[ids][None]})
        # The end-of-text token has the highest id in CLIP's vocabulary.
        text_embeds.append(rows[0, ids.argmax()])
    row_ids = np.unique(token_ids)
    arrays = {
        "prompts": np.asarray(TEXT_PROBE_PROMPTS),
        "token_ids": token_ids.astype(np.int32),
        "row_ids": row_ids.astype(np.int32),
        "rows": table[row_ids],
        "text_embeds": np.stack(text_embeds),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "text_probe.npz"
    # zipfile dates each member 1980-01-01, so the same arrays give the same bytes. np.savez does too, but it adds
    # ZIP64 fields, so its bytes differ from the file TEXT_PROBE_SHA256 pins.
    with zipfile.ZipFile(output, "w") as archive:
        for name, array in arrays.items():
            with archive.open(zipfile.ZipInfo(f"{name}.npy"), "w") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
    print(f"Text probe written: {output}")
    return output


COMPONENT_GRAPHS = {
    "image_encoder": {
        "inputs": {"pixel_values": tuple(INTERFACE["pixel_values"])},
        "outputs": {"image_embeds": tuple(INTERFACE["image_embeds"])},
    },
    "text_encoder": {
        "inputs": {"token_embeds": tuple(INTERFACE["token_embeds"])},
        "outputs": {"text_embeds": tuple(INTERFACE["text_embeds"])},
    },
}


def graph_paths(directory):
    return {component: directory / f"{component}.onnx" for component in COMPONENT_GRAPHS}


def prepare_model(weights, directory):
    """Adapt CLIP in PyTorch, export it, and split the independent encoders for compilation."""
    source = export_model(weights, directory)
    extractor = onnx.utils.Extractor(onnx.load(source))
    paths = graph_paths(directory)
    for component, path in paths.items():
        graph = COMPONENT_GRAPHS[component]
        model = extractor.extract_model(list(graph["inputs"]), list(graph["outputs"]))
        onnx.checker.check_model(model)
        onnx.save(model, path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-probe", type=Path, metavar="ONNX", help="regenerate the text probe from this export")
    args = parser.parse_args()
    if args.text_probe:
        write_text_probe(args.weights, args.text_probe, args.output_dir)
    else:
        prepare_model(args.weights, args.output_dir)


if __name__ == "__main__":
    main()
