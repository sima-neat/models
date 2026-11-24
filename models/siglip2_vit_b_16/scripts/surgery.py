"""Adapt SigLIP 2 in PyTorch, export ONNX and split its image and text encoders.

The image and text encoders return their raw embeddings. The text encoder starts after token lookup,
which the CPU performs, and pools the last of its 64 tokens. ImageNet scoring stays in accuracy.py.
Surgery saves 64 prompts in a temporary sidecar file for text-encoder fidelity checks against FP32.

Use --weights weights.pth --output-dir graphs to prepare the encoders. Surgery verifies
PyTorch/ONNX parity and the committed runtime probes before compilation. To regenerate
those probes, use --weights weights.pth --text-check check.npz --onnx graphs/model.onnx
and update TEXT_CHECK_SHA256 in this script, accuracy.py and performance.py.
"""

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
from calibration import CLASS_NAMES, preprocess_image
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version

IMAGE_SIZE = 224
TEXT_LENGTH = 64
EMBED_DIM = 768
CLASS_COUNT = 1000
FIDELITY_PROMPTS = 64
TOKENIZER_FILE = "tokenizer.json"
TOKEN_EMBEDDINGS_FILE = "token_embeddings.npy"
RUNTIME_FILES = (TOKENIZER_FILE, TOKEN_EMBEDDINGS_FILE)
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "token_embeds": [1, TEXT_LENGTH, EMBED_DIM],
    "image_embeds": [1, EMBED_DIM],
    "text_embeds": [1, EMBED_DIM],
}
INPUTS = ("pixel_values", "token_embeds")
OUTPUTS = ("image_embeds", "text_embeds")

# OpenAI's ImageNet class names and the 7 prompt templates its forward selection chose over the 80, from
# https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# The names follow the ImageNet class index order (sorted WordNet IDs). SigLIP 2 was trained on lowercase text.
# The text-encoder check's classes, one per template: ImageNet classes the 64 fidelity prompts leave out, each named by
# one token, so the check file stores few token embedding rows.
CHECK_CLASSES = (1, 107, 291, 340, 879, 963, 980)
# Fixed fidelity and regression prompts are independent of the ImageNet accuracy recipe.
PROBE_TEMPLATES = (
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
)

TEXT_CHECK_SHA256 = "273bfe20216fb3c1ef681f453335b37b066f687744dec2893154384e3c1305e5"


def validate_config(directory):
    model_type = json.loads((directory / "config.json").read_text(encoding="utf-8")).get("model_type")
    if model_type != "siglip":
        raise ValueError(f"config.json names model_type {model_type!r}, expected 'siglip'")


def load_siglip(source):
    """Restore the upstream model, processor and original assets from the conversion bundle."""
    import torch
    from transformers import AutoProcessor, SiglipConfig, SiglipModel

    bundle = torch.load(source, map_location="cpu", weights_only=True)
    with tempfile.TemporaryDirectory(prefix="siglip2-processor-") as directory:
        directory = Path(directory)
        for name, content in bundle["assets"].items():
            (directory / name).write_bytes(content)
        validate_config(directory)
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    config = SiglipConfig.from_dict(json.loads(bundle["assets"]["config.json"]))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = SiglipModel(config).float().eval()
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model, processor, bundle["assets"]


def write_runtime_files(siglip, assets, directory):
    """Write the original tokenizer and FP32 lookup weights beside the exported text graph."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = assets[TOKENIZER_FILE]
    (directory / TOKENIZER_FILE).write_bytes(tokenizer.encode("utf-8") if isinstance(tokenizer, str) else tokenizer)
    table = siglip.text_model.embeddings.token_embedding.weight.detach().numpy()
    np.save(directory / TOKEN_EMBEDDINGS_FILE, table, allow_pickle=False)


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    require_onnx_opset_version(model)
    if [value.name for value in model.graph.input] != list(INPUTS) or [
        value.name for value in model.graph.output
    ] != list(OUTPUTS):
        raise ValueError(f"Expected inputs {INPUTS} and outputs {OUTPUTS} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]
        ):
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
        raise RuntimeError("onnxsim could not validate the folded SigLIP 2 export")
    onnx.save(model, path)


def fidelity_prompts():
    """The 64 fidelity prompts: the templates in turn, with classes spread over the 1,000."""
    return [
        PROBE_TEMPLATES[index % len(PROBE_TEMPLATES)]
        .format(CLASS_NAMES[index * CLASS_COUNT // FIDELITY_PROMPTS])
        .lower()
        for index in range(FIDELITY_PROMPTS)
    ]


def check_prompts():
    """The text-encoder check's prompts: each template once, with the class CHECK_CLASSES gives it."""
    pairs = zip(PROBE_TEMPLATES, CHECK_CLASSES, strict=True)
    return [template.format(CLASS_NAMES[index]).lower() for template, index in pairs]


def write_text_check(checkpoint, onnx_path, output):
    """Write the text-encoder check: each check prompt's token IDs, the token embedding rows they use, and each
    prompt's text embedding from the FP32 ONNX."""
    import onnxruntime as ort

    siglip, processor, _ = load_siglip(checkpoint)
    tokenizer = processor.tokenizer

    def token_ids(texts):
        # Padded to the end to 64 tokens, as export_model() tokenizes.
        return np.asarray(
            tokenizer(texts, padding="max_length", max_length=TEXT_LENGTH, truncation=True)["input_ids"], np.int32
        )

    texts = check_prompts()
    check_ids, prompt_ids = token_ids(texts), token_ids(fidelity_prompts())
    table = siglip.text_model.embeddings.token_embedding.weight.detach().numpy()
    needed = np.unique(np.concatenate([check_ids, prompt_ids])).tolist()
    rows = {token: table[token] for token in needed}

    def lookup(ids):
        return np.stack([rows[token] for token in ids.tolist()])

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros(INTERFACE["pixel_values"], np.float32)

    def text_embeds(token_embeds):
        (embeds,) = session.run(["text_embeds"], {"pixel_values": pixel_values, "token_embeds": token_embeds[None]})
        return embeds[0]

    # The temporary sidecar holds the fidelity inputs prepared during export.
    with np.load(onnx_path.parent / "prompts.npz", allow_pickle=False) as saved:
        embedded = saved["prompts"]
    np.testing.assert_array_equal(
        np.stack([lookup(ids) for ids in prompt_ids]),
        embedded,
        err_msg="this tokenizer and lookup differ from the export's",
    )
    row_ids = np.unique(check_ids)
    # Given an open file, np.savez keeps the output's name; it dates every entry 1980-01-01, so the same arrays give
    # the same bytes.
    with open(output, "wb") as stream:
        np.savez(
            stream,
            prompts=np.asarray(texts),
            token_ids=check_ids,
            row_ids=row_ids,
            rows=lookup(row_ids),
            text_embeds=np.stack([text_embeds(lookup(ids)) for ids in check_ids]),
        )


def tokenize(tokenizer, texts):
    return tokenizer(list(texts), padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="pt")[
        "input_ids"
    ]


def text_features(siglip, input_ids):
    features = siglip.get_text_features(input_ids=input_ids)
    return getattr(features, "pooler_output", features)


def adapted_model(siglip):
    import torch

    class SigLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.siglip = siglip

        def forward(self, pixel_values, token_embeds):
            image_embeds = self.siglip.vision_model(pixel_values=pixel_values).pooler_output
            text = self.siglip.text_model
            hidden = text.encoder(inputs_embeds=text.embeddings(inputs_embeds=token_embeds)).last_hidden_state
            text_embeds = text.head(text.final_layer_norm(hidden)[:, -1, :])
            return image_embeds, text_embeds

    return SigLIP().eval()


def validate_parity(path, siglip, pixel_values, prompt_ids, prompt_embeds):
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected_image = siglip.get_image_features(pixel_values=pixel_values)
        expected_image = getattr(expected_image, "pooler_output", expected_image).numpy()
        expected_text = text_features(siglip, prompt_ids[:4]).numpy()
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
        np.testing.assert_allclose(text_embeds[0], expected_text[index], rtol=1e-3, atol=1e-3)
    print(f"PyTorch/ONNX check passed: {path}")


def export_model(source, output_dir):
    import torch
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    siglip, processor, assets = load_siglip(source)
    write_runtime_files(siglip, assets, output_dir / "text_encoder")
    prompt_ids = tokenize(processor.tokenizer, fidelity_prompts())
    with torch.inference_mode():
        prompt_embeds = siglip.text_model.embeddings.token_embedding(prompt_ids)
    model = adapted_model(siglip)
    rng = np.random.RandomState(0)
    images = [
        Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8))
        for h, w in ((300, 400), (400, 300), (224, 224))
    ]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    # Check the actual bundled processor against the preprocessing used for compilation fidelity.
    for image, expected in zip(images, pixel_values.numpy(), strict=True):
        content = io.BytesIO()
        image.save(content, format="PNG")
        # Pillow and the processor's Torchvision backend can round bilinear samples by one RGB level.
        np.testing.assert_allclose(
            preprocess_image(content.getvalue()).transpose(2, 0, 1), expected, rtol=0, atol=2 / 255 + 1e-6
        )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (pixel_values[:1], prompt_embeds[:1]),
            str(output),
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            opset_version=onnx_opset_version(),
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)
    validate_parity(output, siglip, pixel_values, prompt_ids, prompt_embeds)
    np.savez(output_dir / "prompts.npz", prompts=prompt_embeds.numpy())
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


def validate_text_check(source):
    """Keep the committed runtime probes aligned with the exported text encoder."""
    import onnxruntime as ort

    content = Path(__file__).with_name("text_encoder_check.npz").read_bytes()
    if hashlib.sha256(content).hexdigest() != TEXT_CHECK_SHA256:
        raise ValueError("text_encoder_check.npz checksum does not match the exported probe contract")
    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        rows = dict(zip(saved["row_ids"].tolist(), saved["rows"]))
        session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        for ids, expected in zip(saved["token_ids"], saved["text_embeds"], strict=True):
            embeddings = np.stack([rows[token] for token in ids.tolist()])[None]
            (actual,) = session.run(
                ["text_embeds"],
                {
                    "pixel_values": np.zeros(INTERFACE["pixel_values"], np.float32),
                    "token_embeds": embeddings,
                },
            )
            np.testing.assert_allclose(actual[0], expected, rtol=1e-4, atol=1e-4)


def prepare_model(weights, directory):
    """Export the adapted model and extract the two independently compiled encoders."""
    source = export_model(weights, directory)
    validate_text_check(source)
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
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-dir", type=Path)
    target.add_argument("--text-check", type=Path, help="write the accuracy stage's text-encoder check to this file")
    parser.add_argument(
        "--onnx",
        type=Path,
        metavar="ONNX",
        help="with --text-check: the exported model.onnx whose text encoder embeds the check prompts",
    )
    args = parser.parse_args()
    if (args.text_check is None) != (args.onnx is None):
        parser.error("--text-check and --onnx go together")
    if args.output_dir:
        prepare_model(args.weights.resolve(), args.output_dir.resolve())
    else:
        write_text_check(args.weights.resolve(), args.onnx.resolve(), args.text_check.resolve())
        checksum = hashlib.sha256(args.text_check.read_bytes()).hexdigest()
        print(
            f"Wrote {args.text_check}; set TEXT_CHECK_SHA256 in surgery.py, accuracy.py and performance.py "
            f"to {checksum}"
        )


if __name__ == "__main__":
    main()
