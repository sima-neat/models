"""Publish DINOv3 ViT-H+/16 from the pinned timm checkpoint as reference.onnx, timm's own forward pass at
1x3x224x224, and as the weight-index.pth surgery.py rebuilds the encoder from.

Conversion changes no computation. reference.onnx is timm's DINOv3, folded with onnxsim, returning the final-norm
class token and the 196 patch tokens without the 4 register tokens, as Meta's DINOv3 returns them. Numerical fidelity
compares each build against it. reference.onnx holds every weight of timm's state dict, as is or transposed, so
weight-index.pth names the initializer of each instead of publishing the 3.4 GB of weights twice, which would exceed
the registry's 5 GiB upload limit. It also holds the rotary sines and cosines of the 14x14 patch grid of a 224x224
image, which timm computes from the input shape.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.conversion import conversion_outputs
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from onnx import numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[1]

ARCHITECTURE = "vit_huge_plus_patch16_dinov3"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "LICENSE.md"])
IMAGE_SIZE = 224
PATCH_SIZE = 16
PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
HIDDEN_SIZE = 1280
HEAD_SIZE = 64
REGISTERS = 4
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "class_embedding": [1, HIDDEN_SIZE],
    "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
}
OUTPUTS = ("class_embedding", "patch_embeddings")
# The weight-index.pth entry with timm's rotary embedding of the 14x14 grid: [196, 2 * head size], sines then cosines.
ROTARY_EMBEDDING = "rope.embedding"
# The weight-index.pth entry mapping each state dict name to its reference.onnx initializer and whether it is stored
# transposed, as each Linear weight is.
INITIALIZERS = "initializers"
SEEDS = range(123, 127)
# ONNX Runtime and PyTorch order their float32 sums differently, which moves a few output values by up to about 5e-4;
# nothing else may differ.
TOLERANCE = 1e-3


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = (config.get("architecture"), config.get("num_features"))
    if declared != (ARCHITECTURE, HIDDEN_SIZE):
        raise ValueError(f"config.json declares {declared}, expected {(ARCHITECTURE, HIDDEN_SIZE)}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    # From the path, the checker finds the external data beside the file.
    onnx.checker.check_model(path)
    require_onnx_opset_version(model)
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected the input and outputs {list(INTERFACE)}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")


def compare_outputs(expected, actual):
    for name, want, got in zip(OUTPUTS, expected, actual, strict=True):
        if list(got.shape) != INTERFACE[name] or not np.isfinite(got).all():
            raise ValueError(f"{name} has shape {list(got.shape)} or non-finite values")
        np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE, err_msg=name)


def fold_constants(exported, path):
    """Fold exported into path, with the weights, above protobuf's 2 GB limit, in one external data file beside it."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(exported), perform_optimization=False)
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded export")
    onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{path.name}.data")


def load_dinov3(checkpoint):
    import timm
    import torch
    from safetensors.torch import load_file

    # At a fixed 224x224 input timm computes the rotary embedding once, so the export holds it as a constant.
    model = timm.create_model(
        ARCHITECTURE, pretrained=False, num_classes=0, img_size=IMAGE_SIZE, dynamic_img_size=False
    )
    # strict=True raises on any missing or unexpected weight.
    model.load_state_dict(load_file(checkpoint), strict=True)
    if model.num_prefix_tokens != 1 + REGISTERS:
        raise ValueError(f"Expected a class token and {REGISTERS} registers, found {model.num_prefix_tokens} tokens")
    return model.to(torch.float32).eval()


def _content(array):
    """A key equal for two arrays exactly when they hold the same values in the same shape and dtype."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(), array.shape, array.dtype.str


def index_initializers(weights, reference):
    """The reference.onnx initializer holding each named array of weights, and whether it holds it transposed."""
    initializers = {}
    for initializer in onnx.load(reference).graph.initializer:
        initializers.setdefault(_content(numpy_helper.to_array(initializer)), initializer.name)

    def find(array):
        if name := initializers.get(_content(array)):
            return name, False
        if array.ndim == 2 and (name := initializers.get(_content(array.T))):
            return name, True
        return None

    index = {name: find(array) for name, array in weights.items()}
    missing = [name for name, found in index.items() if found is None]
    if missing:
        raise ValueError(f"reference.onnx holds no initializer equal to {missing[:4]}")
    return index


def save_weight_index(dinov3, reference, path):
    """Write where reference.onnx holds each weight of timm's state dict, with the rotary embedding of the 224x224
    input."""
    import torch

    with torch.inference_mode():
        rotary = dinov3.rope.get_embed()
    if list(rotary.shape) != [PATCHES, 2 * HEAD_SIZE]:
        raise ValueError(f"timm's rotary embedding has shape {list(rotary.shape)}, not one row per patch")
    weights = {name: value.numpy() for name, value in dinov3.state_dict().items()}
    torch.save({ROTARY_EMBEDDING: rotary.clone(), INITIALIZERS: index_initializers(weights, reference)}, path)
    print(f"Wrote {path}")


def embedding_model(dinov3):
    """timm's DINOv3 returning the package outputs: the final-norm class token and the patch tokens."""
    from torch import nn

    prefix = dinov3.num_prefix_tokens

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            self.dinov3 = dinov3

        def forward(self, pixel_values):
            tokens = self.dinov3.forward_features(pixel_values)
            return tokens[:, 0], tokens[:, prefix:]

    return Embeddings().eval()


def synthetic_pixels(seed):
    return np.random.RandomState(seed).standard_normal(INTERFACE["pixel_values"]).astype(np.float32)


def export_reference(dinov3, path):
    """Export timm's forward pass, fold it and require ONNX Runtime to reproduce timm on seeded inputs."""
    import onnxruntime as ort
    import torch

    model = embedding_model(dinov3)
    # Above 2 GB torch writes each weight to its own file beside the export, so it exports into a directory of its own.
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary, torch.inference_mode():
        exported = Path(temporary) / path.name
        torch.onnx.export(
            model, (torch.from_numpy(synthetic_pixels(SEEDS.start)),), str(exported), input_names=["pixel_values"],
            output_names=list(OUTPUTS), opset_version=onnx_opset_version(), dynamo=False,
        )
        fold_constants(exported, path)
    validate_export(path)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        pixels = synthetic_pixels(seed)
        with torch.inference_mode():
            expected = [output.numpy() for output in model(torch.from_numpy(pixels))]
        compare_outputs(expected, session.run(list(OUTPUTS), {"pixel_values": pixels}))
    print(f"timm/ONNX check passed within {TOLERANCE} on {len(SEEDS)} synthetic inputs: {path}")


def convert(checkpoint, output_dir):
    import torch

    validate_config(checkpoint.parent)
    output_dir.mkdir(parents=True, exist_ok=False)
    weight_index, reference = (output_dir / name for name in conversion_outputs(MODEL_DIR))
    torch.set_num_threads(4)
    dinov3 = load_dinov3(checkpoint)
    export_reference(dinov3, reference)
    save_weight_index(dinov3, reference, weight_index)
    return weight_index, reference


def main():
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
