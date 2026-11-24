"""Republish DINOv3 ConvNeXt Base's pinned timm checkpoint for surgery.py, and export timm's network as the reference.

Conversion changes no computation. weights.pth is the checkpoint's state dict, loaded strictly into timm's
convnext_base; surgery.py rebuilds the network from it. reference.onnx is timm's network at 1x3x224x224 with its exact
GELU, returning the outputs of Meta's DINOv3 ConvNeXt: the final LayerNorm of the 7x7 feature map averaged over its
positions (class_embedding, timm's pooled embedding) and the same LayerNorm at each of the 49 positions
(patch_embeddings). surgery.py checks its export against it, and compile.py measures numerical fidelity against it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.conversion import conversion_outputs
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURE = "convnext_base"
HIDDEN_SIZE = 1024
PATCHES = 49
INTERFACE = {
    "pixel_values": [1, 3, 224, 224],
    "class_embedding": [1, HIDDEN_SIZE],
    "patch_embeddings": [1, PATCHES, HIDDEN_SIZE],
}
OUTPUTS = ("class_embedding", "patch_embeddings")
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", "LICENSE.md"])
SEEDS = range(123, 127)
# ONNX Runtime and PyTorch round float32 sums differently: on the seeded inputs an output value differs by up to
# 1.0e-4, at most 5.3e-5 times one plus its magnitude. The check allows TOLERANCE times one plus its magnitude.
TOLERANCE = 1e-4


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = (config.get("architecture"), config.get("num_classes"), config.get("num_features"))
    if declared != (ARCHITECTURE, 0, HIDDEN_SIZE):
        raise ValueError(f"config.json declares {declared}, expected {(ARCHITECTURE, 0, HIDDEN_SIZE)}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    require_onnx_opset_version(model)
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected the input pixel_values and the outputs {OUTPUTS}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def embeddings(model):
    """timm's network returning the class embedding, which is its pooled output, and the patch embeddings."""
    import torch

    class Embeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            features = self.model.forward_features(pixel_values)
            patches = self.model.head.norm(features).flatten(2).transpose(1, 2)
            return self.model.forward_head(features), patches

    return Embeddings().eval()


def compare_outputs(expected, actual):
    for name, want, got in zip(OUTPUTS, expected, actual, strict=True):
        if list(got.shape) != INTERFACE[name] or not np.isfinite(got).all():
            raise ValueError(f"{name} has shape {list(got.shape)} or non-finite values")
        np.testing.assert_allclose(got, want, rtol=TOLERANCE, atol=TOLERANCE, err_msg=name)


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"pixel_values": INTERFACE["pixel_values"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded export")
    onnx.save(model, path)


def convert(source, output_dir):
    import onnxruntime as ort
    import timm
    import torch
    from safetensors.torch import load_file

    validate_config(source.parent)
    output_dir.mkdir(parents=True, exist_ok=False)
    weights, reference = (output_dir / name for name in conversion_outputs(MODEL_DIR))
    torch.set_num_threads(4)
    model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=0)
    # strict=True raises on any missing or unexpected weight.
    model.load_state_dict(load_file(source), strict=True)
    model = model.eval()
    torch.save(model.state_dict(), weights)

    def synthetic(seed):
        values = np.random.RandomState(seed).standard_normal(INTERFACE["pixel_values"]).astype(np.float32)
        return torch.from_numpy(values)

    reference_model = embeddings(model)
    with torch.inference_mode():
        torch.onnx.export(
            reference_model,
            (synthetic(SEEDS[0]),),
            str(reference),
            input_names=["pixel_values"],
            output_names=list(OUTPUTS),
            opset_version=onnx_opset_version(),
            dynamo=False,
        )
    fold_constants(reference)
    validate_export(reference)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(reference), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = synthetic(seed)
        with torch.inference_mode():
            expected = [value.numpy() for value in reference_model(sample)]
        compare_outputs(expected, session.run(list(OUTPUTS), {"pixel_values": sample.numpy()}))
    print(f"PyTorch/ONNX check passed on {len(SEEDS)} synthetic inputs at tolerance {TOLERANCE}: {reference}")
    return weights, reference


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
