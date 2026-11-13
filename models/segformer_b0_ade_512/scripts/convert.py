"""Export SegFormer-B0 (ADE20K, 512x512) from its Hugging Face checkpoint and check ONNX against PyTorch."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

FILES = ("config.json", "model.safetensors")
EXPORTER = "4.57.1"
OPSET = 17
IMAGE_SIZE = 512
CLASS_COUNT = 150
# The decode head predicts at a quarter of the input resolution; the CPU upsamples the logits.
INTERFACE = {
    "pixel_values": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "logits": [1, CLASS_COUNT, IMAGE_SIZE // 4, IMAGE_SIZE // 4],
}
SEEDS = range(123, 133)


def validate_source(source):
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "config.json" or files != set(FILES):
        raise ValueError(f"the source must be SegFormer's {', '.join(FILES)}, opened at config.json")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"SegFormer export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input pixel_values and output logits")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        # Its optimizer would fuse each token Linear into a 2D Gemm, whose bias the compiler mis-types.
        perform_optimization=False,
        overwrite_input_shapes={"pixel_values": INTERFACE["pixel_values"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded SegFormer export")
    onnx.save(model, path)


def compare_logits(expected, actual):
    """The ONNX logits must match PyTorch's and pick the same class at nearly every pixel."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite logits: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    agreement = float(np.mean(actual.argmax(1) == expected.argmax(1)))
    if agreement < 0.999:
        raise ValueError(f"ONNX and PyTorch agree on only {agreement:.4%} of pixel classes")


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    import transformers
    from transformers import SegformerForSemanticSegmentation

    if transformers.__version__ != EXPORTER:
        raise RuntimeError(f"Expected transformers {EXPORTER}, found {transformers.__version__}")
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    reference = SegformerForSemanticSegmentation.from_pretrained(source.parent, local_files_only=True).eval()
    if reference.config.num_labels != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} ADE20K classes, the checkpoint has {reference.config.num_labels}")

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).logits

    with tempfile.TemporaryDirectory(prefix="segformer-") as temporary:
        exported = Path(temporary) / "model.onnx"
        # A new module starts in training mode, and the export restores that mode to everything inside it.
        torch.onnx.export(
            Logits(reference).eval(),
            (torch.zeros(INTERFACE["pixel_values"]),),
            str(exported),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        exported.replace(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        # Normalized images lie in about [-2.1, 2.6]; the samples cover that range.
        sample = np.random.RandomState(seed).uniform(-2.1, 2.6, INTERFACE["pixel_values"]).astype(np.float32)
        with torch.inference_mode():
            expected = reference(pixel_values=torch.from_numpy(sample)).logits.numpy()
        compare_logits(expected, session.run(["logits"], {"pixel_values": sample})[0])
    print(f"Synthetic-input PyTorch/ONNX logits check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


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
