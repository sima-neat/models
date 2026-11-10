"""Export Megvii's original YOLOX-Nano forward pass, including grid decode and score sigmoid.

The checkpoint is a verified official release asset. Torch Hub supplies the original architecture
at a full upstream commit; no checkpoint is downloaded by Hub. Compiler-specific cuts belong to surgery.py.
"""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = "yolox_nano.pth"
MODEL = "yolox_nano"
UPSTREAM = "Megvii-BaseDetection/YOLOX:419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
IMAGE_SIZE = 416
CLASS_COUNT = 80
ANCHOR_COUNT = sum((IMAGE_SIZE // stride) ** 2 for stride in (8, 16, 32))
INTERFACE = {"images": [1, 3, IMAGE_SIZE, IMAGE_SIZE], "predictions": [1, ANCHOR_COUNT, CLASS_COUNT + 5]}
SEEDS = range(123, 133)


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [13]:
        raise ValueError("YOLOX export requires opset 13")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input images and output predictions")
    for value in values:
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    """Compare every upstream decoded xywh value, objectness and class probability."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite predictions: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def convert(source, output_dir):
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    reference = torch.hub.load(
        UPSTREAM, MODEL, pretrained=False, device="cpu", trust_repo=True, skip_validation=True
    ).eval()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    reference.load_state_dict(checkpoint["model"], strict=True)
    reference.requires_grad_(False)
    # The default eval forward decodes xywh and applies sigmoid, exactly as upstream inference does.
    sample = torch.zeros(INTERFACE["images"], dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            reference,
            sample,
            str(output),
            input_names=["images"],
            output_names=["predictions"],
            opset_version=13,
            dynamic_axes=None,
            do_constant_folding=True,
            dynamo=False,
        )
    validate_export(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = np.random.RandomState(seed).uniform(0, 255, INTERFACE["images"]).astype(np.float32)
        with torch.inference_mode():
            expected = reference(torch.from_numpy(sample)).numpy()
        compare_outputs(expected, session.run(["predictions"], {"images": sample})[0])
    print(f"PyTorch/ONNX output check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
