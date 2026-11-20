"""Export fast-reid's BoT(R50) Market-1501 forward pass: pixel normalization, ResNet-50 and the BNNeck feature.

The checkpoint is fast-reid's verified v0.1.1 release asset. fast-reid is not a pip package, so this script
rebuilds its v0.1.1 inference network from torchvision's ResNet-50, which has the same layers and parameter names:
ResNet-50 with the last stride set to 1, global average pooling and the batch-norm neck, whose output fast-reid
uses as the embedding at test time. The model takes RGB pixel values in [0, 255] and returns the 2048-value
embedding before L2 normalization. Compiler-specific changes would belong to surgery.py; none are needed.
"""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = "market_bot_R50.pth"
HEIGHT, WIDTH, FEATURES = 256, 128, 2048
INTERFACE = {"images": [1, 3, HEIGHT, WIDTH], "embedding": [1, FEATURES]}
OPSET = 17
SEEDS = range(123, 133)
# fast-reid's ResNet stages and the torchvision children they map onto, in order.
BACKBONE = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def upstream_state(state):
    """Rename fast-reid v0.1.1 parameters to this module's; the classifier, used only in training, is dropped."""
    mapped = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            stage, _, rest = key.removeprefix("backbone.").partition(".")
            mapped[f"backbone.{BACKBONE.index(stage)}.{rest}"] = value
        elif key.startswith("heads.bnneck."):
            mapped["bnneck." + key.removeprefix("heads.bnneck.")] = value
        elif key in ("pixel_mean", "pixel_std"):
            mapped[key] = value.reshape(1, 3, 1, 1)
        elif key != "heads.classifier.weight":
            raise ValueError(f"unexpected checkpoint parameter {key}")
    return mapped


def build_network(checkpoint):
    import torch
    import torchvision

    class BoT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            resnet = torchvision.models.resnet50(weights=None)
            # fast-reid's MODEL.BACKBONE.LAST_STRIDE: 1 keeps layer4 at 1/16 of the input.
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
            self.backbone = torch.nn.Sequential(*[getattr(resnet, name) for name in BACKBONE])
            self.bnneck = torch.nn.BatchNorm2d(FEATURES)
            self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
            self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))

        def forward(self, images):
            features = self.backbone((images - self.pixel_mean) / self.pixel_std)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            return self.bnneck(pooled)[..., 0, 0]

    network = BoT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    network.load_state_dict(upstream_state(state), strict=True)
    return network.eval().requires_grad_(False)


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"BoT export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input images and output embedding")
    for value in values:
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    """Compare every embedding value, and the cosine similarity the evaluation ranks by."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite embedding: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    cosine = float(np.sum(actual * expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
    if cosine < 0.99999:
        raise ValueError(f"embedding cosine similarity {cosine:.6f} is below 0.99999")


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    network = build_network(source)
    sample = torch.zeros(INTERFACE["images"], dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            network,
            sample,
            str(output),
            input_names=["images"],
            output_names=["embedding"],
            opset_version=OPSET,
            dynamic_axes=None,
            do_constant_folding=True,
            dynamo=False,
        )
    simplified, valid = simplify(onnx.load(output))
    if not valid:
        raise RuntimeError("onnxsim could not verify the simplified graph")
    onnx.save(simplified, output)
    validate_export(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = np.random.RandomState(seed).uniform(0, 255, INTERFACE["images"]).astype(np.float32)
        with torch.inference_mode():
            expected = network(torch.from_numpy(sample)).numpy()
        compare_outputs(expected, session.run(["embedding"], {"images": sample})[0])
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
