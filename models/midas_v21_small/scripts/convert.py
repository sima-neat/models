"""Export the pinned MiDaS v2.1 Small checkpoint to a static 256x256 ONNX graph.

The network is MiDaS's MidasNet_small (isl-org/MiDaS, MIT): an EfficientNet-Lite3 encoder, here timm's
tf_efficientnet_lite3, and a RefineNet-style decoder that fuses four encoder stages. Its modules keep the
checkpoint's names, so the checkpoint loads strictly.
"""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = "midas_v21_small_256.pt"
INTERFACE = {"pixel_values": [1, 3, 256, 256], "predicted_depth": [1, 256, 256]}
OUTPUTS = ("predicted_depth",)
SEEDS = range(123, 127)
# Each encoder stage's channels, and the decoder's width at that stage (MiDaS's features=64, expanded).
ENCODER_CHANNELS = (32, 48, 136, 384)
DECODER_CHANNELS = (64, 128, 256, 512)
# Operations that compute shapes at run time; the export must fold them all for the fixed input.
SHAPE_OPERATIONS = {"Shape", "ConstantOfShape", "Range", "ScatterND", "Where"}


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def build_network():
    """MidasNet_small as MiDaS defines it for midas_v21_small_256, with the checkpoint's module names."""
    import timm
    import torch
    from torch import nn
    from torch.nn import functional

    class ResidualConvUnit(nn.Module):
        """MiDaS's ResidualConvUnit_custom without batch normalization."""

        def __init__(self, features, activation):
            super().__init__()
            self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.activation = activation

        def forward(self, x):
            return self.conv2(self.activation(self.conv1(self.activation(x)))) + x

    class FeatureFusionBlock(nn.Module):
        """MiDaS's FeatureFusionBlock_custom: add the refined skip, refine, upsample 2x and project."""

        def __init__(self, features, activation, expand):
            super().__init__()
            self.out_conv = nn.Conv2d(features, features // 2 if expand else features, kernel_size=1)
            self.resConfUnit1 = ResidualConvUnit(features, activation)
            self.resConfUnit2 = ResidualConvUnit(features, activation)

        def forward(self, x, skip=None):
            if skip is not None:
                x = x + self.resConfUnit1(skip)
            x = functional.interpolate(self.resConfUnit2(x), scale_factor=2, mode="bilinear", align_corners=True)
            return self.out_conv(x)

    class Upsample(nn.Module):
        """MiDaS's Interpolate(scale_factor=2, mode="bilinear"), which keeps align_corners=False. The authors'
        TorchScript trace of their training code uses True here; both score the same on NYU Depth V2."""

        def forward(self, x):
            return functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    class MidasSmall(nn.Module):
        def __init__(self):
            super().__init__()
            # exportable=True is how MiDaS builds the encoder: TensorFlow "same" padding in an exportable form.
            encoder = timm.create_model("tf_efficientnet_lite3", pretrained=False, exportable=True)
            self.pretrained = nn.Module()
            # MiDaS's first stage is (conv_stem, bn1, act1, block 0, block 1). timm's bn1 applies the ReLU6 itself,
            # so the activation's slot holds an identity and the blocks keep their checkpoint indices.
            self.pretrained.layer1 = nn.Sequential(encoder.conv_stem, encoder.bn1, nn.Identity(), *encoder.blocks[0:2])
            self.pretrained.layer2 = nn.Sequential(*encoder.blocks[2:3])
            self.pretrained.layer3 = nn.Sequential(*encoder.blocks[3:5])
            self.pretrained.layer4 = nn.Sequential(*encoder.blocks[5:9])
            activation = nn.ReLU(False)
            self.scratch = nn.Module()
            for index, (channels, features) in enumerate(zip(ENCODER_CHANNELS, DECODER_CHANNELS), start=1):
                convolution = nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False)
                setattr(self.scratch, f"layer{index}_rn", convolution)
            self.scratch.refinenet4 = FeatureFusionBlock(DECODER_CHANNELS[3], activation, expand=True)
            self.scratch.refinenet3 = FeatureFusionBlock(DECODER_CHANNELS[2], activation, expand=True)
            self.scratch.refinenet2 = FeatureFusionBlock(DECODER_CHANNELS[1], activation, expand=True)
            self.scratch.refinenet1 = FeatureFusionBlock(DECODER_CHANNELS[0], activation, expand=False)
            features = DECODER_CHANNELS[0]
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
                Upsample(),
                nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
                activation,
                nn.Conv2d(32, 1, kernel_size=1),
                # non_negative=True: relative inverse depth is never negative.
                nn.ReLU(True),
                nn.Identity(),
            )

        def forward(self, pixel_values):
            layer1 = self.pretrained.layer1(pixel_values)
            layer2 = self.pretrained.layer2(layer1)
            layer3 = self.pretrained.layer3(layer2)
            layer4 = self.pretrained.layer4(layer3)
            path = self.scratch.refinenet4(self.scratch.layer4_rn(layer4))
            path = self.scratch.refinenet3(path, self.scratch.layer3_rn(layer3))
            path = self.scratch.refinenet2(path, self.scratch.layer2_rn(layer2))
            path = self.scratch.refinenet1(path, self.scratch.layer1_rn(layer1))
            return torch.squeeze(self.scratch.output_conv(path), dim=1)

    return MidasSmall()


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("The export requires opset 17")
    inputs, outputs = [value.name for value in model.graph.input], [value.name for value in model.graph.output]
    if inputs != ["pixel_values"] or outputs != list(OUTPUTS):
        raise ValueError(
            f"Expected the input pixel_values and the output predicted_depth, found {inputs} and {outputs}"
        )
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    unfolded = sorted({node.op_type for node in model.graph.node} & SHAPE_OPERATIONS)
    if unfolded:
        raise ValueError(f"Shape computations survived folding: {unfolded}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    if expected.shape != actual.shape or list(expected.shape) != INTERFACE["predicted_depth"]:
        raise ValueError("Unexpected shape for predicted_depth")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite predicted_depth")
    # Relative inverse depth has no fixed scale, so the tolerance follows the map's own range.
    scale = max(float(np.abs(expected).max()), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4 * scale, err_msg="predicted_depth")


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
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    network = build_network()
    # weights_only refuses anything but tensors; strict=True raises on any missing or unexpected weight.
    network.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
    network = network.cpu().float().eval()

    def synthetic(seed):
        values = np.random.RandomState(seed).standard_normal(INTERFACE["pixel_values"]).astype(np.float32)
        return torch.from_numpy(values)

    with torch.inference_mode():
        torch.onnx.export(
            network,
            (synthetic(SEEDS[0]),),
            str(output),
            input_names=["pixel_values"],
            output_names=list(OUTPUTS),
            opset_version=17,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        sample = synthetic(seed)
        with torch.inference_mode():
            expected = network(sample).numpy()
        (actual,) = session.run(list(OUTPUTS), {"pixel_values": sample.numpy()})
        compare_outputs(expected, actual)
    print(f"PyTorch/ONNX check passed on {len(SEEDS)} synthetic inputs: {output}")
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
