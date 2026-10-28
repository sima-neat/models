"""Export LightStereo-S from its OpenStereo checkpoint with MLA-friendly operators and check ONNX against PyTorch.

The export takes one side-by-side frame, [left | right], and splits it before the network, so an application
passes a single image; the network itself keeps upstream's two views.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "LightStereo-S-SceneFlow.ckpt"
# cfgs/lightstereo/lightstereo_s_sceneflow.yaml evaluates the 960x540 Scene Flow pairs padded to 544x960.
HEIGHT, WIDTH = 544, 960
# The same config sets MAX_DISP 192, so the quarter-resolution cost volume holds 48 disparity bins.
MAX_DISP = 192
BINS = MAX_DISP // 4
VIEW = [1, 3, HEIGHT, WIDTH]
INTERFACE = {"stereo": [1, 3, HEIGHT, 2 * WIDTH], "disparity": [1, 1, HEIGHT, WIDTH]}
INPUTS = ("stereo",)
OUTPUT = "disparity"
OPSET = 17
# Upstream's cost volume writes slices in place and its upsampling unfolds patches, which export as ScatterND
# and index arithmetic the MLA cannot run; edge and wide zero padding do not compile either, and INT8
# InstanceNormalization fails the compiler's simulation check. The rewritten export must hold none of these.
FORBIDDEN_OPS = {
    "ScatterND",
    "Gather",
    "GatherND",
    "Where",
    "Range",
    "Expand",
    "ConstantOfShape",
    "Shape",
    "Pad",
    "InstanceNormalization",
}
SEEDS = range(123, 128)
# The rewrites reorder float sums; on real pairs they differ from upstream by under 0.005 px, a hundredth of
# the model's own error there.
TOLERANCE_PIXELS = 0.01


def validate_source(source: dict[str, Any]) -> None:
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_export(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"LightStereo export requires opset {OPSET}")
    if [v.name for v in model.graph.input] != list(INPUTS) or [v.name for v in model.graph.output] != [OUTPUT]:
        raise ValueError(f"Expected inputs {INPUTS} and output {OUTPUT} in that order")
    for value in [*model.graph.input, *model.graph.output]:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}: {shape}")
    forbidden = sorted({node.op_type for node in model.graph.node} & FORBIDDEN_OPS)
    if forbidden:
        raise ValueError(f"Export still holds operators the rewrite removes: {forbidden}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_disparity(expected: np.ndarray, actual: np.ndarray) -> float:
    """Return the largest disparity difference in pixels, failing on a shape change, NaN or excess error."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite disparity: {actual.shape} vs {expected.shape}")
    error = float(np.abs(actual - expected).max())
    if error > TOLERANCE_PIXELS:
        raise ValueError(f"ONNX disparity differs from PyTorch by {error:.5f} px (limit {TOLERANCE_PIXELS})")
    return error


def build_network(export: bool) -> Any:
    """Build LightStereo-S with OpenStereo's module names, so its checkpoint loads strictly.

    With ``export`` the forward pass uses exact rewrites of the cost volume, edge padding, instance normalization,
    disparity regression and context upsampling; without it, OpenStereo's own operations, as the reference. torch
    and timm are imported here so the tests can load this script without them.
    """
    from functools import partial

    import timm
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Basic(nn.Module):
        """OpenStereo's BasicConv2d and BasicDeconv2d: a bias-free layer, then optional norm and activation."""

        def __init__(self, layer: nn.Module, norm: Any = None, act: Any = None) -> None:
            super().__init__()
            channels = layer.out_channels
            self.block = nn.Sequential(layer, *([norm(channels)] if norm else []), *([act()] if act else []))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    def conv(cin: int, cout: int, norm: Any = None, act: Any = None, **kwargs: Any) -> nn.Module:
        return Basic(nn.Conv2d(cin, cout, bias=False, **kwargs), norm, act)

    def deconv(cin: int, cout: int, norm: Any = None, act: Any = None, **kwargs: Any) -> nn.Module:
        return Basic(nn.ConvTranspose2d(cin, cout, bias=False, **kwargs), norm, act)

    leaky = partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)

    class InstanceNorm(nn.Module):
        """nn.InstanceNorm2d without affine or running statistics, written out: the Model Compiler's INT8 lowering
        of InstanceNormalization fails its own simulation check, while these operators compile."""

        def __init__(self, _channels: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            centered = x - x.mean(dim=(2, 3), keepdim=True)
            variance = (centered * centered).mean(dim=(2, 3), keepdim=True)
            return centered / torch.sqrt(variance + self.eps)

    instance_norm = InstanceNorm if export else nn.InstanceNorm2d

    class FPNLayer(nn.Module):
        def __init__(self, low: int, high: int) -> None:
            super().__init__()
            self.deconv = deconv(low, high, nn.BatchNorm2d, leaky, kernel_size=4, stride=2, padding=1)
            self.conv = conv(high * 2, high, nn.BatchNorm2d, leaky, kernel_size=3, padding=1)

        def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
            return self.conv(torch.cat([high, self.deconv(low)], 1))

    class MobileV2Residual(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int, expansion: int = 4) -> None:
            super().__init__()
            hidden = cin * expansion
            self.residual = stride == 1 and cin == cout
            self.pwconv = nn.Sequential(nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6())
            self.dwconv = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6()
            )
            self.pwliner = nn.Sequential(nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feature = self.pwliner(self.dwconv(self.pwconv(x)))
            return x + feature if self.residual else feature

    class AttentionModule(nn.Module):
        def __init__(self, dim: int, image_dim: int) -> None:
            super().__init__()
            self.conv0 = nn.Conv2d(image_dim, dim, 1)
            for index, size in enumerate((7, 11, 21)):
                self.add_module(f"conv{index}_1", nn.Conv2d(dim, dim, (1, size), padding=(0, size // 2), groups=dim))
                self.add_module(f"conv{index}_2", nn.Conv2d(dim, dim, (size, 1), padding=(size // 2, 0), groups=dim))
            self.conv3 = nn.Conv2d(dim, dim, 1)

        def forward(self, cost: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            attention = self.conv0(image)
            strips = [getattr(self, f"conv{i}_2")(getattr(self, f"conv{i}_1")(attention)) for i in range(3)]
            return self.conv3(attention + sum(strips)) * cost

    class Aggregation(nn.Module):
        def __init__(self, channels: int, image_channels: list[int]) -> None:
            super().__init__()
            c = channels
            self.conv0 = nn.Sequential(MobileV2Residual(c, c, 1))
            self.conv1 = MobileV2Residual(c, c * 2, 2)
            self.conv2 = nn.Sequential(MobileV2Residual(c * 2, c * 2, 1))
            self.conv3 = MobileV2Residual(c * 2, c * 4, 2)
            self.conv4 = nn.Sequential(*(MobileV2Residual(c * 4, c * 4, 1) for _ in range(3)))
            self.conv5 = nn.Sequential(
                nn.ConvTranspose2d(c * 4, c * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm2d(c * 2),
            )
            self.conv6 = nn.Sequential(
                nn.ConvTranspose2d(c * 2, c, 3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm2d(c)
            )
            self.redir1 = MobileV2Residual(c, c, 1)
            self.redir2 = MobileV2Residual(c * 2, c * 2, 1)
            self.att0 = AttentionModule(c, image_channels[0])
            self.att2 = AttentionModule(c * 2, image_channels[1])
            self.att4 = AttentionModule(c * 4, image_channels[2])

        def forward(self, cost: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
            x = self.att0(self.conv0(cost), features[0])
            conv2 = self.att2(self.conv2(self.conv1(x)), features[1])
            conv4 = self.att4(self.conv4(self.conv3(conv2)), features[2])
            conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2))
            return F.relu(self.conv6(conv5) + self.redir1(x))

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mobilenet = timm.create_model("mobilenetv2_100", pretrained=False, features_only=True)
            self.conv_stem, self.bn1, self.act1 = mobilenet.conv_stem, mobilenet.bn1, mobilenet.act1
            self.block0, self.block1, self.block2 = mobilenet.blocks[0], mobilenet.blocks[1], mobilenet.blocks[2]
            self.block3, self.block4 = mobilenet.blocks[3:5], mobilenet.blocks[5]
            self.fpn_layer4, self.fpn_layer3, self.fpn_layer2 = FPNLayer(160, 96), FPNLayer(96, 32), FPNLayer(32, 24)
            # Upstream pads this convolution by replicating edges; the export pads explicitly instead.
            padding = {"padding": 0} if export else {"padding": 1, "padding_mode": "replicate"}
            self.out_conv = conv(24, 24, instance_norm, kernel_size=3, **padding)

        def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
            c2 = self.block1(self.block0(self.act1(self.bn1(self.conv_stem(image)))))
            c3 = self.block2(c2)
            c4 = self.block3(c3)
            p4 = self.fpn_layer4(self.block4(c4), c4)
            p3 = self.fpn_layer3(p4, c3)
            p2 = self.fpn_layer2(p3, c2)
            return [self.out_conv(edge_pad(p2) if export else p2), p3, p4]

    def edge_pad(x: torch.Tensor) -> torch.Tensor:
        """Replicate the outermost row and column, which ONNX would otherwise export as an edge-mode Pad."""
        x = torch.cat([x[:, :, :, :1], x, x[:, :, :, -1:]], 3)
        return torch.cat([x[:, :, :1], x, x[:, :, -1:]], 2)

    def correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Mean channel product of left and right features shifted by each disparity, zero where none overlaps."""
        if not export:
            volume = left.new_zeros(left.shape[0], BINS, *left.shape[2:])
            for d in range(BINS):
                volume[:, d, :, d:] = (left[:, :, :, d:] * right[:, :, :, : left.shape[3] - d]).mean(dim=1)
            return volume
        # The Model Compiler lowers a zero Pad of the right features to a pool it cannot pad by 47 columns, so each
        # shift rolls the features instead and a 0/1 mask zeroes the columns that wrapped around.
        width = left.shape[3]
        products = []
        for d in range(BINS):
            shifted = torch.cat([right[:, :, :, width - d :], right[:, :, :, : width - d]], 3) if d else right
            products.append((left * shifted).mean(dim=1, keepdim=True))
        overlap = (torch.arange(width) >= torch.arange(BINS).view(BINS, 1)).float().view(1, BINS, 1, width)
        return torch.cat(products, 1) * overlap

    class LightStereo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()
            self.cost_agg = Aggregation(BINS, [24, 32, 96])
            self.refine_1 = nn.Sequential(
                conv(24, 24, instance_norm, nn.LeakyReLU, kernel_size=3, padding=1),
                conv(24, 24, instance_norm, nn.ReLU, kernel_size=3, padding=1),
            )
            self.stem_2 = nn.Sequential(
                conv(3, 16, nn.BatchNorm2d, nn.LeakyReLU, kernel_size=3, stride=2, padding=1),
                conv(16, 16, nn.BatchNorm2d, nn.ReLU, kernel_size=3, padding=1),
            )
            self.refine_2 = FPNLayer(24, 16)
            self.refine_3 = deconv(16, 9, kernel_size=4, stride=2, padding=1)
            # Fixed 1x1 and 3x3 kernels for the export: disparity regression at full-resolution scale, the 3x3
            # neighbourhood unfold, and the sum over the neighbourhood. They are not checkpoint weights.
            bins = torch.arange(BINS, dtype=torch.float32)
            self.register_buffer("bins", bins.view(1, BINS, 1, 1), persistent=False)
            self.register_buffer("regression", (bins * 4).view(1, BINS, 1, 1), persistent=False)
            self.register_buffer("neighbours", torch.eye(9).view(9, 1, 3, 3), persistent=False)
            self.register_buffer("neighbour_sum", torch.ones(1, 9, 1, 1), persistent=False)

        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            features_left, features_right = self.backbone(left), self.backbone(right)
            cost = correlation(features_left[0], features_right[0])
            prob = F.softmax(self.cost_agg(cost, features_left), dim=1)
            weights = F.softmax(self.refine_3(self.refine_2(self.refine_1(features_left[0]), self.stem_2(left))), 1)
            if not export:
                disp = (prob * self.bins).sum(1, keepdim=True) * 4.0
                b, _, h, w = disp.shape
                unfolded = F.unfold(disp, kernel_size=3, padding=1).reshape(b, 9, h, w)
                return (F.interpolate(unfolded, (h * 4, w * 4), mode="nearest") * weights).sum(1).unsqueeze(1)
            unfolded = F.conv2d(F.conv2d(prob, self.regression), self.neighbours, padding=1)
            return F.conv2d(F.interpolate(unfolded, scale_factor=4, mode="nearest") * weights, self.neighbour_sum)

    return LightStereo()


def side_by_side(network: Any) -> Any:
    """Wrap the two-view network so it takes one [left | right] frame; the split exports as two Slice nodes."""
    import torch

    class SideBySide(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network

        def forward(self, stereo: torch.Tensor) -> torch.Tensor:
            return self.network(stereo[..., :WIDTH], stereo[..., WIDTH:])

    return SideBySide().eval()


def load_network(checkpoint: Path, export: bool) -> Any:
    import torch

    network = build_network(export)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"]
    network.load_state_dict(state, strict=True)
    return network.eval()


def fold_constants(path: Path) -> None:
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded LightStereo export")
    onnx.save(model, path)


def synthetic_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A normalized noise image and the same image shifted left by a seeded disparity, as a rectified pair."""
    random = np.random.RandomState(seed)
    left = random.normal(0, 1, VIEW).astype(np.float32)
    return left, np.roll(left, -random.randint(1, MAX_DISP), axis=3)


def convert(checkpoint: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    reference, network = load_network(checkpoint, export=False), side_by_side(load_network(checkpoint, export=True))
    example = tuple(torch.zeros(INTERFACE[name]) for name in INPUTS)
    with torch.inference_mode():
        torch.onnx.export(
            network, example, output, input_names=list(INPUTS), output_names=[OUTPUT], opset_version=OPSET, dynamo=False
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    for seed in SEEDS:
        left, right = synthetic_pair(seed)
        with torch.inference_mode():
            expected = reference(torch.from_numpy(left), torch.from_numpy(right)).numpy()
        stereo = np.concatenate([left, right], axis=3)
        errors.append(compare_disparity(expected, session.run([OUTPUT], {"stereo": stereo})[0]))
    print(
        f"Synthetic-pair PyTorch/ONNX disparity check passed with seeds {SEEDS.start}-{SEEDS.stop - 1} "
        f"(max {max(errors):.2e} px): {output}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Conversion runs before compilation.source is pinned, which full validation would reject.
    conversions = load_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
