"""Build LightStereo-M for the MLA from the converted weights, export it to ONNX and check it against OpenStereo.

The network keeps OpenStereo's module names, so the weights load strictly. Its export path replaces five operations
the MLA cannot run with exact rewrites: the correlation cost volume, the edge padding of the backbone's last
convolution, instance normalization, disparity regression and the context upsampling. The export takes one 540x1920
side-by-side frame, [left | right], repeats each view's top row four times as OpenStereo's RightTopPad pads Scene Flow
views to 544x960, runs the two views through the network, and crops the disparity back to the 540x960 left view.

compile.py calls export_network. To get the FP32 ONNX on a host, run:

    python models/lightstereo_m/scripts/surgery.py --weights weights.pth --output model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from model_registry.dependencies import onnx_opset_version, require_onnx_opset_version
from onnx import helper
from scene_flow import HEIGHT, INPUT, WIDTH

# cfgs/lightstereo/lightstereo_m_sceneflow.yaml pads each view to 544x960 above its first row (RightTopPad).
PADDED_HEIGHT = 544
TOP_PAD = PADDED_HEIGHT - HEIGHT
# The same config sets MAX_DISP 192, so the quarter-resolution cost volume holds 48 disparity bins.
MAX_DISP = 192
BINS = MAX_DISP // 4
# It also sets AGGREGATION_BLOCKS [4, 8, 16], the residual blocks the cost aggregation runs at quarter, eighth
# and sixteenth resolution, and EXPANSE_RATIO 4, how much wider each block's hidden layer is than its input.
AGGREGATION_BLOCKS = (4, 8, 16)
EXPANSION = 4
OUTPUT = "disparity"
INTERFACE = {INPUT: [1, 3, HEIGHT, 2 * WIDTH], OUTPUT: [1, 1, HEIGHT, WIDTH]}
# Upstream's cost volume writes slices in place and its upsampling unfolds patches, which export as ScatterND and
# index arithmetic the MLA cannot run. Edge-mode Pad and INT8 InstanceNormalization do not compile either.
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
# The parts of the network an INT8 build can run BF16. The read-out is the two softmaxes and every operation after
# them: the disparity regression over the 48 bins and the context upsampling to full resolution.
BF16_REGIONS = ("readout",)
SEEDS = range(123, 128)
# The rewrites reorder float sums; on real pairs they differ from upstream by under 0.005 px, a hundredth of the
# model's own error there.
TOLERANCE_PIXELS = 0.01


def build_network(export: bool) -> Any:
    """Build LightStereo-M with OpenStereo's module names and timm's MobileNetV2 names, so its weights load strictly.

    With ``export`` the forward pass uses the exact rewrites; without it, OpenStereo's own operations, as the
    reference. torch is imported here so the tests can load this script without it.
    """
    from collections import OrderedDict
    from functools import partial

    import torch
    from torch import nn
    from torch.nn import functional as F

    class InvertedResidual(nn.Module):
        """timm's MobileNetV2 InvertedResidual at expansion 6: pointwise, depthwise 3x3, linear pointwise."""

        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            hidden = cin * 6
            self.conv_pw, self.bn1 = nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden)
            self.conv_dw = nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden)
            self.conv_pwl, self.bn3 = nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)
            self.residual = stride == 1 and cin == cout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = F.relu6(self.bn2(self.conv_dw(F.relu6(self.bn1(self.conv_pw(x))))))
            feature = self.bn3(self.conv_pwl(hidden))
            return x + feature if self.residual else feature

    class DepthwiseSeparable(nn.Module):
        """timm's MobileNetV2 first block: depthwise 3x3, then a linear pointwise convolution."""

        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.conv_dw, self.bn1 = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False), nn.BatchNorm2d(cin)
            self.conv_pw, self.bn2 = nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn2(self.conv_pw(F.relu6(self.bn1(self.conv_dw(x)))))

    def stage(cin: int, cout: int, stride: int, repeats: int) -> nn.Sequential:
        rest = (InvertedResidual(cout, cout, 1) for _ in range(repeats - 1))
        return nn.Sequential(InvertedResidual(cin, cout, stride), *rest)

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
        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            hidden = cin * EXPANSION
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
            quarter, eighth, sixteenth = AGGREGATION_BLOCKS
            self.conv0 = nn.Sequential(*(MobileV2Residual(c, c, 1) for _ in range(quarter)))
            # conv1 and conv3 halve the resolution, and each counts as the first block of its scale.
            self.conv1 = MobileV2Residual(c, c * 2, 2)
            self.conv2 = nn.Sequential(*(MobileV2Residual(c * 2, c * 2, 1) for _ in range(eighth - 1)))
            self.conv3 = MobileV2Residual(c * 2, c * 4, 2)
            self.conv4 = nn.Sequential(*(MobileV2Residual(c * 4, c * 4, 1) for _ in range(sixteenth - 1)))
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
            # timm's mobilenetv2_100 up to its 160-channel stage, under the names upstream gives its blocks.
            self.conv_stem, self.bn1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32)
            self.block0 = nn.Sequential(DepthwiseSeparable(32, 16))
            self.block1, self.block2 = stage(16, 24, 2, 2), stage(24, 32, 2, 3)
            # Upstream slices timm's blocks[3:5], which keeps their indices as module names.
            self.block3 = nn.Sequential(OrderedDict([("3", stage(32, 64, 2, 4)), ("4", stage(64, 96, 1, 3))]))
            self.block4 = stage(96, 160, 2, 3)
            self.fpn_layer4, self.fpn_layer3, self.fpn_layer2 = FPNLayer(160, 96), FPNLayer(96, 32), FPNLayer(32, 24)
            # Upstream pads this convolution by replicating edges; the export pads explicitly instead.
            padding = {"padding": 0} if export else {"padding": 1, "padding_mode": "replicate"}
            self.out_conv = conv(24, 24, instance_norm, kernel_size=3, **padding)

        def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
            c2 = self.block1(self.block0(F.relu6(self.bn1(self.conv_stem(image)))))
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


def load_network(weights: Path, export: bool) -> Any:
    import torch

    network = build_network(export)
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return network.eval()


def side_by_side(network: Any) -> Any:
    """Wrap the two-view network so it takes one 540x1920 [left | right] frame and returns a 540x960 disparity.

    The top row repeated TOP_PAD times above the frame is RightTopPad's edge padding of both views at once.
    """
    import torch

    class SideBySide(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network

        def forward(self, stereo: torch.Tensor) -> torch.Tensor:
            padded = torch.cat([stereo[:, :, :1]] * TOP_PAD + [stereo], 2)
            disparity = self.network(padded[..., :WIDTH], padded[..., WIDTH:])
            return disparity[:, :, TOP_PAD:]

    return SideBySide().eval()


def validate_export(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    require_onnx_opset_version(model)
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected input {INPUT} and output {OUTPUT} in that order")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}: {shape}")
    forbidden = sorted({node.op_type for node in model.graph.node} & FORBIDDEN_OPS)
    if forbidden:
        raise ValueError(f"Export still holds operators the rewrite removes: {forbidden}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("The export must be one self-contained ONNX file")


def readout_operations(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """The two softmaxes and every operation after them, in graph order."""
    readers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node)
    pending = [node for node in graph.node if node.op_type == "Softmax"]
    names = {node.name for node in pending}
    while pending:
        for name in pending.pop().output:
            for reader in readers.get(name, []):
                if reader.name not in names:
                    names.add(reader.name)
                    pending.append(reader)
    return [node for node in graph.node if node.name in names]


def annotate_bf16_regions(source: Path, output: Path, regions: list[str]) -> int:
    """Write a copy of the export whose regions run BF16 in an INT8 build; return the number of marked operations.

    The Model Compiler gives an ai.sima AnnotatePrecision's precision to the operation that produces its input, and
    to that operation's constants. The read-out runs to the graph output, so no INT8 mark has to end it. ONNX
    Runtime cannot run the marked copy.
    """
    unknown = sorted(set(regions) - set(BF16_REGIONS))
    if unknown:
        raise ValueError(f"unknown BF16 regions {unknown}; choose from {list(BF16_REGIONS)}")
    model = onnx.load(source)
    marked = {node.name for node in readout_operations(model.graph)} if "readout" in regions else set()
    nodes = []
    for node in model.graph.node:
        nodes.append(node)
        if node.name in marked:
            (value,) = node.output
            node.output[0] = f"{value}/unannotated"
            nodes.append(
                helper.make_node(
                    "AnnotatePrecision",
                    [node.output[0]],
                    [value],
                    name=f"{node.name}/AnnotatePrecision",
                    domain="ai.sima",
                    precision="bfloat16",
                )
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("ai.sima", 1))
    onnx.save(model, output)
    return len(marked)


def fold_constants(path: Path) -> None:
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={INPUT: INTERFACE[INPUT]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded LightStereo export")
    onnx.save(model, path)


def synthetic_frame(seed: int) -> np.ndarray:
    """A normalized noise frame, [left | right], whose right view is the left one shifted left by a seeded
    disparity, as a rectified pair."""
    random = np.random.RandomState(seed)
    left = random.normal(0, 1, [1, 3, HEIGHT, WIDTH]).astype(np.float32)
    return np.concatenate([left, np.roll(left, -random.randint(1, MAX_DISP), axis=3)], axis=3)


def run_upstream(reference: Any, stereo: np.ndarray) -> np.ndarray:
    """Run upstream's evaluation of one frame: each view edge-padded to 544x960 at the top, the disparity's padded rows
    dropped."""
    import torch

    padded = np.pad(stereo, ((0, 0), (0, 0), (TOP_PAD, 0), (0, 0)), mode="edge")
    with torch.inference_mode():
        disparity = reference(torch.from_numpy(padded[..., :WIDTH]), torch.from_numpy(padded[..., WIDTH:]))
    return disparity.numpy()[:, :, TOP_PAD:]


def compare_disparity(expected: np.ndarray, actual: np.ndarray) -> float:
    """Return the largest disparity difference in pixels, failing on a shape change, NaN or excess error."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite disparity: {actual.shape} vs {expected.shape}")
    error = float(np.abs(actual - expected).max())
    if error > TOLERANCE_PIXELS:
        raise ValueError(f"ONNX disparity differs from PyTorch by {error:.5f} px (limit {TOLERANCE_PIXELS})")
    return error


def export_network(weights: Path, output: Path) -> Path:
    """Export the rewritten network from weights.pth to output, then check it against upstream's operations."""
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    reference, network = load_network(weights, export=False), side_by_side(load_network(weights, export=True))
    with torch.inference_mode():
        torch.onnx.export(
            network,
            (torch.zeros(INTERFACE[INPUT]),),
            output,
            input_names=[INPUT],
            output_names=[OUTPUT],
            opset_version=onnx_opset_version(),
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    errors = []
    for seed in SEEDS:
        stereo = synthetic_frame(seed)
        expected = run_upstream(reference, stereo)
        errors.append(compare_disparity(expected, session.run([OUTPUT], {INPUT: stereo})[0]))
    print(
        f"Synthetic-pair PyTorch/ONNX disparity check passed with seeds {SEEDS.start}-{SEEDS.stop - 1} "
        f"(max {max(errors):.2e} px): {output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="weights.pth from the conversion")
    parser.add_argument("--output", type=Path, required=True, help="ONNX file to write")
    args = parser.parse_args()
    export_network(args.weights.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
