"""Export RAFT-Stereo's realtime model from its checkpoint as one static graph, and check it against the network's
iterative form.

RAFT-Stereo (Lipson, Teed and Deng; Princeton Vision & Learning Lab, princeton-vl/RAFT-Stereo, MIT licence) estimates
the disparity of a rectified stereo pair. This file defines the realtime configuration's network with the upstream
module names, so the pinned checkpoint loads strictly: one backbone shared by both views and the context, features at
1/8 resolution, two GRU levels updated slow-fast, a 4-level correlation pyramid of radius 4, and 7 refinement
iterations, as upstream's realtime evaluation runs it.

The graph takes one side-by-side frame [1, 3, 540, 1920], the left and the right 540x960 image of a Scene Flow pair,
RGB normalized with the ImageNet mean and std as Neat's ImageNet preset does, and returns the left image's disparity
[1, 1, 540, 960]. Inside, it maps the frame to upstream's [-1, 1] range and pads each view to 544 rows, a multiple of
32, by repeating its top and bottom rows twice as upstream's evaluation pads with replication; it removes those rows
from the disparity. To fit the MLA, it rewrites the network exactly:
- the rows are padded by slicing and joining, since edge Pad does not compile for the MLA;
- the side-by-side frame is split in the graph, and the backbone runs once per view at batch 1;
- InstanceNorm is written out as means over each channel's pixels;
- the correlation pyramid pools the right features along the width before the matmul, which pooling's linearity
  makes exact, with the right image's columns on channels;
- each level's bilinear lookup at 9 offsets is a tent-weighted sum over the columns, relu(1 - |p - j|): what
  grid_sample with align_corners=True and zero padding computes on a one-row image. The tent weights are computed
  once per level and iteration over the columns widened by the radius, and each offset reads its own shifted,
  zero-padded copy of the level's correlations, made once before the iterations; |x| is written as
  leaky_relu(x, -1), which the MLA runs without a layout change on each side;
- the refinement state is the x flow, kept small, instead of absolute coordinates, and the y flow, always 0 in
  stereo, is a constant;
- the convex upsampling is a channel softmax, a convolution taking each neighbour's difference from the centre, the
  centre plus the weighted differences, which equals the weighted sum because the weights sum to 1 but rounds smaller
  products, and a pixel shuffle;
- the 7 iterations are unrolled.
"""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "raftstereo-realtime.pth"
OPSET = 18
ITERATIONS = 7
# Each view, and the rows padded above and below it to a multiple of 32.
HEIGHT, WIDTH = 540, 960
PAD = 2
PADDED_HEIGHT = HEIGHT + 2 * PAD
LEVELS, RADIUS = 4, 4
HIDDEN = 128
INTERFACE = {"stereo": [1, 3, HEIGHT, 2 * WIDTH], "disparity": [1, 1, HEIGHT, WIDTH]}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
# Edge Pad does not compile for the MLA, InstanceNormalization is written out, and Abs, which the compiler has no
# layout rule for, costs a transpose on each side of every use; the others are index arithmetic or ops the MLA cannot
# run.
FORBIDDEN_OPS = {
    "Abs",
    "Shape",
    "Size",
    "If",
    "Loop",
    "GridSample",
    "Gather",
    "GatherND",
    "ScatterND",
    "Pad",
    "InstanceNormalization",
}
SEEDS = range(123, 126)
# The rewrites reorder float sums, which 7 iterations compound; on Scene Flow pairs the export differs from upstream
# by under 0.002 px.
TOLERANCE_PIXELS = 0.01


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def network():
    """RAFT-Stereo's realtime network, with upstream's module names."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class InstanceNorm(nn.Module):
        """InstanceNorm2d without affine parameters, written out over each channel's pixels."""

        def forward(self, x):
            centred = x - x.mean((2, 3), keepdim=True)
            return centred / torch.sqrt((centred * centred).mean((2, 3), keepdim=True) + 1e-5)

    class ResidualBlock(nn.Module):
        def __init__(self, in_planes, planes, norm, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, stride=stride)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
            self.relu = nn.ReLU()
            make = (lambda: nn.BatchNorm2d(planes)) if norm == "batch" else InstanceNorm
            self.norm1, self.norm2 = make(), make()
            self.downsample = None
            if stride != 1 or in_planes != planes:
                self.norm3 = make()
                self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride), self.norm3)

        def forward(self, x):
            y = self.relu(self.norm2(self.conv2(self.relu(self.norm1(self.conv1(x))))))
            if self.downsample is not None:
                x = self.downsample(x)
            return self.relu(x + y)

    class MultiBasicEncoder(nn.Module):
        """The shared backbone with batch norms and its context heads at 1/8, 1/16 and 1/32 (unused here)."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.norm1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.in_planes = 64
            self.layer1 = self.layer(64, 1)
            self.layer2 = self.layer(96, 2)
            self.layer3 = self.layer(128, 2)
            self.layer4 = self.layer(128, 2)
            self.layer5 = self.layer(128, 2)
            head = lambda: nn.Sequential(ResidualBlock(128, 128, "batch"), nn.Conv2d(128, HIDDEN, 3, padding=1))  # noqa: E731
            self.outputs08 = nn.ModuleList([head(), head()])
            self.outputs16 = nn.ModuleList([head(), head()])
            self.outputs32 = nn.ModuleList([nn.Conv2d(128, HIDDEN, 3, padding=1) for _ in range(2)])

        def layer(self, planes, stride):
            blocks = nn.Sequential(
                ResidualBlock(self.in_planes, planes, "batch", stride), ResidualBlock(planes, planes, "batch")
            )
            self.in_planes = planes
            return blocks

        def features(self, image):
            x = self.relu1(self.norm1(self.conv1(image)))
            return self.layer3(self.layer2(self.layer1(x)))

    class ConvGRU(nn.Module):
        def __init__(self, inputs):
            super().__init__()
            self.convz = nn.Conv2d(HIDDEN + inputs, HIDDEN, 3, padding=1)
            self.convr = nn.Conv2d(HIDDEN + inputs, HIDDEN, 3, padding=1)
            self.convq = nn.Conv2d(HIDDEN + inputs, HIDDEN, 3, padding=1)

        def forward(self, h, cz, cr, cq, *inputs):
            x = torch.cat(inputs, 1)
            hx = torch.cat([h, x], 1)
            z = torch.sigmoid(self.convz(hx) + cz)
            r = torch.sigmoid(self.convr(hx) + cr)
            q = torch.tanh(self.convq(torch.cat([r * h, x], 1)) + cq)
            return (1 - z) * h + z * q

    class MotionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.convc1 = nn.Conv2d(LEVELS * (2 * RADIUS + 1), 64, 1)
            self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
            self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
            self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv = nn.Conv2d(128, 126, 3, padding=1)

        def forward(self, flow, corr):
            cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
            flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
            return torch.cat([F.relu(self.conv(torch.cat([cor, flo], 1))), flow], 1)

    class FlowHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(HIDDEN, 256, 3, padding=1)
            self.conv2 = nn.Conv2d(256, 2, 3, padding=1)

        def forward(self, x):
            return self.conv2(F.relu(self.conv1(x)))

    class UpdateBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MotionEncoder()
            self.gru08 = ConvGRU(128 + HIDDEN)
            self.gru16 = ConvGRU(HIDDEN)
            self.gru32 = ConvGRU(HIDDEN)  # the 1/32 level, unused with two GRU levels
            self.flow_head = FlowHead()
            self.mask = nn.Sequential(nn.Conv2d(HIDDEN, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 64 * 9, 1))

        def slow(self, net, inp):
            return [net[0], self.gru16(net[1], *inp[1], F.avg_pool2d(net[0], 3, stride=2, padding=1))]

        def forward(self, net, inp, corr, flow):
            net = self.slow(net, inp)
            upsampled = F.interpolate(net[1], net[0].shape[2:], mode="bilinear", align_corners=True)
            h08 = self.gru08(net[0], *inp[0], self.encoder(flow, corr), upsampled)
            return [h08, net[1]], 0.25 * self.mask(h08), self.flow_head(h08)

    class RAFTStereoRealtime(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnet = MultiBasicEncoder()
            self.update_block = UpdateBlock()
            self.context_zqr_convs = nn.ModuleList([nn.Conv2d(HIDDEN, 3 * HIDDEN, 3, padding=1) for _ in range(2)])
            self.conv2 = nn.Sequential(ResidualBlock(128, 128, "instance"), nn.Conv2d(128, 256, 3, padding=1))

        def context(self, left_features):
            outputs08 = [head(left_features) for head in self.cnet.outputs08]
            sixteenth = self.cnet.layer4(left_features)
            outputs16 = [head(sixteenth) for head in self.cnet.outputs16]
            net = [torch.tanh(outputs08[0]), torch.tanh(outputs16[0])]
            inp = [
                list(conv(torch.relu(x[1])).split(HIDDEN, dim=1))
                for x, conv in zip((outputs08, outputs16), self.context_zqr_convs)
            ]
            return net, inp

    return RAFTStereoRealtime


def load(checkpoint):
    import torch

    model = network()()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return model.eval()


def reference(model, left, right):
    """Upstream's iterative forward for the realtime model: both views as a batch of 2, absolute coordinates, the
    grid_sample lookup and unfold's convex upsampling. Images are [1, 3, H, W] in [0, 255]; returns the disparity."""
    import torch
    import torch.nn.functional as F

    images = torch.cat([2 * (left / 255) - 1, 2 * (right / 255) - 1], 0)
    features = model.cnet.features(images)
    net, inp = model.context(features[:1])
    fmap1, fmap2 = model.conv2(features).split(1, 0)
    _, depth, height, width = fmap1.shape
    corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2).reshape(height * width, 1, 1, width) / depth**0.5
    pyramid = [corr]
    for _ in range(LEVELS - 1):
        pyramid.append(F.avg_pool2d(pyramid[-1], [1, 2], stride=[1, 2]))
    x = torch.arange(width, dtype=torch.float32).repeat(height, 1)
    coords = x.clone()
    offsets = torch.linspace(-RADIUS, RADIUS, 2 * RADIUS + 1).view(2 * RADIUS + 1, 1)
    for _ in range(ITERATIONS):
        samples = []
        for level, volume in enumerate(pyramid):
            columns = volume.shape[-1]
            sample_x = offsets + coords.reshape(height * width, 1, 1, 1) / 2**level
            grid = torch.cat([2 * sample_x / (columns - 1) - 1, torch.zeros_like(sample_x)], -1)
            samples.append(F.grid_sample(volume, grid, align_corners=True).view(1, height, width, -1))
        corr_features = torch.cat(samples, -1).permute(0, 3, 1, 2)
        flow = (coords - x)[None, None]
        net = model.update_block.slow(net, inp)
        net, mask, delta = model.update_block(net, inp, corr_features, torch.cat([flow, torch.zeros_like(flow)], 1))
        coords = coords + delta[0, 0]
    flow = (coords - x)[None, None]
    mask = torch.softmax(mask.view(1, 1, 9, 8, 8, height, width), 2)
    neighbours = F.unfold(8 * flow, [3, 3], padding=1).view(1, 1, 9, 1, 1, height, width)
    up = torch.sum(mask * neighbours, 2).permute(0, 1, 4, 2, 5, 3).reshape(1, 1, 8 * height, 8 * width)
    return -up


def static_graph(model):
    """The realtime network as one static graph from the side-by-side frame to the disparity."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    height, width = PADDED_HEIGHT // 8, WIDTH // 8

    class Static(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            # Each level's offset from every pixel's column to each target column at that level's scale, for the
            # columns widened by the radius on both sides: one tent per column covers all 9 offsets of the lookup.
            for level in range(LEVELS):
                columns = width // 2**level
                offsets = (
                    torch.arange(width, dtype=torch.float32)[None, None, None, :] / 2**level
                    - torch.arange(-RADIUS, columns + RADIUS, dtype=torch.float32)[None, :, None, None]
                )
                self.register_buffer(f"offsets{level}", offsets, persistent=False)  # [1, columns + 8, 1, width]
            # Each 3x3 neighbour less the centre; the centre's own channel is zero.
            centre = torch.zeros(9, 1, 3, 3)
            centre[:, 0, 1, 1] = 1
            self.register_buffer("neighbours", torch.eye(9).reshape(9, 1, 3, 3) - centre, persistent=False)
            # ImageNet-normalized RGB back to upstream's 2 * x - 1, for x in [0, 1].
            self.register_buffer("scale", torch.from_numpy(2 * STD).reshape(1, 3, 1, 1), persistent=False)
            self.register_buffer("shift", torch.from_numpy(2 * MEAN - 1).reshape(1, 3, 1, 1), persistent=False)

        def lookup(self, shifted, flow):
            samples = []
            for level, volumes in enumerate(shifted):
                position = getattr(self, f"offsets{level}") + flow / 2**level  # [1, columns + 8, H, W]
                weight = torch.relu(1 - F.leaky_relu(position, negative_slope=-1.0))
                # The tent around column p, read at column i + offset, is the tent around p - offset at column i.
                samples.extend((volume * weight).sum(1, keepdim=True) for volume in volumes)
            return torch.cat(samples, 1)

        def forward(self, stereo):
            image = stereo * self.scale + self.shift
            top, bottom = image[:, :, :1], image[:, :, -1:]
            image = torch.cat([top] * PAD + [image] + [bottom] * PAD, 2)
            left = self.model.cnet.features(image[..., :WIDTH])
            right = self.model.cnet.features(image[..., WIDTH:])
            net, inp = self.model.context(left)
            fmap1, fmap2 = self.model.conv2(left), self.model.conv2(right)
            depth = fmap1.shape[1]
            rows = fmap1.reshape(depth, height, width).permute(1, 2, 0)  # [H, W, D]
            shifted = []
            for level in range(LEVELS):
                pooled = F.avg_pool2d(fmap2, [1, 2**level], stride=[1, 2**level]) if level else fmap2
                columns = width // 2**level
                corr = rows @ pooled.reshape(depth, height, columns).permute(1, 0, 2) / depth**0.5  # [H, W, columns]
                corr = corr.permute(2, 0, 1).reshape(1, columns, height, width)
                # Zero columns, computed from the input like the first flow below, pad the volume by twice the radius.
                zeros = corr[:, : 2 * RADIUS] * 0
                padded = torch.cat([zeros, corr, zeros], 1)
                # Offset k's copy holds column i + k at row i of the widened range, zero outside the volume.
                shifted.append(
                    [padded[:, k + RADIUS : k + RADIUS + columns + 2 * RADIUS] for k in range(-RADIUS, RADIUS + 1)]
                )
            # Zero, but computed from the input, so that the first iteration's lookup is not folded into constants.
            flow = net[0][:, :1] * 0
            zero = flow
            for _ in range(ITERATIONS):
                corr_features = self.lookup(shifted, flow)
                # Slow-fast: the 1/16 GRU updates once more before each full update.
                net = self.model.update_block.slow(net, inp)
                net, mask, delta = self.model.update_block(net, inp, corr_features, torch.cat([flow, zero], 1))
                flow = flow + delta[:, :1]
            mask = torch.softmax(mask.reshape(1, 9, 64 * height, width), 1).reshape(1, 9 * 64, height, width)
            differences = F.conv2d(8 * flow, self.neighbours, padding=1)
            up = 8 * flow + sum(mask[:, k * 64 : (k + 1) * 64] * differences[:, k : k + 1] for k in range(9) if k != 4)
            return -F.pixel_shuffle(up, 8)[:, :, PAD : PAD + HEIGHT]

    return Static().eval()


def known_shapes(model):
    inferred = onnx.shape_inference.infer_shapes(model).graph
    shapes = {}
    for value in [*inferred.value_info, *inferred.input, *inferred.output]:
        dims = value.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = [dim.dim_value for dim in dims]
    return shapes


def fold_constants(path):
    """Fold the export to a static graph: onnxsim leaves Shape nodes of tensors whose shapes are known; each becomes a
    constant and the graph is folded again, until none is left."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(path), perform_optimization=False)
    for _ in range(8):
        if not valid:
            raise RuntimeError("onnxsim could not validate the folded RAFT-Stereo export")
        shapes = known_shapes(model)
        replaced = 0
        for index, node in enumerate(model.graph.node):
            if node.op_type == "Shape" and node.input[0] in shapes:
                attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                value = shapes[node.input[0]][attributes.get("start", 0) : attributes.get("end")]
                constant = onnx.numpy_helper.from_array(np.asarray(value, np.int64))
                model.graph.node[index].CopyFrom(
                    onnx.helper.make_node("Constant", [], list(node.output), value=constant)
                )
                replaced += 1
        if not replaced:
            break
        model, valid = simplify(model, perform_optimization=False)
    onnx.save(model, path)


def validate_export(path, max_constants):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"RAFT-Stereo export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected inputs and outputs {list(INTERFACE)}")
    for value in values:
        tensor = value.type.tensor_type
        if (
            tensor.elem_type != onnx.TensorProto.FLOAT
            or [d.dim_value for d in tensor.shape.dim] != INTERFACE[value.name]
        ):
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")
    unsupported = sorted({node.op_type for node in model.graph.node} & FORBIDDEN_OPS)
    if unsupported:
        raise ValueError(f"The folded export still holds {unsupported}")
    constants = sum(int(np.prod(tensor.dims)) for tensor in model.graph.initializer)
    if constants > max_constants:
        raise ValueError(
            f"The folded export holds {constants} constant values, more than the network's "
            f"{max_constants}: a loop iteration was folded into constants"
        )


def stereo_pair(seed):
    """A smooth seeded texture seen by two cameras, as HWC RGB in [0, 255]: a point at column x of the left view is at
    x - d in the right view, with a disparity d that grows from 8 px at the top of the image to 48 px at the bottom."""
    import torch
    import torch.nn.functional as F

    coarse = torch.from_numpy(np.random.default_rng(seed).uniform(0, 255, (1, 3, HEIGHT // 8, (WIDTH + 64) // 8)))
    texture = F.interpolate(coarse, size=(HEIGHT, WIDTH + 64), mode="bicubic", align_corners=False)[0].numpy()
    texture = texture.transpose(1, 2, 0).astype(np.float32)
    left = texture[:, :WIDTH]
    right = np.empty_like(left)
    for row in range(HEIGHT):
        columns = np.arange(WIDTH, dtype=np.float32) + 8 + 40 * row / HEIGHT
        for channel in range(3):
            right[row, :, channel] = np.interp(columns, np.arange(WIDTH + 64), texture[row, :, channel])
    return np.clip(left, 0, 255), np.clip(right, 0, 255)


def normalize(left, right):
    """The model input for one pair of HWC RGB views in [0, 255]: the side-by-side frame, ImageNet-normalized, NCHW."""
    frame = (np.concatenate([left, right], 1).astype(np.float32) / 255 - MEAN) / STD
    return frame.transpose(2, 0, 1)[None].copy()


def compare_disparity(expected, actual):
    """Return the largest difference between the static graph's disparity and the iterative network's, in pixels."""
    if actual.shape != expected.shape:
        raise ValueError(f"Unexpected disparity shape {actual.shape}; expected {expected.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("The exported disparity holds non-finite values")
    difference = float(np.max(np.abs(actual - expected)))
    if difference > TOLERANCE_PIXELS:
        raise ValueError(f"The exported disparity differs from PyTorch by {difference:.3f} px")
    return difference


def convert(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    model = load(source)
    static = static_graph(model)
    parameters = sum(value.numel() for value in model.state_dict().values())
    example = (torch.rand(*INTERFACE["stereo"]),)
    program = torch.onnx.export(
        static, example, dynamo=True, opset_version=OPSET, input_names=["stereo"], output_names=["disparity"]
    )
    program.save(str(output), external_data=False)
    fold_constants(output)
    # The network's own weights, plus the lookup's offsets and the folded batch norms' small constants.
    validate_export(output, max_constants=parameters + 2_000_000)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    differences = []
    for seed in SEEDS:
        left, right = stereo_pair(seed)
        # Upstream's evaluation pads each view with replication; the reference runs on the padded views.
        padded = lambda image: torch.nn.functional.pad(  # noqa: E731
            torch.from_numpy(image.transpose(2, 0, 1).copy())[None], (0, 0, PAD, PAD), mode="replicate"
        )
        with torch.inference_mode():
            expected = reference(model, padded(left), padded(right))[:, :, PAD : PAD + HEIGHT].numpy()
        differences.append(compare_disparity(expected, session.run(None, {"stereo": normalize(left, right)})[0]))
    print(
        f"Seeded stereo-pair iterative/ONNX disparity check passed with seeds {SEEDS.start}-{SEEDS.stop - 1}, "
        f"largest difference {max(differences):.1e} px: {output}"
    )
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
