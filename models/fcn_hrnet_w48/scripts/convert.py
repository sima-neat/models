"""Export FCN-HRNet-W48 (ADE20K, 512x512) from its MMSegmentation checkpoint and check ONNX against PyTorch.

MMSegmentation's own code needs mmcv, which has no wheel for the conversion environment, so this script rebuilds the
network in plain PyTorch under the checkpoint's own key names and loads every tensor strictly. The rebuild ports
mmseg/models/backbones/hrnet.py, mmseg/models/decode_heads/decode_head.py's resize_concat, fcn_head.py and
mmseg/models/utils/wrappers.py (MMSegmentation, Apache-2.0) as they run at inference, operation for operation: each
lower-resolution fuse input is upsampled by 2**(j - i) and then resized to the target size, and the decode head resizes
all four branches before concatenating them, even where a resize keeps the size. SyncBN becomes BatchNorm2d, and each
fuse sum starts from its first term instead of from zero, which adds nothing. The export is the model's forward pass at
the shipped 512x512 input: the decode head's logits at a quarter of the input resolution, then the resize of those
logits to the input that MMSegmentation's predict_by_feat makes, up to where its post-processing resizes them to each
image and takes the argmax. surgery.py cuts that last resize at compile time.
"""

import argparse
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "fcn_hr48_512x512_160k_ade20k_20200614_214407-a52fc02c.pth"
# Each branch's channels; branch i runs at 1/2**(i + 2) of the input resolution.
WIDTHS = (48, 96, 192, 384)
OPSET = 17
IMAGE_SIZE = 512
CLASS_COUNT = 150
INTERFACE = {
    "image": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
    "logits": [1, CLASS_COUNT, IMAGE_SIZE, IMAGE_SIZE],
}
SEEDS = range(123, 133)


def build_network(widths, class_count):
    """FCN with an HRNetV2 backbone, in MMSegmentation's module layout so the checkpoint's keys load unchanged."""
    import torch
    from torch import nn
    from torch.nn import functional

    def norm(channels):
        return nn.BatchNorm2d(channels, eps=1e-5)

    def resize(tensor, size):
        # MMSegmentation's resize as HRNet and its decode head call it: bilinear, align_corners=False.
        return functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    class Upsample(nn.Module):
        """MMSegmentation's parameter-free Upsample: a resize to the input size times scale_factor, rounded down."""

        def __init__(self, scale_factor):
            super().__init__()
            self.scale_factor = float(scale_factor)

        def forward(self, tensor):
            return resize(tensor, [int(size * self.scale_factor) for size in tensor.shape[-2:]])

    class BasicBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            self.bn1 = norm(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
            self.bn2 = norm(channels)
            self.relu = nn.ReLU()

        def forward(self, tensor):
            out = self.relu(self.bn1(self.conv1(tensor)))
            return self.relu(self.bn2(self.conv2(out)) + tensor)

    class Bottleneck(nn.Module):
        def __init__(self, inplanes, planes, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = norm(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = norm(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
            self.bn3 = norm(planes * 4)
            self.relu = nn.ReLU()
            self.downsample = downsample

        def forward(self, tensor):
            identity = tensor if self.downsample is None else self.downsample(tensor)
            out = self.relu(self.bn1(self.conv1(tensor)))
            out = self.relu(self.bn2(self.conv2(out)))
            return self.relu(self.bn3(self.conv3(out)) + identity)

    class HRModule(nn.Module):
        """Parallel branches of four basic blocks, then every branch receives every other branch's features."""

        def __init__(self, channels):
            super().__init__()
            count = len(channels)
            self.branches = nn.ModuleList(nn.Sequential(*[BasicBlock(c) for _ in range(4)]) for c in channels)
            rows = []
            for i in range(count):
                row = []
                for j in range(count):
                    if j > i:
                        row.append(
                            nn.Sequential(
                                nn.Conv2d(channels[j], channels[i], 1, bias=False),
                                norm(channels[i]),
                                Upsample(2 ** (j - i)),
                            )
                        )
                    elif j == i:
                        row.append(None)
                    else:
                        steps = []
                        for k in range(i - j):
                            out = channels[i] if k == i - j - 1 else channels[j]
                            step = [nn.Conv2d(channels[j], out, 3, 2, 1, bias=False), norm(out)]
                            if k < i - j - 1:
                                step.append(nn.ReLU())
                            steps.append(nn.Sequential(*step))
                        row.append(nn.Sequential(*steps))
                rows.append(nn.ModuleList(row))
            self.fuse_layers = nn.ModuleList(rows) if count > 1 else None
            self.relu = nn.ReLU()

        def forward(self, tensors):
            tensors = [branch(tensor) for branch, tensor in zip(self.branches, tensors, strict=True)]
            if self.fuse_layers is None:
                return tensors
            fused = []
            for i, row in enumerate(self.fuse_layers):
                terms = []
                for j, layer in enumerate(row):
                    if j == i:
                        terms.append(tensors[j])
                    elif j > i:
                        terms.append(resize(layer(tensors[j]), tensors[i].shape[2:]))
                    else:
                        terms.append(layer(tensors[j]))
                # Summed in branch order, as MMSegmentation sums them.
                total = terms[0]
                for term in terms[1:]:
                    total = total + term
                fused.append(self.relu(total))
            return fused

    def transition(previous, current):
        layers = []
        for i, channels in enumerate(current):
            if i < len(previous):
                same = channels == previous[i]
                layers.append(
                    None
                    if same
                    else nn.Sequential(nn.Conv2d(previous[i], channels, 3, 1, 1, bias=False), norm(channels), nn.ReLU())
                )
            else:
                steps = []
                for j in range(i + 1 - len(previous)):
                    out = channels if j == i - len(previous) else previous[-1]
                    steps.append(nn.Sequential(nn.Conv2d(previous[-1], out, 3, 2, 1, bias=False), norm(out), nn.ReLU()))
                layers.append(nn.Sequential(*steps))
        return nn.ModuleList(layers)

    class HRNet(nn.Module):
        """HRNetV2: a stride-4 stem and a bottleneck stage, then stages of one, four and three modules."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
            self.bn1 = norm(64)
            self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
            self.bn2 = norm(64)
            self.relu = nn.ReLU()
            downsample = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), norm(256))
            self.layer1 = nn.Sequential(Bottleneck(64, 64, downsample), *[Bottleneck(256, 64) for _ in range(3)])
            self.transition1 = transition([256], widths[:2])
            self.stage2 = nn.Sequential(HRModule(widths[:2]))
            self.transition2 = transition(widths[:2], widths[:3])
            self.stage3 = nn.Sequential(*[HRModule(widths[:3]) for _ in range(4)])
            self.transition3 = transition(widths[:3], widths)
            self.stage4 = nn.Sequential(*[HRModule(widths) for _ in range(3)])

        @staticmethod
        def branch_inputs(layers, tensors):
            return [
                tensors[i] if layer is None else layer(tensors[i] if i < len(tensors) else tensors[-1])
                for i, layer in enumerate(layers)
            ]

        def forward(self, image):
            features = self.layer1(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(image)))))))
            tensors = [layer(features) for layer in self.transition1]
            stages = ((self.stage2, self.transition2), (self.stage3, self.transition3), (self.stage4, None))
            for stage, layers in stages:
                for module in stage:
                    tensors = module(tensors)
                if layers is not None:
                    tensors = self.branch_inputs(layers, tensors)
            return tensors

    class ConvModule(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn = norm(channels)
            self.activate = nn.ReLU()

        def forward(self, tensor):
            return self.activate(self.bn(self.conv(tensor)))

    class FCNHead(nn.Module):
        """MMSegmentation's FCNHead with resize_concat: all branches resized to branch 0's size, then concatenated."""

        def __init__(self):
            super().__init__()
            channels = sum(widths)
            self.convs = nn.Sequential(ConvModule(channels))
            self.conv_seg = nn.Conv2d(channels, class_count, 1)

        def forward(self, tensors):
            size = tensors[0].shape[2:]
            features = torch.cat([resize(tensor, size) for tensor in tensors], dim=1)
            return self.conv_seg(self.convs(features))

    class FCNHRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = HRNet()
            self.decode_head = FCNHead()

        def forward(self, image):
            # MMSegmentation's predict_by_feat resizes the logits to the network input before post-processing.
            return resize(self.decode_head(self.backbone(image)), image.shape[2:])

    return FCNHRNet()


def validate_source(source):
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"FCN-HRNet export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input image and output logits")
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
        onnx.load(path), overwrite_input_shapes={"image": INTERFACE["image"]}, dynamic_input_shape=False
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded FCN-HRNet export")
    onnx.save(model, path)


def compare_logits(expected, actual):
    """The ONNX logits must match PyTorch's and pick the same class at nearly every pixel."""
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"Unexpected or non-finite logits: {actual.shape} vs {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    agreement = float(np.mean(actual.argmax(1) == expected.argmax(1)))
    if agreement < 0.999:
        raise ValueError(f"ONNX and PyTorch agree on only {agreement:.4%} of pixel classes")


def load_reference(path):
    """The rebuilt network with every checkpoint tensor loaded strictly, through PyTorch's weights-only unpickler."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = (checkpoint.get("meta") or {}).get("CLASSES")
    if classes is not None and len(classes) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} ADE20K classes, the checkpoint names {len(classes)}")
    network = build_network(WIDTHS, CLASS_COUNT)
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.eval()


def convert(source, output_dir):
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    reference = load_reference(source)
    with tempfile.TemporaryDirectory(prefix="fcn-hrnet-") as temporary:
        exported = Path(temporary) / "model.onnx"
        torch.onnx.export(
            reference,
            (torch.zeros(INTERFACE["image"]),),
            str(exported),
            input_names=["image"],
            output_names=["logits"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
        fold_constants(exported)
        validate_export(exported)
        shutil.move(exported, output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for seed in SEEDS:
        # Normalized images lie in about [-2.1, 2.7]; the samples cover that range.
        sample = np.random.RandomState(seed).uniform(-2.1, 2.7, INTERFACE["image"]).astype(np.float32)
        with torch.inference_mode():
            expected = reference(torch.from_numpy(sample)).numpy()
        compare_logits(expected, session.run(["logits"], {"image": sample})[0])
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
