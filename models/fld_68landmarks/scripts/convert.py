"""Export ibug's FAN 2dfan2_alt, a 68-point face landmark network, as an ONNX of its heatmap head.

The source is ibug-group/face_alignment's `2dfan2_alt.pth` (MIT), mirrored byte-for-byte on Hugging Face at
public-data/ibug-face-alignment. FAN (Bulat and Tzimiropoulos, ICCV 2017) is a stem and two stacked hourglasses;
ibug trained this one on 300W-LP. Its network code is reproduced below from ibug's `fan.py` (MIT, Copyright (c) 2020
Jie Shen) with the 2dfan2_alt configuration: crop ratio 0.55, 256x256 input, two modules, 256 hourglass features,
depth 4, max pooling, batch normalization, a 7x7 stride-2 stem and a 2x2 stem pool, 68 landmarks.
The corresponding upstream source is ibug/face_alignment/fan/fan.py at b52975cfec7a08edbb247ac1db0f860827235070.

The export takes one [1, 3, 256, 256] RGB face crop in [0, 1] and returns the last hourglass's heatmaps,
[1, 68, 64, 64]. The crop around the face box and the heatmap-to-coordinate decode stay on the CPU (landmarks.py).
The ONNX is checked against PyTorch on seeded inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]
WEIGHTS = "fan/2dfan2_alt.pth"
INPUT_NAME, OUTPUT_NAME = "image", "heatmaps"
INPUT_SHAPE, OUTPUT_SHAPE = (1, 3, 256, 256), (1, 68, 64, 64)
OPSET = 17
MAX_HEATMAP_ERROR = 1e-4
CONFIG = dict(num_modules=2, hg_num_features=256, hg_depth=4, num_landmarks=68)


def validate_source(source: dict[str, Any]) -> None:
    """The conversion takes exactly the pinned 2dfan2_alt weights, as one file."""
    if source.get("type") != "file" or not str(source.get("uri", "")).endswith(f"/{WEIGHTS}"):
        raise ValueError(f"the conversion source must pin exactly {WEIGHTS}")


def build_network():
    """ibug's FAN with the 2dfan2_alt configuration (batch normalization, max pooling)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def conv3x3(inputs: int, outputs: int) -> nn.Conv2d:
        return nn.Conv2d(inputs, outputs, kernel_size=3, stride=1, padding=1, bias=False)

    class ConvBlock(nn.Module):
        def __init__(self, inputs: int, outputs: int):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(inputs)
            self.conv1 = conv3x3(inputs, outputs // 2)
            self.bn2 = nn.BatchNorm2d(outputs // 2)
            self.conv2 = conv3x3(outputs // 2, outputs // 4)
            self.bn3 = nn.BatchNorm2d(outputs // 4)
            self.conv3 = conv3x3(outputs // 4, outputs // 4)
            self.downsample = (
                nn.Sequential(
                    nn.BatchNorm2d(inputs),
                    nn.ReLU(True),
                    nn.Conv2d(inputs, outputs, kernel_size=1, stride=1, bias=False),
                )
                if inputs != outputs
                else None
            )

        def forward(self, x):
            out1 = self.conv1(F.relu(self.bn1(x), True))
            out2 = self.conv2(F.relu(self.bn2(out1), True))
            out3 = self.conv3(F.relu(self.bn3(out2), True))
            residual = self.downsample(x) if self.downsample is not None else x
            return torch.cat((out1, out2, out3), 1) + residual

    class HourGlass(nn.Module):
        def __init__(self, depth: int, features: int):
            super().__init__()
            self.depth = depth
            for level in range(depth, 0, -1):
                self.add_module(f"b1_{level}", ConvBlock(features, features))
                self.add_module(f"b2_{level}", ConvBlock(features, features))
                if level == 1:
                    self.add_module(f"b2_plus_{level}", ConvBlock(features, features))
                self.add_module(f"b3_{level}", ConvBlock(features, features))

        def level(self, depth: int, x):
            up1 = self._modules[f"b1_{depth}"](x)
            low1 = self._modules[f"b2_{depth}"](F.max_pool2d(x, 2))
            low2 = self.level(depth - 1, low1) if depth > 1 else self._modules[f"b2_plus_{depth}"](low1)
            low3 = self._modules[f"b3_{depth}"](low2)
            return up1 + F.interpolate(low3, scale_factor=2, mode="nearest")

        def forward(self, x):
            return self.level(self.depth, x)

    class FAN(nn.Module):
        def __init__(self, num_modules: int, hg_num_features: int, hg_depth: int, num_landmarks: int):
            super().__init__()
            self.num_modules = num_modules
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = ConvBlock(64, 128)
            self.conv3 = ConvBlock(128, 128)
            self.conv4 = ConvBlock(128, hg_num_features)
            for i in range(num_modules):
                self.add_module(f"m{i}", HourGlass(hg_depth, hg_num_features))
                self.add_module(f"top_m_{i}", ConvBlock(hg_num_features, hg_num_features))
                self.add_module(f"conv_last{i}", nn.Conv2d(hg_num_features, hg_num_features, kernel_size=1))
                self.add_module(f"bn_end{i}", nn.BatchNorm2d(hg_num_features))
                self.add_module(f"l{i}", nn.Conv2d(hg_num_features, num_landmarks, kernel_size=1))
                if i < num_modules - 1:
                    self.add_module(f"bl{i}", nn.Conv2d(hg_num_features, hg_num_features, kernel_size=1))
                    self.add_module(f"al{i}", nn.Conv2d(num_landmarks, hg_num_features, kernel_size=1))

        def forward(self, x):
            x = F.max_pool2d(self.conv2(F.relu(self.bn1(self.conv1(x)), True)), 2)
            previous = self.conv4(self.conv3(x))
            heatmaps = None
            for i in range(self.num_modules):
                features = self._modules[f"top_m_{i}"](self._modules[f"m{i}"](previous))
                features = F.relu(self._modules[f"bn_end{i}"](self._modules[f"conv_last{i}"](features)), True)
                heatmaps = self._modules[f"l{i}"](features)
                if i < self.num_modules - 1:
                    previous = previous + self._modules[f"bl{i}"](features) + self._modules[f"al{i}"](heatmaps)
            return heatmaps

    return FAN(**CONFIG)


def load(weights: Path):
    import torch

    network = build_network()
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return network.eval()


def validate_export(path: Path) -> None:
    """The exported ONNX has the declared one-input, one-output interface."""
    import onnx

    graph = onnx.load(str(path)).graph
    initializers = {item.name for item in graph.initializer}

    def shapes(values: Any) -> dict[str, list[int]]:
        return {
            item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
            for item in values
            if item.name not in initializers
        }

    expected = ({INPUT_NAME: list(INPUT_SHAPE)}, {OUTPUT_NAME: list(OUTPUT_SHAPE)})
    if (shapes(graph.input), shapes(graph.output)) != expected:
        raise ValueError(f"unexpected FAN interface: {shapes(graph.input)} -> {shapes(graph.output)}")


def convert(weights: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch

    network = load(weights)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.onnx.export(
        network,
        torch.zeros(INPUT_SHAPE),
        str(output),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamo=False,
    )
    validate_export(output)

    inputs = torch.rand((4, *INPUT_SHAPE[1:]), generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected = network(inputs).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    exported = np.concatenate([session.run(None, {INPUT_NAME: inputs[i : i + 1].numpy()})[0] for i in range(4)])
    error = float(np.abs(exported - expected).max())
    print(f"max |ONNX - PyTorch| heatmap error over {len(inputs)} inputs: {error:.2e}")
    if error > MAX_HEATMAP_ERROR:
        raise ValueError(f"the export differs from PyTorch by {error:.2e}")
    return output


def main() -> None:
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
