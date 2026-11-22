"""Build DINOv3 ConvNeXt Large for the MLA from the converted weights, export it to ONNX and check it against the
reference.

The network is timm's convnext_large written out in plain PyTorch, with timm's module names so the weights load
strictly, and the outputs of convert.py's reference.onnx. One change is for the compiler: each GELU is its tanh
approximation, x * sigmoid(x * (x * x * b + a)). The Model Compiler fuses the exact erf GELU into one operation whose
BF16 erf loses fidelity: quantized to w16a16 by Model Compiler 3.0.0.dev0+develop.2291, the Large model's outputs
keep cosine similarity 0.9951 (class) and 0.9758 (patches) with the exact reference on 16 of the imagenet-2012 cal
images at positions 51 to 100 of the seed-42 shuffled order, and 0.9999 and 0.9995 with the tanh form. The tanh form
runs as BF16 multiplies, one multiply-add and a sigmoid. In float, on the seeded inputs export_network checks, its
least cosine similarity with the exact reference is 0.999973.

A second change is for the code generator: the first stage's MLPs run as MLP_SLICES[0] slices of fc1's output
channels, each through its own GELU and its slice of fc2's input channels, and the slices' outputs are added. The sum
is the same MLP. With INT8 weights, Model Compiler 3.0.0.dev0+develop.2291 splits those MLPs into 4 channel parts
itself (conv_bounded_split), and the w8a16 package then keeps cosine similarity only 0.515 (class) and 0.377
(patches) with the reference on the device, on the imagenet-2012 cal images at positions 51 to 100 of the seed-42
shuffled order, against 0.9985 and 0.9920 in its simulation on 16 of them. Sliced here, the code generator leaves the
MLPs whole and the package keeps 0.9987 and 0.9920 on the device. The w16a16 package keeps 0.9999 and 0.9994 on the
device either way, and sliced its MLA stage time falls from 29515 to 27977 microseconds, measured on one board at a
600 MHz MLA clock.

compile.py calls export_network. To get the FP32 ONNX on a host, run:

    python models/dinov3_convnext_large/scripts/surgery.py --weights weights.pth --reference reference.onnx \
        --output model.onnx
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx

DEPTHS = (3, 3, 27, 3)
WIDTHS = (192, 384, 768, 1536)
HIDDEN_SIZE = WIDTHS[-1]
PATCHES = 49
INPUT = "pixel_values"
OUTPUTS = ("class_embedding", "patch_embeddings")
INTERFACE = {INPUT: [1, 3, 224, 224], OUTPUTS[0]: [1, HIDDEN_SIZE], OUTPUTS[1]: [1, PATCHES, HIDDEN_SIZE]}
OPSET = 17
# timm's LayerNorm and LayerNorm2d default.
EPSILON = 1e-6
# GELU's tanh approximation 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), written as
# x * sigmoid(GELU_SCALE (x + GELU_CUBIC x^3)).
GELU_SCALE = 2 * math.sqrt(2 / math.pi)
GELU_CUBIC = 0.044715
SEEDS = range(123, 127)
# The least cosine similarity of each output with the exact reference on the synthetic inputs, just under the
# 0.999973 the tanh GELU measures.
MIN_COSINE = 0.99996
# Slices of each stage's MLP; the first stage's 192 -> 768 -> 192 MLPs run as four 192-channel slices.
MLP_SLICES = (4, 1, 1, 1)


def build_network() -> Any:
    """timm's convnext_large with num_classes=0 and tanh-form GELUs, returning the class and patch embeddings. torch is
    imported here so the tests can load this script without it."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class LayerNorm2d(nn.LayerNorm):
        """A LayerNorm over the channels of an NCHW map, as timm's LayerNorm2d."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def gelu(x: torch.Tensor) -> torch.Tensor:
        # In this order the compiler fuses x * x * (GELU_CUBIC * GELU_SCALE) + GELU_SCALE into one multiply-add.
        # Spelled x * sigmoid(x * (a + b * x * x)) instead, it quantizes to one operation more; on the board the
        # ConvNeXt Tiny w16a16 MLA stage took 3116 us with either spelling, measured alternately in one session, so the
        # shorter form stays.
        return x * torch.sigmoid(x * (x * x * (GELU_CUBIC * GELU_SCALE) + GELU_SCALE))

    class Mlp(nn.Module):
        def __init__(self, width: int, slices: int) -> None:
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)
            self.slices = slices

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.slices == 1:
                return self.fc2(gelu(self.fc1(x)))
            size = self.fc1.out_features // self.slices
            parts = [slice(start, start + size) for start in range(0, self.fc1.out_features, size)]
            # fc2's bias goes with the first slice.
            outputs = [
                F.linear(
                    gelu(F.linear(x, self.fc1.weight[part], self.fc1.bias[part])),
                    self.fc2.weight[:, part],
                    self.fc2.bias if index == 0 else None,
                )
                for index, part in enumerate(parts)
            ]
            total = outputs[0]
            for output in outputs[1:]:
                total = total + output
            return total

    class Block(nn.Module):
        """A depthwise 7x7 convolution, then a LayerNorm and MLP over the channels of each position, scaled and added
        to the input."""

        def __init__(self, width: int, slices: int) -> None:
            super().__init__()
            self.conv_dw = nn.Conv2d(width, width, 7, padding=3, groups=width)
            self.norm = nn.LayerNorm(width, eps=EPSILON)
            self.mlp = Mlp(width, slices)
            self.gamma = nn.Parameter(torch.ones(width))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.mlp(self.norm(self.conv_dw(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            return x + y * self.gamma.reshape(1, -1, 1, 1)

    class Stage(nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            width = WIDTHS[index]
            if index:
                self.downsample = nn.Sequential(
                    LayerNorm2d(WIDTHS[index - 1], eps=EPSILON), nn.Conv2d(WIDTHS[index - 1], width, 2, stride=2)
                )
            else:
                self.downsample = nn.Identity()
            self.blocks = nn.Sequential(*(Block(width, MLP_SLICES[index]) for _ in range(DEPTHS[index])))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.blocks(self.downsample(x))

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = LayerNorm2d(HIDDEN_SIZE, eps=EPSILON)

    class ConvNeXt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, WIDTHS[0], 4, stride=4), LayerNorm2d(WIDTHS[0], eps=EPSILON))
            self.stages = nn.Sequential(*(Stage(index) for index in range(len(DEPTHS))))
            self.head = Head()

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.stages(self.stem(pixel_values))
            class_embedding = self.head.norm(features.mean((2, 3), keepdim=True)).flatten(1)
            return class_embedding, self.head.norm(features).flatten(2).transpose(1, 2)

    return ConvNeXt()


def load_network(weights: Path) -> Any:
    import torch

    network = build_network()
    network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return network.eval()


def validate_export(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"The export requires opset {OPSET}")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError(f"Expected the input {INPUT} and the outputs {OUTPUTS}")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}: {shape}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("The export must be one self-contained ONNX file")


def fold_constants(path: Path) -> None:
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={INPUT: INTERFACE[INPUT]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded export")
    onnx.save(model, path)


def synthetic(seed: int) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal(INTERFACE[INPUT]).astype(np.float32)


def cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = (np.asarray(value, np.float64).ravel() for value in (expected, actual))
    return float(expected @ actual / (np.linalg.norm(expected) * np.linalg.norm(actual)))


def compare_outputs(name: str, expected: np.ndarray, actual: np.ndarray) -> float:
    """Return the output's cosine similarity with the reference, failing on a shape change, NaN or a low cosine."""
    if expected.shape != actual.shape or list(actual.shape) != INTERFACE[name]:
        raise ValueError(f"Unexpected shape for {name}: {actual.shape}")
    if not np.isfinite(actual).all():
        raise ValueError(f"Non-finite {name}")
    similarity = cosine(expected, actual)
    if similarity < MIN_COSINE:
        raise ValueError(f"{name} has cosine similarity {similarity:.6f} with the reference (least {MIN_COSINE})")
    return similarity


def export_network(weights: Path, reference: Path, output: Path) -> Path:
    """Export the network from weights.pth to output, then check it against reference.onnx in ONNX Runtime."""
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    network = load_network(weights)
    with torch.inference_mode():
        torch.onnx.export(
            network,
            (torch.from_numpy(synthetic(SEEDS.start)),),
            str(output),
            input_names=[INPUT],
            output_names=list(OUTPUTS),
            opset_version=OPSET,
            dynamo=False,
        )
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        for path in (reference, output)
    ]
    similarities = []
    for seed in SEEDS:
        expected, actual = (session.run(list(OUTPUTS), {INPUT: synthetic(seed)}) for session in sessions)
        similarities += [
            compare_outputs(name, value, candidate) for name, value, candidate in zip(OUTPUTS, expected, actual)
        ]
    print(
        f"Reference check passed on {len(SEEDS)} synthetic inputs, least cosine similarity {min(similarities):.6f}: "
        f"{output}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="weights.pth from the conversion")
    parser.add_argument("--reference", type=Path, required=True, help="reference.onnx from the conversion")
    parser.add_argument("--output", type=Path, required=True, help="ONNX file to write")
    args = parser.parse_args()
    export_network(*(path.expanduser().resolve() for path in (args.weights, args.reference, args.output)))


if __name__ == "__main__":
    main()
