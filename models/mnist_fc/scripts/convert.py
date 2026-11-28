"""Export dacorvo/mnist-mlp, a three-layer MLP for MNIST, as a 4D ONNX for the Modalix MLA.

The upstream model normalizes the image with MNIST's mean 0.1307 and std 0.3081, flattens its 28x28 pixels, and
runs Linear(784, 256), ReLU, Linear(256, 256), ReLU, Linear(256, 10) and a softmax. The export keeps every tensor
4D, so nothing flattens:

- the first Linear is a 28x28 convolution over the [1, 1, 28, 28] image, which on that input is the same dot
  product over the row-major flattened pixels. The normalization folds into it: weights / std, and
  bias - mean / std * the sum of each output's weights. The input is the image scaled to [0, 1], as for mnist_cnn;
- the other two Linears are 1x1 convolutions;
- the output is the 10 logits as [1, 10, 1, 1]. The softmax does not change which digit scores highest, so it is
  left to the application.

The export is checked against the upstream layers, evaluated in NumPy on the safetensors weights, on seeded images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from onnx import TensorProto, helper, numpy_helper

MODEL_DIR = Path(__file__).resolve().parents[1]
FILES = ("config.json", "model.safetensors")
MEAN, STD = np.float32(0.1307), np.float32(0.3081)
IMAGE_SIZE, INPUT_SIZE, HIDDEN_SIZE, CLASSES = 28, 784, 256, 10
INPUT_NAME, OUTPUT_NAME = "image", "logits"
OPSET = 17
LAYERS = ("input_layer", "mid_layer", "output_layer")
# The 4D export and the upstream layers differ only by float32 rounding.
MAX_LOGIT_ERROR = 1e-4


def validate_source(source: dict[str, Any]) -> None:
    """The conversion takes exactly the pinned MLP config and weights."""
    files = {item["path"] for item in source.get("files", [])}
    if source.get("type") != "file_set" or source.get("entrypoint") != "config.json" or files != set(FILES):
        raise ValueError(f"the mnist-mlp conversion source must pin exactly {', '.join(FILES)}, opened at config.json")


def check_config(config: dict[str, Any]) -> None:
    expected = {"input_size": INPUT_SIZE, "hidden_size": HIDDEN_SIZE, "output_size": CLASSES}
    found = {key: config.get(key) for key in expected}
    if found != expected or config.get("architectures") != ["MLP"]:
        raise ValueError(f"unexpected mnist-mlp config: {found}, architectures {config.get('architectures')}")


def check_weights(weights: dict[str, np.ndarray]) -> None:
    shapes = {
        "input_layer.weight": (HIDDEN_SIZE, INPUT_SIZE), "input_layer.bias": (HIDDEN_SIZE,),
        "mid_layer.weight": (HIDDEN_SIZE, HIDDEN_SIZE), "mid_layer.bias": (HIDDEN_SIZE,),
        "output_layer.weight": (CLASSES, HIDDEN_SIZE), "output_layer.bias": (CLASSES,),
    }
    found = {name: tuple(value.shape) for name, value in weights.items()}
    if found != shapes:
        raise ValueError(f"unexpected mnist-mlp weights: {found}")


def reference_logits(weights: dict[str, np.ndarray], images: np.ndarray) -> np.ndarray:
    """The upstream MLP before its softmax, on [N, 28, 28] images in [0, 1]."""
    hidden = ((images - MEAN) / STD).reshape(len(images), INPUT_SIZE)
    for layer in LAYERS[:-1]:
        hidden = np.maximum(hidden @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"], 0)
    return hidden @ weights["output_layer.weight"].T + weights["output_layer.bias"]


def convolutions(weights: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each layer's convolution weight [out, in, kH, kW] and bias, the normalization folded into the first."""
    first, first_bias = weights["input_layer.weight"], weights["input_layer.bias"]
    layers = [(
        (first / STD).reshape(HIDDEN_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE),
        first_bias - MEAN / STD * first.sum(axis=1),
    )]
    for layer in LAYERS[1:]:
        weight = weights[f"{layer}.weight"]
        layers.append((weight[:, :, None, None], weights[f"{layer}.bias"]))
    return [(weight.astype(np.float32), bias.astype(np.float32)) for weight, bias in layers]


def build_graph(weights: dict[str, np.ndarray]) -> onnx.ModelProto:
    """The MLP as three convolutions, with ReLUs between them."""
    nodes, initializers, value = [], [], INPUT_NAME
    for index, (weight, bias) in enumerate(convolutions(weights)):
        name = LAYERS[index]
        initializers += [numpy_helper.from_array(weight, f"{name}.weight"),
                         numpy_helper.from_array(bias, f"{name}.bias")]
        output = OUTPUT_NAME if index == len(LAYERS) - 1 else f"{name}.output"
        nodes.append(helper.make_node("Conv", [value, f"{name}.weight", f"{name}.bias"], [output], name=f"{name}.conv",
                                      kernel_shape=list(weight.shape[2:])))
        if index < len(LAYERS) - 1:
            nodes.append(helper.make_node("Relu", [output], [f"{name}.relu"], name=f"{name}.relu"))
            output = f"{name}.relu"
        value = output
    graph = helper.make_graph(
        nodes, "mnist_fc",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 1, IMAGE_SIZE, IMAGE_SIZE])],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, CLASSES, 1, 1])],
        initializers,
    )
    opsets = [helper.make_opsetid("", OPSET)]
    model = helper.make_model(graph, opset_imports=opsets, producer_name="mnist_fc convert.py")
    onnx.checker.check_model(model, full_check=True)
    return model


def validate_export(path: Path) -> None:
    """The exported ONNX has the declared one-input, one-output 4D interface."""
    graph = onnx.load(str(path)).graph
    initializers = {item.name for item in graph.initializer}

    def shapes(values: Any) -> dict[str, list[int]]:
        return {item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
                for item in values if item.name not in initializers}

    expected = ({INPUT_NAME: [1, 1, IMAGE_SIZE, IMAGE_SIZE]}, {OUTPUT_NAME: [1, CLASSES, 1, 1]})
    if (shapes(graph.input), shapes(graph.output)) != expected:
        raise ValueError(f"unexpected mnist_fc interface: {shapes(graph.input)} -> {shapes(graph.output)}")


def convert(main_file: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    from safetensors.numpy import load_file

    source = main_file.parent
    check_config(json.loads((source / "config.json").read_text(encoding="utf-8")))
    weights = load_file(str(source / "model.safetensors"))
    check_weights(weights)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    onnx.save(build_graph(weights), str(output))
    validate_export(output)

    images = np.random.default_rng(0).random((32, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    exported = np.stack([session.run(None, {INPUT_NAME: image[None, None]})[0].reshape(CLASSES) for image in images])
    error = float(np.abs(exported - reference_logits(weights, images)).max())
    print(f"max |ONNX - upstream| logit error over {len(images)} images: {error:.2e}")
    if error > MAX_LOGIT_ERROR:
        raise ValueError(f"the export differs from the upstream MLP by {error:.2e}")
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
