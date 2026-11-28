"""Export PointNet++ SSG (S3DIS) from its safetensors checkpoint as one ONNX holding its eight MLA stages.

The graph has one input and one output per component (sa1-sa4, fp4-fp1), unconnected to each other: the CPU
steps between them (sampling, grouping, interpolation) run outside the graph, in pointcloud.py. compile.py
extracts each component from this ONNX. The export is checked against PyTorch on a synthetic block run through
the whole pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model
from pointcloud import (
    CLASSES,
    COMPONENTS,
    FEATURES,
    FP_CHANNELS,
    INPUT_SHAPES,
    NEIGHBORS,
    OUTPUT_SHAPES,
    POINTS,
    SA_CHANNELS,
    SA_POINTS,
    SA_RADII,
    input_name,
    output_name,
    segment,
    synthetic_block,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json"])
OPSET = 17
BATCH_NORM_EPS = 1e-5
# The checkpoint's hyperparameters this export implements.
HPARAMS = {
    "in_channels": FEATURES,
    "num_classes": CLASSES,
    "sa_channels": [list(channels) for _, channels in SA_CHANNELS],
    "fp_channels": [list(channels) for _, channels in FP_CHANNELS],
    "head_channels": [128],
    "ratios": [0.25] * 4,
    "radii": list(SA_RADII),
    "num_neighbors": [NEIGHBORS] * 4,
    "use_pos": True,
    "normalize_pos": False,
    "skip_input": False,
    "fp_k": 3,
}


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    hparams = config.get("hparams", {})
    differing = {key: hparams.get(key) for key, value in HPARAMS.items() if hparams.get(key) != value}
    if config.get("task") != "semantic-segmentation" or differing:
        raise ValueError(f"config.json is not the PointNet++ SSG S3DIS model this export implements: {differing}")
    if [count / points for count, points in zip(SA_POINTS, (POINTS, *SA_POINTS))] != HPARAMS["ratios"]:
        raise ValueError("the fixed point counts do not follow the checkpoint's sampling ratios")


def build_stages(weights):
    """The eight stages as PyTorch modules, each consuming every checkpoint tensor it names."""
    import torch
    from torch import nn

    used = set()

    def tensor(name):
        used.add(name)
        return torch.from_numpy(np.asarray(weights[name], np.float32))

    def mlp(prefix, channels, plain_last=False):
        """1x1 convolutions with batch norm and ReLU; with plain_last the last convolution stands alone."""
        layers = []
        for i, (cin, cout) in enumerate(zip(channels, channels[1:])):
            conv = nn.Conv2d(cin, cout, 1)
            conv.weight.data = tensor(f"{prefix}.lins.{i}.weight").reshape(cout, cin, 1, 1)
            conv.bias.data = tensor(f"{prefix}.lins.{i}.bias")
            layers.append(conv)
            if plain_last and i == len(channels) - 2:
                break
            norm = nn.BatchNorm2d(cout, eps=BATCH_NORM_EPS)
            for field in ("weight", "bias", "running_mean", "running_var"):
                getattr(norm, field).data = tensor(f"{prefix}.norms.{i}.module.{field}")
            layers += [norm, nn.ReLU()]
        return nn.Sequential(*layers)

    class SetAbstraction(nn.Module):
        """[1, C + 3, S, 32] grouped neighbours -> shared MLP -> max over the neighbours -> [1, C', S, 1]."""

        def __init__(self, level):
            super().__init__()
            cin, channels = SA_CHANNELS[level]
            self.mlp = mlp(f"encoder.sa_blocks.{level}.convs.0.local_nn", [cin + 3, *channels])

        def forward(self, grouped):
            return self.mlp(grouped).amax(dim=3, keepdim=True)

    class FeaturePropagation(nn.Module):
        """[1, C_interp + C_skip, N, 1] -> MLP; the finest level also runs the classifier head."""

        def __init__(self, block):
            super().__init__()
            cin, channels = FP_CHANNELS[block]
            self.mlp = mlp(f"decoder.fp_blocks.{block}.mlp", [cin, *channels])
            self.head = mlp("head", [channels[-1], 128, CLASSES], plain_last=True) if block == 3 else nn.Identity()

        def forward(self, stacked):
            return self.head(self.mlp(stacked))

    # decoder.fp_blocks.0 upsamples the coarsest level: fp4 in yanx27's names.
    stages = {f"sa{level + 1}": SetAbstraction(level) for level in range(4)}
    stages.update({f"fp{4 - block}": FeaturePropagation(block) for block in range(4)})
    unused = {name for name in weights if not name.endswith("num_batches_tracked")} - used
    if unused:
        raise ValueError(f"checkpoint tensors the export does not use: {sorted(unused)}")
    return {name: stages[name].eval() for name in COMPONENTS}


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [OPSET]:
        raise ValueError(f"PointNet++ export requires opset {OPSET}")
    expected_inputs = [input_name(name) for name in COMPONENTS]
    expected_outputs = [output_name(name) for name in COMPONENTS]
    names = ([v.name for v in model.graph.input], [v.name for v in model.graph.output])
    if names != (expected_inputs, expected_outputs):
        raise ValueError("Expected one input and one output per component, in component order")
    for component in COMPONENTS:
        for value, shape in ((model.graph.input, INPUT_SHAPES), (model.graph.output, OUTPUT_SHAPES)):
            (item,) = [v for v in value if v.name in {input_name(component), output_name(component)}]
            tensor = item.type.tensor_type
            dims = tuple(d.dim_value for d in tensor.shape.dim)
            if tensor.elem_type != onnx.TensorProto.FLOAT or dims != shape[component]:
                raise ValueError(f"Unexpected dtype or shape for {item.name}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare(expected, actual):
    if expected.shape != (POINTS, CLASSES) or actual.shape != expected.shape:
        raise ValueError(f"Unexpected logits shape: {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite logits")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, err_msg="logits")
    agreement = float((actual.argmax(1) == expected.argmax(1)).mean())
    if agreement < 0.999:
        raise ValueError(f"ONNX labels agree with PyTorch on only {agreement:.4f} of the points")
    return agreement


def convert(source, output_dir):
    import onnxruntime as ort
    import torch
    from safetensors.numpy import load_file

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    validate_config(source.parent)
    stages = build_stages(load_file(source))

    class Stages(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = torch.nn.ModuleList(stages[name] for name in COMPONENTS)

        def forward(self, *inputs):
            return tuple(stage(value) for stage, value in zip(self.stages, inputs))

    with torch.inference_mode():
        torch.onnx.export(
            Stages().eval(),
            tuple(torch.zeros(INPUT_SHAPES[name]) for name in COMPONENTS),
            str(output),
            input_names=[input_name(name) for name in COMPONENTS],
            output_names=[output_name(name) for name in COMPONENTS],
            opset_version=OPSET,
            dynamo=False,
        )
    validate_export(output)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    # The graph needs every input; each run fills the other components' inputs with zeros.
    zeros = {input_name(name): np.zeros(INPUT_SHAPES[name], np.float32) for name in COMPONENTS}

    def onnx_run(component, value):
        (result,) = session.run([output_name(component)], {**zeros, input_name(component): value})
        return result

    def torch_run(component, value):
        with torch.inference_mode():
            return stages[component](torch.from_numpy(value)).numpy()

    block = synthetic_block()
    agreement = compare(segment(block, torch_run), segment(block, onnx_run))
    print(f"PyTorch/ONNX pipeline check passed ({agreement:.4f} of labels agree): {output}")
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
