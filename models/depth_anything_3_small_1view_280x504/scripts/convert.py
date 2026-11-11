"""Export the unmodified DA3-Small forward pass to ONNX at the shipped input shape.

Only static constant folding runs here; surgery.py owns all MLA-specific rewrites.
The Cartesian-product symbolic fills an unsupported PyTorch-to-ONNX export operator.
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata
from preprocess import HEIGHT, WIDTH

MODEL_DIR = Path(__file__).resolve().parents[1]
PRESET = "da3-small"
PACKAGE = "depth_anything_3"
WHEEL = "depth_anything_3-0.1.1-py3-none-any.whl"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", WHEEL])
MISSING_KEYS = {f"head.scratch.output_conv2_aux.{level}.2.{name}" for level in (1, 2, 3) for name in ("weight", "bias")}
INTERFACE = {"image": [1, 3, HEIGHT, WIDTH], "depth": [1, 1, HEIGHT, WIDTH], "depth_conf": [1, 1, HEIGHT, WIDTH]}


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the file set {SOURCE}, model.yaml declares {declared}")


def unpack_package(wheel, directory):
    """Unpack the pinned package; requirements-convert.txt provides the dependencies used by its forward pass."""
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.startswith(f"{PACKAGE}/")]
        if not members or any(".." in Path(name).parts for name in members):
            raise ValueError(f"Unexpected contents in {wheel.name}")
        archive.extractall(directory, members)
    sys.path.insert(0, str(directory))


def build_network(directory):
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    declared = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = load_config(MODEL_REGISTRY[PRESET])
    # The Hugging Face mixin builds the network from the wheel preset that config.json names.
    if declared.get("model_name") != PRESET or OmegaConf.to_container(config) != declared.get("config"):
        raise ValueError(f"config.json does not describe the wheel's {PRESET} preset")
    network = create_object(config).eval()
    state = load_file(directory / "model.safetensors")
    if not all(key.startswith("model.") for key in state):
        raise ValueError("Expected the mixin's model. prefix on every checkpoint tensor")
    loaded = network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != MISSING_KEYS:
        raise ValueError(f"Unexpected checkpoint keys: {loaded}")
    return network


def sample_inputs():
    """Three normalized noise images and one smooth RGB gradient."""
    from preprocess import MEAN, STD

    samples = [
        np.random.RandomState(seed).uniform(0, 1, (HEIGHT, WIDTH, 3)).astype(np.float32) for seed in range(123, 126)
    ]
    y, x = np.meshgrid(
        np.linspace(0, 1, HEIGHT, dtype=np.float32), np.linspace(0, 1, WIDTH, dtype=np.float32), indexing="ij"
    )
    samples.append(np.stack([x, y, (x + y) / 2], axis=-1))
    return [np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)[None]) for image in samples]


def cartesian_product(graph, values):
    """Lower a pair of 1-D tensors in torch.cartesian_prod to ONNX broadcasting, in the same row order."""
    import torch
    from torch.onnx import symbolic_helper

    rows, columns = symbolic_helper._unpack_list(values)

    def constant(values):
        return graph.op("Constant", value_t=torch.tensor(values, dtype=torch.int64))

    shape = graph.op("Concat", graph.op("Shape", rows), graph.op("Shape", columns), axis_i=0)
    left = graph.op("Expand", graph.op("Unsqueeze", rows, constant([1])), shape)
    right = graph.op("Expand", graph.op("Unsqueeze", columns, constant([0])), shape)
    pairs = graph.op(
        "Concat", graph.op("Unsqueeze", left, constant([2])), graph.op("Unsqueeze", right, constant([2])), axis_i=2
    )
    return graph.op("Reshape", pairs, constant([-1, 2]))


def validate_export(path):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    if [version.version for version in model.opset_import if not version.domain] != [17]:
        raise ValueError("expected ONNX opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("expected image, depth and depth_conf")
    for value in values:
        tensor = value.type.tensor_type
        shape = [dimension.dim_value for dimension in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"unexpected input/output: {value.name}")


def convert(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="da3-upstream-") as temporary:
        unpack_package(source.parent / WHEEL, Path(temporary))
        network = build_network(source.parent)

        class Upstream(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image):
                result = self.model(image[:, None], ref_view_strategy="first")
                return result.depth, result.depth_conf

        upstream = Upstream(network).eval()
        samples = sample_inputs()
        with torch.inference_mode():
            references = [[value.numpy() for value in upstream(torch.from_numpy(sample))] for sample in samples]
            torch.onnx.register_custom_op_symbolic("aten::cartesian_prod", cartesian_product, 17)
            try:
                torch.onnx.export(
                    upstream,
                    (torch.from_numpy(samples[0]),),
                    str(output),
                    opset_version=17,
                    input_names=["image"],
                    output_names=["depth", "depth_conf"],
                    dynamo=False,
                )
            finally:
                torch.onnx.unregister_custom_op_symbolic("aten::cartesian_prod", 17)
    model, valid = simplify(onnx.load(output), perform_optimization=False)
    if not valid:
        raise RuntimeError("ONNX constant folding failed")
    onnx.save(model, output)
    validate_export(output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    for sample, expected in zip(samples, references, strict=True):
        for reference, actual in zip(expected, session.run(None, {"image": sample}), strict=True):
            if not np.isfinite(actual).all():
                raise ValueError("non-finite ONNX output")
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
    print(f"Exported the original {PRESET} forward pass: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
