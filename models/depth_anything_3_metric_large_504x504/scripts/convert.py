"""Export Depth Anything 3 Metric Large for one 504x504 view and check ONNX against PyTorch."""

import argparse
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

PRESET = "da3metric-large"
PACKAGE = "depth_anything_3"
WHEEL = "depth_anything_3-0.1.1-py3-none-any.whl"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", WHEEL])
HEIGHT, WIDTH = 504, 504
INTERFACE = {"image": [1, 3, HEIGHT, WIDTH], "depth": [1, 1, HEIGHT, WIDTH], "sky": [1, 1, HEIGHT, WIDTH]}
OUTPUTS = ("depth", "sky")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# Upstream replaces the depth of sky pixels (sky >= 0.3) after the head; the export stops before that.
SKY_THRESHOLD = 0.3
# The positional table and every traced shape must fold into constants.
UNFOLDED_OPS = {"Constant", "Shape", "Range", "ScatterND", "Sin", "Cos"}
SEEDS = range(123, 126)


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the file set {SOURCE}, model.yaml declares {declared}")


def validate_export(path):
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    if [item.version for item in model.opset_import if not item.domain] != [17]:
        raise ValueError("Depth Anything 3 export requires opset 17")
    values = [*model.graph.input, *model.graph.output]
    if [value.name for value in values] != list(INTERFACE):
        raise ValueError("Expected input image and outputs depth, sky in that order")
    for value in values:
        tensor = value.type.tensor_type
        shape = [d.dim_value for d in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or shape != INTERFACE[value.name]:
            raise ValueError(f"Unexpected dtype or shape for {value.name}")
    unfolded = sorted({node.op_type for node in model.graph.node} & UNFOLDED_OPS)
    if unfolded:
        raise ValueError(f"The folded export still computes {unfolded}")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("This conversion workflow requires one self-contained ONNX file")


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        if reference.shape != candidate.shape or list(reference.shape) != INTERFACE[name]:
            raise ValueError(f"Unexpected shape for {name}")
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"Non-finite output: {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def synthetic_inputs():
    """Normalized noise, and a smooth gradient image that looks more like a scene to the network."""
    noise = [np.random.RandomState(seed).uniform(0, 1, (HEIGHT, WIDTH, 3)).astype(np.float32) for seed in SEEDS]
    rows, columns = np.linspace(0, 1, HEIGHT, dtype=np.float32), np.linspace(0, 1, WIDTH, dtype=np.float32)
    y, x = np.meshgrid(rows, columns, indexing="ij")
    gradient = np.stack([x, y, (x + y) / 2], axis=-1)
    return [np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)[None]) for image in [*noise, gradient]]


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={"image": INTERFACE["image"]},
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("onnxsim could not validate the folded Depth Anything 3 export")
    onnx.save(model, path)


def unpack_package(wheel, directory):
    """Unpack only the pure-Python package tree; its declared dependencies are not needed to export."""
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
    network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True)
    return network


def head_outputs(network, image):
    """Upstream DepthAnything3Net.forward for one view, up to its sky fill."""
    features, _ = network.backbone(image[:, None], cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    head = network._process_depth_head(features, HEIGHT, WIDTH)
    return head.depth, head.sky


def check_sky_boundary(network, image, depth, sky):
    """The full upstream forward differs from the head outputs only in the depth of sky pixels."""
    upstream = network(image[:, None], ref_view_strategy="first")
    non_sky = sky < SKY_THRESHOLD
    np.testing.assert_array_equal(upstream.sky.numpy(), sky, err_msg="sky")
    np.testing.assert_array_equal(upstream.depth.numpy()[non_sky], depth[non_sky], err_msg="non-sky depth")


def freeze_for_export(network):
    """Write the fixed-shape graph explicitly; the arithmetic stays the same."""
    import torch

    vit = network.backbone.pretrained
    settings = (vit.alt_start, vit.rope_start, vit.num_register_tokens, vit.cat_token, network.head.pos_embed)
    if settings != (-1, -1, 0, False, False):
        raise ValueError(f"Unexpected {PRESET} configuration")
    for block in vit.blocks:
        # The explicit softmax form of the same attention that scaled_dot_product_attention computes.
        block.attn.fused_attn = False
    # The bicubically interpolated position table depends only on the fixed patch grid.
    patches = (HEIGHT // vit.patch_size) * (WIDTH // vit.patch_size)
    table = vit.interpolate_pos_encoding(torch.zeros(1, 1 + patches, vit.embed_dim), HEIGHT, WIDTH).detach().clone()
    vit.interpolate_pos_encoding = types.MethodType(lambda self, *args: table, vit)


def convert(source, output_dir):
    import onnxruntime as ort
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "model.onnx"
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="depth-anything-3-") as temporary:
        unpack_package(source.parent / WHEEL, Path(temporary))
        network = build_network(source.parent)
        samples = synthetic_inputs()
        references = []
        with torch.inference_mode():
            for sample in samples:
                image = torch.from_numpy(sample)
                depth, sky = (value.numpy() for value in head_outputs(network, image))
                check_sky_boundary(network, image, depth, sky)
                references.append((depth, sky))

            freeze_for_export(network)

            class Exported(torch.nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, image):
                    return head_outputs(self.network, image)

            torch.onnx.export(
                Exported(network).eval(),
                (torch.from_numpy(samples[0]),),
                str(output),
                input_names=["image"],
                output_names=list(OUTPUTS),
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
    del network
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for sample, expected in zip(samples, references, strict=True):
        compare_outputs(expected, session.run(list(OUTPUTS), {"image": sample}))
    print(f"Synthetic-input PyTorch/ONNX check passed on {len(samples)} inputs: {output}")
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
