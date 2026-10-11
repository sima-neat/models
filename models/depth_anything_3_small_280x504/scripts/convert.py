"""Export Depth Anything 3 Small for two 280x504 views and check ONNX against PyTorch."""

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

PRESET = "da3-small"
PACKAGE = "depth_anything_3"
WHEEL = "depth_anything_3-0.1.1-py3-none-any.whl"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", WHEEL])
VIEWS, HEIGHT, WIDTH = 2, 280, 504
INPUTS = ("image0", "image1")
OUTPUTS = ("depth", "depth_conf", "pose_enc")
INTERFACE = {
    "image0": [1, 3, HEIGHT, WIDTH],
    "image1": [1, 3, HEIGHT, WIDTH],
    "depth": [1, VIEWS, HEIGHT, WIDTH],
    "depth_conf": [1, VIEWS, HEIGHT, WIDTH],
    "pose_enc": [1, VIEWS, 9],
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# The checkpoint has no LayerNorms for the auxiliary ray head's unused pyramid levels; upstream's
# non-strict load leaves their defaults, and the export does not use that head.
MISSING_KEYS = {f"head.scratch.output_conv2_aux.{level}.2.{name}" for level in (1, 2, 3) for name in ("weight", "bias")}
# The positional tables and every traced shape must fold into constants, and the camera token is
# written by concatenation rather than ScatterND.
UNFOLDED_OPS = {"Constant", "Shape", "Range", "ScatterND", "Sin", "Cos"}
SEEDS = range(123, 126)
# The gradient pair is two crops of one wider image, this many pixels apart: a short camera baseline.
PAIR_SHIFT = 32


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
        raise ValueError("Expected inputs image0, image1 and outputs depth, depth_conf, pose_enc in that order")
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


def normalize(image):
    return np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1)[None])


def synthetic_inputs():
    """Pairs of normalized noise views, and two shifted crops of a smooth gradient that looks more like a scene."""
    pairs = []
    for seed in SEEDS:
        views = np.random.RandomState(seed).uniform(0, 1, (VIEWS, HEIGHT, WIDTH, 3)).astype(np.float32)
        pairs.append([normalize(view) for view in views])
    rows = np.linspace(0, 1, HEIGHT, dtype=np.float32)
    columns = np.linspace(0, 1, WIDTH + PAIR_SHIFT, dtype=np.float32)
    y, x = np.meshgrid(rows, columns, indexing="ij")
    gradient = np.stack([np.sin(6 * x) ** 2, y, (x + y) / 2], axis=-1)
    pairs.append([normalize(gradient[:, offset : offset + WIDTH]) for offset in (0, PAIR_SHIFT)])
    return pairs


def fold_constants(path):
    from onnxsim import simplify

    model, valid = simplify(
        onnx.load(path),
        perform_optimization=False,
        overwrite_input_shapes={name: INTERFACE[name] for name in INPUTS},
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
    loaded = network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != MISSING_KEYS:
        raise ValueError(f"Unexpected checkpoint keys: {loaded}")
    return network


def head_outputs(network, views):
    """Upstream DepthAnything3Net.forward for two views, up to its pose decoding."""
    features, _ = network.backbone(views, cam_token=None, export_feat_layers=[], ref_view_strategy="first")
    head = network._process_depth_head(features, HEIGHT, WIDTH)
    return head.depth, head.depth_conf, network.cam_dec(features[-1][1])


def check_upstream(network, views, depth, confidence, pose):
    """The full upstream forward returns the same depth and confidence, and decodes pose_enc into its cameras."""
    import torch
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse

    upstream = network(views, ref_view_strategy="first")
    np.testing.assert_array_equal(upstream.depth.numpy(), depth, err_msg="depth")
    np.testing.assert_array_equal(upstream.depth_conf.numpy(), confidence, err_msg="depth_conf")
    camera_to_world, intrinsics = pose_encoding_to_extri_intri(torch.from_numpy(pose), (HEIGHT, WIDTH))
    np.testing.assert_array_equal(upstream.extrinsics.numpy(), affine_inverse(camera_to_world).numpy(), err_msg="pose")
    np.testing.assert_array_equal(upstream.intrinsics.numpy(), intrinsics.numpy(), err_msg="intrinsics")


def freeze_for_export(network):
    """Write the fixed-shape graph explicitly; the arithmetic stays the same."""
    import torch
    from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

    vit = network.backbone.pretrained
    settings = (vit.alt_start, vit.rope_start, vit.patch_start_idx, vit.num_register_tokens, vit.cat_token)
    if settings != (4, 4, 1, 0, True) or not network.head.pos_embed:
        raise ValueError(f"Unexpected {PRESET} configuration")
    # Upstream reorders the views around a selected reference only from this many views on.
    if VIEWS >= THRESH_FOR_REF_SELECTION:
        raise ValueError("Reference-view selection would reorder the views")
    for block in vit.blocks:
        # The explicit softmax form of the same attention that scaled_dot_product_attention computes.
        block.attn.fused_attn = False
    # The bicubically interpolated position table and the RoPE positions depend only on the fixed patch grid.
    patches = (HEIGHT // vit.patch_size) * (WIDTH // vit.patch_size)
    table = vit.interpolate_pos_encoding(torch.zeros(1, 1 + patches, vit.embed_dim), HEIGHT, WIDTH).detach().clone()
    vit.interpolate_pos_encoding = types.MethodType(lambda self, *args: table, vit)
    positions, global_positions = vit._prepare_rope(1, VIEWS, HEIGHT, WIDTH, torch.device("cpu"))

    def intermediate_layers(self, x, n=1, export_feat_layers=(), **kwargs):
        # Upstream _get_intermediate_layers_not_chunked without camera conditioning or auxiliary
        # features. The camera token replaces token 0 by concatenation instead of index assignment.
        if kwargs.get("cam_token") is not None or export_feat_layers:
            raise ValueError("The export supports no camera conditioning or auxiliary features")
        x = self.prepare_tokens_with_masks(x)
        total = len(self.blocks)
        taken = range(total - n, total) if isinstance(n, int) else n
        output, local_x = [], x
        for index, block in enumerate(self.blocks):
            rope = index >= self.rope_start and self.rope is not None
            if index == self.alt_start:
                camera = torch.cat([self.camera_token[:, :1], self.camera_token[:, 1:].expand(1, VIEWS - 1, -1)], dim=1)
                x = torch.cat([camera[:, :, None], x[:, :, 1:]], dim=2)
            if index >= self.alt_start and index % 2 == 1:
                x = self.process_attention(x, block, "global", pos=global_positions if rope else None, attn_mask=None)
            else:
                x = self.process_attention(x, block, "local", pos=positions if rope else None)
                local_x = x
            if index in taken:
                features = torch.cat([local_x, x], dim=-1)
                output.append((features[:, :, 0], features))
        return output, []

    vit._get_intermediate_layers_not_chunked = types.MethodType(intermediate_layers, vit)

    head = network.head
    add_pos_embed = head._add_pos_embed
    tables = {}

    def head_pos_embed(self, x, W, H, ratio=0.1):
        # The UV sin/cos table depends only on the feature-map shape. int() turns traced sizes into the
        # values the eager check run cached, so tracing finds the same tables.
        key = (tuple(int(size) for size in x.shape[1:]), W, H, ratio)
        if key not in tables:
            tables[key] = add_pos_embed(torch.zeros((1, *key[0]), dtype=x.dtype), W, H, ratio).detach().clone()
        return x + tables[key]

    head._add_pos_embed = types.MethodType(head_pos_embed, head)


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
                views = torch.from_numpy(np.stack(sample, axis=1))
                result = [value.numpy() for value in head_outputs(network, views)]
                check_upstream(network, views, *result)
                references.append(result)

            freeze_for_export(network)

            class Exported(torch.nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, image0, image1):
                    return head_outputs(self.network, torch.stack((image0, image1), dim=1))

            exported = Exported(network).eval()
            for sample, expected in zip(samples, references, strict=True):
                compare_outputs(expected, [value.numpy() for value in exported(*map(torch.from_numpy, sample))])
            torch.onnx.export(
                exported,
                tuple(map(torch.from_numpy, samples[0])),
                str(output),
                input_names=list(INPUTS),
                output_names=list(OUTPUTS),
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
    del network, exported
    fold_constants(output)
    validate_export(output)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    for sample, expected in zip(samples, references, strict=True):
        compare_outputs(expected, session.run(list(OUTPUTS), dict(zip(INPUTS, sample, strict=True))))
    print(f"Synthetic-input PyTorch/ONNX check passed on {len(samples)} view pairs: {output}")
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
