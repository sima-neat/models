"""Specialize one DA3-Small view in PyTorch, export ONNX, and check both outputs against upstream.

QKV is split before reshaping, fixed position tables are cached, the camera token
is concatenated, and the final learned depth/confidence projections are separate.
All changes retain the checkpoint's computation; no LayerNorm affine terms are folded.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from preprocess import HEIGHT, WIDTH

PRESET = "da3-small"
VIEWS = 1
INPUT_SHAPES = {"image": [1, 3, HEIGHT, WIDTH]}
OUTPUT_SHAPES = {"depth": [1, 1, HEIGHT, WIDTH], "depth_conf": [1, 1, HEIGHT, WIDTH]}
INTERFACE = {**INPUT_SHAPES, **OUTPUT_SHAPES}
INPUTS = tuple(INPUT_SHAPES)
OUTPUTS = tuple(OUTPUT_SHAPES)


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported Depth Anything 3 graph: {message}")


def _shape(value) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _index(model):
    shapes = {value.name: _shape(value) for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    shapes.update({name: list(value.shape) for name, value in constants.items()})
    from collections import defaultdict

    producers, users = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            if name:
                users[name].append(node)
    # A graph output is used outside the graph and must keep its producer.
    for value in model.graph.output:
        users[value.name].append(None)
    return shapes, constants, producers, users


def _validate_interface(model):
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    _require(inputs == INPUT_SHAPES and outputs == OUTPUT_SHAPES, f"interface {inputs} -> {outputs}")


def _prune(model):
    """Drop nodes and initializers that no longer reach a graph output."""
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [value for value in model.graph.initializer if value.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def _finish(model):
    _prune(model)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def _bypass(model, node, producers, users, outputs) -> bool:
    """Remove a node whose output equals its first input; a graph output keeps its name."""
    source, result = node.input[0], node.output[0]
    if result in outputs:
        producer = producers.get(source)
        if producer is None or len(users[source]) != 1:
            return False
        producer.output[list(producer.output).index(source)] = result
    else:
        for user in users[result]:
            for index, name in enumerate(user.input):
                if name == result:
                    user.input[index] = source
    model.graph.node.remove(node)
    return True


def _collapse_data_movement(model):
    """Collapse reshape and transpose chains, then remove identity Reshapes, Squeezes and Casts."""
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph
    values = (*graph.input, *graph.value_info, *graph.output)
    # No rewrite here changes what a value name holds, so one shape inference serves them all.
    shapes = {value.name: _shape(value) for value in values}
    floats = {value.name for value in values if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT}
    producers = {}
    for node in graph.node:
        producer = producers.get(node.input[0]) if node.input else None
        if (
            node.op_type == "Reshape"
            and producer is not None
            and producer.op_type in ("Reshape", "Squeeze", "Unsqueeze")
        ):
            # A reshape reads only the row-major element order, which the previous reshape kept.
            node.input[0] = producer.input[0]
        elif node.op_type == "Transpose" and producer is not None and producer.op_type == "Transpose":
            first, second = _attributes(producer)["perm"], _attributes(node)["perm"]
            node.input[0] = producer.input[0]
            permutation = next(item for item in node.attribute if item.name == "perm")
            del permutation.ints[:]
            permutation.ints.extend(first[axis] for axis in second)
        for name in node.output:
            producers[name] = node
    _prune(model)
    outputs = {value.name for value in graph.output}
    while True:
        _, _, producers, users = _index(model)
        for node in graph.node:
            identity = (
                node.op_type == "Cast" and _attributes(node)["to"] == onnx.TensorProto.FLOAT and node.input[0] in floats
            )
            identity |= (
                node.op_type in ("Reshape", "Squeeze", "Unsqueeze")
                and shapes.get(node.input[0]) is not None
                and shapes.get(node.input[0]) == shapes.get(node.output[0])
            )
            if identity and _bypass(model, node, producers, users, outputs):
                break
        else:
            return _finish(model)


def freeze_for_export(network, reference_threshold):
    """Write the fixed-shape graph explicitly; the arithmetic stays the same."""
    import torch

    vit = network.backbone.pretrained
    settings = (vit.alt_start, vit.rope_start, vit.patch_start_idx, vit.num_register_tokens, vit.cat_token)
    if settings != (4, 4, 1, 0, True) or not network.head.pos_embed:
        raise ValueError(f"Unexpected {PRESET} configuration")
    # Upstream reorders the views around a selected reference only from this many views on.
    if VIEWS >= reference_threshold:
        raise ValueError("Reference-view selection would reorder the views")
    for block in vit.blocks:
        # The explicit softmax form of the same attention that scaled_dot_product_attention computes.
        block.attn.forward = types.MethodType(attention_forward, block.attn)
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
                # A single view uses only the reference token. Avoid an empty Expand/Concat,
                # whose ONNX constant folding can incorrectly retain the second camera token.
                camera = self.camera_token[:, :1]
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


def attention_forward(self, x, pos=None, attn_mask=None):
    """Upstream's evaluation attention with fixed Q/K/V splits instead of a 5-D transpose and Gather."""
    _require(attn_mask is None and not self.training, "unmasked evaluation attention")
    batch, tokens, channels = x.shape
    q, k, v = self.qkv(x).split(int(channels), dim=-1)
    shape = (batch, tokens, self.num_heads, channels // self.num_heads)
    q, k, v = (value.reshape(shape).transpose(1, 2) for value in (q, k, v))
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q, k = self.rope(q, pos), self.rope(k, pos)
    weights = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    result = (self.attn_drop(weights) @ v).transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(result))


def split_head_projections(head):
    """Copy the two learned weight rows into independent 32→1 projections, before exp/exp+1."""
    import torch

    projection = head.scratch.output_conv2[-1]
    _require(
        isinstance(projection, torch.nn.Conv2d)
        and tuple(projection.weight.shape) == (2, 32, 1, 1)
        and projection.bias is not None
        and projection.groups == 1,
        "the learned 32→2 final projection",
    )
    _require(
        head.activation == "exp" and head.conf_activation == "expp1" and head.down_ratio == 1,
        "full-resolution exp depth and exp+1 confidence",
    )
    for index, name in enumerate(("depth_projection", "confidence_projection")):
        layer = torch.nn.Conv2d(32, 1, kernel_size=1)
        with torch.no_grad():
            layer.weight.copy_(projection.weight[index : index + 1])
            layer.bias.copy_(projection.bias[index : index + 1])
        setattr(head, name, layer.eval())
    head.scratch.output_conv2[-1] = torch.nn.Identity()
    head._forward_impl = types.MethodType(head_forward, head)


def head_forward(self, feats, height, width, patch_start_idx):
    """The upstream main depth branch, with separate NCHW outputs and the unused ray branch omitted."""
    import torch

    batch, _, channels = feats[0].shape
    rows, columns = HEIGHT // self.patch_size, WIDTH // self.patch_size
    resized = []
    for stage, take in enumerate(self.intermediate_layer_idx):
        value = self.norm(feats[take][:, patch_start_idx:])
        value = value.permute(0, 2, 1).reshape(batch, channels, rows, columns)
        value = self.projects[stage](value)
        value = self._add_pos_embed(value, WIDTH, HEIGHT)
        resized.append(self.resize_layers[stage](value))
    fused, _ = self._fuse(resized)
    fused = torch.nn.functional.interpolate(fused, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=True)
    fused = self._add_pos_embed(fused, WIDTH, HEIGHT)
    features = self.scratch.output_conv2(fused)
    return {
        self.head_main: torch.exp(self.depth_projection(features)).squeeze(1),
        f"{self.head_main}_conf": (torch.exp(self.confidence_projection(features)) + 1).squeeze(1),
    }


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


def compare_outputs(expected, actual):
    for name, reference, candidate in zip(OUTPUTS, expected, actual, strict=True):
        _require(list(reference.shape) == list(candidate.shape) == OUTPUT_SHAPES[name], f"shape of {name}")
        _require(np.isfinite(reference).all() and np.isfinite(candidate).all(), f"finite {name}")
        np.testing.assert_allclose(candidate, reference, rtol=1e-3, atol=1e-3, err_msg=name)


def validate_export(path):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    _validate_interface(model)
    _require([op.version for op in model.opset_import if not op.domain] == [17], "ONNX opset 17")
    values = [*model.graph.input, *model.graph.output]
    _require(all(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for value in values), "float32 interface")
    forbidden = {"Constant", "Shape", "Range", "ScatterND", "Einsum", "Sin", "Cos", "Gather"}
    _require(not ({node.op_type for node in model.graph.node} & forbidden), "static graph without dynamic selection")
    shapes = [_shape(value) for value in (*model.graph.value_info, *model.graph.output)]
    _require(
        all(0 < len(shape) <= 4 and shape[0] == 1 and all(d > 0 for d in shape) for shape in shapes),
        "static batch-one tensors of at most four dimensions",
    )


def prepare_model(source: Path, output_dir: Path) -> Path:
    import onnxruntime as ort
    import torch
    from onnxsim import simplify
    from torch.package import PackageImporter

    torch.set_num_threads(4)
    package = PackageImporter(source)
    _require(package.load_text("da3", "preset.txt") == PRESET, "the packaged da3-small preset")
    network = package.load_pickle("da3", "model.pkl", map_location="cpu").float().eval()
    samples = sample_inputs()
    references = []
    with torch.inference_mode():
        for sample in samples:
            original = network(torch.from_numpy(sample)[:, None], ref_view_strategy="first")
            references.append([original.depth.numpy(), original.depth_conf.numpy()])
        threshold = package.import_module("depth_anything_3.utils.constants").THRESH_FOR_REF_SELECTION
        freeze_for_export(network, threshold)
        split_head_projections(network.head)

        class Exported(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image):
                features, _ = self.model.backbone(
                    image[:, None], cam_token=None, export_feat_layers=[], ref_view_strategy="first"
                )
                output = self.model._process_depth_head(features, HEIGHT, WIDTH)
                return output.depth, output.depth_conf

        exported = Exported(network).eval()
        for sample, expected in zip(samples, references, strict=True):
            compare_outputs(expected, [value.numpy() for value in exported(torch.from_numpy(sample))])
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "depth_anything_3_small.onnx"
        torch.onnx.export(
            exported,
            (torch.from_numpy(samples[0]),),
            str(path),
            opset_version=17,
            input_names=list(INPUTS),
            output_names=list(OUTPUTS),
            dynamo=False,
        )
    model, valid = simplify(onnx.load(path), perform_optimization=False)
    _require(valid, "ONNX constant folding")
    model = _collapse_data_movement(model)
    onnx.save(model, path)
    validate_export(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for sample, expected in zip(samples, references, strict=True):
        compare_outputs(expected, session.run(list(OUTPUTS), {"image": sample}))
    print(f"Original PyTorch / optimized PyTorch / ONNX check passed on {len(samples)} samples: {path}")
    return path
