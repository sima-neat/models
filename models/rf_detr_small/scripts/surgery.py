"""Split RF-DETR around TopK and optimize each partition's attention layout."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT = "input"
FEATURES = "backbone_features"
SCORES = "proposal_scores"
BOXES = "proposal_boxes"
PROPOSALS = "proposals"
OUTPUTS = ["dets", "labels"]

BOUNDARY_SHAPES = {
    FEATURES: [1, 256, 32, 32],
    SCORES: [1, 1024],
    BOXES: [1, 1024, 4],
    PROPOSALS: [1, 300, 4],
}


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def _validate_source(model: onnx.ModelProto) -> None:
    inputs = {value.name: _shape(value) for value in model.graph.input}
    outputs = {value.name: _shape(value) for value in model.graph.output}
    if inputs != {INPUT: [1, 3, 512, 512]}:
        raise ValueError(f"unexpected RF-DETR input interface: {inputs}")
    expected_outputs = {"dets": [1, 300, 4], "labels": [1, 300, 91]}
    if outputs != expected_outputs:
        raise ValueError(f"unexpected RF-DETR output interface: {outputs}")


def _rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for values in (node.input, node.output):
            for index, value in enumerate(values):
                if value == old:
                    values[index] = new
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == old:
            value.name = new


def _replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    _require(old not in {value.name for value in model.graph.output}, f"cannot bypass public output {old}")
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new


def _set_value_info(model: onnx.ModelProto, name: str, shape: list[int]) -> None:
    values = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value in values:
        if value.name == name:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            del value.type.tensor_type.shape.dim[:]
            for size in shape:
                value.type.tensor_type.shape.dim.add(dim_value=size)
            return
    model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported RF-DETR graph: {message}")


def _attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _index(model):
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    shapes = {value.name: _shape(value) for value in values}
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    producers, users = {}, defaultdict(list)
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
        for name in node.input:
            users[name].append(node)
        if node.op_type == "Constant" and "value" in _attributes(node):
            constants[node.output[0]] = numpy_helper.to_array(_attributes(node)["value"])
    shapes.update({name: list(value.shape) for name, value in constants.items()})
    # Public intermediate outputs must never disappear or change interpretation.
    for value in model.graph.output:
        users[value.name].append(None)
    return shapes, constants, producers, users


def _producer(producers, value, kind):
    node = producers.get(value)
    _require(node is not None and node.op_type == kind, f"expected {kind} producing {value}")
    return node


def _next(users, node, kind):
    following = users[node.output[0]]
    _require(
        len(following) == 1 and following[0] is not None and following[0].op_type == kind,
        f"expected private {node.name} -> {kind}",
    )
    return following[0]


def _scalar(constants, name):
    value = constants.get(name)
    return None if value is None or value.size != 1 else float(value.reshape(()))


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


def _require_folded(model: onnx.ModelProto) -> None:
    # convert.py folds the export with onnxsim; matching relies on static shapes and constants.
    unfolded = sorted({node.op_type for node in model.graph.node if node.op_type in ("Constant", "Shape")})
    _require(not unfolded, f"source still has {unfolded} nodes; fold it with onnxsim first")


def _remove_noop_masks(model: onnx.ModelProto, expected: dict[str, int]) -> onnx.ModelProto:
    """Where(all-False constant, X, Y) is Y, and a ScatterND that overwrites every row is its updates."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    counts = defaultdict(int)
    for node in list(model.graph.node):
        if node.op_type == "Where":
            condition, _, otherwise = node.input
            _require(condition in constants, f"non-constant mask at {node.name}")
            _require(not constants[condition].any(), f"mask at {node.name} selects elements")
            _require(shapes.get(otherwise) == shapes.get(node.output[0]), f"broadcasting Where at {node.name}")
            _replace_uses(model, node.output[0], otherwise)
            counts[node.op_type] += 1
        elif node.op_type == "ScatterND":
            data, indices, updates = node.input
            _require(data in constants and indices in constants, f"non-constant ScatterND {node.name}")
            index = constants[indices]
            _require(shapes.get(updates) == shapes.get(data), f"partial ScatterND {node.name}")
            grid = np.stack(np.meshgrid(*[np.arange(size) for size in index.shape[:-1]], indexing="ij"), -1)
            _require(
                index.shape[:-1] == tuple(shapes[data][: index.shape[-1]]) and np.array_equal(index, grid),
                f"ScatterND {node.name} does not overwrite every position in order",
            )
            _replace_uses(model, node.output[0], updates)
            counts[node.op_type] += 1
    _require(dict(counts) == expected, f"expected no-op masks {expected}, found {dict(counts)}")
    return _finish(model)


def _remove_identities(model: onnx.ModelProto, expected: dict[str, int]) -> onnx.ModelProto:
    """Bypass same-type Casts, same-shape Expands, and Gathers that read one slice of a singleton stack.

    The decoder stacks every layer's final LayerNorm for auxiliary losses and keeps only the last one,
    so bypassing that Gather also removes two LayerNorms whose results are never used.
    """
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    types = {value.name: value.type.tensor_type.elem_type for value in (*model.graph.input, *model.graph.value_info)}
    counts = defaultdict(int)
    for node in list(model.graph.node):
        source = None
        if node.op_type == "Cast" and types.get(node.input[0]) == _attributes(node)["to"]:
            source = node.input[0]
        elif node.op_type == "Expand" and shapes.get(node.input[0]) == shapes.get(node.output[0]):
            source = node.input[0]
        elif node.op_type == "Gather" and node.input[1] in constants and constants[node.input[1]].ndim == 0:
            rank = len(shapes[node.input[0]])
            axis = _attributes(node).get("axis", 0) % rank
            index = int(constants[node.input[1]])

            def singleton(name):
                unsqueeze = producers.get(name)
                if unsqueeze is None or unsqueeze.op_type != "Unsqueeze" or unsqueeze.input[1] not in constants:
                    return None
                axes = [value % rank for value in constants[unsqueeze.input[1]].reshape(-1).tolist()]
                return unsqueeze.input[0] if axes == [axis] else None

            stacked = producers.get(node.input[0])
            if stacked is not None and stacked.op_type == "Unsqueeze" and index in (0, -1):
                source = singleton(node.input[0])
            elif stacked is not None and stacked.op_type == "Concat" and _attributes(stacked)["axis"] % rank == axis:
                pieces = [singleton(name) for name in stacked.input]
                if all(pieces):
                    source = pieces[index]
        if source is not None:
            _require(shapes.get(source) == shapes.get(node.output[0]), f"identity changes shape at {node.name}")
            _replace_uses(model, node.output[0], source)
            counts[node.op_type] += 1
    _require(dict(counts) == expected, f"expected identities {expected}, found {dict(counts)}")
    return _finish(model)


def _cut_points(model: onnx.ModelProto) -> dict[str, str]:
    """Find the partition boundaries by structure; the export already decodes all boxes before TopK."""
    shapes, constants, producers, users = _index(onnx.shape_inference.infer_shapes(model))
    topks = [node for node in model.graph.node if node.op_type == "TopK"]
    gathers = [node for node in model.graph.node if node.op_type == "GatherElements"]
    _require(len(topks) == 1 and len(gathers) == 1, "one TopK and one GatherElements")
    topk, gather = topks[0], gathers[0]
    scores = _producer(producers, topk.input[0], "ReduceMax")
    _require(shapes[scores.output[0]] == BOUNDARY_SHAPES[SCORES], "TopK scores shape")
    _producer(producers, gather.input[1], "Expand")
    _require(
        shapes[gather.input[0]] == BOUNDARY_SHAPES[BOXES] and shapes[gather.output[0]] == BOUNDARY_SHAPES[PROPOSALS],
        "gathered proposal boxes",
    )
    features = [
        name
        for node in model.graph.node
        if "/backbone/" in node.name
        for name in node.output
        if shapes.get(name) == BOUNDARY_SHAPES[FEATURES]
        and any(user is not None and "/backbone/" not in user.name for user in users[name])
    ]
    _require(len(features) == 1, f"one backbone feature map crosses the cut, found {features}")
    return {features[0]: FEATURES, scores.output[0]: SCORES, gather.input[0]: BOXES, gather.output[0]: PROPOSALS}


def _optimize_window_attention(model: onnx.ModelProto) -> onnx.ModelProto:
    """Commute identical window/head permutations through unmasked attention."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    rewritten = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or "/backbone/" not in softmax.name:
            continue
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), "last-axis softmax required")
        scale = _producer(producers, softmax.input[0], "Mul")
        _require(_next(users, scale, "Softmax") is softmax, "private score scale")
        scores, factor = scale.input if _scalar(constants, scale.input[1]) is not None else scale.input[::-1]
        _require(_scalar(constants, factor) is not None, f"scalar score scale at {scale.name}")
        qk = _producer(producers, scores, "MatMul")
        _require(_next(users, qk, "Mul") is scale, "private attention scores")
        qt, kt = (producers.get(name) for name in qk.input)
        windows, heads, tokens, dim = shapes[qk.input[0]]
        if windows == 1:
            continue  # Global attention already has no window/head interchange.
        pv = _next(users, softmax, "MatMul")
        _require(pv.input[0] == softmax.output[0], "probabilities must be the left PV operand")
        vt = producers.get(pv.input[1])
        output = _next(users, pv, "Transpose")
        for node, consumer in ((qt, qk), (kt, qk), (vt, pv)):
            _require(node is not None and node.op_type == "Transpose", "Q/K/V come from Transposes")
            _require(_next(users, node, consumer.op_type) is consumer, f"shared {node.name}")
            _require(shapes.get(node.input[0]) == [windows, tokens, heads, dim], f"producer shape at {node.name}")
        _require(shapes.get(kt.output[0]) == [windows, heads, dim, tokens], "K shape")
        changes = (
            (qt, [0, 2, 1, 3], [2, 0, 1, 3]),
            (kt, [0, 2, 3, 1], [2, 0, 3, 1]),
            (vt, [0, 2, 1, 3], [2, 0, 1, 3]),
            (output, [0, 2, 1, 3], [1, 2, 0, 3]),
        )
        for node, old, new in changes:
            _require(_attributes(node).get("perm") == old, f"unexpected permutation at {node.name}")
            attribute = next(a for a in node.attribute if a.name == "perm")
            del attribute.ints[:]
            attribute.ints.extend(new)
        rewritten += 1
    _require(rewritten == 9, f"expected 9 window blocks, found {rewritten}")
    return _finish(model)


def _fold_backbone_score_scale(model: onnx.ModelProto, expected: int) -> onnx.ModelProto:
    """Scale the Q projection instead of every [windows, heads, tokens, tokens] attention score."""
    _, constants, producers, users = _index(model)
    initializers = {value.name: value for value in model.graph.initializer}
    folded = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or "/backbone/" not in softmax.name:
            continue
        scale = _producer(producers, softmax.input[0], "Mul")
        _require(_next(users, scale, "Softmax") is softmax, "private score scale")
        scores, factor = scale.input if _scalar(constants, scale.input[1]) is not None else scale.input[::-1]
        value = _scalar(constants, factor)
        # A power of two changes only float exponents, so the folded graph is exact.
        _require(value is not None and value > 0 and math.frexp(value)[0] == 0.5, f"power-of-two scale at {scale.name}")
        qk = _producer(producers, scores, "MatMul")
        _require(_next(users, qk, "Mul") is scale, "private attention scores")
        # Q = Transpose(Reshape(Add(bias, MatMul(hidden, weight)))).
        transpose = _producer(producers, qk.input[0], "Transpose")
        reshape = _producer(producers, transpose.input[0], "Reshape")
        add = _producer(producers, reshape.input[0], "Add")
        bias = next((name for name in add.input if name in initializers), None)
        projection = _producer(producers, next(name for name in add.input if name != bias), "MatMul")
        weight = projection.input[1]
        for node, consumer in ((transpose, qk), (reshape, transpose), (add, reshape), (projection, add)):
            _require(_next(users, node, consumer.op_type) is consumer, f"private Q path at {node.name}")
        for name in (weight, bias):
            _require(name in initializers and len(users[name]) == 1, f"private Q parameter at {projection.name}")
            array = numpy_helper.to_array(initializers[name])
            _require(array.dtype == np.float32, f"float32 Q parameter {name}")
            initializers[name].CopyFrom(numpy_helper.from_array(array * np.float32(value), name))
        softmax.input[0] = scores
        folded += 1
    _require(folded == expected, f"expected {expected} backbone attention blocks, found {folded}")
    return _finish(model)


def _optimize_decoder_attention(model: onnx.ModelProto) -> onnx.ModelProto:
    """Keep self-attention/head merge/projection in NCHW; leave deformable attention alone."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    replacements, removed = {}, set()
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or "/self_attn/" not in softmax.name:
            continue
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), "last-axis softmax required")
        scale = _producer(producers, softmax.input[0], "Mul")
        _require(_next(users, scale, "Softmax") is softmax, "private score scale")
        scores, factor = scale.input if _scalar(constants, scale.input[1]) is not None else scale.input[::-1]
        _require(_scalar(constants, factor) is not None, "scalar score scale")
        qk = _producer(producers, scores, "MatMul")
        _require(_next(users, qk, "Mul") is scale, "private attention scores")
        pv = _next(users, softmax, "MatMul")
        _require(pv.input[0] == softmax.output[0], "probabilities must be the left PV operand")
        merge = _next(users, pv, "Transpose")
        # onnxsim may merge the [1, T, H*D] and [T, H*D] reshapes into one.
        flattening = [_next(users, merge, "Reshape")]
        while (
            (following := users[flattening[-1].output[0]])
            and following[0] is not None
            and following[0].op_type == "Reshape"
        ):
            flattening.append(_next(users, flattening[-1], "Reshape"))
        # onnxsim 0.7 fuses the projection into 2-D Gemm + Reshape; the SDK's 0.4 keeps 3-D MatMul + Add.
        following = users[flattening[-1].output[0]]
        _require(len(following) == 1 and following[0] is not None, f"private {flattening[-1].name} projection")
        projection = following[0]
        if projection.op_type == "Gemm":
            options = _attributes(projection)
            _require(
                options.get("alpha", 1) == options.get("beta", 1) == 1
                and options.get("transA", 0) == 0
                and options.get("transB", 0) == 0
                and len(projection.input) == 3,
                "unsupported Gemm",
            )
            weight_name, bias_name = projection.input[1:]
            restore = _next(users, projection, "Reshape")
        else:
            _require(projection.op_type == "MatMul", f"expected Gemm or MatMul after {flattening[-1].name}")
            restore = _next(users, projection, "Add")
            weight_name = projection.input[1]
            bias_name = next(name for name in restore.input if name != projection.output[0])
        q, k = qk.input
        v = pv.input[1]
        shape = shapes.get(q, [])
        _require(len(shape) == 4 and min(shape) > 0, "static decoder Q shape required")
        batch, heads, tokens, dim = shape
        _require(
            batch == 1 and shapes.get(k) == [1, heads, dim, tokens] and shapes.get(v) == shape,
            "expected batch-one equal-head self-attention",
        )
        _require(
            _attributes(merge).get("perm") == [0, 2, 1, 3]
            and shapes.get(flattening[-1].output[0]) in ([tokens, heads * dim], [1, tokens, heads * dim]),
            "unexpected head merge",
        )
        _require(weight_name in constants and bias_name in constants, "constant projection weight/bias required")
        weight, bias = constants[weight_name], constants[bias_name]
        channels = weight.shape[1]
        _require(weight.shape[0] == heads * dim and bias.shape == (channels,), "projection shape mismatch")
        _require(shapes.get(restore.output[0]) == [1, tokens, channels], "restored sequence shape")
        prefix = softmax.name + "/nchw/"
        _require(not any(name.startswith(prefix) for name in (*producers, *constants)), "rewrite name collision")
        nodes = []

        def node(kind, inputs, name, outputs=None, **attributes):
            outputs = outputs or [prefix + name]
            nodes.append(helper.make_node(kind, inputs, outputs, name=prefix + name, **attributes))
            return outputs[0]

        def constant(name, value):
            name = prefix + name
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))
            return name

        def channels_first(value, name):
            # [1,H,T,D] (Q, V) or [1,H,D,T] (K) -> [1,D,H,T], folding the exporter's head Transpose into it.
            target = [0, 2, 1, 3] if value == k else [0, 3, 1, 2]
            if value in constants:  # Layer 0 V comes from the learned queries and is already folded.
                _require(len(users[value]) == 1, f"shared constant {value}")
                return constant(name, np.ascontiguousarray(constants[value].transpose(target)))
            source = producers[value]
            if source.op_type == "Transpose" and len(users[value]) == 1:
                perm = _attributes(source)["perm"]
                return node("Transpose", [source.input[0]], name, perm=[perm[axis] for axis in target])
            return node("Transpose", [value], name, perm=target)

        q4, k4, v4 = channels_first(q, "q"), channels_first(k, "k"), channels_first(v, "v")
        scores = node("Einsum", [q4, k4], "scores", equation="nchw,nchq->nqhw")
        scaled = node("Mul", [scores, factor], "scaled")
        probabilities = node("Softmax", [scaled], "probabilities", axis=1)
        values = node("Einsum", [probabilities, v4], "values", equation="nchw,nqhc->nqhw")
        split = [prefix + f"head{index}" for index in range(heads)]
        node("Split", [values, constant("split_sizes", np.ones(heads, dtype=np.int64))], "split", outputs=split, axis=2)
        packed = node("Concat", split, "packed", axis=1)
        # Gemm or MatMul computes X @ W + b with W stored [in, out]; a 1x1 Conv needs [out, in, 1, 1].
        weights = constant("weight", np.ascontiguousarray(weight.T.reshape(channels, heads * dim, 1, 1)))
        projected = node("Conv", [packed, weights, bias_name], "projection", kernel_shape=[1, 1])
        sequence = node("Transpose", [projected], "sequence", perm=[0, 3, 2, 1])
        node(
            "Reshape",
            [sequence, constant("output_shape", np.array([1, tokens, channels], dtype=np.int64))],
            "output",
            outputs=list(restore.output),
        )
        # Insert at the original output Reshape, after all Q/K/V and bias producers.
        replacements[restore.output[0]] = nodes
        removed.update(n.output[0] for n in (qk, scale, softmax, pv, merge, *flattening, projection, restore))
    _require(len(replacements) == 3, f"expected 3 decoder self-attention blocks, found {len(replacements)}")
    nodes = []
    for original in model.graph.node:
        nodes.extend(replacements.get(original.output[0], []))
        if original.output[0] not in removed:
            nodes.append(original)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return _finish(model)


def prepare_model(source: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    from onnxsim import simplify

    output_dir.mkdir(parents=True, exist_ok=True)
    full = onnx.load(source)
    onnx.checker.check_model(full)
    _validate_source(full)
    _require_folded(full)
    full = _remove_noop_masks(full, {"Where": 5, "ScatterND": 1})
    full = _remove_identities(full, {"Cast": 4, "Expand": 2, "Gather": 2})
    for old, new in _cut_points(full).items():
        _rename_value(full, old, new)
    for name, shape in BOUNDARY_SHAPES.items():
        _set_value_info(full, name, shape)
    full = onnx.shape_inference.infer_shapes(full)
    _validate_source(full)
    full.ir_version = 8
    onnx.checker.check_model(full)

    full_path = output_dir / "rf_detr_small_full.onnx"
    backbone_path = output_dir / "rf_detr_small_backbone.onnx"
    transformer_path = output_dir / "rf_detr_small_transformer.onnx"
    onnx.save(full, full_path)
    onnx.utils.extract_model(str(full_path), str(backbone_path), [INPUT], [FEATURES, SCORES, BOXES])
    onnx.utils.extract_model(str(full_path), str(transformer_path), [FEATURES, PROPOSALS], OUTPUTS)

    transformer = onnx.load(transformer_path)
    transformer, valid = simplify(
        transformer,
        overwrite_input_shapes={
            FEATURES: BOUNDARY_SHAPES[FEATURES],
            PROPOSALS: BOUNDARY_SHAPES[PROPOSALS],
        },
        dynamic_input_shape=False,
    )
    if not valid:
        raise RuntimeError("RF-DETR transformer simplification failed validation")
    # onnxsim turns the singleton feature-level axis of the reference points into a Gather.
    transformer = _remove_identities(transformer, {"Gather": 1})
    transformer = _optimize_decoder_attention(transformer)
    transformer.ir_version = 8
    onnx.checker.check_model(transformer)
    onnx.save(transformer, transformer_path)

    backbone = _fold_backbone_score_scale(_optimize_window_attention(onnx.load(backbone_path)), 12)
    backbone.ir_version = 8
    onnx.checker.check_model(backbone)
    onnx.save(backbone, backbone_path)
    return full_path, backbone_path, transformer_path


def validate_float_split(
    source: Path,
    backbone: Path,
    transformer: Path,
    image: np.ndarray,
) -> None:
    import onnxruntime as ort
    from calibration import stable_topk_gather

    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    backbone_session = ort.InferenceSession(str(backbone), providers=["CPUExecutionProvider"])
    transformer_session = ort.InferenceSession(str(transformer), providers=["CPUExecutionProvider"])
    sample = image[None]
    features, scores, boxes = backbone_session.run(None, {INPUT: sample})
    proposals = stable_topk_gather(scores, boxes)
    expected = reference.run(None, {INPUT: sample})
    actual = transformer_session.run(None, {FEATURES: features, PROPOSALS: proposals})
    for name, expected_value, actual_value in zip(OUTPUTS, expected, actual, strict=True):
        # Gemm -> NCHW Conv preserves the projection but can change float32
        # accumulation order. This tolerance covers the validated ONNX rewrite.
        if not np.allclose(actual_value, expected_value, rtol=2e-4, atol=2e-4):
            maximum_error = float(np.max(np.abs(actual_value - expected_value)))
            raise RuntimeError(f"RF-DETR split output {name} does not match the original graph: {maximum_error}")
