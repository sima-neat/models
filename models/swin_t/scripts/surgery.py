"""Fold the Swin-T ONNX export to its static input and rewrite it for the Model Compiler."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

INPUT = "pixel_values"
OUTPUT = "logits"
INPUT_SHAPE = [1, 3, 224, 224]
OUTPUT_SHAPE = [1, 1000]
# The last stage holds a 7x7 grid of 768-channel tokens.
FINAL_GRID, FINAL_CHANNELS = 7, 768
# Swin-T has 2, 2, 6 and 2 blocks per stage, and every second block shifts its windows except in the
# last stage, whose 7x7 grid is a single window.
BLOCKS = 12
SHIFTED_BLOCKS = 5
# Blocks with more than one window: all but the last stage's two.
WINDOWED_BLOCKS = 10
WINDOW = 7


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unexpected Swin-T graph: {message}")


def _attributes(node: onnx.NodeProto) -> dict:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}


def _constant(model: onnx.ModelProto, name: str, value: np.ndarray) -> str:
    model.graph.initializer.append(numpy_helper.from_array(value, name))
    return name


def _index(model: onnx.ModelProto):
    """Constants by name, the node producing each value, and the nodes reading each value."""
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    users = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            users[name].append(node)
    return constants, producers, users


def _producer(producers: dict, value: str, kind: str) -> onnx.NodeProto:
    node = producers.get(value)
    _require(node is not None and node.op_type == kind, f"{value} must come from a {kind}")
    return node


def _only_user(users: dict, node: onnx.NodeProto, kind: str) -> onnx.NodeProto:
    readers = users[node.output[0]]
    _require(len(readers) == 1 and readers[0].op_type == kind, f"{node.name} must feed only a {kind}")
    return readers[0]


def _scalar(constants: dict, name: str) -> float | None:
    value = constants.get(name)
    return float(value) if value is not None and value.size == 1 else None


def _replace(model: onnx.ModelProto, name: str, value: np.ndarray, users: dict) -> None:
    """Overwrite a constant that only one node reads."""
    _require(len(users[name]) == 1, f"constant {name} is shared")
    (initializer,) = [item for item in model.graph.initializer if item.name == name]
    initializer.CopyFrom(numpy_helper.from_array(value.astype(np.float32), name))


def windows_in_four_dimensions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Partition and merge windows with 4-D tensors.

    The export splits [1, H, W, C] into [1, H/7, 7, W/7, 7, C], swaps the two middle axes and flattens
    the windows. Each window row's 7xC values stay contiguous, so [H/7, 7, W/7, 7C] with its middle axes
    swapped is the same permutation; window merging is its inverse. Both drop a 6-D transpose.
    """
    constants, producers, users = _index(model)
    shapes = _shapes(model)
    rewritten = 0
    for transpose in model.graph.node:
        if transpose.op_type != "Transpose" or _attributes(transpose)["perm"] != [0, 1, 3, 2, 4, 5]:
            continue
        split = _producer(producers, transpose.input[0], "Reshape")
        _only_user(users, split, "Transpose")
        one, outer, inner, across, last, channels = shapes[split.output[0]]
        _require(one == 1 and last == WINDOW, f"6-D window layout at {split.name}")
        shape = np.asarray([outer, inner, across, last * channels], np.int64)
        split.input[1] = _constant(model, f"{split.output[0]}/four_dimensional_shape", shape)
        del transpose.attribute[:]
        transpose.attribute.append(helper.make_attribute("perm", [0, 2, 1, 3]))
        rewritten += 1
    # Each block partitions once and merges once.
    _require(rewritten == 2 * BLOCKS, f"expected {2 * BLOCKS} window transposes, found {rewritten}")
    return _finish(model)


def merge_shift_masks(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add each shifted block's window mask to its relative position bias, ahead of time.

    The export adds the [1, heads, 49, 49] bias, reshapes the scores to 5-D, adds the
    [1, windows, 1, 49, 49] mask and reshapes back; one constant [windows, heads, 49, 49] Add is the same sum.
    """
    constants, producers, users = _index(model)
    shapes = _shapes(model)
    merged = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        back = producers.get(softmax.input[0])
        if back is None or back.op_type != "Reshape":
            continue
        masked = _producer(producers, back.input[0], "Add")
        mask = next((name for name in masked.input if name in constants), None)
        _require(mask is not None, f"constant window mask at {masked.name}")
        split = _producer(producers, next(name for name in masked.input if name != mask), "Reshape")
        biased = _producer(producers, split.input[0], "Add")
        bias = next((name for name in biased.input if name in constants), None)
        _require(bias is not None, f"constant position bias at {biased.name}")
        for node, consumer in ((biased, "Reshape"), (split, "Add"), (masked, "Reshape"), (back, "Softmax")):
            _only_user(users, node, consumer)
        windows, heads, tokens, _ = shapes[biased.output[0]]
        _require(constants[mask].shape == (1, windows, 1, tokens, tokens), f"window mask shape at {masked.name}")
        combined = constants[bias].reshape(1, heads, tokens, tokens) + constants[mask][0]
        _replace(model, bias, combined, users)
        softmax.input[0] = biased.output[0]
        merged += 1
    _require(merged == SHIFTED_BLOCKS, f"expected {SHIFTED_BLOCKS} shifted blocks, found {merged}")
    return _finish(model)


def fold_score_scale(model: onnx.ModelProto) -> onnx.ModelProto:
    """Scale the query projection instead of every [windows, heads, 49, 49] attention score."""
    constants, producers, users = _index(model)
    folded = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        biased = _producer(producers, softmax.input[0], "Add")
        scaled = next(producers[name] for name in biased.input if name not in constants)
        _require(scaled.op_type == "Div", f"scaled scores at {scaled.name}")
        _only_user(users, scaled, "Add")
        scores, divisor = scaled.input
        value = _scalar(constants, divisor)
        _require(value is not None and value > 0, f"scalar score scale at {scaled.name}")
        product = _producer(producers, scores, "MatMul")
        _only_user(users, product, "Div")
        # Q = Transpose(Reshape(Add(bias, Split(MatMul(hidden, [Wq Wk Wv]))[0]))); onnxsim fuses the projections.
        transpose = _producer(producers, product.input[0], "Transpose")
        reshape = _producer(producers, transpose.input[0], "Reshape")
        add = _producer(producers, reshape.input[0], "Add")
        bias = next(name for name in add.input if name in constants)
        query = next(name for name in add.input if name != bias)
        split = _producer(producers, query, "Split")
        projection = _producer(producers, split.input[0], "MatMul")
        _only_user(users, projection, "Split")
        for node, consumer in ((add, "Reshape"), (reshape, "Transpose"), (transpose, "MatMul")):
            _only_user(users, node, consumer)
        _require(len(users[query]) == 1, f"private query at {split.name}")
        sizes = _attributes(split)["split"] if "split" in _attributes(split) else constants[split.input[1]].tolist()
        _require(_attributes(split).get("axis") in (-1, 2) and len(set(sizes)) == 1, f"Q/K/V split at {split.name}")
        index = list(split.output).index(query)
        width = sizes[0]
        weight = constants[projection.input[1]].copy()
        weight[:, index * width : (index + 1) * width] /= np.float32(value)
        _replace(model, projection.input[1], weight, users)
        _replace(model, bias, constants[bias] / np.float32(value), users)
        biased.input[list(biased.input).index(scaled.output[0])] = scores
        folded += 1
    _require(folded == BLOCKS, f"expected {BLOCKS} attention blocks, found {folded}")
    return _finish(model)


def fold_gelu_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """Compute fc2(z (1 + erf(z))) / sqrt(2) with z = fc1(x) / sqrt(2), the same GELU.

    The export's 0.5 h (1 + erf(h / sqrt(2))) spends a Div and a Mul on every hidden activation;
    folding both constants into the fc1 and fc2 weights leaves Erf, Add and Mul.
    """
    constants, producers, users = _index(model)
    scale = np.float32(1 / math.sqrt(2))
    folded = 0
    for erf in model.graph.node:
        if erf.op_type != "Erf":
            continue
        divide = _producer(producers, erf.input[0], "Div")
        _only_user(users, divide, "Erf")
        hidden, divisor = divide.input
        root_two = _scalar(constants, divisor)
        _require(
            root_two is not None and np.isclose(root_two, math.sqrt(2)), f"GELU divides by sqrt(2) at {divide.name}"
        )
        shift = _only_user(users, erf, "Add")
        _require(_scalar(constants, next(n for n in shift.input if n != erf.output[0])) == 1.0, f"GELU at {shift.name}")
        gate = _only_user(users, shift, "Mul")
        _require(sorted(gate.input) == sorted([hidden, shift.output[0]]), f"GELU gate at {gate.name}")
        half = _only_user(users, gate, "Mul")
        _require(_scalar(constants, next(n for n in half.input if n != gate.output[0])) == 0.5, f"GELU at {half.name}")
        fc2 = _only_user(users, half, "MatMul")
        _require(fc2.input[0] == half.output[0] and len(users[hidden]) == 2, f"GELU operands at {fc2.name}")
        add = _producer(producers, hidden, "Add")
        bias = next(name for name in add.input if name in constants)
        fc1 = _producer(producers, next(name for name in add.input if name != bias), "MatMul")
        _only_user(users, fc1, "Add")
        for name in (fc1.input[1], bias, fc2.input[1]):
            _replace(model, name, constants[name] * scale, users)
        erf.input[0] = hidden
        fc2.input[0] = gate.output[0]
        folded += 1
    _require(folded == BLOCKS, f"expected {BLOCKS} GELUs, found {folded}")
    return _finish(model)


def _finish(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drop nodes and initializers nothing reads, then re-infer shapes."""
    graph = model.graph
    live = {value.name for value in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(name in live for name in node.output):
            kept.append(node)
            live.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    initializers = [item for item in graph.initializer if item.name in live]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    return onnx.shape_inference.infer_shapes(model)


def fold(source: Path) -> onnx.ModelProto:
    """Fix the export's dynamic shapes to one 224x224 image and fold every shape computation."""
    from onnxsim import simplify

    model, valid = simplify(onnx.load(source), overwrite_input_shapes={INPUT: INPUT_SHAPE})
    _require(valid, "onnxsim could not validate the folded graph")
    model = onnx.shape_inference.infer_shapes(model)
    shapes = _shapes(model)
    _require(shapes[INPUT] == INPUT_SHAPE and shapes[OUTPUT] == OUTPUT_SHAPE, "input or output interface")
    unfolded = sorted({node.op_type for node in model.graph.node} & {"Shape", "ScatterND", "Range", "Where"})
    _require(not unfolded, f"shape computations survived folding: {unfolded}")
    return model


def pool_as_image(model: onnx.ModelProto) -> onnx.ModelProto:
    """Average the final tokens as a 7x7 image.

    The export pools [1, 49, 768] tokens with a 1-D GlobalAveragePool on [1, 768, 49], which the
    Model Compiler's layout conversion rejects; a 2-D GlobalAveragePool over [1, 768, 7, 7] is the same mean.
    """
    shapes = _shapes(model)
    pools = [node for node in model.graph.node if node.op_type == "GlobalAveragePool"]
    _require(len(pools) == 1, "one GlobalAveragePool")
    (pool,) = pools
    (transpose,) = [node for node in model.graph.node if pool.input[0] in node.output]
    _require(transpose.op_type == "Transpose" and _attributes(transpose)["perm"] == [0, 2, 1], "token transpose")
    tokens = transpose.input[0]
    _require(shapes[tokens] == [1, FINAL_GRID * FINAL_GRID, FINAL_CHANNELS], "final token shape")
    grid = "pool/grid"
    shape = _constant(model, "pool/grid_shape", np.asarray([1, FINAL_GRID, FINAL_GRID, FINAL_CHANNELS], np.int64))
    index = list(model.graph.node).index(transpose)
    model.graph.node.insert(index, helper.make_node("Reshape", [tokens, shape], [grid], name="pool/Reshape"))
    transpose.input[0] = grid
    del transpose.attribute[:]
    transpose.attribute.append(helper.make_attribute("perm", [0, 3, 1, 2]))
    model = _finish(model)
    _require(_shapes(model)[pool.output[0]] == [1, FINAL_CHANNELS, 1, 1], "2-D pool output")
    return model


def heads_before_windows(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run window attention in head/window order instead of window/head order.

    Q, K and V become [heads, windows, 49, dim] rather than [windows, heads, 49, dim], the position bias
    (with any window mask) is transposed to match, and the attention output is permuted straight back to
    [windows, 49, heads, dim]; RF-DETR's backbone (#129) needed fewer layout conversions this way.
    """
    constants, producers, users = _index(model)
    shapes = _shapes(model)
    rewritten = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax":
            continue
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), f"last-axis softmax at {softmax.name}")
        biased = _producer(producers, softmax.input[0], "Add")
        bias = next(name for name in biased.input if name in constants)
        product = _producer(producers, next(name for name in biased.input if name != bias), "MatMul")
        windows, heads, tokens, _ = shapes[product.output[0]]
        if windows == 1:
            continue  # A single window has no window/head interchange.
        query, key = (_producer(producers, name, "Transpose") for name in product.input)
        weighted = _only_user(users, softmax, "MatMul")
        _require(weighted.input[0] == softmax.output[0], f"probabilities must be the left operand at {weighted.name}")
        value = _producer(producers, weighted.input[1], "Transpose")
        merge = _only_user(users, weighted, "Transpose")
        _only_user(users, product, "Add")
        _only_user(users, biased, "Softmax")
        for node in (query, key, value):
            _require(len(users[node.output[0]]) == 1, f"shared projection at {node.name}")
        changes = (
            (query, [0, 2, 1, 3], [2, 0, 1, 3]),
            (key, [0, 2, 3, 1], [2, 0, 3, 1]),
            (value, [0, 2, 1, 3], [2, 0, 1, 3]),
            (merge, [0, 2, 1, 3], [1, 2, 0, 3]),
        )
        for node, old, new in changes:
            _require(_attributes(node)["perm"] == old, f"unexpected permutation at {node.name}")
            del node.attribute[:]
            node.attribute.append(helper.make_attribute("perm", new))
        table = constants[bias]
        _require(table.ndim == 4 and table.shape[1] == heads, f"[windows, heads, 49, 49] bias at {biased.name}")
        _replace(model, bias, table.transpose(1, 0, 2, 3), users)
        rewritten += 1
    _require(rewritten == WINDOWED_BLOCKS, f"expected {WINDOWED_BLOCKS} windowed blocks, found {rewritten}")
    return _finish(model)


# Rewrites in the order prepare_model applies them; merging the masks first leaves the score scale beside the bias.
REWRITES = (
    pool_as_image,
    windows_in_four_dimensions,
    merge_shift_masks,
    fold_score_scale,
    fold_gelu_constants,
    heads_before_windows,
)


def prepare_model(source: Path, output: Path, rewrites=REWRITES) -> Path:
    """Write the graph the recipe imports: the folded export with every rewrite applied."""
    model = fold(source)
    for rewrite in rewrites:
        model = rewrite(model)
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return output


def max_difference(source: Path, prepared: Path, image: np.ndarray) -> float:
    """Largest logit difference between the source and the prepared graph on one NCHW image."""
    import onnxruntime as ort

    outputs = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run([OUTPUT], {INPUT: image})[0]
        for path in (source, prepared)
    ]
    return float(np.max(np.abs(outputs[0] - outputs[1])))
