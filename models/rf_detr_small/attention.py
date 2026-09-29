"""Guarded ONNX layout rewrites for the pinned RF-DETR export, before calibration.

Window attention keeps producer head order. Decoder self-attention uses NCHW
through its output projection. Neither rewrite depends on a Flash kernel.
"""

from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported RF-DETR attention: {message}")


def _attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _index(model):
    shapes = {
        value.name: [d.dim_value for d in value.type.tensor_type.shape.dim]
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
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


def _finish(model):
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def optimize_window_attention(model, *, expected_blocks=9):
    """Commute identical window/head permutations through unmasked attention."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    rewritten = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or not softmax.name.startswith("/backbone/"):
            continue
        qk = _producer(producers, softmax.input[0], "MatMul")
        q, k = [_producer(producers, value, "Mul") for value in qk.input]
        shape = shapes.get(q.output[0], [])
        _require(len(shape) == 4 and min(shape) > 0, f"static Q shape at {q.name}")
        windows, heads, tokens, dim = shape
        if windows == 1:
            continue  # Global attention already has no window/head interchange.
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), "last-axis softmax required")
        qt, kt = [_producer(producers, node.input[0], "Transpose") for node in (q, k)]
        _require(_next(users, q, "MatMul") is qk and _next(users, k, "MatMul") is qk, "shared Q/K scaling")
        _require(_next(users, qk, "Softmax") is softmax, "shared attention scores")
        pv = _next(users, softmax, "MatMul")
        _require(pv.input[0] == softmax.output[0], "probabilities must be the left PV operand")
        vt = _producer(producers, pv.input[1], "Transpose")
        output = _next(users, pv, "Transpose")
        _require(_next(users, kt, "Mul") is k and _next(users, vt, "MatMul") is pv, "shared K/V transpose")
        _require(
            shapes.get(k.output[0]) == [windows, heads, dim, tokens] and shapes.get(vt.output[0]) == shape,
            "Q/K/V group or token mismatch",
        )
        _require(
            all(shapes.get(node.input[0]) == [windows, tokens, heads, dim] for node in (qt, kt, vt)),
            "Q/K/V producer shapes differ",
        )
        _require(all(shapes.get(node.input[1]) in ([], [1]) for node in (q, k)), "non-scalar Q/K scaling")
        # The exporter also uses Shape(Q)[-1:] to calculate its scalar scale.
        for observer in users[qt.output[0]]:
            if observer is q:
                continue
            _require(
                observer is not None and observer.op_type == "Shape" and not observer.attribute, "shared Q transpose"
            )
            sliced = _next(users, observer, "Slice")
            _require(
                len(sliced.input) == 3
                and constants.get(sliced.input[1], np.array([])).tolist() == [-1]
                and constants.get(sliced.input[2], np.array([])).tolist() == [np.iinfo(np.int64).max],
                "Q shape observer must read only the unchanged head dimension",
            )
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
    _require(rewritten == expected_blocks, f"expected {expected_blocks} window blocks, found {rewritten}")
    return _finish(model)


def optimize_decoder_attention(model, *, expected_blocks=3):
    """Keep self-attention/head merge/projection in NCHW; leave deformable attention alone."""
    model = onnx.shape_inference.infer_shapes(model)
    shapes, constants, producers, users = _index(model)
    replacements, removed = {}, set()
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or "/self_attn/" not in softmax.name:
            continue
        qk = _producer(producers, softmax.input[0], "MatMul")
        _require(_next(users, qk, "Softmax") is softmax, "shared attention scores")
        pv = _next(users, softmax, "MatMul")
        _require(pv.input[0] == softmax.output[0], "probabilities must be the left PV operand")
        transpose = _next(users, pv, "Transpose")
        reshape = _next(users, transpose, "Reshape")
        projection = _next(users, reshape, "Gemm")
        q, k = qk.input
        v = pv.input[1]
        shape = shapes.get(q, [])
        _require(len(shape) == 4 and min(shape) > 0, "static decoder Q shape required")
        batch, heads, tokens, dim = shape
        _require(
            batch == 1 and shapes.get(k) == [1, heads, dim, tokens] and shapes.get(v) == shape,
            "expected batch-one equal-head self-attention",
        )
        _require(_attributes(softmax).get("axis", -1) in (-1, 3), "last-axis softmax required")
        _require(
            _attributes(transpose).get("perm") == [2, 0, 1, 3]
            and shapes.get(reshape.output[0]) == [tokens, heads * dim],
            "unexpected head merge",
        )
        options = _attributes(projection)
        _require(
            options.get("alpha", 1) == options.get("beta", 1) == 1
            and options.get("transA", 0) == 0
            and options.get("transB", 0) == 1,
            "unsupported Gemm",
        )
        _require(
            len(projection.input) == 3 and projection.input[1] in constants and projection.input[2] in constants,
            "constant projection weight/bias required",
        )
        weight, bias = (constants[name] for name in projection.input[1:])
        _require(
            weight.ndim == 2 and weight.shape[1] == heads * dim and bias.shape == (weight.shape[0],),
            "projection shape mismatch",
        )
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

        q4 = node("Transpose", [q], "q", perm=[0, 3, 1, 2])
        k4 = node("Transpose", [k], "k", perm=[0, 2, 1, 3])
        v4 = node("Transpose", [v], "v", perm=[0, 3, 1, 2])
        scores = node("Einsum", [q4, k4], "scores", equation="nchw,nchq->nqhw")
        probabilities = node("Softmax", [scores], "probabilities", axis=1)
        values = node("Einsum", [probabilities, v4], "values", equation="nchw,nqhc->nqhw")
        split = [prefix + f"head{index}" for index in range(heads)]
        node("Split", [values, constant("split_sizes", np.ones(heads, dtype=np.int64))], "split", outputs=split, axis=2)
        packed = node("Concat", split, "packed", axis=1)
        weights = constant("weight", np.ascontiguousarray(weight.reshape(weight.shape[0], heads * dim, 1, 1)))
        projected = node("Conv", [packed, weights, projection.input[2]], "projection", kernel_shape=[1, 1])
        unpacked = node("Transpose", [projected], "unpacked", perm=[3, 0, 2, 1])
        node(
            "Reshape",
            [unpacked, constant("output_shape", np.array([tokens, weight.shape[0]], dtype=np.int64))],
            "output",
            outputs=list(projection.output),
        )
        # Insert at the original projection, after all Q/K/V and bias producers.
        replacements[projection.output[0]] = nodes
        removed.update(n.output[0] for n in (qk, softmax, pv, transpose, reshape, projection))
    _require(
        len(replacements) == expected_blocks,
        f"expected {expected_blocks} decoder self-attention blocks, found {len(replacements)}",
    )
    nodes = []
    for original in model.graph.node:
        nodes.extend(replacements.get(original.output[0], []))
        if original.output[0] not in removed:
            nodes.append(original)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for n in nodes for name in n.input} | {v.name for v in model.graph.output}
    initializers = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return _finish(model)
