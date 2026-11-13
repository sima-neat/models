"""Split the validated static D-FINE Nano export at its CPU proposal-selection boundary.

Shape freezing and the proposal split preserve numerical operations. The decoder then
gets three rewrites for the compiler: per-head attention sampling and a batch-preserving
Integral, which reorder no arithmetic, and the bounded reference logits below, a measured
approximation that compile.py checks against the original floating model.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from model_registry.fetch import sha256_file
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

MODEL_ID = "dfine_n"
IMAGE_SIZE = 640
QUERY_COUNT = 300
CLASSES = 80
# Decoder width, attention heads and the channels each head samples.
WIDTH = 128
HEADS = 8
HEAD_CHANNELS = WIDTH // HEADS
# The two feature levels, at strides 16 and 32, in attention-memory order, and the
# sampling points each level takes per query and head.
LEVELS = ((IMAGE_SIZE // 16, IMAGE_SIZE // 16), (IMAGE_SIZE // 32, IMAGE_SIZE // 32))
POINTS = (6, 6)
TOKENS = sum(height * width for height, width in LEVELS)
DECODER_LAYERS = 3
# Integral distribution bins per box side: max_num_bins + 1.
BINS = 33
INPUT_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
BACKBONE_OUTPUTS = {
    "attention_memory": (1, TOKENS, WIDTH),
    "query_content": (1, TOKENS, WIDTH),
    "proposal_scores": (1, TOKENS),
    "reference_logits": (1, TOKENS, 4),
}
DECODER_INPUTS = {
    "attention_memory": (1, TOKENS, WIDTH),
    "query_content": (1, QUERY_COUNT, WIDTH),
    "reference_logits": (1, QUERY_COUNT, 4),
}
OUTPUTS = {"pred_boxes": (1, QUERY_COUNT, 4), "pred_logits": (1, QUERY_COUNT, CLASSES)}
BF16_REGIONS = ("backbone", "nonconv", "outputs")
LAYOUT_OPS = {
    "Identity",
    "Reshape",
    "Transpose",
    "Flatten",
    "Slice",
    "Split",
    "Concat",
    "Squeeze",
    "Unsqueeze",
    "Tile",
    "Expand",
}
INDEX_ROUTING = {"Identity", "Unsqueeze", "Reshape", "Expand", "Tile"}


def require(condition, message):
    if not condition:
        raise ValueError(f"Unsupported D-FINE graph: {message}")


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def shape(value):
    dims = value.type.tensor_type.shape.dim
    return tuple(d.dim_value for d in dims) if all(d.HasField("dim_value") for d in dims) else None


@dataclass(frozen=True)
class GraphIndex:
    """Lookups over one ONNX graph: typed values, each value's producer and readers, and constants."""

    model: onnx.ModelProto
    values: dict[str, onnx.ValueInfoProto]
    producers: dict[str, onnx.NodeProto]
    users: dict[str, list[onnx.NodeProto]]
    constants: dict[str, np.ndarray]

    def shape(self, name):
        return shape(self.values[name]) if name in self.values else None

    def producer(self, name, op_type):
        node = self.producers.get(name)
        require(node is not None and node.op_type == op_type, f"expected {op_type} producing {name}")
        return node

    def sole_user(self, node, op_type):
        readers = self.users.get(node.output[0], [])
        require(len(readers) == 1 and readers[0].op_type == op_type, f"expected {node.name} -> one {op_type}")
        return readers[0]


def index(model):
    values = {v.name: v for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    producers = {name: node for node in model.graph.node for name in node.output}
    users = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant" and "value" in attrs(node):
            constants[node.output[0]] = numpy_helper.to_array(attrs(node)["value"])
    return GraphIndex(model, values, producers, users, constants)


def extract(model, inputs, outputs):
    """Extract a subgraph with inferred intermediate shapes.

    onnx 1.17's Extractor inferred shapes itself; later releases keep only the value_info
    already present, so the rewrites below would see no intermediate shapes.
    """
    return onnx.utils.Extractor(onnx.shape_inference.infer_shapes(model)).extract_model(inputs, outputs)


def validate_source(model):
    metadata = {p.key: p.value for p in model.metadata_props}
    require(metadata.get("model_id") == MODEL_ID, "model identity mismatch")
    require(
        metadata.get("box_format") == "normalized_cxcywh" and metadata.get("class_output") == "logits",
        "unexpected public output semantics",
    )
    require([(x.domain, x.version) for x in model.opset_import] == [("", 17)], "expected ONNX opset 17")
    require(
        {x.name: shape(x) for x in model.graph.input} == {"input": INPUT_SHAPE},
        "expected static batch-one 640 RGB input",
    )
    require(
        [(x.name, shape(x)) for x in model.graph.output] == list(OUTPUTS.items()),
        "unexpected public output shapes/order",
    )
    require(
        all(x.type.tensor_type.elem_type == TensorProto.FLOAT for x in (*model.graph.input, *model.graph.output)),
        "expected float32 public I/O",
    )
    onnx.checker.check_model(model)


def infer_and_freeze_shapes(model):
    """Use existing symbolic inference without merging/guessing unknown dimensions."""
    source = deepcopy(model)
    operations = [x.SerializeToString() for x in source.graph.node]
    weights = [x.SerializeToString() for x in source.graph.initializer]
    inferred = SymbolicShapeInference.infer_shapes(source, auto_merge=False, guess_output_rank=False)
    require(operations == [x.SerializeToString() for x in inferred.graph.node], "shape inference changed operations")
    require(weights == [x.SerializeToString() for x in inferred.graph.initializer], "shape inference changed weights")
    values = index(inferred).values
    frozen = []
    for node in inferred.graph.node:
        if node.op_type != "Shape" or node.input[0] not in values:
            continue
        dimensions = shape(values[node.input[0]])
        if dimensions is None:
            continue
        attributes = attrs(node)
        dimensions = dimensions[attributes.get("start", 0) : attributes.get("end", len(dimensions))]
        output, name = node.output[0], node.name
        node.CopyFrom(
            helper.make_node(
                "Constant", [], [output], name=name, value=numpy_helper.from_array(np.asarray(dimensions, np.int64))
            )
        )
        frozen.append(output)
    return inferred, frozen


def ancestors(value, producers, stops=()):
    seen, pending = set(), [value]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in stops and name in producers:
            pending.extend(producers[name].input)
    return seen


def follows_indices(value, root, producers):
    while value != root:
        node = producers.get(value)
        if node is None or node.op_type not in INDEX_ROUTING:
            return False
        value = node.input[0]
    return True


def rename(model, mapping):
    for node in model.graph.node:
        for fields in (node.input, node.output):
            for i, name in enumerate(fields):
                fields[i] = mapping.get(name, name)
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.name = mapping.get(value.name, value.name)


def split_model(model):
    """Return backbone, decoder and evidence of the matched proposal contract."""
    validate_source(model)
    model, frozen = infer_and_freeze_shapes(model)
    graph = index(model)
    values, producers, constants = graph.values, graph.producers, graph.constants

    def matches(name, dimensions, dtype=TensorProto.FLOAT):
        return name in values and shape(values[name]) == dimensions and values[name].type.tensor_type.elem_type == dtype

    topks = [
        n
        for n in model.graph.node
        if n.op_type == "TopK"
        and n.input[1] in constants
        and constants[n.input[1]].size == 1
        and constants[n.input[1]].item() == QUERY_COUNT
    ]
    require(len(topks) == 1, "expected one Top300 proposal selector")
    topk = topks[0]
    require(
        attrs(topk).get("axis", -1) == 1 and attrs(topk).get("largest", 1) == 1 and attrs(topk).get("sorted", 1) == 1,
        "unexpected Top300 attributes",
    )
    scores, indices = topk.input[0], topk.output[1]
    require(
        matches(scores, (1, TOKENS)) and matches(indices, (1, QUERY_COUNT), TensorProto.INT64),
        "unexpected proposal count or Top300 index type",
    )
    gathers = [
        n for n in model.graph.node if n.op_type == "GatherElements" and follows_indices(n.input[1], indices, producers)
    ]
    require(len(gathers) == 2, "expected exactly two Top300 gathers")
    by_width = {}
    for node in gathers:
        require(attrs(node).get("axis", 0) == 1, "unexpected proposal gather axis")
        dimensions = graph.shape(node.input[0])
        require(dimensions in ((1, TOKENS, WIDTH), (1, TOKENS, 4)), "unexpected gathered tensor shape")
        size = dimensions[-1]
        require(size not in by_width and matches(node.output[0], (1, QUERY_COUNT, size)), "ambiguous gathered content")
        require(matches(node.input[1], (1, QUERY_COUNT, size), TensorProto.INT64), "unexpected gather index layout")
        by_width[size] = node
    content, references = by_width[WIDTH], by_width[4]
    scorer = producers.get(scores)
    require(
        scorer is not None
        and scorer.op_type == "ReduceMax"
        and attrs(scorer).get("axes") in ([-1], [2])
        and attrs(scorer).get("keepdims", 1) == 0
        and matches(scorer.input[0], (1, TOKENS, CLASSES)),
        "expected maximum raw class logit",
    )
    reference_add = producers.get(references.input[0])
    require(
        reference_add is not None and reference_add.op_type == "Add",
        "expected unactivated bbox-plus-anchor reference logits",
    )
    require(
        sum(content.input[0] in ancestors(x, producers) for x in reference_add.input) == 1,
        "expected one content-dependent bbox delta and one independent anchor",
    )
    level_shapes = [(1, height * width, WIDTH) for height, width in LEVELS]
    memory_candidates = [
        n
        for n in model.graph.node
        if n.op_type == "Concat"
        and attrs(n).get("axis") == 1
        and matches(n.output[0], (1, TOKENS, WIDTH))
        and [graph.shape(x) for x in n.input] == level_shapes
    ]
    require(len(memory_candidates) == 1, "expected one ordered multi-level attention memory")
    memory = memory_candidates[0].output[0]
    require(
        memory != content.input[0] and memory in ancestors(content.input[0], producers),
        "query content must be transformed from attention memory",
    )
    require(
        producers.get(content.input[0]) is not None and producers[content.input[0]].op_type == "LayerNormalization",
        "expected transformed query normalization",
    )
    require(content.input[0] in ancestors(scores, producers), "proposal scores do not use query content")
    require(
        content.input[0] in ancestors(references.input[0], producers), "proposal references do not use query content"
    )

    # Prove the permitted routing copies each selected index across its last axis.
    # Shape freezing removes all model-data dependencies from this tiny integer graph.
    index_graph = extract(model, [indices], [n.input[1] for n in gathers])
    require([x.name for x in index_graph.graph.input] == [indices], "index expansion depends on model data")
    expected_indices = np.arange(QUERY_COUNT, dtype=np.int64)[None]
    expanded = ReferenceEvaluator(index_graph).run(None, {indices: expected_indices})
    for node, actual in zip(gathers, expanded, strict=True):
        expected = np.broadcast_to(expected_indices[..., None], (1, QUERY_COUNT, graph.shape(node.input[0])[-1]))
        require(np.array_equal(actual, expected), "index routing changes proposal order or values")

    backbone_outputs = [memory, content.input[0], scores, references.input[0]]
    decoder_inputs = [memory, content.output[0], references.output[0]]
    needed = set().union(*(ancestors(name, producers, decoder_inputs) for name in OUTPUTS))
    require("input" not in needed, "decoder retains a backbone dependency")
    backbone = extract(model, ["input"], backbone_outputs)
    decoder = extract(model, decoder_inputs, list(OUTPUTS))
    require(all(name in needed for name in decoder_inputs), "unused decoder boundary")
    require(not any(n.op_type == "GatherElements" for n in decoder.graph.node), "unresolved dynamic gather")
    require(not any(n.op_type == "TopK" for n in backbone.graph.node), "proposal selection remained in backbone")
    rename(backbone, dict(zip(backbone_outputs, BACKBONE_OUTPUTS, strict=True)))
    rename(decoder, dict(zip(decoder_inputs, DECODER_INPUTS, strict=True)))
    for component in (backbone, decoder):
        onnx.checker.check_model(component, full_check=True)
    return (
        backbone,
        decoder,
        {
            "model_id": MODEL_ID,
            "tokens": TOKENS,
            "width": WIDTH,
            "frozen_shapes": frozen,
            "source_boundaries": backbone_outputs,
            "source_indices": indices,
        },
    )


def _constant_scalar(graph, name):
    """The value of a scalar the frozen graph computes from constants alone."""
    if name in graph.constants:
        value = graph.constants[name]
    else:
        value = ReferenceEvaluator(extract(graph.model, [], [name])).run(None, {})[0]
    require(np.asarray(value).size == 1, f"expected a scalar constant: {name}")
    return int(np.asarray(value).reshape(()))


def _int64(name, values):
    return numpy_helper.from_array(np.asarray(values, np.int64), name)


def _splice(model, old, new_nodes, initializers):
    """Replace the node producing a tensor with new nodes that produce the same name."""
    nodes = list(model.graph.node)
    position = [node.name for node in nodes].index(old.name)
    nodes[position].output[0] = old.output[0] + "/upstream"
    del model.graph.node[:]
    model.graph.node.extend(nodes[:position] + new_nodes + nodes[position:])
    model.graph.initializer.extend(initializers)


def _prune(model):
    """Drop what no longer reaches an output; keep the public inputs and outputs."""
    model = extract(model, [v.name for v in model.graph.input], [v.name for v in model.graph.output])
    onnx.checker.check_model(model, full_check=True)
    return model


def _attention_levels(graph, concat):
    """Match one deformable-attention sampling: its value, grid and weight tensors and level spans."""
    samples = [graph.producer(name, "GridSample") for name in concat.input]
    require(len(samples) == len(LEVELS), "expected one GridSample per feature level")
    mul = graph.sole_user(concat, "Mul")
    reduce = graph.sole_user(mul, "ReduceSum")
    reshape = graph.sole_user(reduce, "Reshape")
    transpose = graph.sole_user(reshape, "Transpose")
    total = sum(POINTS)
    require(
        graph.shape(concat.output[0]) == (HEADS, HEAD_CHANNELS, QUERY_COUNT, total)
        and attrs(reduce).get("keepdims", 1) == 0
        and reduce.input[1] in graph.constants
        and graph.constants[reduce.input[1]].tolist() in ([-1], [3])
        and graph.shape(reduce.output[0]) == (HEADS, HEAD_CHANNELS, QUERY_COUNT)
        and attrs(transpose).get("perm") == [0, 2, 1]
        and graph.shape(transpose.output[0]) == (1, QUERY_COUNT, WIDTH),
        "unexpected attention weighting or reduction",
    )
    weight_view = mul.input[1] if mul.input[0] == concat.output[0] else mul.input[0]
    weight_reshape = graph.producer(weight_view, "Reshape")
    weight_transpose = graph.producer(weight_reshape.input[0], "Transpose")
    weights = weight_transpose.input[0]
    require(
        graph.shape(weight_view) == (HEADS, 1, QUERY_COUNT, total)
        and attrs(weight_transpose).get("perm") == [0, 2, 1, 3]
        and graph.shape(weights) == (1, QUERY_COUNT, HEADS, total),
        "unexpected attention weight layout",
    )
    values, grids, spans = set(), set(), []
    for level, (sample, (height, width)) in enumerate(zip(samples, LEVELS, strict=True)):
        policy = attrs(sample)
        require(
            policy.get("mode", b"bilinear") == b"bilinear"
            and policy.get("padding_mode", b"zeros") == b"zeros"
            and policy.get("align_corners", 0) == 0,
            "unexpected attention interpolation",
        )
        value_reshape = graph.producer(sample.input[0], "Reshape")
        value_slice = graph.producer(value_reshape.input[0], "Slice")
        flat = graph.producer(value_slice.input[0], "Reshape")
        value_transpose = graph.producer(flat.input[0], "Transpose")
        steps = graph.constants.get(value_slice.input[4]) if len(value_slice.input) > 4 else np.ones(1)
        require(
            graph.shape(sample.input[0]) == (HEADS, HEAD_CHANNELS, height, width)
            and graph.shape(value_slice.output[0]) == (HEADS, HEAD_CHANNELS, height * width)
            and len(value_slice.input) >= 4
            and graph.constants.get(value_slice.input[3]) is not None
            and graph.constants[value_slice.input[3]].tolist() in ([2], [-1])
            and steps is not None
            and steps.tolist() == [1]
            and graph.shape(flat.output[0]) == (HEADS, HEAD_CHANNELS, TOKENS)
            and attrs(value_transpose).get("perm") == [0, 2, 3, 1]
            and graph.shape(value_transpose.input[0]) == (1, TOKENS, HEADS, HEAD_CHANNELS),
            f"unexpected attention value layout at level {level}",
        )
        values.add(value_transpose.input[0])
        spans.append((_constant_scalar(graph, value_slice.input[1]), _constant_scalar(graph, value_slice.input[2])))
        split = graph.producer(sample.input[1], "Split")
        grid_reshape = graph.producer(split.input[0], "Reshape")
        grid_transpose = graph.producer(grid_reshape.input[0], "Transpose")
        require(
            list(split.output).index(sample.input[1]) == level
            and attrs(split).get("axis") in (-2, 2)
            and len(split.input) == 2
            and graph.constants.get(split.input[1]) is not None
            and graph.constants[split.input[1]].tolist() == list(POINTS)
            and graph.shape(grid_reshape.output[0]) == (HEADS, QUERY_COUNT, total, 2)
            and attrs(grid_transpose).get("perm") == [0, 2, 1, 3, 4]
            and graph.shape(grid_transpose.input[0]) == (1, QUERY_COUNT, HEADS, total, 2),
            f"unexpected attention grid layout at level {level}",
        )
        grids.add(grid_transpose.input[0])
    require(len(values) == 1 and len(grids) == 1, "levels sample different value or grid tensors")
    expected = [(sum(h * w for h, w in LEVELS[:i]), sum(h * w for h, w in LEVELS[: i + 1])) for i in range(len(LEVELS))]
    require(spans == expected, "levels do not partition the attention memory in order")
    return samples, transpose, values.pop(), grids.pop(), weights


def per_head_attention(model):
    """Sample each attention head on its own, as the compiler needs batch-one GridSample.

    Upstream folds the eight heads into the GridSample batch. Each head samples, weights
    and reduces its own channels, so separating the heads reorders no arithmetic.
    """
    model = deepcopy(model)
    graph = index(model)
    concats = {}
    for node in model.graph.node:
        if node.op_type == "GridSample":
            reader = graph.sole_user(node, "Concat")
            require(attrs(reader).get("axis") in (-1, 3), "expected the sampled levels concatenated last")
            concats[reader.name] = reader
    require(len(concats) == DECODER_LAYERS, f"expected {DECODER_LAYERS} deformable-attention samplings")
    for concat in concats.values():
        samples, transpose, value, grid, weights = _attention_levels(graph, concat)
        base = transpose.name
        initializers = [_int64(f"{base}/head_{head}", head) for head in range(HEADS)]
        initializers += [_int64(f"{base}/axis_1", [1]), _int64(f"{base}/axis_2", [2]), _int64(f"{base}/last", [-1])]
        begin = 0
        for level, ((height, width), count) in enumerate(zip(LEVELS, POINTS, strict=True)):
            initializers += [
                _int64(f"{base}/level_{level}/first_token", [begin]),
                _int64(f"{base}/level_{level}/last_token", [begin + height * width]),
                _int64(f"{base}/level_{level}/first_point", [sum(POINTS[:level])]),
                _int64(f"{base}/level_{level}/last_point", [sum(POINTS[:level]) + count]),
                _int64(f"{base}/level_{level}/shape", [1, HEAD_CHANNELS, height, width]),
            ]
            begin += height * width
        nodes, heads = [], []
        for head in range(HEADS):
            prefix = f"{base}/head_{head}"
            nodes += [
                helper.make_node("Gather", [value, prefix], [f"{prefix}/value"], name=f"{prefix}/value", axis=2),
                helper.make_node(
                    "Transpose", [f"{prefix}/value"], [f"{prefix}/channels"], name=f"{prefix}/channels", perm=[0, 2, 1]
                ),
                helper.make_node("Gather", [grid, prefix], [f"{prefix}/grid"], name=f"{prefix}/grid", axis=2),
                helper.make_node("Gather", [weights, prefix], [f"{prefix}/weights"], name=f"{prefix}/weights", axis=2),
                helper.make_node(
                    "Unsqueeze", [f"{prefix}/weights", f"{base}/axis_1"], [f"{prefix}/weight"], name=f"{prefix}/weight"
                ),
            ]
            sampled = []
            for level, sample in enumerate(samples):
                name = f"{prefix}/level_{level}"
                span = f"{base}/level_{level}"
                nodes += [
                    helper.make_node(
                        "Slice",
                        [f"{prefix}/channels", f"{span}/first_token", f"{span}/last_token", f"{base}/axis_2"],
                        [f"{name}/tokens"],
                        name=f"{name}/tokens",
                    ),
                    helper.make_node(
                        "Reshape", [f"{name}/tokens", f"{span}/shape"], [f"{name}/map"], name=f"{name}/map"
                    ),
                    helper.make_node(
                        "Slice",
                        [f"{prefix}/grid", f"{span}/first_point", f"{span}/last_point", f"{base}/axis_2"],
                        [f"{name}/grid"],
                        name=f"{name}/grid",
                    ),
                    helper.make_node(
                        "GridSample",
                        [f"{name}/map", f"{name}/grid"],
                        [f"{name}/sampled"],
                        name=f"{name}/sampled",
                        **attrs(sample),
                    ),
                ]
                sampled.append(f"{name}/sampled")
            nodes += [
                helper.make_node("Concat", sampled, [f"{prefix}/sampled"], name=f"{prefix}/sampled", axis=-1),
                helper.make_node(
                    "Mul", [f"{prefix}/sampled", f"{prefix}/weight"], [f"{prefix}/weighted"], name=f"{prefix}/weighted"
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}/weighted", f"{base}/last"],
                    [f"{prefix}/reduced"],
                    name=f"{prefix}/reduced",
                    keepdims=0,
                ),
                helper.make_node(
                    "Transpose", [f"{prefix}/reduced"], [f"{prefix}/output"], name=f"{prefix}/output", perm=[0, 2, 1]
                ),
            ]
            heads.append(f"{prefix}/output")
        nodes.append(helper.make_node("Concat", heads, [transpose.output[0]], name=f"{base}/heads", axis=-1))
        _splice(model, transpose, nodes, initializers)
    return _prune(model)


def batch_preserving_integral(model):
    """Keep the batch axis through the Integral's softmax and projection.

    Upstream folds the queries and box sides into a (1200, 33) matrix, which the compiler
    would read as a batch. The same softmax and projection run on a (1, 1200, 33) view.
    """
    model = deepcopy(model)
    graph = index(model)
    rows = QUERY_COUNT * 4
    rewritten = 0
    for softmax in [n for n in model.graph.node if n.op_type == "Softmax"]:
        if graph.shape(softmax.input[0]) != (rows, BINS):
            continue
        require(attrs(softmax).get("axis", -1) in (1, -1), "unexpected Integral softmax axis")
        corners = graph.producer(softmax.input[0], "Reshape")
        matmul = graph.sole_user(softmax, "MatMul")
        folded = graph.sole_user(matmul, "Reshape")
        output = graph.sole_user(folded, "Reshape")
        project = matmul.input[1]
        require(
            graph.shape(corners.input[0]) == (1, QUERY_COUNT, 4 * BINS)
            and matmul.input[0] == softmax.output[0]
            and graph.shape(project) == (BINS,)
            and graph.shape(folded.output[0]) == (QUERY_COUNT, 4)
            and graph.shape(output.output[0]) == (1, QUERY_COUNT, 4),
            "unexpected Integral projection",
        )
        base = output.name
        nodes = [
            helper.make_node(
                "Reshape", [corners.input[0], f"{base}/rows"], [f"{base}/corners"], name=f"{base}/corners"
            ),
            helper.make_node(
                "Softmax", [f"{base}/corners"], [f"{base}/probabilities"], name=f"{base}/probabilities", axis=-1
            ),
            helper.make_node("Reshape", [project, f"{base}/column"], [f"{base}/project"], name=f"{base}/project"),
            helper.make_node(
                "MatMul", [f"{base}/probabilities", f"{base}/project"], [f"{base}/expected"], name=f"{base}/expected"
            ),
            helper.make_node(
                "Reshape", [f"{base}/expected", f"{base}/boxes"], [output.output[0]], name=f"{base}/sides"
            ),
        ]
        initializers = [
            _int64(f"{base}/rows", [1, rows, BINS]),
            _int64(f"{base}/column", [BINS, 1]),
            _int64(f"{base}/boxes", [1, QUERY_COUNT, 4]),
        ]
        _splice(model, output, nodes, initializers)
        rewritten += 1
    require(rewritten == DECODER_LAYERS, f"expected {DECODER_LAYERS} Integral projections")
    return _prune(model)


def bounded_reference_logits(model):
    """Replace the decoder's inverse-sigmoid reference chain, not arbitrary Log ops.

    AFE lowers Log to a calibrated INT8 lookup even under a BF16 policy. Its wide
    positive input range loses useful reference values. For this checked chain,
    clamp the original logits to +/- -log(eps). FP32 sigmoid saturation makes this
    an approximation, not an algebraically exact floating-point cancellation.
    """
    model = deepcopy(model)
    graph = index(model)
    producers, constants = graph.producers, graph.constants
    logs = [n for n in model.graph.node if n.op_type == "Log"]
    require(len(logs) == 1, "expected one reference inverse-sigmoid Log")
    log = logs[0]
    division = producers.get(log.input[0])
    require(division is not None and division.op_type == "Div", "expected inverse-sigmoid ratio")
    clips = [producers.get(x) for x in division.input]
    require(all(n is not None and n.op_type == "Clip" for n in clips), "expected clamped ratio operands")
    epsilon = np.float32(1e-5)
    for node in clips:
        require(
            len(node.input) >= 2
            and node.input[1] in constants
            and np.asarray(constants[node.input[1]]).size == 1
            and np.asarray(constants[node.input[1]]).item() == epsilon,
            "unexpected inverse-sigmoid epsilon",
        )
        if len(node.input) > 2 and node.input[2]:
            require(
                node.input[2] in constants and np.asarray(constants[node.input[2]]).item() == 1,
                "unexpected inverse-sigmoid upper bound",
            )
    sigmoid_clip = producers.get(clips[0].input[0])
    complement = producers.get(clips[1].input[0])
    require(sigmoid_clip is not None and sigmoid_clip.op_type == "Clip", "expected bounded sigmoid")
    require(
        complement is not None
        and complement.op_type == "Sub"
        and complement.input[1] == sigmoid_clip.output[0]
        and complement.input[0] in constants
        and np.asarray(constants[complement.input[0]]).item() == 1,
        "unexpected inverse-sigmoid complement",
    )
    require(
        len(sigmoid_clip.input) == 3
        and all(x in constants for x in sigmoid_clip.input[1:])
        and [np.asarray(constants[x]).item() for x in sigmoid_clip.input[1:]] == [0, 1],
        "unexpected sigmoid clamp",
    )
    sigmoid = producers.get(sigmoid_clip.input[0])
    require(
        sigmoid is not None and sigmoid.op_type == "Sigmoid" and list(sigmoid.input) == ["reference_logits"],
        "Log is not the reference-logit chain",
    )
    names = {name for n in model.graph.node for name in (*n.input, *n.output)} | set(constants)
    lower, upper = "dfine_reference_logit_min", "dfine_reference_logit_max"
    require(lower not in names and upper not in names, "reference bound name collision")
    bound = np.float32(-np.log(epsilon))
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(-bound, np.float32), lower),
            numpy_helper.from_array(np.array(bound, np.float32), upper),
        ]
    )
    replacement = helper.make_node(
        "Clip", ["reference_logits", lower, upper], list(log.output), name="bounded_reference_logits"
    )
    log.CopyFrom(replacement)
    # Prune only dead ancestors. Keep the exact public three-input/two-output ABI.
    return _prune(model)


def validate_decoder(decoder):
    """Check the rewritten decoder samples batch-one head maps with the upstream interpolation."""
    values = index(decoder).values
    samples = [node for node in decoder.graph.node if node.op_type == "GridSample"]
    require(len(samples) == DECODER_LAYERS * HEADS * len(LEVELS), "expected one GridSample per layer, head and level")
    maps = {(1, HEAD_CHANNELS, height, width) for height, width in LEVELS}
    for node in samples:
        dimensions = shape(values[node.input[0]]) if node.input[0] in values else None
        require(dimensions in maps, "unexpected attention geometry")
        policy = attrs(node)
        require(
            policy.get("mode", b"bilinear") == b"bilinear"
            and policy.get("padding_mode", b"zeros") == b"zeros"
            and policy.get("align_corners", 0) == 0,
            "unexpected attention interpolation",
        )


def prepare_model(source, destination):
    """Write the two component graphs: the split backbone and the rewritten decoder."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    backbone, decoder, _ = split_model(onnx.load(source))
    decoder = per_head_attention(decoder)
    decoder = batch_preserving_integral(decoder)
    decoder = bounded_reference_logits(decoder)
    validate_decoder(decoder)
    paths = [destination / f"{name}.onnx" for name in ("backbone", "decoder")]
    for model, path in zip((backbone, decoder), paths, strict=True):
        onnx.save(model, path)
    return tuple(paths)


def validate_bf16_regions(regions):
    if (
        not isinstance(regions, list)
        or not regions
        or not all(isinstance(region, str) for region in regions)
        or len(regions) != len(set(regions))
        or not set(regions) <= set(BF16_REGIONS)
    ):
        raise ValueError(f"Expected distinct BF16 regions from {BF16_REGIONS}")


def region_precision_map(graph, regions):
    """Freeze a compact mixed policy onto the exact prepared graph, as RF-DETR does.

    ``backbone`` is the HGNet feature extractor, not the encoder neck/projections.
    ``nonconv`` keeps other floating operations BF16, matching the INC screening
    domain. ``outputs`` also promotes numerical producers behind terminal views;
    marking only a terminal Concat does not establish its inputs' precision.
    """
    validate_bf16_regions(regions)
    model = onnx.load(graph)
    names = [node.name for node in model.graph.node]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("Expected unique named operations in the prepared backbone")
    floats = {
        value.name
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    }
    nodes = {
        node.name: "int8"
        for node in model.graph.node
        if node.op_type != "Constant" and (node.op_type == "Conv" or any(v in floats for v in node.output))
    }
    resolved = set()
    for node in model.graph.node:
        if node.name not in nodes:
            continue
        matches = set()
        if "/backbone/" in node.name:
            matches.add("backbone")
        if node.op_type != "Conv":
            matches.add("nonconv")
        if matches.intersection(regions):
            resolved.update(matches.intersection(regions))
            nodes[node.name] = "bfloat16"
    if "outputs" in regions:
        producers = {value: node for node in model.graph.node for value in node.output}
        pending, visited = [value.name for value in model.graph.output], set()
        while pending:
            node = producers.get(pending.pop())
            if node is None or node.name in visited:
                continue
            visited.add(node.name)
            if node.name in nodes:
                nodes[node.name] = "bfloat16"
                resolved.add("outputs")
            if node.op_type in LAYOUT_OPS:
                pending.extend(node.input if node.op_type == "Concat" else node.input[:1])
    if resolved != set(regions):
        raise ValueError(f"BF16 regions did not resolve: {sorted(set(regions) - resolved)}")
    if set(nodes.values()) != {"int8", "bfloat16"}:
        raise ValueError("BF16 regions must retain mixed arithmetic")
    return {"graph_sha256": sha256_file(graph)[0], "nodes": nodes}


def annotate_backbone(graph, precision_map):
    """Apply a frozen INC map to exactly the graph it was selected for, using the bundled exporter."""
    if (
        not isinstance(precision_map, dict)
        or set(precision_map) != {"graph_sha256", "nodes"}
        or precision_map["graph_sha256"] != sha256_file(graph)[0]
        or not isinstance(precision_map["nodes"], dict)
        or not precision_map["nodes"]
        or not all(isinstance(value, str) for value in precision_map["nodes"].values())
        or set(precision_map["nodes"].values()) != {"int8", "bfloat16"}
    ):
        raise ValueError("Expected a mixed INC precision map bound to this prepared backbone")
    model = onnx.load(graph)
    names = [node.name for node in model.graph.node]
    if len(names) != len(set(names)) or not set(precision_map["nodes"]) <= set(names):
        raise ValueError("INC precision map has unknown or ambiguous operation names")
    from neural_compressor.annotate_precision import annotate_graph

    # The bundled exporter groups fused chains but rebuilds the graph without value_info.
    # Replace only nodes, retaining the original shapes, metadata and public contract.
    annotated = annotate_graph(model.graph, precision_map["nodes"])
    model.graph.ClearField("node")
    model.graph.node.extend(annotated.node)
    model.opset_import.append(onnx.helper.make_opsetid("ai.sima", 1))
    onnx.checker.check_model(model)
    onnx.save(model, graph)
