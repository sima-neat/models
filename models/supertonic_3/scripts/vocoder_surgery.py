#!/usr/bin/env python3
"""Create the fixed-shape, MLA-oriented Supertonic vocoder graph.

The upstream graph uses rank-three Conv1D activations and expresses the
144-channel-to-24-channel temporal expansion as Reshape/Transpose/Reshape.
For the fixed 192-latent deployment profile, that shuffle is exactly a
DepthToSpace(blocksize=6, mode=CRD). Keeping its six-row height in the first
convolution lets that convolution consume the shuffle without another reshape.

The final waveform Transpose/Reshape is deliberately removed. The graph emits
logical NCHW frames [1, 512, 1, 1152]. When the public output is tessellated as
HWC/HWC16, its bytes are already ordered as [time, 512] and may be consumed as
one flat PCM waveform without a host transpose.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from graph_tools import compare, prune
from onnx import TensorProto, helper, numpy_helper

VOCODER_SOURCE_SHA256 = "085de76dd8e8d5836d6ca66826601f615939218f90e519f70ee8a36ed2a4c4ba"
BATCH_SIZE = 1
LATENT_CHANNELS = 144
LATENT_LENGTH = 192
CHUNK_COMPRESS_FACTOR = 6
AE_LATENT_CHANNELS = LATENT_CHANNELS // CHUNK_COMPRESS_FACTOR
EXPANDED_LENGTH = LATENT_LENGTH * CHUNK_COMPRESS_FACTOR
PCM_PER_FRAME = 512


def _replace_ints_attribute(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, values))


def _set_shape(value_info: onnx.ValueInfoProto, dimensions: list[int]) -> None:
    tensor_type = value_info.type.tensor_type
    del tensor_type.shape.dim[:]
    for size in dimensions:
        tensor_type.shape.dim.add().dim_value = size


def _initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _replace_initializer(graph: onnx.GraphProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(graph.initializer):
        if initializer.name == name:
            graph.initializer[index].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(value), name=name))
            return
    raise KeyError(f"initializer not found: {name}")


def _replace_edge_pads(graph: onnx.GraphProto) -> int:
    """Replace fixed temporal edge padding with exact Slice/Concat replication."""

    constants = {
        "vocoder.edge.left.starts": np.asarray([0], dtype=np.int64),
        "vocoder.edge.left.ends": np.asarray([1], dtype=np.int64),
        "vocoder.edge.right.starts": np.asarray([EXPANDED_LENGTH - 1], dtype=np.int64),
        "vocoder.edge.right.ends": np.asarray([EXPANDED_LENGTH], dtype=np.int64),
        "vocoder.edge.axes": np.asarray([3], dtype=np.int64),
        "vocoder.edge.steps": np.asarray([1], dtype=np.int64),
    }
    for name, value in constants.items():
        graph.initializer.append(numpy_helper.from_array(value, name=name))

    initializers = _initializer_map(graph)
    rewritten: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "Pad":
            rewritten.append(node)
            continue
        mode = next(
            (attribute.s for attribute in node.attribute if attribute.name == "mode"),
            b"constant",
        )
        pads = numpy_helper.to_array(initializers[node.input[1]]).astype(np.int64)
        if mode != b"edge" or pads.shape != (8,):
            raise ValueError(f"unexpected Pad contract for {node.name}")
        if np.any(pads[[0, 1, 2, 4, 5, 6]] != 0):
            raise ValueError(f"only temporal edge padding is supported: {node.name}")
        left_width, right_width = int(pads[3]), int(pads[7])
        left = f"{node.output[0]}/left_edge"
        right = f"{node.output[0]}/right_edge"
        rewritten.extend(
            [
                helper.make_node(
                    "Slice",
                    [
                        node.input[0],
                        "vocoder.edge.left.starts",
                        "vocoder.edge.left.ends",
                        "vocoder.edge.axes",
                        "vocoder.edge.steps",
                    ],
                    [left],
                    name=f"{node.name}/left_edge",
                ),
                helper.make_node(
                    "Slice",
                    [
                        node.input[0],
                        "vocoder.edge.right.starts",
                        "vocoder.edge.right.ends",
                        "vocoder.edge.axes",
                        "vocoder.edge.steps",
                    ],
                    [right],
                    name=f"{node.name}/right_edge",
                ),
                helper.make_node(
                    "Concat",
                    [left] * left_width + [node.input[0]] + [right] * right_width,
                    list(node.output),
                    axis=3,
                    name=f"{node.name}/edge_concat",
                ),
            ]
        )
        count += 1
    del graph.node[:]
    graph.node.extend(rewritten)
    return count


def staticize_and_simplify(source: Path) -> onnx.ModelProto:
    from onnxsim import simplify

    model = onnx.load(source.as_posix())
    simplified, check = simplify(
        model,
        check_n=0,
        overwrite_input_shapes={"latent": [BATCH_SIZE, LATENT_CHANNELS, LATENT_LENGTH]},
    )
    if not check:
        raise RuntimeError("onnxsim did not validate the fixed vocoder graph")
    return simplified


def optimize(model: onnx.ModelProto, output: Path) -> dict[str, Any]:
    graph = model.graph
    original_counts = Counter(node.op_type for node in graph.node)
    nodes_by_name = {node.name: node for node in graph.node}

    required_nodes = {
        "/Div",
        "/Reshape",
        "/Transpose",
        "/Reshape_1",
        "/Mul",
        "/decoder/embed/net/Conv",
        "/decoder/head/layer2/Conv",
        "/decoder/head/Transpose",
        "/decoder/head/Reshape",
    }
    missing = required_nodes - nodes_by_name.keys()
    if missing:
        raise ValueError(f"unexpected vocoder topology; missing nodes: {sorted(missing)}")

    if len(graph.input) != 1 or graph.input[0].name != "latent":
        raise ValueError("expected one upstream input named 'latent'")
    _set_shape(graph.input[0], [BATCH_SIZE, LATENT_CHANNELS, 1, LATENT_LENGTH])

    # Replace the rank-changing input pixel shuffle with a native 4D operator.
    removed_names = {
        "/Reshape",
        "/Transpose",
        "/Reshape_1",
        "/decoder/head/Transpose",
        "/decoder/head/Reshape",
    }
    div = nodes_by_name["/Div"]
    depth_to_space_output = "/vocoder/DepthToSpace_output_0"
    depth_to_space = helper.make_node(
        "DepthToSpace",
        inputs=[div.output[0]],
        outputs=[depth_to_space_output],
        name="/vocoder/DepthToSpace",
        blocksize=CHUNK_COMPRESS_FACTOR,
        mode="CRD",
    )
    nodes_by_name["/Mul"].input[0] = depth_to_space_output

    rewritten_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.name in removed_names:
            continue
        rewritten_nodes.append(node)
        if node.name == "/Div":
            rewritten_nodes.append(depth_to_space)
    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    # Affine parameters follow DepthToSpace's [N, 4, 6, T] representation.
    initializers = _initializer_map(graph)
    for name in ("tts.ae.latent_std", "tts.ae.latent_mean"):
        value = numpy_helper.to_array(initializers[name])
        _replace_initializer(
            graph,
            name,
            value.reshape(BATCH_SIZE, 4, CHUNK_COMPRESS_FACTOR, 1),
        )

    # Lift every Conv1D to Conv2D. The embedding convolution consumes the
    # [4, 6] DepthToSpace channel/height pair with a 6x7 kernel.
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        initializers = _initializer_map(graph)
        weight = numpy_helper.to_array(initializers[node.input[1]])
        if node.name == "/decoder/embed/net/Conv":
            if list(weight.shape[1:]) != [AE_LATENT_CHANNELS, 7]:
                raise ValueError(f"unexpected embedding weight shape: {weight.shape}")
            weight_4d = weight.reshape(weight.shape[0], 4, 6, 7)
            kernel_shape = [6, 7]
        else:
            if weight.ndim != 3:
                raise ValueError(f"expected Conv1D weights for {node.name}: {weight.shape}")
            weight_4d = np.expand_dims(weight, axis=2)
            kernel_shape = [1, int(weight.shape[-1])]
        _replace_initializer(graph, node.input[1], weight_4d)

        attributes = {attribute.name: attribute for attribute in node.attribute}
        dilation = list(attributes["dilations"].ints)
        stride = list(attributes["strides"].ints)
        _replace_ints_attribute(node, "kernel_shape", kernel_shape)
        _replace_ints_attribute(node, "dilations", [1, int(dilation[0])])
        _replace_ints_attribute(node, "strides", [1, int(stride[0])])
        _replace_ints_attribute(node, "pads", [0, 0, 0, 0])

    # Lift NCT <-> NTC LayerNorm layout changes to NCHW <-> NTHC. These are
    # the only transposes retained because they define channel-wise LayerNorm.
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        old_perm = list(attributes["perm"].ints)
        if old_perm != [0, 2, 1]:
            raise ValueError(f"unexpected retained transpose {node.name}: {old_perm}")
        _replace_ints_attribute(node, "perm", [0, 3, 2, 1])

    # Lift temporal Pad from NCW to NCHW while preserving edge semantics.
    converted_pad_initializers: set[str] = set()
    for node in graph.node:
        if node.op_type != "Pad":
            continue
        pads_name = node.input[1]
        if pads_name in converted_pad_initializers:
            continue
        initializers = _initializer_map(graph)
        pads = numpy_helper.to_array(initializers[pads_name]).astype(np.int64)
        if pads.shape != (6,):
            raise ValueError(f"unexpected rank-three pads for {node.name}: {pads}")
        pads_4d = np.asarray(
            [pads[0], pads[1], 0, pads[2], pads[3], pads[4], 0, pads[5]],
            dtype=np.int64,
        )
        _replace_initializer(graph, pads_name, pads_4d)
        converted_pad_initializers.add(pads_name)

    edge_pads_replaced = _replace_edge_pads(graph)

    # Per-channel residual scale was [C, 1] for NCW broadcasting.
    for initializer in list(graph.initializer):
        if not initializer.name.endswith(".gamma"):
            continue
        value = numpy_helper.to_array(initializer)
        if value.shape == (1, 512, 1):
            _replace_initializer(graph, initializer.name, value.reshape(1, 512, 1, 1))

    # Expose the final NCHW convolution result. HWC/HWC16 tessellation makes
    # its raw bytes time-major and therefore identical to the removed flatten.
    head = next(node for node in graph.node if node.name == "/decoder/head/layer2/Conv")
    head.output[0] = "wav_frames"
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                "wav_frames",
                TensorProto.FLOAT,
                [BATCH_SIZE, PCM_PER_FRAME, 1, EXPANDED_LENGTH],
            )
        ]
    )

    # The edge-pad rewrite leaves its replaced Slice nodes in place; only unused initializers go.
    removed_initializers = prune(model, nodes=False)["initializers"]
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)

    activation_shapes: dict[str, list[int]] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        activation_shapes[value_info.name] = [int(dimension.dim_value) for dimension in tensor_type.shape.dim]
    node_outputs = {name for node in model.graph.node for name in node.output if name}
    non_4d = {
        name: activation_shapes[name]
        for name in node_outputs
        if name in activation_shapes and len(activation_shapes[name]) != 4
    }
    if non_4d:
        raise ValueError(f"non-4D data activations remain: {non_4d}")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output.as_posix())
    final_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "status": "passed",
        "source_profile": {
            "input": [BATCH_SIZE, LATENT_CHANNELS, LATENT_LENGTH],
            "output": [BATCH_SIZE, LATENT_LENGTH * CHUNK_COMPRESS_FACTOR * PCM_PER_FRAME],
        },
        "compiled_contract": {
            "input": [BATCH_SIZE, LATENT_CHANNELS, 1, LATENT_LENGTH],
            "output": [BATCH_SIZE, PCM_PER_FRAME, 1, EXPANDED_LENGTH],
            "output_physical_order": "HWC/HWC16 => time-major 512-sample frames",
        },
        "nodes_before": dict(sorted(original_counts.items())),
        "nodes_after": dict(sorted(final_counts.items())),
        "transposes_retained_for_layer_norm": final_counts["Transpose"],
        "reshape_nodes": final_counts["Reshape"],
        "depth_to_space_nodes": final_counts["DepthToSpace"],
        "edge_pads_replaced": edge_pads_replaced,
        "unused_initializers_removed": removed_initializers,
        "all_data_activations_rank4": True,
        "output": str(output),
    }


def validate_equivalence(source: Path, optimized: Path, reference_cases: list[Path]) -> dict[str, Any]:
    """Compare source waveform bytes with flattened HWC-order output frames."""

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    source_session = ort.InferenceSession(source.as_posix(), options, providers=["CPUExecutionProvider"])
    optimized_session = ort.InferenceSession(optimized.as_posix(), options, providers=["CPUExecutionProvider"])

    comparisons: list[dict[str, Any]] = []
    for case_path in reference_cases:
        with np.load(case_path) as case:
            latent = np.asarray(case["final_latent_natural"], dtype=np.float32)
        natural_length = int(latent.shape[-1])
        if natural_length > LATENT_LENGTH:
            raise ValueError(f"{case_path.name} latent length {natural_length} exceeds {LATENT_LENGTH}")
        latent = np.pad(
            latent,
            ((0, 0), (0, 0), (0, LATENT_LENGTH - natural_length)),
        ).astype(np.float32)

        source_waveform = source_session.run(None, {"latent": latent})[0]
        frames = optimized_session.run(None, {"latent": latent[:, :, None, :]})[0]
        flat_waveform = frames.transpose(0, 2, 3, 1).reshape(BATCH_SIZE, -1)
        comparison = {
            "case": case_path.stem,
            "natural_latent_length": natural_length,
            "source_output_shape": list(source_waveform.shape),
            "optimized_output_shape": list(frames.shape),
            **compare(source_waveform, flat_waveform),
        }
        comparison["passed"] = bool(
            comparison["shape_equal"]
            and comparison["finite"]
            and comparison["max_absolute_error"] <= 3e-5
            and comparison["relative_l2"] <= 1e-5
            and comparison["cosine_similarity"] >= 0.999999999
        )
        comparisons.append(comparison)

    if not all(comparison["passed"] for comparison in comparisons):
        raise ValueError(f"vocoder equivalence validation failed: {comparisons}")
    return {
        "status": "passed",
        "thresholds": {
            "max_absolute_error": 3e-5,
            "relative_l2": 1e-5,
            "min_cosine_similarity": 0.999999999,
        },
        "comparisons": comparisons,
        "max_absolute_error": max(comparison["max_absolute_error"] for comparison in comparisons),
        "max_relative_l2": max(comparison["relative_l2"] for comparison in comparisons),
        "min_cosine_similarity": min(comparison["cosine_similarity"] for comparison in comparisons),
    }
