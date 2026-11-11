"""Graph pruning and output comparison shared by the Supertonic surgery and compile recipe."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx


def prune(model: onnx.ModelProto, *, nodes: bool = True, inputs: bool = False) -> dict[str, int]:
    """Keep the nodes the graph outputs depend on and the initializers they read; with inputs, also drop the
    graph inputs nothing reads. Without nodes, keep every node and drop only initializers no node reads. Node
    order is kept."""

    if nodes:
        needed = {value.name for value in model.graph.output}
        kept_reversed: list[onnx.NodeProto] = []
        for node in reversed(model.graph.node):
            if any(output in needed for output in node.output):
                kept_reversed.append(node)
                needed.update(value for value in node.input if value)
        removed = {"nodes": len(model.graph.node) - len(kept_reversed)}
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept_reversed))
    else:
        needed = {value for node in model.graph.node for value in node.input if value}
        removed = {"nodes": 0}

    initializers = [value for value in model.graph.initializer if value.name in needed]
    removed["initializers"] = len(model.graph.initializer) - len(initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)

    if inputs:
        graph_inputs = [value for value in model.graph.input if value.name in needed]
        removed["inputs"] = len(model.graph.input) - len(graph_inputs)
        del model.graph.input[:]
        model.graph.input.extend(graph_inputs)
    return removed


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Shape, finiteness, and the error of candidate against reference, in float64; errors are None when the
    shapes differ or a value is not finite."""

    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    result: dict[str, Any] = {
        "shape": list(actual.shape),
        "shape_equal": expected.shape == actual.shape,
        "finite": bool(np.isfinite(expected).all() and np.isfinite(actual).all()),
    }
    if not result["shape_equal"] or not result["finite"]:
        return {**result, "cosine_similarity": None, "relative_l2": None, "mean_absolute_error": None,
                "max_absolute_error": None}
    delta = actual - expected
    tiny = np.finfo(np.float64).tiny
    expected_norm, actual_norm = float(np.linalg.norm(expected)), float(np.linalg.norm(actual))
    return {
        **result,
        "cosine_similarity": float(np.vdot(expected.ravel(), actual.ravel()) / max(expected_norm * actual_norm, tiny)),
        "relative_l2": float(np.linalg.norm(delta) / max(expected_norm, tiny)),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }
