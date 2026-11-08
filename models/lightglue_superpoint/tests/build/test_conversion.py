"""Checks of the conversion's match comparison, and of the converted ONNX's interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def _assignment(count: int, matched: int) -> np.ndarray:
    """An assignment in which the first `matched` keypoints of image 0 match image 1's in reverse order."""
    assignment = np.full((count, count), 1e-4, np.float32)
    for row in range(matched):
        assignment[row, count - 1 - row] = 0.9
    return assignment


def test_matches_are_mutual_best_pairs_above_the_threshold(converter: ModuleType) -> None:
    assignment = _assignment(8, 3)
    assignment[5, 0] = 0.05  # row 5's best pair, below the threshold
    assignment[6, 1], assignment[7, 1] = 0.5, 0.6  # column 1's best is row 7, so row 6's best pair is not mutual

    assert converter.matches(assignment).tolist() == [[0, 7], [1, 6], [2, 5], [7, 1]]


def test_the_comparison_accepts_rounding_and_rejects_other_matches(converter: ModuleType) -> None:
    count = converter.KEYPOINTS
    expected = _assignment(count, 300)

    converter.compare_assignment(expected, expected + 5e-4)
    moved = expected.copy()
    moved[:10] = np.roll(moved[:10], 1, axis=1)  # ten keypoints match the wrong partner
    with pytest.raises(AssertionError, match="assignment"):
        converter.compare_assignment(expected, moved)
    with pytest.raises(ValueError, match="only 100 matches"):
        converter.compare_assignment(_assignment(count, 100), _assignment(count, 100))
    nan = expected.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        converter.compare_assignment(expected, nan)


def test_the_synthetic_pair_shares_most_keypoints(converter: ModuleType) -> None:
    (pixels0, descriptors0), (pixels1, descriptors1) = converter.sample_pair(123)

    assert pixels0.shape == pixels1.shape == (converter.KEYPOINTS, 2)
    np.testing.assert_allclose(np.linalg.norm(descriptors1, axis=1), 1, atol=1e-5)
    # Each shared keypoint's descriptor stays close to its partner's.
    similarity = descriptors1 @ descriptors0.T
    assert np.sum(similarity.max(1) > 0.9) >= converter.KEYPOINTS * 3 // 4
    normalized = converter.normalized(np.asarray([[0, 0], [640, 480], [320, 240]], np.float32))
    np.testing.assert_allclose(normalized, [[-1, -0.75], [1, 0.75], [0, 0]])


def test_the_converted_matcher_has_the_static_interface(converted: Path, converter: ModuleType) -> None:
    converter.validate_export(converted)
    model = onnx.load(converted, load_external_data=False)
    assert [node.op_type for node in model.graph.node].count("Softmax") >= 2


def test_the_rotary_embedding_scale_and_mask_are_folded_into_the_projections(converted: Path) -> None:
    """The self-attention folds leave no rotate_half (step-2 slices and negations) and no all-zero mask Add."""
    model = onnx.load(converted, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    steps = [
        onnx.numpy_helper.to_array(initializers[node.input[4]]).tolist()
        for node in model.graph.node
        if node.op_type == "Slice" and len(node.input) > 4 and node.input[4] in initializers
    ]
    zeros = {name for name, tensor in initializers.items() if not onnx.numpy_helper.to_array(tensor).any()}
    zero_adds = [node for node in model.graph.node if node.op_type == "Add" and zeros.intersection(node.input)]

    assert "Neg" not in {node.op_type for node in model.graph.node}
    assert all(step == [1] for step in steps)
    assert zero_adds == []
