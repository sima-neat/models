from pathlib import Path
import numpy as np

from model_registry.model_tests import measured_shortfalls


def test_accuracy_meets_expected(accuracy, model_dir: Path, build_profile):
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
    result = accuracy["datasets"]["coco-2017-val"]
    assert result["evaluated_samples"] == result["available_samples"] == 5000


def test_scores_are_probabilities_without_an_extra_sigmoid(load_script):
    accuracy = load_script("accuracy")
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    outputs[3][0, 20, 20, 7] = .75
    result = accuracy.decode(outputs, 42, 640, 640, np.arange(80))
    assert result.shape == (1, 7)
    assert result[0, 0] == 42 and result[0, -1] == 7
    assert result[0, -2] == .75


def test_probability_contract_rejects_logits(load_script):
    import pytest
    accuracy = load_script("accuracy")
    outputs = [np.zeros(shape, np.float32) for shape in accuracy.OUTPUT_SHAPES]
    outputs[3][0, 0, 0, 0] = -3
    with pytest.raises(ValueError, match="probabilities"):
        accuracy.decode(outputs, 1, 640, 640, np.arange(80))
