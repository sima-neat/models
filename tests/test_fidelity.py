"""Numerical fidelity of an ONNX build: tools/model_registry/fidelity.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from model_registry.fidelity import compare_with_onnx, fidelity_report, write_fidelity_report

ROOT = Path(__file__).resolve().parents[1]
FIDELITY_SCHEMA = ROOT / "schemas" / "numerical-fidelity.schema.json"
FIDELITY = {
    "status": "diagnostic",
    "dataset": {"id": "imagenet-2012", "variant": "cal", "sha256": "e" * 64, "size_bytes": 1},
    "selection": {"method": "seeded_random_without_replacement", "seed": 42, "offset": 20, "samples": 2},
    "outputs": ["logits", "boxes"],
}


class Quantized:
    """A quantized graph that returns the given outputs for each execute call."""

    def __init__(self, outputs: list[list[np.ndarray]]) -> None:
        self.outputs = iter(outputs)
        self.inputs: list[dict[str, np.ndarray]] = []

    def execute(self, inputs: dict[str, np.ndarray], use_jax: bool) -> list[np.ndarray]:
        assert use_jax
        self.inputs.append(inputs)
        return next(self.outputs)


@pytest.fixture
def onnx_source(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, np.ndarray]]:
    """Stand in for onnxruntime and afe: an ONNX with the outputs "scores" [1, 2] and "features" [1, 2, 3], whose
    values are the first input value. Returns the feeds the session received."""
    feeds: list[dict[str, np.ndarray]] = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            assert providers == ["CPUExecutionProvider"]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="scores"), SimpleNamespace(name="features")]

        def run(self, names: None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            feeds.append(feed)
            value = float(next(iter(feed.values())).flat[0])
            return [np.full((1, 2), value, np.float32), np.full((1, 2, 3), value, np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setitem(sys.modules, "afe", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "afe.apis", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "afe.apis.defines", SimpleNamespace(InputName=str))
    return feeds


def test_compare_pairs_each_onnx_output_with_the_quantized_output_that_holds_it(
    tmp_path: Path, onnx_source: list[dict[str, np.ndarray]]
) -> None:
    samples = [np.full((4, 5, 3), value, np.float32) for value in (1.0, 2.0)]
    # The compiler returns its outputs in its own order, and the scores with two more axes of length one.
    quantized = Quantized([[np.full((1, 2, 3), value), np.full((1, 1, 1, 2), value + 0.5)] for value in (1.0, 2.0)])

    compared = compare_with_onnx(tmp_path / "model.onnx", quantized, samples, "image", ["scores", "features"])

    # The ONNX takes each sample channels first, the quantized graph channels last, both with a batch axis.
    assert [feed["image"].shape for feed in onnx_source] == [(1, 3, 4, 5)] * 2
    assert [inputs["image"].shape for inputs in quantized.inputs] == [(1, 4, 5, 3)] * 2
    assert [list(sample) for sample in compared] == [["scores", "features"]] * 2
    expected, actual = compared[1]["scores"]
    np.testing.assert_array_equal(expected, np.full((1, 2), 2.0))
    np.testing.assert_array_equal(actual, np.full((1, 2), 2.5))
    assert compared[1]["features"][1].shape == (1, 2, 3)


@pytest.mark.parametrize(
    ("names", "outputs", "error", "message"),
    [
        (["scores"], [np.zeros((1, 2)), np.zeros((1, 2, 3))], ValueError, "has the outputs ['scores', 'features']"),
        (
            ["scores", "features"],
            [np.zeros((1, 2)), np.zeros((1, 7))],
            RuntimeError,
            "0 quantized outputs match features of shape (1, 2, 3): [(1, 2), (1, 7)]",
        ),
        (
            ["scores", "features"],
            [np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2, 3))],
            RuntimeError,
            "2 quantized outputs match scores of shape (1, 2)",
        ),
    ],
)
def test_compare_rejects_outputs_it_cannot_pair(
    tmp_path: Path,
    onnx_source: list[dict[str, np.ndarray]],
    names: list[str],
    outputs: list[np.ndarray],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error) as raised:
        sample = np.zeros((4, 5, 3), np.float32)
        compare_with_onnx(tmp_path / "model.onnx", Quantized([outputs]), [sample], "image", names)

    assert message in str(raised.value)


def test_write_fidelity_report_writes_the_models_declared_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    compared = [{"logits": (np.array([[3.0, 4.0]]), np.array([[3.0, 4.0]]))}] * 5

    report = write_fidelity_report(ROOT / "models" / "resnet_18", "w8a8", compared, tmp_path)

    assert json.loads((tmp_path / "fidelity.json").read_text(encoding="utf-8")) == report
    Draft202012Validator(json.loads(FIDELITY_SCHEMA.read_text(encoding="utf-8"))).validate(report)
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("resnet_18", "w8a8", 5)
    assert report["reference"] == "ONNX source (ONNX Runtime)"
    assert report["candidate"] == "Quantized AFE graph (JAX execute)"
    assert capsys.readouterr().out == "Fidelity logits: cosine 1.00000, normalized RMSE 0.00000\n"


def test_fidelity_report_sums_each_output_over_every_sample() -> None:
    compared = [
        {"logits": (np.array([3.0, 4.0]), np.array([3.0, 4.0])), "boxes": (np.array([[1.0, 1.0]]), np.zeros((1, 2)))},
        {"logits": (np.array([0.0, 0.0]), np.array([0.0, 2.0])), "boxes": (np.zeros((1, 2)), np.zeros((1, 2)))},
    ]

    report = fidelity_report(
        "fixture", "w8a8", FIDELITY, compared, reference="ONNX source (ONNX Runtime)", candidate="Quantized graph"
    )

    Draft202012Validator(json.loads(FIDELITY_SCHEMA.read_text(encoding="utf-8"))).validate(report)
    assert {key: value for key, value in report.items() if key != "metrics"} == {
        "schema_version": 1,
        "model_id": "fixture",
        "build_profile_id": "w8a8",
        "status": "diagnostic",
        "reference": "ONNX source (ONNX Runtime)",
        "candidate": "Quantized graph",
        "dataset": {"id": "imagenet-2012", "variant": "cal", "sha256": "e" * 64},
        "selection": FIDELITY["selection"],
        "samples": 2,
    }
    logits, boxes = report["metrics"]
    # logits: expected (3, 4, 0, 0), actual (3, 4, 0, 2). Error energy 4 over 4 values, expected energy 25.
    assert logits == {
        "name": "logits",
        "cosine_similarity": pytest.approx(25 / np.sqrt(25 * 29)),
        "normalized_rmse": pytest.approx(np.sqrt(4 / 4) / np.sqrt(25 / 4)),
        "mean_absolute_error": pytest.approx(2 / 4),
        "max_absolute_error": 2.0,
    }
    # boxes: the candidate is all zeros, so it has no direction to agree with. The largest error is in sample 0.
    assert boxes == {
        "name": "boxes",
        "cosine_similarity": 0.0,
        "normalized_rmse": pytest.approx(1.0),
        "mean_absolute_error": pytest.approx(2 / 4),
        "max_absolute_error": 1.0,
    }


@pytest.mark.parametrize(
    ("compared", "error", "message"),
    [
        ([{"logits": (np.zeros(2), np.zeros(2))}], None, None),
        ([], ValueError, "a fidelity report needs at least one compared sample"),
        (
            [{"logits": (np.ones(2), np.ones(2))}, {"boxes": (np.ones(2), np.ones(2))}],
            ValueError,
            "fidelity sample 1 compares ['boxes'], the first sample ['logits']",
        ),
        ([{"logits": (np.ones((1, 2)), np.ones(2))}], RuntimeError, "logits of fidelity sample 0 has shape (2,)"),
        ([{"logits": (np.ones(2), np.array([1.0, np.nan]))}], RuntimeError, "contains non-finite values"),
    ],
)
def test_fidelity_report_rejects_samples_it_cannot_compare(
    compared: list[dict[str, tuple[np.ndarray, np.ndarray]]], error: type[Exception] | None, message: str | None
) -> None:
    def report() -> dict[str, Any]:
        return fidelity_report("fixture", "w8a8", FIDELITY, compared, reference="ONNX", candidate="AFE")

    if error is None:
        # Two all-zero outputs agree exactly.
        assert report()["metrics"][0]["cosine_similarity"] == 1.0
    else:
        with pytest.raises(error) as raised:
            report()
        assert message in str(raised.value)
