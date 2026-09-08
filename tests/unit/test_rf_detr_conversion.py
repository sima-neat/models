from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rf_detr_convert", ROOT / "models/rf_detr_small/convert.py")
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def outputs() -> list[np.ndarray]:
    return [np.zeros(converter.INTERFACE[name], dtype=np.float32) for name in converter.OUTPUTS]


def test_comparison_rejects_proposal_permutation() -> None:
    reference = outputs()
    reference[0][0, 0, 0] = 0.2
    reference[0][0, 1, 0] = 0.8
    candidate = [value.copy() for value in reference]
    candidate[0][:, [0, 1]] = candidate[0][:, [1, 0]]

    records = converter.compare_outputs("real-image", reference, candidate)

    assert not records[0]["passed"]
    assert records[0]["max_abs_error"] > 0.5
    assert records[0]["atol"] == records[0]["rtol"] == 1e-3


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_comparison_rejects_matching_nonfinite_values(invalid: float) -> None:
    reference = outputs()
    reference[0][0, 0, 0] = invalid

    records = converter.compare_outputs("nonfinite", reference, reference)

    assert not records[0]["passed"]
    assert not records[0]["finite"]
    json.dumps(records, allow_nan=False)


def test_comparison_rejects_broadcastable_wrong_shape() -> None:
    reference = outputs()
    candidate = [reference[0][:, :1], reference[1]]
    assert not converter.compare_outputs("shape", reference, candidate)[0]["passed"]


def test_comparison_records_small_finite_error() -> None:
    reference = outputs()
    candidate = [value + 1e-5 for value in reference]

    records = converter.compare_outputs("sample", reference, candidate)

    assert all(record["passed"] for record in records)
    assert all(0 < record["max_abs_error"] < 1e-3 for record in records)


def test_onnx_interface_requires_output_names_in_order(tmp_path: Path) -> None:
    values = {
        name: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    }
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(
                name,
                TensorProto.FLOAT,
                converter.INTERFACE[name],
                np.zeros(converter.INTERFACE[name]).ravel(),
            ),
        )
        for name in converter.OUTPUTS
    ]
    graph = helper.make_graph(nodes, "fixture", [values["input"]], [values["labels"], values["dets"]])
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)

    with pytest.raises(RuntimeError, match="dets then labels"):
        converter.validate_export(path)


def test_required_real_image_is_not_silently_skipped(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="validation.png"):
        converter._images(tmp_path, tmp_path)


def test_failed_comparison_writes_failure_evidence_and_rejects_reuse(tmp_path: Path, monkeypatch) -> None:
    def failed_export(source, destination, report):
        reference = outputs()
        candidate = [reference[0] + 0.5, reference[1]]
        report["comparisons"] = converter.compare_outputs("real-image", reference, candidate)
        raise RuntimeError("ONNX differs from unmodified PyTorch")

    monkeypatch.setattr(converter, "export_model", failed_export)
    destination = tmp_path / "output"
    with pytest.raises(RuntimeError, match="differs"):
        converter.convert(tmp_path / "model.safetensors", destination)
    report = json.loads((destination / "validation.json").read_text())
    assert report["status"] == "failed"
    assert not report["comparisons"][0]["passed"]
    with pytest.raises(FileExistsError):
        converter.convert(tmp_path / "model.safetensors", destination)
    assert json.loads((destination / "validation.json").read_text()) == report
