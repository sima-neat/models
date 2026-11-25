"""Tests that the operator audit and the shape freeze of the neat-compile-model skill run as its SKILL.md calls them."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "neat-compile-model" / "scripts"
DATABASE = json.loads((ROOT / "skills" / "neat-compile-model" / "data" / "supported_operators.json").read_text())


def _write_model(path: Path, op_type: str, batch: str | int = 1, output_shape: list | None = None) -> Path:
    """Write a one-node opset 17 model that applies op_type to a [batch, 3, 8, 8] input."""
    shape = [batch, 3, 8, 8]
    graph = helper.make_graph(
        [helper.make_node(op_type, ["x"], ["y"])],
        "skill_script_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape or shape)],
    )
    # IR 8 is opset 17's version. onnx 1.23 writes IR 14 by default, which the checker in onnxsim 0.7.3 rejects.
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), path)
    return path


def _run(script: str, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
    # onnx_static_simplify.py runs the onnxsim command, which installs beside this Python.
    environment = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    command = [sys.executable, str(SCRIPTS / script), *map(str, arguments)]
    return subprocess.run(command, capture_output=True, text=True, env=environment, check=False)


def _audit(model: Path, dtype: str) -> tuple[int, dict]:
    result = _run("model_surgery_guard.py", "audit-model", "--model", model, "--dtype", dtype, "--json")
    return result.returncode, json.loads(result.stdout)


def test_audit_passes_a_graph_of_supported_operators(tmp_path: Path) -> None:
    code, report = _audit(_write_model(tmp_path / "relu.onnx", "Relu"), "int8")

    assert code == 0
    assert report["release"] == DATABASE["release"]
    assert [entry["operator"] for entry in report["supported"]] == ["Relu"]


def test_audit_fails_an_operator_the_database_does_not_list(tmp_path: Path) -> None:
    assert "Celu" not in DATABASE["operators"]

    code, report = _audit(_write_model(tmp_path / "celu.onnx", "Celu"), "int8")

    assert code == 2
    assert [entry["operator"] for entry in report["unknown"]] == ["Celu"]


def test_audit_fails_an_operator_unsupported_at_the_requested_precision(tmp_path: Path) -> None:
    assert DATABASE["operators"]["ReduceLogSum"]["bfloat16"] != "Y"

    code, report = _audit(_write_model(tmp_path / "reduce.onnx", "ReduceLogSum", output_shape=[1, 1, 1, 1]), "bfloat16")

    assert code == 2
    assert [entry["operator"] for entry in report["unsupported"]] == ["ReduceLogSum"]


def test_query_op_prints_an_entry_and_fails_an_unlisted_operator() -> None:
    listed = _run("model_surgery_guard.py", "query-op", "--op", "Relu")
    unlisted = _run("model_surgery_guard.py", "query-op", "--op", "Celu")

    assert listed.returncode == 0
    assert json.loads(listed.stdout)["operator"] == "Relu"
    assert unlisted.returncode == 1


def test_static_simplify_freezes_a_symbolic_batch(tmp_path: Path) -> None:
    if shutil.which("onnxsim", path=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}") is None:
        pytest.skip("onnxsim is not installed")
    source = _write_model(tmp_path / "dynamic.onnx", "Relu", batch="batch")
    frozen = tmp_path / "static.onnx"

    result = _run("onnx_static_simplify.py", "--input", source, "--output", frozen, "--replace", "batch=1")

    assert result.returncode == 0, result.stderr
    dims = onnx.load(frozen).graph.input[0].type.tensor_type.shape.dim
    assert [dim.dim_value for dim in dims] == [1, 3, 8, 8]
