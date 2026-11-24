"""Numerical fidelity of an ONNX build: compare the quantized graph with its ONNX source and write the report.

A model's scripts/compile.py imports this module when model.yaml declares numerical_fidelity, so it is a compile
fingerprint input of those builds, and an edit recompiles them. onnxruntime and the Model Compiler's afe package
are imported where they are used, so the registry can import the report layout without them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .model import load_model

REPORT_FILE = "fidelity.json"
# The model browser and sima-cli choose a renderer by it (schemas/numerical-fidelity.schema.json).
FIDELITY_SCHEMA_VERSION = 1


def compare_with_onnx(
    onnx_path: Path,
    quantized: Any,
    samples: Iterable[np.ndarray],
    input_name: str,
    output_names: Sequence[str],
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Run each sample through the ONNX source and the quantized graph, and pair their outputs.

    Args:
        onnx_path: The ONNX the quantized graph is compared with.
        quantized: The quantized Model Compiler model.
        samples: The preprocessed samples as the compiler takes them: float32, channels last, without the batch
            axis. The ONNX receives each one channels first.
        input_name: The graph's one input.
        output_names: The name the report gives each ONNX output, in the graph's output order.

    Returns:
        One entry per sample, in order. Each maps an output name to its ONNX value and its quantized value.

    Raises:
        ValueError: output_names does not name every ONNX output.
        RuntimeError: no quantized output, or more than one, matches an ONNX output.
    """
    import onnxruntime
    from afe.apis.defines import InputName

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    graph_outputs = [output.name for output in session.get_outputs()]
    if len(graph_outputs) != len(output_names):
        raise ValueError(
            f"{onnx_path.name} has the outputs {graph_outputs}, and the comparison names {list(output_names)}"
        )
    compared = []
    for sample in samples:
        expected = session.run(None, {input_name: sample.transpose(2, 0, 1)[None]})
        executed = quantized.execute(inputs={InputName(input_name): sample[None]}, use_jax=True)
        # Copied, because the compiler may reuse an output's storage on the next execute.
        actual = [np.array(output) for output in executed]
        compared.append(
            {name: (value, _matching_output(name, value, actual)) for name, value in zip(output_names, expected)}
        )
    return compared


def _matching_output(name: str, expected: np.ndarray, actual: list[np.ndarray]) -> np.ndarray:
    """The quantized output that holds the ONNX output: the one of its shape, or else the one of its size.

    The compiler returns its outputs in its own order and may keep axes of length one that the ONNX drops.
    """
    matches = [output for output in actual if output.shape == expected.shape] or [
        output for output in actual if output.size == expected.size
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{len(matches)} quantized outputs match {name} of shape {expected.shape}: "
            f"{[output.shape for output in actual]}"
        )
    return matches[0].reshape(expected.shape)


def write_fidelity_report(
    model_dir: Path,
    profile_id: str,
    compared: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
    *,
    reference: str = "ONNX source (ONNX Runtime)",
    candidate: str = "Quantized AFE graph (JAX execute)",
) -> dict[str, Any]:
    """Write the build's fidelity.json into output_dir, which must exist, and print each output's agreement.

    Args:
        model_dir: The model's folder. Its model.yaml declares numerical_fidelity.
        profile_id: The build profile that was compiled.
        compared: One entry per fidelity sample, as compare_with_onnx returns them.
        output_dir: The build's output directory.
        reference: What computed the expected values.
        candidate: What computed the actual values.

    Returns:
        The report it wrote.
    """
    model = load_model(model_dir)
    report = fidelity_report(
        model["id"], profile_id, model["numerical_fidelity"], compared, reference=reference, candidate=candidate
    )
    (output_dir / REPORT_FILE).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Printed, because the Model Compiler's logging setup can drop a recipe's INFO records.
    for metric in report["metrics"]:
        print(
            f"Fidelity {metric['name']}: cosine {metric['cosine_similarity']:.5f}, "
            f"normalized RMSE {metric['normalized_rmse']:.5f}",
            flush=True,
        )
    return report


def fidelity_report(
    model_id: str,
    profile_id: str,
    fidelity: Mapping[str, Any],
    compared: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    *,
    reference: str,
    candidate: str,
) -> dict[str, Any]:
    """The numerical fidelity report of one build, laid out as schemas/numerical-fidelity.schema.json requires.

    Each metric covers one output over every sample: the cosine similarity, the RMS error divided by the
    expected values' RMS, and the mean and the largest absolute error.

    Args:
        model_id: The model the build belongs to.
        profile_id: The build profile that was compiled.
        fidelity: The numerical_fidelity section of model.yaml.
        compared: One entry per fidelity sample, in selection order. Each maps an output name, in the order the
            report lists the metrics, to its expected value from the reference and its actual value from the
            candidate.
        reference: What computed the expected values, such as "ONNX source (ONNX Runtime)".
        candidate: What computed the actual values, such as "Quantized AFE graph (JAX execute)".

    Raises:
        ValueError: compared is empty, or its samples name different outputs.
        RuntimeError: an output's expected and actual values differ in shape or hold a non-finite value.
    """
    if not compared:
        raise ValueError("a fidelity report needs at least one compared sample")
    names = list(compared[0])
    for index, outputs in enumerate(compared):
        if list(outputs) != names:
            raise ValueError(f"fidelity sample {index} compares {list(outputs)}, the first sample {names}")
    return {
        "schema_version": FIDELITY_SCHEMA_VERSION,
        "model_id": model_id,
        "build_profile_id": profile_id,
        "status": fidelity["status"],
        "reference": reference,
        "candidate": candidate,
        "dataset": {key: fidelity["dataset"][key] for key in ("id", "variant", "sha256")},
        "selection": fidelity["selection"],
        "samples": len(compared),
        "metrics": [_output_metric(name, [outputs[name] for outputs in compared]) for name in names],
    }


def _output_metric(name: str, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """One output's agreement over every sample, summed one sample at a time in selection order."""
    count = 0
    dot = expected_squared = actual_squared = error_squared = error_absolute = max_absolute_error = 0.0
    for index, (expected_value, actual_value) in enumerate(pairs):
        expected = np.asarray(expected_value, dtype=np.float64)
        actual = np.asarray(actual_value, dtype=np.float64)
        if expected.shape != actual.shape:
            raise RuntimeError(f"{name} of fidelity sample {index} has shape {actual.shape}, expected {expected.shape}")
        if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
            raise RuntimeError(f"{name} of fidelity sample {index} contains non-finite values")
        error = actual - expected
        absolute = np.abs(error)
        count += expected.size
        dot += float(np.dot(expected.ravel(), actual.ravel()))
        expected_squared += float(np.sum(np.square(expected)))
        actual_squared += float(np.sum(np.square(actual)))
        error_squared += float(np.sum(np.square(error)))
        error_absolute += float(np.sum(absolute))
        max_absolute_error = max(max_absolute_error, float(np.max(absolute)))
    expected_rms = max(np.sqrt(expected_squared / count), np.finfo(np.float64).eps)
    norm = np.sqrt(expected_squared * actual_squared)
    # Two all-zero outputs agree. An all-zero output against a nonzero one has no direction, so it scores 0.
    cosine = dot / norm if norm else float(expected_squared == actual_squared)
    return {
        "name": name,
        "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
        "normalized_rmse": float(np.sqrt(error_squared / count) / expected_rms),
        "mean_absolute_error": float(error_absolute / count),
        "max_absolute_error": max_absolute_error,
    }
