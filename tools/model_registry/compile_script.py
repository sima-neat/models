"""The registry formats a model's scripts/compile.py shares: its command line, its build parameters, the MPK
manifest, and the numerical fidelity report.

The recipe stays in the model: quantization, the compile call, calibration, graph surgery, running the source and
the quantized graph for fidelity, and the MPK audit. Every ONNX compile imports this module, so it is in
COMPILE_TOOLING and an edit recompiles every ONNX build. It imports only what the Model Compiler environment has.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .model import load_model, profile_component_for

# The model browser and sima-cli choose a renderer by it (schemas/numerical-fidelity.schema.json).
FIDELITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompileArguments:
    """What scripts/run_model_compilation.sh passes to compile.py, with each path resolved."""

    model: Path
    output_dir: Path
    build_profile: str
    dataset: Path | None
    fidelity_dataset: Path | None


def parse_arguments(description: str, model_dir: Path, argv: Sequence[str] | None = None) -> CompileArguments:
    """Parse the ONNX recipe command line and stop, through parser.error, on one the runner would not pass.

    --model must be an existing .onnx file, each dataset given must exist, and --fidelity-dataset is required
    when the model's model.yaml declares numerical_fidelity.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, required=True, help="the verified ONNX source")
    parser.add_argument("--output-dir", type=Path, required=True, help="the build's output directory")
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True, help="the profile's one invocation, named after it")
    parser.add_argument("--dataset", type=Path, help="dataset-registry calibration archive")
    parser.add_argument("--fidelity-dataset", type=Path, help="dataset-registry fidelity archive")
    args = parser.parse_args(argv)
    if args.compile_invocation != args.build_profile:
        parser.error("a build profile compiles in one invocation named after it")

    # Check the suffix before resolving, so a model.onnx symbolic link to a file without one passes.
    if args.model.suffix != ".onnx":
        parser.error(f"--model must be the verified ONNX source, not {args.model}")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    if dataset is not None and not dataset.exists():
        parser.error(f"dataset does not exist: {dataset}")
    if args.fidelity_dataset is None and load_model(model_dir).get("numerical_fidelity") is not None:
        parser.error("model.yaml declares numerical_fidelity; pass --fidelity-dataset")
    fidelity_dataset = args.fidelity_dataset.expanduser().resolve() if args.fidelity_dataset else None
    if fidelity_dataset is not None and not fidelity_dataset.exists():
        parser.error(f"fidelity dataset does not exist: {fidelity_dataset}")
    output_dir = args.output_dir.expanduser().resolve()
    return CompileArguments(model, output_dir, args.build_profile, dataset, fidelity_dataset)


def component_parameters(model_dir: Path, profile_id: str, component_id: str) -> dict[str, Any]:
    """One component's parameters in a build profile, as model.yaml declares them.

    Raises:
        ValidationError: model.yaml declares no such build profile, or the profile has no such component.
    """
    _, binding = profile_component_for(load_model(model_dir), profile_id, component_id)
    return binding["parameters"]


def read_mpk_manifest(archive: Path) -> tuple[list[str], dict[str, Any]]:
    """The member names of an MPK archive and its one _mpk.json manifest, refusing a link or a path outside it."""
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe MPK member: {member.name}")
        names = [member.name for member in members]
        manifests = [name for name in names if name.endswith("_mpk.json")]
        if len(manifests) != 1:
            raise RuntimeError(f"expected one MPK manifest, found {manifests}")
        source = stream.extractfile(manifests[0])
        if source is None:
            raise RuntimeError(f"cannot read MPK manifest: {manifests[0]}")
        return names, json.load(source)


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
