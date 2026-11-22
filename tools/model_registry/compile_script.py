"""What every model's scripts/compile.py shares: its command line, its build parameters, the MPK manifest, and the
log level it passes to the Model Compiler.

The recipe stays in the model: quantization, the compile call, calibration preprocessing, graph surgery, and the
MPK audit. Every ONNX compile imports this module, so it is in COMPILE_TOOLING and an edit recompiles every ONNX
build. It imports only what the Model Compiler environment has. fidelity.py holds the numerical fidelity
measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .model import load_model, profile_component_for


def compiler_log_level() -> int:
    """The log level a recipe passes to the Model Compiler.

    DEBUG in a GitHub Actions job, so a failed CI compile leaves its full log. INFO anywhere else, so a compile on
    a developer's machine stays readable.
    """
    return logging.DEBUG if os.environ.get("GITHUB_ACTIONS") == "true" else logging.INFO


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
