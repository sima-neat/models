"""Prepare four source files and bind the publisher's verified pins to model YAML.

Preparation checks interfaces and copies bytes, not checkpoint lineage or numerical
equivalence. Neither command uploads artifacts, approves a model or activates it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from model_registry.fetch import Fetcher, sha256_file
from model_registry.source import SourceSpec

sys.path.insert(0, str(Path(__file__).parent / "model/scripts"))
from validation import COMPONENTS, validate_graph  # noqa: E402

MODEL_ID = "lewam_pusht"
CHECKPOINT = "e649a703b224e4144fa068b73ae6ab3735cf60b81ed2ffe36898c1e26474d5fb"
SOURCE_REVISION = "83c47030a80cee28f638680b76998d2f6a410914"
RECEIPT = "publication-input.json"


def require_new(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)


def prepare(graphs: dict[str, Path], output: Path) -> dict:
    """Write publisher-ready files once, committing the directory only when complete."""
    require_new(output)
    if set(graphs) != set(COMPONENTS):
        raise ValueError("Exactly front, body, head, policy graphs are required")
    if any(path.is_symlink() for path in graphs.values()):
        raise ValueError("Source symlinks are not publication inputs")
    files = {name: {"path": f"{name}/model.onnx", **validate_graph(graphs[name], name)} for name in COMPONENTS}
    receipt = {
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT,
        "source_revision": SOURCE_REVISION,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lewam-publication-", dir=output.parent) as temporary:
        staged = Path(temporary) / "input"
        for name, row in files.items():
            target = staged / row["path"]
            target.parent.mkdir(parents=True)
            shutil.copyfile(graphs[name], target)
            if sha256_file(target) != (row["sha256"], row["size_bytes"]):
                raise ValueError(f"Source changed while preparing publication input: {name}")
        (staged / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        staged.rename(output)
    return receipt


def read_prepared(directory: Path) -> dict:
    """Verify the local input receipt against every graph before binding remote pins."""
    receipt = json.loads((directory / RECEIPT).read_text())
    if (receipt.get("model_id"), receipt.get("checkpoint_sha256"), receipt.get("source_revision")) != (
        MODEL_ID,
        CHECKPOINT,
        SOURCE_REVISION,
    ):
        raise ValueError("Unexpected prepared source identity")
    if set(receipt.get("files", {})) != set(COMPONENTS):
        raise ValueError("Exactly four prepared sources required")
    for name in COMPONENTS:
        relative = f"{name}/model.onnx"
        path = directory / relative
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Source symlinks are not publication inputs")
        if receipt["files"][name] != {"path": relative, **validate_graph(path, name)}:
            raise ValueError(f"Prepared source receipt mismatch: {name}")
    return receipt


def bind(directory: Path, pins: dict[str, dict], cache: Path, output: Path) -> dict:
    """Write an ordinary compilation.source fragment only after all downloads verify."""
    require_new(output)
    if set(pins) != set(COMPONENTS):
        raise ValueError("Exactly four publication pins required")
    receipt = read_prepared(directory)
    specs = {}
    for name in COMPONENTS:
        source = pins[name].get("source", {})
        if source.get("type") != "file":
            raise ValueError("Each publisher pin must describe one file")
        spec = SourceSpec.from_mapping(source)
        original = receipt["files"][name]
        if (spec.sha256, spec.size_bytes) != (original["sha256"], original["size_bytes"]):
            raise ValueError(f"Published source differs from prepared bytes: {name}")
        specs[name] = spec
    fetcher = Fetcher(cache)
    for spec in specs.values():
        fetcher.fetch(spec)
    fragment = {
        "compilation": {
            "source": {
                "main_file": "front.onnx",
                "files": {
                    f"{name}.onnx": {"uri": spec.uri, "sha256": spec.sha256, "size_bytes": spec.size_bytes}
                    for name, spec in specs.items()
                },
            }
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lewam-pins-", dir=output.parent) as temporary:
        staged = Path(temporary) / "source.yaml"
        staged.write_text(yaml.safe_dump(fragment, sort_keys=False))
        # A same-filesystem hard link commits the complete file without replacing a concurrent writer.
        os.link(staged, output)
    return fragment


def named_paths(values: list[str]) -> dict[str, Path]:
    pairs = [value.split("=", 1) for value in values]
    if any(len(pair) != 2 or not all(pair) for pair in pairs) or len(dict(pairs)) != len(pairs):
        raise ValueError("Duplicate or malformed component=path")
    return {name: Path(path) for name, path in pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--graph", action="append", required=True, help="component=local ONNX")
    prep.add_argument("--output", type=Path, required=True)
    binding = commands.add_parser("bind")
    binding.add_argument("--prepared", type=Path, required=True)
    binding.add_argument("--pin", action="append", required=True, help="component=returned source.yaml")
    binding.add_argument("--cache", type=Path, required=True)
    binding.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = named_paths(args.graph if args.command == "prepare" else args.pin)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "prepare":
        prepare(paths, args.output)
    else:
        bind(
            args.prepared,
            {name: yaml.safe_load(path.read_text()) for name, path in paths.items()},
            args.cache,
            args.output,
        )


if __name__ == "__main__":
    main()
