"""Publish RAFT-Stereo's pinned realtime checkpoint as the weights.pth that surgery.py builds the network from.

Conversion only changes the file format: princeton-vl saved raftstereo-realtime.pth from a DataParallel wrapper, so
every tensor is named under module.; the conversion drops that prefix, and surgery.py rebuilds and exports the network
for the MLA at compile time.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "raftstereo-realtime.pth"
WRAPPER = "module."


def validate_source(source: dict[str, Any]) -> None:
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def unwrap(state: dict[str, Any]) -> dict[str, Any]:
    """The network's state dict under its own module names, from the DataParallel checkpoint's."""
    wrapped = [name for name in state if not name.startswith(WRAPPER)]
    if not state or wrapped:
        raise ValueError(f"Expected every tensor of {CHECKPOINT} under {WRAPPER!r}; found {wrapped[:3]}")
    return {name.removeprefix(WRAPPER): value for name, value in state.items()}


def convert(checkpoint: Path, output_dir: Path) -> Path:
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    (weights,) = conversion_outputs(MODEL_DIR)
    output = output_dir / weights
    torch.save(unwrap(torch.load(checkpoint, map_location="cpu", weights_only=True)), output)
    print(f"Wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
