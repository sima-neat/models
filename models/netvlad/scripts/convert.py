"""Publish Patch-NetVLAD's pitts_WPCA4096 checkpoint as the float32 weights.pth that surgery.py builds NetVLAD from.

Conversion only changes the file format: surgery.py rebuilds the network for the MLA at compile time.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "pitts_WPCA4096.pth.tar"
FEATURE_DIM = 512
CLUSTERS = 64
DESCRIPTOR_DIM = 4096
# Checkpoint tensors besides VGG16's encoder; vladv1 has no assignment bias.
STATE_SHAPES = {
    "pool.centroids": (CLUSTERS, FEATURE_DIM),
    "pool.conv.weight": (CLUSTERS, FEATURE_DIM, 1, 1),
    "WPCA.0.weight": (DESCRIPTOR_DIM, CLUSTERS * FEATURE_DIM, 1, 1),
    "WPCA.0.bias": (DESCRIPTOR_DIM,),
}


def validate_source(source: dict[str, Any]) -> None:
    declared = (source.get("type"), PurePosixPath(str(source.get("uri", ""))).name)
    if declared != ("file", CHECKPOINT):
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared}")


def validate_state(state: dict[str, Any]) -> None:
    """Require the vladv1 NetVLAD and 4096-D WPCA tensors surgery.py rebuilds, and no others besides VGG16."""
    shapes = {name: tuple(value.shape) for name, value in state.items() if not name.startswith("encoder.")}
    if shapes != STATE_SHAPES:
        raise ValueError(f"Unexpected NetVLAD checkpoint tensors: {shapes}")


def convert(checkpoint: Path, output_dir: Path) -> Path:
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    (weights,) = conversion_outputs(MODEL_DIR)
    output = output_dir / weights
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    validate_state(state)
    # The checkpoint stores WPCA in float64; upstream's float32 modules cast it when they load it, as this does.
    torch.save({name: value.float().contiguous() for name, value in state.items()}, output)
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
