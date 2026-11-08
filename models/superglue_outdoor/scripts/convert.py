"""Publish SuperGlue outdoor's pinned checkpoint as the weights.pth that surgery.py builds the matcher from.

Conversion only changes the file format: surgery.py rebuilds the matcher for the MLA at compile time.
"""

import argparse
from pathlib import Path, PurePosixPath

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT = "model.safetensors"


def validate_source(source):
    declared = PurePosixPath(str(source.get("uri", ""))).name
    if declared != CHECKPOINT:
        raise ValueError(f"Expected the checkpoint file {CHECKPOINT}, model.yaml declares {declared!r}")


def convert(checkpoint, output_dir):
    import torch
    from safetensors.torch import load_file

    output_dir.mkdir(parents=True, exist_ok=False)
    (weights,) = conversion_outputs(MODEL_DIR)
    output = output_dir / weights
    torch.save(load_file(str(checkpoint)), output)
    print(f"Wrote {output}")
    return output


def main():
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
