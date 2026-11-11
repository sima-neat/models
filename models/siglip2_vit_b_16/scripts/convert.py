"""Bundle the pinned SigLIP 2 checkpoint and processor assets for surgery.py."""

import argparse
from pathlib import Path

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[1]


def convert(source, output_dir, metadata):
    import torch
    from safetensors.torch import load_file

    assets = {
        item["path"]: (source.parent / item["path"]).read_bytes()
        for item in metadata["files"]
        if item["path"] != metadata["entrypoint"]
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.save({"state_dict": load_file(str(source), device="cpu"), "assets": assets}, output)
    print(f"Wrote {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_model(MODEL_DIR)["conversion"]
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    metadata = conversions[args.conversion]["source"]
    source, *_ = Fetcher().fetch_source(metadata)
    convert(source, args.output_dir.resolve(), metadata)


if __name__ == "__main__":
    main()
