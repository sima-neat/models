"""Bundle the original checkpoint and pinned tokenizer assets without graph changes.

The inference adapters, fixed prompt mask, decoder cut and precision annotations
are applied by surgery.py during compilation, not by conversion.
"""
import argparse
from pathlib import Path

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]


def convert(source, output_dir):
    import torch

    metadata = load_metadata(MODEL_DIR)["conversion"]["default"]["source"]
    assets = {item["path"]: (source.parent / item["path"]).read_text(encoding="utf-8")
              for item in metadata["files"] if item["path"] != metadata["entrypoint"]}
    state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("expected the original YOLO-World state_dict")
    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.save({"state_dict": state, "assets": assets}, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR)["conversion"]
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source, *_ = Fetcher().fetch_source(conversions[args.conversion]["source"])
    print(convert(source, args.output_dir.resolve()))


if __name__ == "__main__":
    main()
