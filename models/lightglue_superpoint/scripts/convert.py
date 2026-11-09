"""Publish LightGlue's matcher weights from the pinned Hugging Face checkpoint as the weights.pth surgery.py reads.

Conversion only changes the file format: surgery.py rebuilds the matcher for the MLA at compile time. The checkpoint
also holds SuperPoint, its keypoint detector, which the weights leave out. config.json must declare the architecture
surgery.py writes out.
"""

import argparse
import json
from pathlib import Path

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_and_validate_model

MODEL_DIR = Path(__file__).resolve().parents[1]

SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json"])
# The config.json values that decide the matcher's architecture; surgery.py builds LightGlue with them.
ARCHITECTURE = {
    "model_type": "lightglue",
    "descriptor_dim": 256,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 9,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "hidden_act": "gelu",
    "attention_bias": True,
    "filter_threshold": 0.1,
}
DETECTOR_PREFIX = "keypoint_detector."


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the Hugging Face file set {SOURCE}, model.yaml declares {declared}")


def validate_config(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    declared = {key: config.get(key) for key in ARCHITECTURE}
    if declared != ARCHITECTURE:
        raise ValueError(f"config.json declares {declared}, expected {ARCHITECTURE}")


def convert(checkpoint, output_dir):
    import torch
    from safetensors.torch import load_file

    validate_config(checkpoint.parent)
    output_dir.mkdir(parents=True, exist_ok=False)
    (weights,) = conversion_outputs(MODEL_DIR)
    output = output_dir / weights
    tensors = load_file(str(checkpoint))
    torch.save({name: value for name, value in tensors.items() if not name.startswith(DETECTOR_PREFIX)}, output)
    print(f"Wrote {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default", help="conversion entry in model.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_and_validate_model(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}; model.yaml declares: {', '.join(sorted(conversions))}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
