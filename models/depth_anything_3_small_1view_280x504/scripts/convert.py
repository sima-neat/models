"""Package the unmodified DA3-Small checkpoint and its pinned upstream Python modules.

Torch's package format keeps model code with its weights. surgery.py can rebuild
and export it without installing DA3's unrelated application dependencies.
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from model_registry.conversion import conversion_outputs
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[1]
PRESET = "da3-small"
PACKAGE = "depth_anything_3"
WHEEL = "depth_anything_3-0.1.1-py3-none-any.whl"
SOURCE = ("file_set", "model.safetensors", ["model.safetensors", "config.json", WHEEL])
# Upstream intentionally leaves these unused auxiliary ray-head normalizations at their defaults.
MISSING_KEYS = {f"head.scratch.output_conv2_aux.{level}.2.{name}" for level in (1, 2, 3) for name in ("weight", "bias")}


def validate_source(source):
    declared = (source.get("type"), source.get("entrypoint"), [item.get("path") for item in source.get("files", [])])
    if declared != SOURCE:
        raise ValueError(f"Expected the file set {SOURCE}, model.yaml declares {declared}")


def unpack_package(wheel, directory):
    """Unpack only the pure-Python package tree; its declared dependencies are not needed to export."""
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.startswith(f"{PACKAGE}/")]
        if not members or any(".." in Path(name).parts for name in members):
            raise ValueError(f"Unexpected contents in {wheel.name}")
        archive.extractall(directory, members)
    sys.path.insert(0, str(directory))


def build_network(directory):
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    declared = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config = load_config(MODEL_REGISTRY[PRESET])
    # The Hugging Face mixin builds the network from the wheel preset that config.json names.
    if declared.get("model_name") != PRESET or OmegaConf.to_container(config) != declared.get("config"):
        raise ValueError(f"config.json does not describe the wheel's {PRESET} preset")
    network = create_object(config).eval()
    state = load_file(directory / "model.safetensors")
    if not all(key.startswith("model.") for key in state):
        raise ValueError("Expected the mixin's model. prefix on every checkpoint tensor")
    loaded = network.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != MISSING_KEYS:
        raise ValueError(f"Unexpected checkpoint keys: {loaded}")
    return network


def convert(source: Path, output_dir: Path) -> Path:
    import torch
    from torch.package import PackageExporter

    output_dir.mkdir(parents=True, exist_ok=False)
    (filename,) = conversion_outputs(MODEL_DIR)
    output = output_dir / filename
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="da3-upstream-") as temporary:
        unpack_package(source.parent / WHEEL, Path(temporary))
        network = build_network(source.parent)
        with PackageExporter(output) as package:
            package.intern("depth_anything_3.**")
            package.extern("**")
            package.save_pickle("da3", "model.pkl", network)
            package.save_text("da3", "preset.txt", PRESET)
    print(f"Packaged the original {PRESET} network: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion", default="default")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    conversions = load_metadata(MODEL_DIR).get("conversion", {})
    if args.conversion not in conversions:
        parser.error(f"unknown conversion {args.conversion!r}")
    source = conversions[args.conversion]["source"]
    validate_source(source)
    path, *_ = Fetcher().fetch_source(source)
    convert(path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
