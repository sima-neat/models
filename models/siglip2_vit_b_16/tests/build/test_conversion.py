"""Conversion preserves the upstream tensors and processor assets, including the binary tokenizer."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[2]


def test_conversion_bundles_assets_of_the_selected_source(monkeypatch, tmp_path, load_script):
    recipe = load_script("convert")
    source = tmp_path / "alternate.safetensors"
    source.write_bytes(b"checkpoint")
    assets = {"alternate.json": b'{"selected": true}', "tokenizer.model": b"\x00\xff\x80"}
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)
    metadata = {"entrypoint": source.name, "files": [{"path": name} for name in (source.name, *assets)]}
    monkeypatch.setattr(recipe, "load_model", lambda model_dir: {"conversion": {"alternate": {"source": metadata}}})
    monkeypatch.setattr(recipe, "conversion_outputs", lambda model_dir: ("weights.pth",))
    fetcher = MagicMock()
    fetcher.fetch_source.return_value = [source]
    monkeypatch.setattr(recipe, "Fetcher", lambda: fetcher)
    torch, load_file = MagicMock(), MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=load_file))
    monkeypatch.setattr("sys.argv", ["convert.py", "--conversion", "alternate", "--output-dir", str(tmp_path / "out")])

    recipe.main()

    fetcher.fetch_source.assert_called_once_with(metadata)
    load_file.assert_called_once_with(str(source), device="cpu")
    torch.save.assert_called_once_with(
        {"state_dict": load_file.return_value, "assets": assets}, tmp_path / "out/weights.pth"
    )


def test_converted_checkpoint_preserves_weights_and_processor_assets(converted: Path):
    import torch

    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    weights, assets = checkpoint["state_dict"], checkpoint["assets"]
    source = load_model(MODEL_DIR)["conversion"]["default"]["source"]
    files = {item["path"]: item for item in source["files"] if item["path"] != source["entrypoint"]}
    assert set(assets) == set(files)
    for name, content in assets.items():
        assert (hashlib.sha256(content).hexdigest(), len(content)) == (files[name]["sha256"], files[name]["size_bytes"])
    assert json.loads(assets["config.json"])["model_type"] == "siglip"
    processor = json.loads(assets["preprocessor_config.json"])
    assert processor["size"] == {"height": 224, "width": 224}
    assert processor["resample"] == 2
    assert processor["do_resize"] and processor["do_rescale"] and processor["do_normalize"]
    assert processor["rescale_factor"] == 1 / 255
    assert processor["image_mean"] == processor["image_std"] == [0.5] * 3
    assert weights["text_model.embeddings.token_embedding.weight"].shape == (256000, 768)
    assert weights["vision_model.embeddings.patch_embedding.weight"].shape == (768, 3, 16, 16)
    assert weights["text_model.head.weight"].shape == (768, 768)
    assert weights["vision_model.head.probe"].shape == (1, 1, 768)
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in weights.values())
