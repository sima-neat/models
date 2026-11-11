"""The conversion publishes the upstream tensors and byte-exact processor assets, without graph surgery."""

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[2]


def test_conversion_bundles_assets_of_the_selected_source(monkeypatch, tmp_path, load_script):
    recipe = load_script("convert")
    source = tmp_path / "alternate.bin"
    source.write_bytes(b"checkpoint")
    (tmp_path / "alternate.json").write_text('{"selected": true}')
    metadata = {"entrypoint": source.name, "files": [{"path": source.name}, {"path": "alternate.json"}]}
    monkeypatch.setattr(recipe, "load_metadata", lambda model_dir: {"conversion": {"alternate": {"source": metadata}}})
    monkeypatch.setattr(recipe, "conversion_outputs", lambda model_dir: ("weights.pth",))
    fetcher = MagicMock()
    fetcher.fetch_source.return_value = [source]
    monkeypatch.setattr(recipe, "Fetcher", lambda: fetcher)
    torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("sys.argv", ["convert.py", "--conversion", "alternate", "--output-dir", str(tmp_path / "out")])

    recipe.main()

    fetcher.fetch_source.assert_called_once_with(metadata)
    torch.load.assert_called_once_with(source, map_location="cpu", weights_only=True)
    torch.save.assert_called_once_with(
        {"state_dict": torch.load.return_value, "assets": {"alternate.json": '{"selected": true}'}},
        tmp_path / "out/weights.pth",
    )


def test_converted_checkpoint_preserves_weights_and_processor_assets(converted: Path):
    import torch

    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    weights, assets = checkpoint["state_dict"], checkpoint["assets"]
    source = load_model(MODEL_DIR)["conversion"]["default"]["source"]
    files = {item["path"]: item for item in source["files"] if item["path"] != source["entrypoint"]}
    assert set(assets) == set(files)
    for name, content in assets.items():
        data = content.encode("utf-8")
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (files[name]["sha256"], files[name]["size_bytes"])
    assert json.loads(assets["config.json"])["model_type"] == "clip"
    assert weights["visual_projection.weight"].shape == (512, 768)
    assert weights["text_projection.weight"].shape == (512, 512)
    assert weights["text_model.embeddings.token_embedding.weight"].shape == (49408, 512)
    for name, value in weights.items():
        if name.endswith(".position_ids"):
            assert torch.equal(value, torch.arange(value.shape[1])[None])
        else:
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
