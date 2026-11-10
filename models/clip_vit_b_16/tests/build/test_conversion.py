"""The conversion publishes the upstream tensors and byte-exact processor assets, without graph surgery."""

import hashlib
import json
from pathlib import Path

from model_registry.model import load_model

MODEL_DIR = Path(__file__).resolve().parents[2]


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
