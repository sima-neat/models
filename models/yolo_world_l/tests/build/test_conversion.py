from pathlib import Path


def test_converted_checkpoint_preserves_both_encoders(converted: Path):
    import torch

    (path,) = converted.glob("*.pth") if converted.is_dir() else [converted]
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state = bundle["state_dict"]
    assert "backbone.image_model.stem.conv.weight" in state
    assert "backbone.text_model.model.text_model.embeddings.token_embedding.weight" in state
    assert any(name.endswith("tokenizer.json") for name in bundle["assets"])
