"""Checks of the conversion's checkpoint handling and of the weights bundle it publishes."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from model_registry.model import load_model as load_metadata

MODEL_DIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_source_must_be_the_pinned_file_set(converter):
    source = load_metadata(MODEL_DIR)["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="checkpoints and CLIP tokenizer"):
        converter.validate_source({**source, "files": source["files"][:1]})


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("features.0.0.c.weight", "stem.conv1.c.weight"),
        ("features.0.2.bn.bias", "stem.conv2.bn.bias"),
        ("features.1.token_mixer.0.conv.c.weight", "stages.0.blocks.0.token_mixer.conv.c.weight"),
        ("features.2.channel_mixer.m.2.bn.weight", "stages.0.blocks.1.channel_mixer.conv2.bn.weight"),
        ("features.3.token_mixer.1.fc1.weight", "stages.1.downsample.pre_block.se.fc1.weight"),
        ("features.4.token_mixer.0.c.weight", "stages.1.downsample.spatial_downsample.c.weight"),
        ("features.4.channel_mixer.m.0.c.weight", "stages.1.downsample.ffn.conv1.c.weight"),
        ("features.20.channel_mixer.m.0.c.weight", "stages.2.blocks.11.channel_mixer.conv1.c.weight"),
        ("features.21.channel_mixer.m.0.c.weight", "stages.3.downsample.pre_block.channel_mixer.conv1.c.weight"),
        ("features.24.token_mixer.0.conv.c.weight", "stages.3.blocks.1.token_mixer.conv.c.weight"),
    ],
)
def test_trunk_keys_map_onto_timm_repvit(key, expected, converter):
    assert converter.trunk_key(f"trunk.model.backbone.model.{key}") == expected


def test_fine_tuned_checkpoint_supplies_only_the_trunk_and_neck(tmp_path, converter):
    torch = pytest.importorskip("torch")
    state = {
        "backbone.vision_backbone.trunk.model.backbone.model.features.0.0.c.weight": torch.ones(1),
        "backbone.vision_backbone.trunk.model.head.0.weight": torch.ones(2),
        "backbone.vision_backbone.convs.0.conv_1x1.weight": torch.ones(3),
        "backbone.language_backbone.encoder.embedding_layer.weight": torch.ones(4),
        "transformer.encoder.layers.0.linear1.weight": torch.ones(5),
    }
    path = tmp_path / "efficientsam3_repvit.pt"
    torch.save({"model": state, "steps": 1}, path)

    assert sorted(converter.backbone_weights(path)) == [
        "backbone.vision_backbone.convs.0.conv_1x1.weight",
        "backbone.vision_backbone.trunk.model.backbone.model.stem.conv1.c.weight",
        "backbone.vision_backbone.trunk.model.head.0.weight",
    ]


def test_trunk_key_rejects_unknown_layers(converter):
    with pytest.raises(KeyError, match="Unmapped"):
        converter.trunk_key("trunk.features.1.unknown.weight")


def test_bundle_holds_the_weights_settings_and_upstream_cases(converted: Path, converter):
    torch = pytest.importorskip("torch")
    bundle = torch.load(converted, map_location="cpu", weights_only=True)

    assert sorted(bundle) == ["assets", "cases", "position_encoding", "settings", "state_dict"]
    assert sorted(bundle["assets"]) == ["merges.txt", "vocab.json"]
    source = load_metadata(MODEL_DIR)["conversion"]["default"]["source"]
    pins = {item["path"]: item["sha256"] for item in source["files"]}
    for name, content in bundle["assets"].items():
        assert hashlib.sha256(content).hexdigest() == pins[name]
    assert bundle["settings"] == {
        "grid": 72,
        "text_heads": 16,
        "heads": 8,
        "norm_eps": 1e-5,
        "groups": 8,
        "score_scale": 0.0625,
        "score_limit": 12.0,
    }
    assert bundle["position_encoding"].shape == (1, 256, 72, 72)
    weights = bundle["state_dict"]
    assert weights["backbone.language_backbone.encoder.token_embedding.weight"].shape == (49408, 1024)
    assert weights["backbone.vision_backbone.trunk.model.backbone.model.stem.conv1.c.weight"].shape == (32, 3, 3, 3)
    assert weights["backbone.vision_backbone.trunk.model.head.3.weight"].shape == (1024, 1024, 3, 3)
    assert weights["transformer.decoder.query_embed.weight"].shape == (200, 256)
    assert weights["segmentation_head.instance_seg_head.weight"].shape == (256, 256, 1, 1)
    for name, value in weights.items():
        assert torch.isfinite(value).all(), name
    # surgery.py must reproduce the upstream model's outputs on these cases.
    assert [case["prompt"] for case in bundle["cases"]] == [prompt for _, prompt in converter.CASES]
    shapes = {
        "pixel_values": (1, 3, 1008, 1008),
        "text_features": (16, 256),
        "scores": (200,),
        "boxes_px": (200, 4),
        "mask_logits": (200, 288, 288),
    }
    for case in bundle["cases"]:
        assert {name: tuple(case[name].shape) for name in shapes} == shapes
        assert all(case[name].dtype == torch.float32 and torch.isfinite(case[name]).all() for name in shapes)
        assert 0 <= case["scores"].min() and case["scores"].max() <= 1
