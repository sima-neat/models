"""The conversion preserves the original network; surgery.py owns the static single-view export."""

from pathlib import Path

import pytest
from model_registry.model import load_model


def test_source_has_exact_weights_config_and_upstream_code(model_dir, load_script):
    converter = load_script("convert")
    source = load_model(model_dir)["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="file set"):
        converter.validate_source({**source, "files": source["files"][:-1]})


def test_conversion_packages_an_unmodified_da3_small_model(converted: Path):
    torch = pytest.importorskip("torch")
    package = torch.package.PackageImporter(converted)
    assert package.load_text("da3", "preset.txt") == "da3-small"
    network = package.load_pickle("da3", "model.pkl", map_location="cpu")
    assert not network.training
    assert len(network.backbone.pretrained.blocks) == 12
    assert network.head.scratch.output_conv2[-1].weight.shape == (2, 32, 1, 1)
    assert not hasattr(network.head, "depth_projection")
    assert all(parameter.dtype == torch.float32 for parameter in network.parameters())
