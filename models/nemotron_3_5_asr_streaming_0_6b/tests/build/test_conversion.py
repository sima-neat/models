"""Conversion preserves the checkpoint; graph adaptations belong to compilation."""

import hashlib
import shutil
import sys
from unittest.mock import MagicMock

import pytest
from model_registry.conversion import conversion_fingerprint
from model_registry.fetch import Fetcher
from model_registry.fingerprints import COMPILE_TOOLING, compile_fingerprint
from model_registry.model import load_model


def test_conversion_packages_unchanged_tensors_and_assets(tmp_path, load_script, monkeypatch):
    recipe = load_script("convert")
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"weights")
    (tmp_path / "config.json").write_text('{"original": true}')
    metadata = {"entrypoint": source.name, "files": [{"path": source.name}, {"path": "config.json"}]}
    torch, safetensors = MagicMock(), MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors)
    result = recipe.convert(source, tmp_path / "out", metadata)
    safetensors.load_file.assert_called_once_with(source)
    torch.save.assert_called_once_with(
        {"state_dict": safetensors.load_file.return_value, "assets": {"config.json": '{"original": true}'}},
        result,
    )
    assert result.name == "weights.pth"


def test_converted_checkpoint_preserves_every_source_tensor_and_asset(converted, model_dir):
    import torch
    from safetensors import safe_open

    checkpoint = torch.load(converted, map_location="cpu", weights_only=True)
    source = load_model(model_dir)["conversion"]["default"]["source"]
    original, *_ = Fetcher().fetch_source(source)
    with safe_open(original, framework="pt") as tensors:
        assert set(checkpoint["state_dict"]) == set(tensors.keys())
        for name, value in checkpoint["state_dict"].items():
            expected = tensors.get_tensor(name)
            assert value.dtype == expected.dtype and torch.equal(value, expected)
    assets = {item["path"]: item for item in source["files"] if item["path"] != source["entrypoint"]}
    assert set(checkpoint["assets"]) == set(assets)
    for name, content in checkpoint["assets"].items():
        data = content.encode("utf-8")
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (assets[name]["sha256"], assets[name]["size_bytes"])


@pytest.mark.parametrize("changed_file", ["surgery.py", "compile.py"])
def test_compile_changes_do_not_invalidate_conversion(tmp_path, model_dir, changed_file):
    root = model_dir.parents[1]
    for name in ("schemas", "taxonomies", "datasets"):
        shutil.copytree(root / name, tmp_path / name)
    target = tmp_path / "models" / model_dir.name
    shutil.copytree(model_dir, target)
    for name in COMPILE_TOOLING:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, tmp_path / name)
    model = load_model(target)
    original = conversion_fingerprint(tmp_path, model)
    selected = {model["id"]: {"conversion_fingerprint": original, "sha256": "a" * 64}}

    def compiled():
        return compile_fingerprint(tmp_path, model["id"], "w8a16", "w8a16", "test", conversion_sources=selected)

    before = compiled()
    changed = target / "scripts" / changed_file
    contents = changed.read_text()
    if changed_file == "compile.py":
        assert "transformers==5.16.1" in contents
        contents = contents.replace("transformers==5.16.1", "transformers==5.16.2")
    else:
        contents += "\n# changed compiler adaptation\n"
    changed.write_text(contents)
    assert conversion_fingerprint(tmp_path, model) == original
    assert compiled() != before


def test_checkpoint_roundtrip_does_not_change_tensors_or_assets(tmp_path, load_script):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    convert, surgery = load_script("convert"), load_script("surgery")
    source = tmp_path / "model.safetensors"
    expected = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "count": torch.tensor([3])}
    safetensors.save_file(expected, source)
    text = '{"model_type": "original"}\r\n'
    (tmp_path / "config.json").write_bytes(text.encode("utf-8"))
    metadata = {"entrypoint": source.name, "files": [{"path": source.name}, {"path": "config.json"}]}
    converted = convert.convert(source, tmp_path / "converted", metadata)
    restored = tmp_path / "restored"
    surgery.restore_checkpoint(converted, restored)
    assert (restored / "config.json").read_bytes().decode("utf-8") == text
    actual = safetensors.load_file(restored / source.name)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert value.dtype == actual[name].dtype and torch.equal(actual[name], value)
