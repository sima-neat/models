from pathlib import Path

from model_registry.artifact import verify_llima


def test_deployment_contains_language_and_vision(compiled):
    deployment = Path(compiled) / "model" / "deployment"
    names = {item["path"] for item in verify_llima(deployment)}
    assert any("language" in name and name.endswith(".elf") for name in names)
    assert any("vision" in name and name.endswith(".elf") for name in names)
    assert "devkit/tokenizer.json" in names
    assert "devkit/vlm_config.json" in names
    assert "compile.sh" in names
