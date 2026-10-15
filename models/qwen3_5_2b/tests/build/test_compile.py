import json
from pathlib import Path

from model_registry.artifact import verify_llima


def test_deployment_contains_language_and_vision(compiled):
    deployment = Path(compiled) / "model" / "deployment"
    names = {item["path"] for item in verify_llima(deployment)}
    assert any("language" in name and name.endswith(".elf") for name in names)
    assert any("vision" in name and name.endswith(".elf") for name in names)
    assert "devkit/tokenizer.json" in names
    assert any(name.startswith("npy_files/") and name.endswith(".npy") for name in names)
    cfg = json.loads((deployment / "devkit/vlm_config.json").read_text())
    assert cfg["pipeline_cfg"]["max_num_tokens"] == 4096
    assert cfg["pipeline_cfg"]["language_group_size"] == 128
    assert cfg["vm_cfg"]["image_size"] == [448, 448]
