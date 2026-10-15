import json
import zipfile
from pathlib import Path

from model_registry.artifact import verify_llima


def test_deployment_contains_language_and_vision(compiled):
    archive = Path(compiled) / "model" / "qwen3_5_2b_llima.zip"
    names = {item["path"] for item in verify_llima(archive)}
    assert any("language" in name and name.endswith(".elf") for name in names)
    assert any("vision" in name and name.endswith(".elf") for name in names)
    with zipfile.ZipFile(archive) as source:
        cfg = json.loads(source.read("devkit/vlm_config.json"))
    assert cfg["pipeline_cfg"]["max_num_tokens"] == 4096
    assert cfg["pipeline_cfg"]["language_group_size"] == 128
    assert cfg["vm_cfg"]["image_size"] == [448, 448]
