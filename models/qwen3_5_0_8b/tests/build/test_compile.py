import json
from pathlib import Path

import yaml
from model_registry.artifact import verify_llima


def test_deployment_contains_language_and_vision(compiled):
    metadata = yaml.safe_load((Path(__file__).resolve().parents[2] / "model.yaml").read_text())
    profile = next(item for item in metadata["build_profiles"] if item["id"] == Path(compiled).name)
    parameters = next(item["parameters"] for item in profile["components"] if item["id"] == "model")
    deployment = Path(compiled) / "model" / "deployment"
    names = {item["path"] for item in verify_llima(deployment)}
    assert any("language" in name and name.endswith(".elf") for name in names)
    assert any("vision" in name and name.endswith(".elf") for name in names)
    assert "devkit/tokenizer.json" in names
    cfg = json.loads((deployment / "devkit/vlm_config.json").read_text())
    assert cfg["pipeline_cfg"]["max_num_tokens"] == parameters["max_num_tokens"]
    assert cfg["pipeline_cfg"]["language_group_size"] == parameters["language_group_size"]
    assert cfg["vm_cfg"]["image_size"] == [parameters["input_height"], parameters["input_width"]]
