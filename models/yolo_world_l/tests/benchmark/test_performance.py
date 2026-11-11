from pathlib import Path

import pytest

from model_registry.model_tests import measured_shortfalls


def test_performance_keeps_encoder_and_detector_separate(performance, model_dir: Path, build_profile):
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []
    assert performance["scope"] == "components"
    assert "metrics" not in performance
    assert set(performance["components"]) == {"text_encoder", "detector"}
    for component in performance["components"].values():
        assert component["metrics"]["throughput_fps"] > 0
        plan = component["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]
        assert component["mla_layer_stats"]["status"] == "ok"
        assert component["mla_stage_stats"]["status"] == "ok"


@pytest.mark.parametrize("script_name", ["performance", "accuracy"])
def test_component_arguments_are_complete_and_unique(load_script, script_name):
    script = load_script(script_name)
    assert set(script.parse_mpks(["text_encoder=text.tgz", "detector=image.tgz"])) == {"text_encoder", "detector"}
    for values in (["detector=image.tgz"], ["detector=a", "detector=b", "text_encoder=c"]):
        with pytest.raises(ValueError):
            script.parse_mpks(values)


@pytest.mark.parametrize("script_name", ["performance", "accuracy"])
@pytest.mark.parametrize("layout", ["compiled", "installed", "explicit"])
def test_prompt_assets_load_from_files_beside_unchanged_components(
    tmp_path, load_script, monkeypatch, layout, script_name
):
    import json
    import sys
    from types import SimpleNamespace

    import numpy as np

    script = load_script(script_name)
    assets = tmp_path / "package/prompt_assets"
    assets.mkdir(parents=True)
    # Sparse on disk: exercise the real NPY loader without generating 96 MiB of fixture values.
    table = np.lib.format.open_memmap(assets / "token_embedding_table.npy", mode="w+", dtype=np.float32,
                                    shape=(49408, 512))
    table[49407, 0] = 3.5
    del table
    (assets / "tokenizer.json").write_text('{"fixture": true}')
    tokenizer = SimpleNamespace(no_truncation=lambda: None, enable_padding=lambda **kwargs: None)

    def from_file(path):
        assert Path(path) == assets / "tokenizer.json"
        assert json.loads(Path(path).read_text()) == {"fixture": True}
        return tokenizer

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=from_file)))
    archive = (tmp_path / "package/text_encoder/encoder_mpk.tar.gz" if layout == "compiled" else
               tmp_path / "package/encoder_mpk.tar.gz" if layout == "installed" else tmp_path / "elsewhere/encoder.tgz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"not opened: compiler archive is independent of CPU resources")
    actual_tokenizer, actual_table = script.prompt_assets(archive, tmp_path / "package" if layout == "explicit" else None)
    assert actual_tokenizer is tokenizer
    assert actual_table[49407, 0] == 3.5
    (assets / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError):
        script.prompt_assets(archive, tmp_path / "package" if layout == "explicit" else None)
