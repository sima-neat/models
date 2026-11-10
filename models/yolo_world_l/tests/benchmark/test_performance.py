from pathlib import Path

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


def test_component_arguments_are_complete_and_unique(load_script):
    import pytest
    script = load_script("performance")
    assert set(script.parse_mpks(["text_encoder=text.tgz", "detector=image.tgz"])) == {"text_encoder", "detector"}
    for values in (["detector=image.tgz"], ["detector=a", "detector=b", "text_encoder=c"]):
        with pytest.raises(ValueError):
            script.parse_mpks(values)
