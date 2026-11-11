from model_registry.model_tests import measured_shortfalls


def test_performance_meets_expected_values(performance, model_dir, build_profile):
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_includes_execution_plan_and_mla_stats(performance):
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]
    assert component["mla_layer_stats"]["status"] == "ok"
    assert component["mla_stage_stats"]["status"] == "ok"
