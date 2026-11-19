"""Check measured end-to-end speed and the actual accelerator route."""

from pathlib import Path

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_baseline(performance, model_dir: Path, build_profile: str):
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_model_execution_plan(performance):
    plan = performance["components"]["model"]["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_performance_gate_rejects_slowdown(model_dir: Path):
    report = {"metrics": {"throughput_fps": 0.01, "real_time_factor": 1.0}}
    assert measured_shortfalls(model_dir, "performance", "w8a16", report)
