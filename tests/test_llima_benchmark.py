"""GenAI measurement boundaries and registered compilation reuse."""

import json
from pathlib import Path

import pytest
from model_registry import llima_benchmark as bench
from model_registry.board_telemetry import summarize
from model_registry.errors import ValidationError
from model_registry.report_publication import check_report
from model_report import from_generation

from tests.test_report_publication import publish, registered_build

ROOT = Path(__file__).resolve().parents[1]


def text_workload(length):
    requests = [
        {
            "phase": "warmup" if i == 0 else "measure",
            "input_tokens": length,
            "token_durations_s": [0.1, 0.02, 0.04],
            "generated_tokens": 3,
            "decode_tokens": 2,
            "ttft_s": 0.1,
            "decode_time_s": 0.06,
        }
        for i in range(6)
    ]
    workload = {"id": f"text_n{length}", "kind": "text", "input_tokens": length, "requests": requests}
    workload.update(telemetry=summarize({}), throughput_duration_ms=0.0)
    workload["metrics"] = bench.workload_metrics(workload)
    return workload


def report():
    workloads = [text_workload(length) for length in bench.input_buckets(4096)]
    image = {
        "id": "image",
        "kind": "image_text",
        "image_size": [448, 448],
        "telemetry": summarize({}),
        "throughput_duration_ms": 0.0,
        "requests": [
            {
                "phase": "warmup" if i == 0 else "measure",
                "ttft_s": 0.13,
                "streamed_tps_mean": 75.0,
                "generated_tokens": None,
            }
            for i in range(6)
        ],
    }
    image["metrics"] = bench.workload_metrics(image)
    return {
        "status": "passed",
        "settings": {"max_context": 4096, "image_size": [448, 448]},
        "runtime": {"mole_version": "0.6.0+test"},
        "timing": {},
        "workloads": [*workloads, image],
    }


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (1152, [128]),
        (4096, [128, 256, 512, 1024, 2048, 3072]),
        (8192, [128, 256, 512, 1024, 2048, 4096, 7168]),
    ],
)
def test_buckets_cover_headroom_boundary(context, expected):
    assert bench.input_buckets(context) == expected


def test_text_rate_uses_actual_decode_time_and_excludes_ttft():
    workload = text_workload(128)
    metrics = bench.workload_metrics(workload)
    assert metrics["ttft_ms"]["value"] == pytest.approx(100)
    assert metrics["prefill_tps"]["value"] == pytest.approx(1280)
    assert metrics["decode_tps"]["value"] == pytest.approx(2 / 0.06)
    assert metrics["decode_tps"]["sample_count"] == 5


@pytest.mark.parametrize("damage", ["missing_bucket", "missing_run", "nan", "counts", "summary", "size"])
def test_failed_or_inconsistent_measurements_are_not_publishable(damage):
    data = report()
    if damage == "missing_bucket":
        data["workloads"].pop(1)
    if damage == "missing_run":
        data["workloads"][0]["requests"].pop()
    if damage == "nan":
        data["workloads"][0]["requests"][1]["ttft_s"] = float("nan")
    if damage == "counts":
        data["workloads"][0]["requests"][1]["generated_tokens"] += 1
    if damage == "summary":
        data["workloads"][0]["metrics"]["decode_tps"]["value"] = 999
    if damage == "size":
        data["workloads"][-1]["image_size"] = [1024, 1024]
    with pytest.raises(ValidationError):
        bench.validate_report(data)


def test_exact_prompt_lengths_preserve_native_assistant_suffix():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["enable_thinking"] is False
            return [1000, *messages[0]["content"].encode(), 1001, 1002]

    for length in (128, 256, 3072):
        tokens = bench._prompt(Tokenizer(), length)
        assert len(tokens) == length and tokens[0] == 1000 and tokens[-2:] == [1001, 1002]








def generation_report():
    data = report()
    data["settings"].update(repetitions=5, warmup=1)
    for workload in data["workloads"]:
        workload["telemetry"] = summarize({
            "metrics": [{"key": "power_current_watts", "unit": "W"}],
            "samples": [{"values": {"power_current_watts": 14.0}}],
        })
    return from_generation(data, {"llima": "0.6.0", "mole": "0.6.0", "build": {}}).document


def test_generation_uses_the_unified_report_and_checks_every_workload(tmp_path):
    data = generation_report()
    path = tmp_path / "model-report.json"
    path.write_text(json.dumps(data))
    assert check_report(path, ROOT, "qwen3_5_0_8b", "w4a16", "performance") == data
    assert data["schema_version"] == 1
    data["generation"]["workloads"][-1]["telemetry"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="required performance metrics were not measured"):
        check_report(path, ROOT, "qwen3_5_0_8b", "w4a16", "performance")


def test_generation_publication_uses_native_inputs_and_keeps_workload_dimensions(tmp_path):
    import hashlib

    compiled = tmp_path / "_compiled"
    registered_build(compiled, "qwen3_5_0_8b", "w4a16", ["model"])
    profile = compiled / "qwen3_5_0_8b/w4a16"
    manifest = profile / "model/deployment-manifest.json"
    manifest.write_text("{}")
    (profile / "compile-outputs.json").write_text(json.dumps({"outputs": [{
        "component_id": "model", "artifact_path": "model/deployment-manifest.json",
        "artifact_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }]}))
    calls, result = publish(tmp_path, "qwen3_5_0_8b", "w4a16", "performance", generation_report())
    assert result["run_type"] == "model_performance"
    inputs = [body for _, path, body in calls if "/inputs/" in path]
    assert inputs[0]["source_reference"].endswith("/artifacts/llima")
    measurements = next(body["measurements"] for _, path, body in calls if path.endswith("/measurements/batch"))
    assert {row["dimensions"]["workload_id"] for row in measurements} == {w["id"] for w in report()["workloads"]}
    image = [row for row in measurements if row["dimensions"]["workload_id"] == "image"]
    assert all("input_tokens" not in row["dimensions"] for row in image)
    assert {row["name"] for row in image} == {
        "model.generation.ttft", "model.generation.decode_rate", "board.power.avg", "board.power.peak",
    }
