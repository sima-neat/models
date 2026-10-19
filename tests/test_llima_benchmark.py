"""GenAI measurement boundaries and registered compilation reuse."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from model_registry import llima_benchmark as bench
from model_registry.benchmark_publication import _measurements, load_benchmark_result, validate_benchmark_set
from model_registry.board_telemetry import summarize
from model_registry.errors import ValidationError

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


def test_board_measurements_keep_the_workload_dimension():
    data = report()
    data["workloads"][0]["telemetry"] = summarize(
        {
            "metrics": [{"key": "cpu_usage_pct", "unit": "%"}, {"key": "power_current_watts", "unit": "W"}],
            "samples": [{"values": {"cpu_usage_pct": 12.5, "power_current_watts": 14.0}}],
        }
    )
    data["workloads"][0]["avg_power_watts"] = 14.0
    result = {
        "benchmark_report": data,
        "benchmark_type": "model.generative",
        "model": {"id": "qwen3_5_0_8b", "build_profile_id": "w4a16", "component_id": "model"},
    }
    readings = _measurements(result, "a" * 64)
    cpu = next(row for row in readings if row["name"] == "board.cpu_usage.avg")
    assert cpu["value"] == 12.5 and cpu["unit"] == "%"
    assert cpu["dimensions"]["workload_id"] == "text_n128"
    assert cpu["dimensions"]["input_tokens"] == "128"
    assert sum(row["name"] == "board.power.avg" for row in readings) == 1


@pytest.mark.parametrize("missing_workload", [None, "text_n128", "image"])
def test_required_telemetry_is_checked_for_every_workload(tmp_path, missing_workload):
    data = report()
    for workload in data["workloads"]:
        if workload["id"] != missing_workload:
            workload["avg_power_watts"] = 14.0
            workload["telemetry"] = summarize({
                "metrics": [{"key": "cpu_usage_pct", "unit": "%"}],
                "samples": [{"values": {"cpu_usage_pct": 12.5}}],
            })
    result = {
        "benchmark_report": data,
        "benchmark_type": "model.generative",
        "model": {"id": "qwen3_5_0_8b", "build_profile_id": "w4a16", "component_id": "model"},
        "provenance": {"models_commit": "a" * 40, "compile_run_id": "00000000-0000-4000-8000-000000000001"},
    }
    path = tmp_path / "model/performance-result.json"
    path.parent.mkdir()
    path.write_text(json.dumps(result))
    with (
        patch("model_registry.benchmark_publication.load_benchmark_result", return_value=result),
        patch("model_registry.benchmark_publication.load_component_performance_config", return_value={
            "required_metrics": ["ttft_ms", "decode_tps", "avg_cpu_usage_pct", "avg_power_watts"],
        }),
    ):
        args = dict(repository=ROOT, result_root=tmp_path, model_id="qwen3_5_0_8b", build_profile_id="w4a16",
                    component_ids=["model"], git_sha="a" * 40)
        if missing_workload:
            with pytest.raises(ValidationError, match="required benchmark metric .* was not measured"):
                validate_benchmark_set(**args)
        else:
            assert validate_benchmark_set(**args)["count"] == 1


def test_genai_import_and_measurements_preserve_workload_dimensions(tmp_path):
    data = report()
    artifact = tmp_path / "deployment-manifest.json"
    artifact.write_text("{}")
    manifest = {
        "model": {
            "id": "qwen3_5_0_8b",
            "build_profile_id": "w4a16",
            "component_id": "model",
            "path": str(artifact),
            "sha256": bench.sha256_file(artifact)[0],
            "uri": "vulcan://runs/00000000-0000-4000-8000-000000000001/artifacts/llima",
        },
        "compatibility": {"platform_version": "3.0.0"},
        "provenance": {"compile_run_id": "00000000-0000-4000-8000-000000000001", "models_commit": "a" * 40},
    }
    manifest_path, report_path, result_path = [
        tmp_path / name for name in ("manifest.json", "report.json", "result.json")
    ]
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(data))
    env = {"buildinfo": "DISTRO_VERSION=3.0.0\n", "fingerprint_version": 2}
    with (
        patch("model_registry.core_benchmark.runner_environment", return_value=env),
        patch.object(bench.subprocess, "check_output", return_value="0.6.0+test"),
    ):
        bench.import_benchmark(manifest_path, report_path, result_path, ROOT / "schemas/performance-result.schema.json")
    result = load_benchmark_result(result_path, ROOT / "schemas/performance-result.schema.json")
    assert result["schema_version"] == 6
    assert result["benchmark_report"] == data
    assert result["execution_plan"] is None
    readings = _measurements(result, result["identity"]["fingerprint"])
    assert len(readings) == 20
    assert {row["dimensions"]["workload_id"] for row in readings} == {w["id"] for w in data["workloads"]}
    assert all(row["unit"] in ("ms", "tokens/s") for row in readings)
    image = [row for row in readings if row["dimensions"]["workload_id"] == "image"]
    assert len(image) == 2 and all("input_tokens" not in row["dimensions"] for row in image)
    bad = copy.deepcopy(result)
    bad["benchmark_report"]["workloads"].pop()
    result_path.write_text(json.dumps(bad))
    with pytest.raises(ValidationError, match="workloads"):
        load_benchmark_result(result_path, ROOT / "schemas/performance-result.schema.json")
