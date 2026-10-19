"""Sentinel sample selection, lifecycle ownership, and workload-specific publication."""

from unittest.mock import patch

import pytest
from model_registry import board_telemetry as board
from model_registry.benchmark_metrics import measurement_name, report_metrics


def sentinel_run():
    units = {
        "power_current_watts": "W",
        "power_rail_mla_watts": "W",
        "rtsn_0": "C",
        "rtsn_1": "C",
        "rtsn_7": "C",
        "lm96163_temp1": "C",
        "cpu_usage_pct": "%",
        "cpu_core_10_usage_pct": "%",
        "cpu_core_2_usage_pct": "%",
        "cpu_core_0_usage_pct": "%",
        "mla_mem_allocated_mb": "MB",
        "ev74_cma_used_mb": "MB",
        "linux_mem_used_mb": "MB",
    }
    values = {
        "power_current_watts": 12.0,
        "power_rail_mla_watts": 2.0,
        "rtsn_0": 40.0,
        "rtsn_1": 42.0,
        "rtsn_7": 99.0,
        "lm96163_temp1": 38.0,
        "cpu_usage_pct": 10.0,
        "cpu_core_0_usage_pct": 2.0,
        "cpu_core_2_usage_pct": 80.0,
        "cpu_core_10_usage_pct": 5.0,
        "mla_mem_allocated_mb": 1024.0,
        "ev74_cma_used_mb": 2048.0,
        "linux_mem_used_mb": 500.0,
    }
    return {
        "metadata": {"id": "owned", "sentinel_version": "test", "sample_interval_ms": 2000},
        "metrics": [{"key": key, "unit": unit, "label": key} for key, unit in units.items()],
        "samples": [
            {"values": values},
            {"values": {**values, "rtsn_0": 44.0, "cpu_core_2_usage_pct": None, "linux_mem_used_mb": 600.0}},
        ],
    }


def test_sentinel_summary_matches_vision_sensor_and_memory_semantics():
    telemetry = board.summarize(sentinel_run())
    assert telemetry["die_temperature_max_c"]["mean"] == 43.0  # Excludes duplicate/trip channels rtsn_7+.
    assert [row["key"] for row in telemetry["cpu_cores_usage_pct"]] == [
        "cpu_core_0_usage_pct",
        "cpu_core_2_usage_pct",
        "cpu_core_10_usage_pct",
    ]
    assert telemetry["cpu_cores_usage_pct"][1]["samples"] == 1
    metrics = report_metrics({"telemetry": telemetry, "avg_power_watts": 12.0})
    assert metrics["avg_cpu_usage_pct"]["value"] == 10.0
    assert metrics["peak_cpu_core_usage_pct"]["value"] == 80.0
    assert metrics["avg_linux_memory_mib"]["value"] == 550.0
    assert metrics["mla_memory_peak_mib"]["unit"] == "MiB"
    assert metrics["peak_power_watts"]["details"]["breakdown"][0]["key"] == "power_rail_mla_watts"
    assert measurement_name("text_n128.avg_cpu_usage_pct") == "board.cpu_usage.avg"
    assert measurement_name("image.ttft_ms") == "model.generation.ttft"


def test_trace_excludes_warmup_and_stops_only_its_id_even_if_generation_raises():
    sampler = board.BoardTelemetry("text_n128")
    with patch.object(board, "request", side_effect=[{"trace": {"id": "owned"}}, {}, sentinel_run()]) as request:
        with sampler.measure(enabled=False):
            pass
        assert request.call_count == 0
        with pytest.raises(ValueError, match="generation failed"), sampler.measure():
            raise ValueError("generation failed")
    assert [call.args[:2] for call in request.call_args_list] == [
        ("POST", "/v1/traces"),
        ("POST", "/v1/traces/owned/stop"),
        ("GET", "/v1/runs/owned"),
    ]
    assert sampler.report()["telemetry"]["samples"] == 2


def test_busy_sentinel_does_not_stop_another_trace_or_fail_generation():
    sampler = board.BoardTelemetry("image")
    with patch.object(board, "request", side_effect=RuntimeError("another trace is active")) as request:
        for _ in range(2):
            with sampler.measure():
                pass
    request.assert_called_once()
    report = sampler.report()
    assert report["telemetry"]["status"] == "another trace is active"
    assert report["telemetry"]["samples"] == 0
    assert report_metrics(report) == {}
