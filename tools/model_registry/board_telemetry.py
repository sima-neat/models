"""Sentinel traces around measured inference, summarized like Core's board telemetry."""

from __future__ import annotations

import http.client
import json
import math
import socket
import statistics
import time
import uuid
from contextlib import contextmanager
from urllib.parse import quote

SENTINEL_SOCKET = "/run/simaai-sentinel/api.sock"


def request(method: str, path: str, payload=None):
    connection = http.client.HTTPConnection("localhost", timeout=5)
    try:
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.settimeout(5)
        connection.sock.connect(SENTINEL_SOCKET)
        connection.request(
            method,
            path,
            body=json.dumps(payload) if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        document = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(document.get("error", f"Sentinel HTTP {response.status}"))
        return document
    finally:
        connection.close()


def summarize(run: dict) -> dict:
    """Use the same sensor selection, units and statistics as Core's BenchmarkTelemetry."""
    samples = run.get("samples", [])
    definitions = {row["key"]: row for row in run.get("metrics", [])}

    def numeric(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    def metric(key, values=None):
        if values is None:
            values = [row["values"].get(key) for row in samples]
        values = [value for value in values if numeric(value)]
        definition = definitions.get(key, {})
        return {
            "key": key,
            "label": definition.get("label", key),
            "unit": definition.get("unit", ""),
            "samples": len(values),
            "mean": statistics.mean(values) if values else 0.0,
            "minimum": min(values) if values else 0.0,
            "maximum": max(values) if values else 0.0,
        }

    def memory(key):
        return {**metric(key), "unit": "MiB"}  # Sentinel reports MiB but labels it MB.

    die_keys = [f"rtsn_{index}" for index in range(7)]  # Other rtsn channels repeat the same sites.
    hottest = [
        max((row["values"][key] for key in die_keys if numeric(row["values"].get(key))), default=None)
        for row in samples
    ]
    die_max = metric("die_max", hottest)
    die_max.update(label="Die max", unit=definitions.get("rtsn_0", {}).get("unit", ""))
    cpu_keys = sorted(
        (key for key in definitions if key.startswith("cpu_core_") and key.endswith("_usage_pct")),
        key=lambda key: int(key.removeprefix("cpu_core_").removesuffix("_usage_pct")),
    )
    metadata = run.get("metadata", {})
    return {
        "status": "ok" if samples else "Sentinel recorded no samples during measured requests",
        "trace_id": metadata.get("id", ""),
        "sentinel_version": metadata.get("sentinel_version", ""),
        "sample_interval_ms": metadata.get("sample_interval_ms", 0),
        "samples": len(samples),
        "power_som_watts": metric("power_current_watts"),
        "power_mla_watts": metric("power_rail_mla_watts"),
        "power_rails_watts": [
            metric(key) for key in definitions if key.startswith("power_rail_") and key.endswith("_watts")
        ],
        "die_temperatures_c": [metric(key) for key in die_keys],
        "die_temperature_max_c": die_max,
        "board_temperatures_c": [metric(key) for key in ("lm96163_temp1", "lm96163_temp2")],
        "cpu_usage_pct": metric("cpu_usage_pct"),
        "cpu_cores_usage_pct": [metric(key) for key in cpu_keys],
        "mla_memory_mib": memory("mla_mem_allocated_mb"),
        "ev74_memory_mib": memory("ev74_mem_allocated_mb"),
        "linux_memory_used_mib": memory("linux_mem_used_mb"),
    }


class BoardTelemetry:
    """Collect owned traces; required telemetry failures stop the benchmark immediately."""

    def __init__(self, workload: str):
        self.workload = workload
        self.runs = []
        self.duration_ms = 0.0

    @contextmanager
    def measure(self, enabled: bool = True):
        trace_id = None
        if enabled:
            trace_id = request(
                "POST",
                "/v1/traces",
                {
                    "name": f"llima-benchmark-{uuid.uuid4().hex}",
                    "note": self.workload,
                    "tags": ["llima-benchmark", self.workload],
                },
            )["trace"]["id"]
        started = time.monotonic()
        try:
            yield
        finally:
            if trace_id is not None:
                self.duration_ms += (time.monotonic() - started) * 1000
                request("POST", f"/v1/traces/{quote(trace_id, safe='')}/stop")
                self.runs.append(request("GET", f"/v1/runs/{quote(trace_id, safe='')}"))

    def report(self) -> dict:
        run = self.runs[0] if self.runs else {}
        telemetry = summarize(
            {**run, "samples": [sample for row in self.runs for sample in row.get("samples", [])]}
        )
        telemetry["trace_ids"] = [row["metadata"]["id"] for row in self.runs]
        result = {"telemetry": telemetry, "throughput_duration_ms": self.duration_ms}
        if telemetry["power_som_watts"]["samples"]:
            result["avg_power_watts"] = telemetry["power_som_watts"]["mean"]
        return result
