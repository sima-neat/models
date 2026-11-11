from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sample(**changes):
    values = dict(status="ok", samples=1, power_samples=1, unit="W", mean=8.0)
    values.update(changes)
    return SimpleNamespace(
        status=values["status"],
        samples=values["samples"],
        power_som_watts=SimpleNamespace(samples=values["power_samples"], unit=values["unit"], mean=values["mean"]),
    )


@pytest.fixture
def preflight(monkeypatch):
    path = ROOT / "scripts/check_benchmark_telemetry.py"
    spec = importlib.util.spec_from_file_location("benchmark_telemetry_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ticks = itertools.count(0.0, 1.0)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
    return module


def monitor_module(monkeypatch, snapshots, *, final=None, error=None):
    events = []
    pending = iter(snapshots)
    latest = snapshots[-1]

    class Monitor:
        def start(self):
            events.append("start")

        def read(self):
            events.append("read")
            if error is not None:
                raise error
            return next(pending, latest)

        def stop(self):
            events.append("stop")
            return final if final is not None else latest

    monkeypatch.setitem(sys.modules, "pyneat", SimpleNamespace(TelemetryMonitor=Monitor))
    return events


def test_preflight_waits_for_fresh_power_and_closes_monitor(preflight, monkeypatch, capsys):
    events = monitor_module(monkeypatch, [sample(status="not started", samples=0, power_samples=0), sample()])
    assert preflight.main() == 0
    assert events == ["start", "read", "read", "stop"]
    assert "fresh SoM power" in capsys.readouterr().out


@pytest.mark.parametrize(
    "changes",
    [
        dict(status="could not connect"),
        dict(samples=0, power_samples=0),
        dict(power_samples=0),
        dict(samples=1, power_samples=2),
        dict(unit="mW"),
        dict(mean=0),
        dict(mean=-1),
        dict(mean=float("nan")),
        dict(mean=float("inf")),
    ],
)
def test_preflight_rejects_unavailable_or_invalid_power(preflight, monkeypatch, changes):
    events = monitor_module(monkeypatch, [sample(**changes)])
    with pytest.raises(SystemExit, match="requires sampled SoM power"):
        preflight.main()
    assert events[0] == "start" and events[-1] == "stop"


def test_preflight_validates_the_final_snapshot(preflight, monkeypatch):
    events = monitor_module(monkeypatch, [sample()], final=sample(power_samples=0))
    with pytest.raises(SystemExit, match="power samples=0"):
        preflight.main()
    assert events == ["start", "read", "stop"]


def test_preflight_closes_monitor_when_reading_fails(preflight, monkeypatch):
    events = monitor_module(monkeypatch, [sample()], error=RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"):
        preflight.main()
    assert events == ["start", "read", "stop"]
