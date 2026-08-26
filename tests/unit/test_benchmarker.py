from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tools" / "runtime"


def load_runtime_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, RUNTIME / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_runtime_module("benchmarker")


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def make_mpk(
    path: Path,
    *,
    mla_stages: int = 1,
    extra_elfs: int = 0,
    sdk_version: str = "2.1.3",
    batches: list[int] | None = None,
) -> str:
    plugins = []
    batches = batches or [1] * mla_stages
    with tarfile.open(path, "w:gz") as archive:
        for index in range(mla_stages):
            elf = f"stage-{index}.elf"
            add_tar_bytes(archive, elf, f"elf {index}".encode())
            plugins.append(
                {
                    "name": f"mla-{index}",
                    "processor": "MLA",
                    "resources": {"executable": elf},
                    "config_params": {"actual_batch_size": batches[index]},
                }
            )
        for index in range(extra_elfs):
            add_tar_bytes(archive, f"support-{index}.elf", b"support")
        manifest = {
            "name": "fixture-model",
            "model_sdk_version": sdk_version,
            "plugins": plugins,
        }
        add_tar_bytes(archive, "fixture_mpk.json", json.dumps(manifest).encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest(path: Path, sha256: str, *, trials: int = 2) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": {
            "id": "fixture_model",
            "variant_id": "modalix_bf16",
            "path": str(path),
            "sha256": sha256,
        },
        "compatibility": {"platform_version": "2.1.3", "neat_version": "0.4.0"},
        "benchmark": {"samples": 25, "trials": trials, "include_plugin_latency": True},
        "model_options": {"decode_type": "pyneat:BoxDecodeType.YoloV26", "top_k": 100},
        "provenance": {"models_commit": "abc123"},
    }


def fake_pyneat() -> ModuleType:
    module = ModuleType("pyneat")
    module.__version__ = "0.4.0+unit-test"
    module.BoxDecodeType = SimpleNamespace(YoloV26=object())
    module.models = []

    class ModelOptions:
        decode_type = None
        top_k = None

    class BenchmarkOptions:
        num_samples = 100
        include_plugin_latency = False
        original_width = None
        original_height = None

    class Model:
        def __init__(self, path: str, options: Any = None) -> None:
            self.path = path
            self.options = options
            self.calls = 0
            module.models.append(self)

        def info(self) -> Any:
            return SimpleNamespace(
                selection=SimpleNamespace(selected_post_kind="boxdecode"),
                output_topology=SimpleNamespace(physical_outputs=2, logical_outputs=1, packed_outputs=True),
            )

        def input_specs(self) -> list[str]:
            return [
                SimpleNamespace(
                    shape=[224, 224, 3],
                    rank=3,
                    dtypes=[SimpleNamespace(name="Float32")],
                    image_format=None,
                    device=None,
                    preferred_device=None,
                    allowed_devices=[],
                    allow_composite=True,
                    required_segment_names=[],
                )
            ]

        def output_specs(self) -> list[str]:
            return [SimpleNamespace(shape=[1, 1000], rank=2, dtypes=[SimpleNamespace(name="Float32")])]

        def benchmark(self, options: Any) -> Any:
            self.calls += 1
            assert options.num_samples == 25
            assert options.include_plugin_latency is True
            return SimpleNamespace(
                latency_ms=10.0 + self.calls,
                fps=100.0 + (10.0 * self.calls),
                avg_power_watts=0.0,
                energy_joules=0.0,
            )

    module.ModelOptions = ModelOptions
    module.BenchmarkOptions = BenchmarkOptions
    module.Model = Model
    return module


def validate(schema_name: str, value: dict[str, Any]) -> None:
    schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(value)


def test_benchmarks_complete_multi_elf_graph_and_emits_valid_result(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk, mla_stages=2, extra_elfs=1))
    validate("performance-test-v1.schema.json", manifest)

    runtime = fake_pyneat()
    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        power_preflight={
            "status": "unavailable",
            "reason": "permission_denied",
            "runtime_user": "sima",
            "required_group": "i2c",
            "devices": [{"path": "/dev/i2c-3", "readable": False, "writable": False}],
        },
    )

    assert result["status"] == "passed"
    assert result["tool"] == {"name": "benchmarker", "version": "source"}
    assert result["model"]["contract"] == {
        "name": "fixture-model",
        "model_sdk_version": "2.1.3",
        "elf_count": 3,
        "mla_stage_count": 2,
        "compiled_batch_size": 1,
        "mla_stages": ["mla-0", "mla-1"],
    }
    assert result["metrics"]["latency_avg_ms"]["value"] == 11.5
    assert result["metrics"]["latency_avg_ms"]["minimum"] == 11.0
    assert result["metrics"]["latency_avg_ms"]["maximum"] == 12.0
    assert result["metrics"]["latency_avg_ms"]["standard_deviation"] == 0.5
    assert result["metrics"]["throughput_fps"]["value"] == 115.0
    assert result["metrics"]["latency_p50_ms"]["status"] == "unsupported"
    assert result["metrics"]["avg_power_watts"]["status"] == "unavailable"
    assert result["metrics"]["avg_power_watts"]["reason"] == "permission_denied"
    assert result["identity"]["fingerprint"] is not None
    assert result["environment"]["fingerprint"]
    assert result["power_measurement"]["scope"] == "board_pmbus_rails"
    assert result["model"]["resolved"]["output_topology"] == {
        "physical": 2,
        "logical": 1,
        "packed": True,
    }
    assert result["model"]["resolved"]["input_specs"][0]["shape"] == [224, 224, 3]
    assert result["model"]["resolved"]["input_specs"][0]["dtypes"] == ["Float32"]
    assert runtime.models[0].options.decode_type is runtime.BoxDecodeType.YoloV26
    assert runtime.models[0].options.top_k == 100
    validate("performance-result-v1.schema.json", result)


def test_measured_power_includes_cross_trial_statistics_and_energy_per_inference(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk), trials=2)
    runtime = fake_pyneat()
    base_model = runtime.Model

    class PoweredModel(base_model):
        def benchmark(self, options: Any) -> Any:
            report = super().benchmark(options)
            report.avg_power_watts = 10.0 + self.calls
            report.energy_joules = 2.5 * self.calls
            return report

    runtime.Model = PoweredModel
    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        power_preflight={
            "status": "available",
            "reason": None,
            "runtime_user": "sima",
            "required_group": "i2c",
            "devices": [{"path": "/dev/i2c-3", "readable": True, "writable": True}],
        },
    )

    assert result["metrics"]["avg_power_watts"]["value"] == 11.5
    assert result["metrics"]["energy_joules"]["value"] == 3.75
    assert result["metrics"]["energy_per_inference_joules"]["value"] == pytest.approx(0.15)
    assert result["metrics"]["energy_per_inference_joules"]["standard_deviation"] == pytest.approx(0.05)
    validate("performance-result-v1.schema.json", result)


def test_rejects_checksum_mismatch(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    make_mpk(mpk)
    manifest = make_manifest(mpk, "0" * 64)

    with pytest.raises(benchmark.ModelError, match="checksum mismatch"):
        benchmark.run_benchmark(manifest, pyneat=fake_pyneat(), actual_platform="2.1.3")


def test_rejects_missing_mla_and_inconsistent_multi_stage_batch(tmp_path: Path) -> None:
    no_mla = tmp_path / "no-mla.tar.gz"
    make_mpk(no_mla, mla_stages=0, extra_elfs=1)
    with pytest.raises(benchmark.CompatibilityError, match="does not contain an MLA"):
        benchmark.inspect_mpk(no_mla)

    mixed_batch = tmp_path / "mixed-batch.tar.gz"
    make_mpk(mixed_batch, mla_stages=2, batches=[1, 2])
    with pytest.raises(benchmark.CompatibilityError, match="inconsistent compiled batch"):
        benchmark.inspect_mpk(mixed_batch)


@pytest.mark.parametrize(
    ("platform", "neat", "message"),
    [
        ("2.2.0", "0.4.0", "platform version mismatch"),
        ("2.1.3", "0.5.0", "Neat version mismatch"),
    ],
)
def test_rejects_runtime_version_mismatch(platform: str, neat: str, message: str) -> None:
    runtime = fake_pyneat()
    runtime.__version__ = neat
    with pytest.raises(benchmark.CompatibilityError, match=message):
        benchmark.require_runtime_compatibility("2.1.3", "0.4.0", runtime, platform)


def test_manifest_rejects_version_not_supported_by_benchmarker(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk))
    manifest["compatibility"]["neat_version"] = "0.5.0"
    path = tmp_path / "test.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(benchmark.CompatibilityError, match="supports Neat 0.4.0"):
        benchmark.load_manifest(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [(None, "unexpected"), ("model", "unexpected"), ("benchmark", "unexpected")],
)
def test_manifest_rejects_unknown_fields(tmp_path: Path, section: str | None, field: str) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk))
    target = manifest if section is None else manifest[section]
    target[field] = True
    path = tmp_path / "test.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(benchmark.ConfigurationError, match="unknown field"):
        benchmark.load_manifest(path)


def test_detects_devkit_platform_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buildinfo = tmp_path / "buildinfo"
    buildinfo.write_text("DISTRO_NAME = eLxr\nDISTRO_VERSION = 2.1.3\n", encoding="utf-8")
    monkeypatch.delenv("SIMA_PLATFORM_VERSION", raising=False)
    assert benchmark.detect_platform_version(buildinfo) == "2.1.3"

    monkeypatch.setenv("SIMA_PLATFORM_VERSION", "2.1.3-local")
    assert benchmark.detect_platform_version(buildinfo) == "2.1.3-local"


def test_cli_writes_machine_readable_error_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk))
    manifest["model"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "test.json"
    output = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("SIMA_PLATFORM_VERSION", "2.1.3")

    assert benchmark.main(["--manifest", str(manifest_path), "--output", str(output)]) == 4
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert result["failure_classification"] == "model"
    validate("performance-result-v1.schema.json", result)


def test_cli_writes_schema_valid_error_for_invalid_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "invalid.json"
    output = tmp_path / "result.json"
    manifest_path.write_text('{"schema_version": 1, "unexpected": true}\n', encoding="utf-8")
    monkeypatch.setenv("SIMA_PLATFORM_VERSION", "2.1.3")

    assert benchmark.main(["--manifest", str(manifest_path), "--output", str(output)]) == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert result["failure_classification"] == "configuration"
    assert result["model"]["id"] is None
    validate("performance-result-v1.schema.json", result)


def test_power_preflight_warns_with_actionable_runtime_user(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    device = tmp_path / "i2c-3"
    device.touch(mode=0o000)
    preflight = benchmark.power_telemetry_preflight((device,))
    preflight["status"] = "unavailable"
    preflight["reason"] = "permission_denied"
    preflight["runtime_user"] = "sima"

    benchmark.warn_power_preflight(preflight)

    warning = capsys.readouterr().err
    assert "user 'sima'" in warning
    assert "i2c group" in warning
