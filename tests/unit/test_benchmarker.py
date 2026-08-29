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
    module.InputKind = SimpleNamespace(Image=object())
    module.PreprocessColorFormat = SimpleNamespace(BGR=object())
    module.models = []

    class ModelOptions:
        decode_type = None
        top_k = None

        def __init__(self) -> None:
            self.preprocess = SimpleNamespace(
                kind=None,
                color_convert=SimpleNamespace(input_format=None),
            )

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


def test_build_model_applies_nested_model_options(tmp_path: Path) -> None:
    runtime = fake_pyneat()
    model = benchmark.build_model(
        runtime,
        tmp_path / "model.tar.gz",
        {
            "preprocess": {
                "kind": "pyneat:InputKind.Image",
                "color_convert": {"input_format": "pyneat:PreprocessColorFormat.BGR"},
            },
            "decode_type": "pyneat:BoxDecodeType.YoloV26",
        },
    )

    assert model.options.preprocess.kind is runtime.InputKind.Image
    assert model.options.preprocess.color_convert.input_format is runtime.PreprocessColorFormat.BGR
    assert model.options.decode_type is runtime.BoxDecodeType.YoloV26


def test_bounded_runner_materializes_dynamic_image_geometry() -> None:
    runtime = fake_pyneat()
    image_format = object()
    wrapped: list[tuple[Any, bool, Any]] = []

    class Tensor:
        @staticmethod
        def from_numpy(array: Any, *, copy: bool, image_format: Any) -> Any:
            wrapped.append((array, copy, image_format))
            return "image-tensor"

    runtime.Tensor = Tensor
    model = runtime.Model("model.tar.gz")
    model.input_specs = lambda: [
        SimpleNamespace(
            shape=[-1, -1, 3],
            dtypes=[SimpleNamespace(name="UInt8")],
            image_format=image_format,
        )
    ]

    inputs = benchmark._synthetic_runner_inputs(
        runtime,
        model,
        {"original_width": 1920, "original_height": 1080},
    )

    assert inputs == ["image-tensor"]
    assert wrapped[0][0].shape == (1080, 1920, 3)
    assert wrapped[0][1:] == (True, image_format)


def test_bounded_runner_preserves_static_image_geometry() -> None:
    runtime = fake_pyneat()
    image_format = object()
    wrapped: list[Any] = []

    class Tensor:
        @staticmethod
        def from_numpy(array: Any, *, copy: bool, image_format: Any) -> Any:
            wrapped.append(array)
            return "image-tensor"

    runtime.Tensor = Tensor
    model = runtime.Model("model.tar.gz")
    model.input_specs = lambda: [
        SimpleNamespace(
            shape=[640, 640, 3],
            dtypes=[SimpleNamespace(name="UInt8")],
            image_format=image_format,
        )
    ]

    benchmark._synthetic_runner_inputs(
        runtime,
        model,
        {"original_width": 1280, "original_height": 720},
    )

    assert wrapped[0].shape == (640, 640, 3)


def add_bounded_runner(runtime: ModuleType) -> None:
    class RunOptions:
        startup_preflight = True
        board_power_enabled = False

        def enable_board_power(self) -> None:
            self.board_power_enabled = True

    class MeasureOptions:
        pass

    class MeasurementScope:
        def __init__(self, throughput: bool) -> None:
            self.throughput = throughput

        def stop(self) -> Any:
            payload = {
                "measurement": {
                    "end_to_end": {"avg_ms": 4.5, "p50_ms": 4.4, "p95_ms": 4.7},
                    "throughput_inferences_per_s": 222.0,
                },
                "graph_metrics": {
                    "power": {"total_avg_watts": 7.5, "energy_joules": 0.2},
                },
            }
            return SimpleNamespace(to_json=lambda: json.dumps(payload))

    class Runner:
        def __init__(self, options: RunOptions) -> None:
            self.options = options
            self.run_calls = 0
            self.push_calls = 0
            self.pull_calls = 0
            self.closed = False

        def run(self, inputs: list[Any], timeout_ms: int) -> None:
            self.run_calls += 1

        def start_measurement(self, options: MeasureOptions) -> MeasurementScope:
            return MeasurementScope("throughput" in options.title)

        def push(self, inputs: list[Any]) -> bool:
            self.push_calls += 1
            return True

        def pull(self, timeout_ms: int) -> object:
            self.pull_calls += 1
            return object()

        def close(self) -> None:
            self.closed = True

    runtime.runners = []
    model_type = runtime.Model

    def build(self: Any, inputs: list[Any], *, run_options: RunOptions) -> Runner:
        runner = Runner(run_options)
        runtime.runners.append(runner)
        return runner

    model_type.build = build
    runtime.RunOptions = RunOptions
    runtime.MeasureOptions = MeasureOptions


def validate(schema_name: str, value: dict[str, Any]) -> None:
    schema = json.loads((ROOT / "schemas" / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(value)


def sequence_memory_sampler(values: list[int]) -> Any:
    remaining = iter(values)
    return benchmark.MlaMemorySampler(
        reader=lambda: (next(remaining), None),
        interval_seconds=60,
    )


def test_parses_sentinel_mla_allocator_summary_and_classifies_failures(tmp_path: Path) -> None:
    assert benchmark.parse_mla_memory_bytes(
        "| Total buffers allocated: 2 | Total allocated size: 0x01800000 |"
    ) == 24 * 1024 * 1024

    malformed = tmp_path / "simaai-mem"
    malformed.write_text("allocator unavailable\n", encoding="utf-8")
    assert benchmark.read_mla_memory_bytes(malformed) == (None, "parse_error")
    assert benchmark.read_mla_memory_bytes(tmp_path / "missing") == (None, "device_not_found")


def test_mla_sampler_reports_peak_delta_and_sampling_context() -> None:
    mib = 1024 * 1024
    sampler = sequence_memory_sampler([10 * mib, 14 * mib, 23 * mib, 18 * mib])
    sampler.start()
    sampler.sample()
    sampler.sample()
    sampler.stop()

    metrics = sampler.metrics()
    assert metrics["mla_memory_baseline_mib"]["value"] == 10
    assert metrics["mla_memory_peak_mib"]["value"] == 23
    assert metrics["mla_memory_peak_delta_mib"]["value"] == 13
    assert metrics["mla_memory_final_mib"]["value"] == 18
    assert metrics["mla_memory_peak_delta_mib"]["aggregation"] == "maximum"
    assert metrics["mla_memory_peak_delta_mib"]["details"] == {
        "source": "/dev/simaai-mem:Total allocated size",
        "scope": "board_allocator",
        "phase": "model_load_and_benchmark",
        "sampling_interval_ms": 60000,
        "read_attempts": 4,
        "read_failures": {},
    }


def test_mla_sampler_keeps_memory_unavailable_without_turning_it_into_zero() -> None:
    sampler = benchmark.MlaMemorySampler(
        reader=lambda: (None, "permission_denied"),
        interval_seconds=60,
    )
    sampler.start()
    sampler.stop()

    measurement = sampler.metrics()["mla_memory_peak_delta_mib"]
    assert measurement["status"] == "unavailable"
    assert measurement["reason"] == "permission_denied"
    assert measurement["value"] is None


def test_mla_sampler_treats_zero_as_valid_and_preserves_intermittent_failures() -> None:
    mib = 1024 * 1024
    outcomes = iter([(0, None), (None, "read_error"), (8 * mib, None), (4 * mib, None)])
    sampler = benchmark.MlaMemorySampler(reader=lambda: next(outcomes), interval_seconds=60)
    sampler.start()
    sampler.sample()
    sampler.sample()
    sampler.stop()

    measurement = sampler.metrics()["mla_memory_peak_delta_mib"]
    assert measurement["status"] == "measured"
    assert measurement["value"] == 8
    assert measurement["details"]["read_attempts"] == 4
    assert measurement["details"]["read_failures"] == {"read_error": 1}


def test_mla_sampler_marks_failed_terminal_sample_unavailable() -> None:
    mib = 1024 * 1024
    outcomes = iter([(2 * mib, None), (7 * mib, None), (None, "read_error")])
    sampler = benchmark.MlaMemorySampler(reader=lambda: next(outcomes), interval_seconds=60)
    sampler.start()
    sampler.sample()
    sampler.stop()

    metrics = sampler.metrics()
    assert metrics["mla_memory_peak_mib"]["status"] == "measured"
    assert metrics["mla_memory_peak_mib"]["value"] == 7
    assert metrics["mla_memory_final_mib"]["status"] == "unavailable"
    assert metrics["mla_memory_final_mib"]["reason"] == "read_error"
    assert metrics["mla_memory_final_mib"]["value"] is None
    assert metrics["mla_memory_final_mib"]["details"]["read_failures"] == {"read_error": 1}


def test_benchmarks_complete_multi_elf_graph_and_emits_valid_result(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk, mla_stages=2, extra_elfs=1))
    validate("performance-test-v1.schema.json", manifest)

    runtime = fake_pyneat()
    mib = 1024 * 1024
    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        mla_memory_sampler=sequence_memory_sampler(
            [10 * mib, 14 * mib, 15 * mib, 18 * mib, 16 * mib, 12 * mib]
        ),
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
    assert result["metrics"]["mla_memory_peak_delta_mib"]["value"] == 8
    assert result["metrics"]["mla_memory_peak_delta_mib"]["unit"] == "MiB"
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
    assert [sample["phase"] for sample in result["environment"]["temperature_samples"]] == [
        "start",
        "trial_complete",
        "trial_complete",
        "end",
    ]
    assert [sample["trial"] for sample in result["environment"]["temperature_samples"]] == [
        None,
        1,
        2,
        None,
    ]
    validate("performance-result-v3.schema.json", result)


def test_measures_model_load_and_time_to_first_inference_with_one_startup_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk), trials=2)
    runtime = fake_pyneat()
    add_bounded_runner(runtime)
    clock_values = iter([1_000_000_000, 1_025_000_000, 1_180_000_000])
    mib = 1024 * 1024
    startup_order: list[str] = []
    original_measure_startup = benchmark.measure_startup
    original_resolved_model = benchmark.resolved_model

    def measured_startup(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        startup_order.append("startup")
        return original_measure_startup(*args, **kwargs)

    def inspected_model(model: Any) -> dict[str, Any]:
        startup_order.append("resolved_model")
        return original_resolved_model(model)

    monkeypatch.setattr(benchmark, "measure_startup", measured_startup)
    monkeypatch.setattr(benchmark, "resolved_model", inspected_model)

    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        power_preflight={"status": "available", "reason": None},
        mla_memory_sampler=sequence_memory_sampler(
            [10 * mib, 14 * mib, 18 * mib, 20 * mib, 16 * mib, 12 * mib]
        ),
        clock_ns=lambda: next(clock_values),
    )

    assert result["metrics"]["model_load_ms"]["value"] == 25.0
    assert result["metrics"]["model_load_ms"]["aggregation"] == "single"
    first = result["metrics"]["time_to_first_inference_ms"]
    assert first["value"] == 180.0
    assert first["sample_count"] == 1
    assert first["details"]["includes"] == [
        "model_construction",
        "input_preparation",
        "graph_build",
        "first_inference",
    ]
    assert first["details"]["cache_state"] == "uncontrolled_after_sha256_validation"
    assert first["details"]["model_source_pre_read"] == "sha256_validation"
    assert first["details"]["startup_preflight"] is True
    assert first["details"]["model_source"]["path"] == str(mpk.resolve())
    assert len(runtime.runners) == 1
    assert runtime.runners[0].run_calls == 1
    assert runtime.runners[0].options.startup_preflight is True
    assert runtime.runners[0].closed
    assert startup_order == ["startup", "resolved_model"]
    validate("performance-result-v3.schema.json", result)


def test_startup_probe_failure_is_nonfatal(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk), trials=1)
    runtime = fake_pyneat()
    clock_values = iter([1_000_000_000, 1_010_000_000])
    mib = 1024 * 1024

    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        power_preflight={"status": "available", "reason": None},
        mla_memory_sampler=sequence_memory_sampler(
            [10 * mib, 14 * mib, 18 * mib, 16 * mib, 12 * mib]
        ),
        clock_ns=lambda: next(clock_values),
    )

    assert result["status"] == "passed"
    assert result["metrics"]["model_load_ms"]["value"] == 10.0
    first = result["metrics"]["time_to_first_inference_ms"]
    assert first["status"] == "unavailable"
    assert first["reason"] == "startup_probe_failed"
    assert first["details"]["error_type"] == "AttributeError"
    validate("performance-result-v3.schema.json", result)


def test_temperature_snapshot_disambiguates_duplicate_sensor_labels(tmp_path: Path) -> None:
    for index, source in enumerate(("sima_thermal", "soc_thermal")):
        sensor = tmp_path / f"hwmon{index}"
        sensor.mkdir()
        (sensor / "name").write_text(source, encoding="utf-8")
        (sensor / "temp1_input").write_text(str(58000 + index * 1000), encoding="utf-8")
        (sensor / "temp1_label").write_text("temp1", encoding="utf-8")

    assert benchmark._temperature_snapshot(tmp_path) == [
        {
            "sensor_id": "hwmon0/temp1",
            "sensor": "temp1",
            "source": "sima_thermal",
            "celsius": 58.0,
        },
        {
            "sensor_id": "hwmon1/temp1",
            "sensor": "temp1",
            "source": "soc_thermal",
            "celsius": 59.0,
        },
    ]


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
    validate("performance-result-v3.schema.json", result)


def test_bounded_runner_uses_explicit_warmup_and_sample_counts(tmp_path: Path) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk), trials=3)
    manifest["benchmark"] = {
        "mode": "bounded_runner",
        "warmup_samples": 1,
        "samples": 5,
        "trials": 3,
        "include_plugin_latency": False,
    }
    validate("performance-test-v1.schema.json", manifest)
    runtime = fake_pyneat()
    add_bounded_runner(runtime)
    mib = 1024 * 1024

    result = benchmark.run_benchmark(
        manifest,
        pyneat=runtime,
        actual_platform="2.1.3",
        power_preflight={"status": "available", "reason": None},
        mla_memory_sampler=sequence_memory_sampler(
            [20 * mib, 24 * mib, 28 * mib, 32 * mib, 40 * mib, 36 * mib, 28 * mib]
        ),
    )

    assert runtime.models[0].calls == 0
    assert len(runtime.runners) == 4
    assert runtime.runners[0].run_calls == 1
    assert runtime.runners[0].options.startup_preflight is True
    assert all(runner.run_calls == 6 for runner in runtime.runners[1:])
    assert all(runner.push_calls == 5 and runner.pull_calls == 5 for runner in runtime.runners[1:])
    assert all(runner.options.startup_preflight is False for runner in runtime.runners[1:])
    assert all(runner.options.board_power_enabled for runner in runtime.runners[1:])
    assert all(runner.closed for runner in runtime.runners)
    assert result["configuration"]["mode"] == "bounded_runner"
    assert result["configuration"]["warmup_samples"] == 1
    assert result["metrics"]["latency_p50_ms"]["value"] == 4.4
    assert result["metrics"]["latency_p95_ms"]["value"] == 4.7
    assert result["metrics"]["avg_power_watts"]["value"] == 7.5
    assert result["metrics"]["mla_memory_peak_delta_mib"]["value"] == 20
    assert result["power_measurement"]["source"] == "pyneat.ModelRunner.measure"
    validate("performance-result-v3.schema.json", result)


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
    assert result["model"]["contract"]["name"] == "fixture-model"
    assert result["model"]["contract"]["mla_stage_count"] == 1
    validate("performance-result-v3.schema.json", result)


def test_cli_preserves_runtime_details_and_completed_trials_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mpk = tmp_path / "model.tar.gz"
    manifest = make_manifest(mpk, make_mpk(mpk), trials=2)
    manifest_path = tmp_path / "test.json"
    output = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = fake_pyneat()
    base_model = runtime.Model

    class FailingSecondTrialModel(base_model):
        def benchmark(self, options: Any) -> Any:
            if self.calls == 1:
                raise RuntimeError("second trial failed")
            return super().benchmark(options)

    runtime.Model = FailingSecondTrialModel
    preflight = {
        "status": "available",
        "reason": None,
        "runtime_user": "sima",
        "required_group": "i2c",
        "devices": [{"path": "/dev/i2c-3", "readable": True, "writable": True}],
    }
    monkeypatch.setenv("SIMA_PLATFORM_VERSION", "2.1.3")
    monkeypatch.setattr(benchmark, "import_pyneat", lambda: runtime)
    monkeypatch.setattr(benchmark, "power_telemetry_preflight", lambda: preflight)
    mib = 1024 * 1024
    memory_sampler = sequence_memory_sampler([5 * mib, 11 * mib, 17 * mib, 9 * mib])
    monkeypatch.setattr(benchmark, "MlaMemorySampler", lambda: memory_sampler)

    assert benchmark.main(["--manifest", str(manifest_path), "--output", str(output)]) == 4
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "error"
    assert result["failure_classification"] == "benchmark"
    assert result["compatibility"]["actual_neat_version"] == "0.4.0+unit-test"
    assert result["model"]["contract"]["name"] == "fixture-model"
    assert result["model"]["resolved"]["postprocess"] == "boxdecode"
    assert len(result["trials"]) == 1
    assert result["trials"][0]["latency_avg_ms"] == 11.0
    assert result["metrics"]["latency_avg_ms"]["value"] == 11.0
    assert result["metrics"]["latency_avg_ms"]["sample_count"] == 1
    assert result["metrics"]["mla_memory_peak_delta_mib"]["value"] == 12
    assert result["metrics"]["mla_memory_peak_delta_mib"]["sample_count"] == 4
    assert result["environment"]["observations"]["end"] is not None
    validate("performance-result-v3.schema.json", result)


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
    validate("performance-result-v3.schema.json", result)


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
