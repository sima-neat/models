from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    layer_stats, stage_stats = component["mla_layer_stats"], component["mla_stage_stats"]
    assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok")
    assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), "Core returned no MLA layer latency"
    assert stage_stats["stages"], "Core returned no MLA stage latency"


def test_report_names_the_full_landmark_boundary(performance):
    workload = performance["workload"]
    assert "CPU crop/resize" in workload and "native BF16 normalization" in workload
    assert "landmark decode" in workload
    assert "face detection excluded" in workload and "HEATMAP-ONLY" not in workload
    (component,) = performance["components"].values()
    (prepared,) = component["execution_plan"]["edges"][0]["tensors"]
    assert prepared["shape"][-3:] == [256, 256, 3]  # 720p is outside the prepared-image Neat graph.
    assert prepared["dtype"] == "uint8"


def test_benchmark_maps_the_face_box_not_the_padding(load_script):
    import numpy as np

    script = load_script("performance")
    frame, face_box = script.benchmark_frame(np.full((101, 199, 3), 200, np.uint8))
    scale = min(1280 / 199, 720 / 101)
    width, height = round(199 * scale), round(101 * scale)
    left, top = (1280 - width) // 2, (720 - height) // 2
    assert frame.shape == (720, 1280, 3)
    np.testing.assert_array_equal(face_box, [left, top, left + width, top + height])
    assert (frame[top : top + height, left : left + width] == 200).all()
    assert face_box.tolist() != [0, 0, 1280, 720]


def test_diagnostic_input_capacity_is_real_720p(load_script):
    options = load_script("performance").image_options()
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)


def test_prepared_crop_options_only_normalize_at_the_bf16_boundary(load_script):
    script = load_script("performance")
    options = script.prepared_rgb_options()
    assert options.preprocess.kind == script.pyneat.InputKind.Image
    assert options.preprocess.enable == options.preprocess.normalize.enable == script.pyneat.AutoFlag.On
    assert options.preprocess.resize.enable == options.preprocess.color_convert.enable == script.pyneat.AutoFlag.Off
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (256, 256)
    assert options.preprocess.normalize.has_explicit_stats is True
    assert options.preprocess.normalize.mean == [0.0] * 3
    assert options.preprocess.normalize.stddev == [float(script.U8_BF16_STDDEV)] * 3


def test_all_256_pixels_match_reference_bf16_round_to_nearest_even(load_script):
    script = load_script("performance")
    pixels = np.arange(256, dtype=np.float32)
    # Independent reference: cast the unchanged normalized source to BF16 RNE.
    bits = (pixels / np.float32(255)).view(np.uint32)
    expected = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    # Existing Core public normalize stats -> multiplier -> EV74 BF16 truncation.
    scale = np.float32(1.0 / (float(script.U8_BF16_STDDEV) * 255.0))
    assert scale == np.float32(129 / 32768)
    actual = ((pixels * scale).view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)
    # Negative control: ordinary FP32 1/255 is not equivalent through a truncating converter.
    naive = ((pixels * np.float32(1 / 255)).view(np.uint32) >> 16).astype(np.uint16)
    assert np.count_nonzero(naive != expected) == 254


def test_prepare_input_preserves_exact_crop_bytes_and_declares_rgb(load_script, monkeypatch):
    script = load_script("performance")
    landmarks = load_script("landmarks")
    image = np.random.default_rng(31).integers(0, 256, (47, 63, 3), dtype=np.uint8)
    face_box = np.array([-5, -3, 37, 45])
    reference, reference_box = landmarks.crop(image, face_box)
    calls = []

    def from_numpy(array, **kwargs):
        calls.append((array, kwargs))
        return "prepared-image"

    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", from_numpy)
    inputs, box = script.prepare_input(image, face_box)
    assert inputs == ["prepared-image"]
    ((pixels, kwargs),) = calls
    assert pixels.dtype == np.uint8 and pixels.shape == (256, 256, 3)
    np.testing.assert_array_equal(pixels.astype(np.float32) / np.float32(255), reference)
    np.testing.assert_array_equal(box, reference_box)
    assert kwargs == {
        "copy": True,
        "layout": script.pyneat.TensorLayout.HWC,
        "image_format": script.pyneat.PixelFormat.RGB,
    }


@pytest.mark.parametrize(
    ("dtype", "quant_needed"),
    [("INT8", True), ("EVXX_FLOAT32", False), ("INT16", False), ("", False), ("EVXX_BFLOAT16", True)],
)
def test_prepared_crop_rejects_other_model_contracts_before_input_or_runner(
    load_script, monkeypatch, tmp_path, dtype, quant_needed
):
    from types import SimpleNamespace

    script = load_script("performance")
    monkeypatch.setattr(script, "read_rgb", lambda _: "image")
    monkeypatch.setattr(script, "benchmark_frame", lambda _: ("frame", "box"))

    def unexpected(*args, **kwargs):
        pytest.fail("a rejected model must not prepare tensors or build a runner")

    model = SimpleNamespace(
        preprocess_requirements=lambda: SimpleNamespace(output_dtype=dtype, quant_needed=quant_needed),
        build=unexpected,
    )
    monkeypatch.setattr(script, "prepare_input", unexpected)
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: model)
    with pytest.raises(ValueError, match="requires BF16 model inputs without quantization"):
        script.benchmark_pipeline(Path("model"), Path("image"), tmp_path)


def test_sequential_frame_includes_prepare_and_same_decode(load_script, monkeypatch):
    from types import SimpleNamespace

    script = load_script("performance")
    calls = []
    monkeypatch.setattr(script, "prepare_input", lambda frame, box: (calls.append("prepare") or ["crop"], "box"))
    monkeypatch.setattr(script, "consume_outputs", lambda tensors, box: calls.append(("decode", tensors, box)))
    runner = SimpleNamespace(run=lambda inputs, timeout_ms: calls.append(("run", inputs)) or ["heatmap"])
    script.infer_one(runner, "frame", "face")
    assert calls == ["prepare", ("run", ["crop"]), ("decode", ["heatmap"], "box")]


def test_performance_borrows_heatmaps_only_until_decode_finishes(load_script, monkeypatch):
    import numpy as np

    script = load_script("performance")
    calls = []
    heatmaps = np.ones((68, 64, 64), np.float32)
    heatmaps.flags.writeable = False

    def adapt(tensor, *, copy):
        calls.append((tensor, copy))
        return heatmaps

    monkeypatch.setattr(script, "heatmaps_of", adapt)
    points, scores = script.consume_outputs(["tensor"], np.array([0, 0, 256, 256]))
    assert calls == [("tensor", False)]
    assert not np.shares_memory(points, heatmaps) and not np.shares_memory(scores, heatmaps)
    assert (heatmaps == 1).all()


def _throughput_fixture(script, monkeypatch, *, decode_failure=False, push_failure=False):
    import queue
    import threading
    from types import SimpleNamespace

    import numpy as np

    monkeypatch.setattr(script, "WARMUP_FRAMES", 2)
    monkeypatch.setattr(script, "MEASURED_FRAMES", 3)
    monkeypatch.setattr(script, "MIN_DURATION_S", 0)
    events = []
    box = np.array([0, 0, 10, 10])
    monkeypatch.setattr(script, "crop_rgb", lambda frame, face: ("prepared", box))
    monkeypatch.setattr(script, "prepare_input", lambda frame, face: (events.append("prepare") or ["input"], box))

    def consume(tensors, output_box):
        events.append("decode")
        assert tensors == ["heatmap"]
        np.testing.assert_array_equal(output_box, box)
        if decode_failure:
            raise ValueError("decode fault")

    monkeypatch.setattr(script, "consume_outputs", consume)

    class Monitor:
        active = False

        def start(self):
            self.active = True
            events.append("monitor-start")

        def running(self):
            return self.active

        def stop(self):
            self.active = False
            events.append("monitor-stop")
            return "telemetry"

    monitor = Monitor()
    monkeypatch.setattr(script.pyneat, "TelemetryMonitor", lambda: monitor)

    class Runner:
        def __init__(self):
            self.queue = queue.Queue(maxsize=4)
            self.closed = threading.Event()
            self.pushed = 0

        def push(self, inputs):
            if push_failure:
                return False
            while not self.closed.is_set():
                try:
                    self.queue.put(["heatmap"], timeout=0.01)
                    self.pushed += 1
                    return True
                except queue.Full:
                    pass
            return False

        def pull(self, timeout):
            while True:
                try:
                    return SimpleNamespace(tensors=self.queue.get(timeout=0.01))
                except queue.Empty:
                    if self.closed.is_set():
                        return SimpleNamespace(tensors=[])

        def close_input(self):
            self.closed.set()

    return Runner(), monitor, events


def test_throughput_decodes_every_output_and_stops_telemetry(load_script, monkeypatch):
    script = load_script("performance")
    runner, monitor, events = _throughput_fixture(script, monkeypatch)
    result = script.measure_throughput(runner, "frame", "face")
    assert result.frames >= 3 and result.telemetry == "telemetry"
    assert events.count("prepare") == events.count("decode") == runner.pushed
    assert events.count("monitor-start") == events.count("monitor-stop") == 1
    assert not monitor.running() and runner.closed.is_set()


def test_throughput_propagates_decode_error_without_live_threads(load_script, monkeypatch):
    import threading

    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch, decode_failure=True)
    with pytest.raises(ValueError, match="decode fault"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()
    assert not any(t.name in {"fan-produce", "fan-consume"} for t in threading.enumerate())


def test_throughput_rejects_failed_submission(load_script, monkeypatch):
    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch, push_failure=True)
    with pytest.raises(RuntimeError, match="rejected"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()


def test_benchmark_closes_runner_when_latency_raises(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    script = load_script("performance")
    closed = []
    runner = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(script, "read_rgb", lambda _: "image")
    monkeypatch.setattr(script, "benchmark_frame", lambda _: ("frame", "box"))
    monkeypatch.setattr(script, "prepare_input", lambda *args: (["input"], "box"))
    monkeypatch.setattr(
        script.pyneat,
        "Model",
        lambda *args: SimpleNamespace(
            preprocess_requirements=lambda: SimpleNamespace(output_dtype="EVXX_BFLOAT16", quant_needed=False),
            build=lambda *a, **k: runner,
        ),
    )

    def failed(*args):
        raise RuntimeError("latency fault")

    monkeypatch.setattr(script, "measure_latency", failed)
    with pytest.raises(RuntimeError, match="latency fault"):
        script.benchmark_pipeline(Path("model"), Path("image"), tmp_path)
    assert closed == [True]


def test_benchmark_uses_a_seeded_annotated_face(load_script, tmp_path):
    import random

    script = load_script("performance")
    split = tmp_path / "test_data"
    split.mkdir()
    names = [f"{index}_1_0.jpg" for index in range(3)]
    for name in names:
        (split / name).write_bytes(b"image")
        (split / name).with_suffix(".pts").write_text("annotation")
    (split / "unannotated.jpg").write_bytes(b"image")
    (split / "._0_1_0.jpg").write_bytes(b"metadata")
    assert script.sample_face(tmp_path).name == random.Random(0).choice(names)


def test_throughput_joins_started_producer_if_consumer_cannot_start(load_script, monkeypatch):
    import threading

    import pytest

    script = load_script("performance")
    runner, monitor, _ = _throughput_fixture(script, monkeypatch)
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == "fan-consume":
            raise RuntimeError("thread creation fault")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="thread creation fault"):
        script.measure_throughput(runner, "frame", "face")
    assert runner.closed.is_set() and not monitor.running()
    assert not any(t.name in {"fan-produce", "fan-consume"} for t in threading.enumerate())
