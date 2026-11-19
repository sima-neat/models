from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == ["image_encoder", "prompt_decoder"]
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id


def test_real_frame_selection_is_repeatable_and_uses_rgb(load_script, tmp_path):
    import cv2
    import numpy as np

    script = load_script("performance")
    images = tmp_path / "val2017"
    images.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(images / name), np.full((30, 40, 3), (5, 60, 200), np.uint8))
    image = script.sample_image(tmp_path)
    assert image.name == "b.jpg"
    frame = script.read_frame(image)
    assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8
    assert frame[360, 640, 0] > frame[360, 640, 2]  # JPEG BGR converted to RGB.


def test_serial_path_prepares_each_frame_and_handoffs_only_between_components(load_script, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    script = load_script("performance")
    events = []
    monkeypatch.setattr(script, "prepare_prompt", lambda frame, encoding: events.append("prompt") or ["prompt"])
    monkeypatch.setattr(script, "handoff", lambda tensors: events.append("handoff") or tensors)
    monkeypatch.setattr(script, "finish", lambda tensors, frame: events.append("mask") or "final-mask")

    def run(name):
        return lambda inputs, **kw: events.append((name, inputs)) or ["output"]

    runners = [SimpleNamespace(run=run(name)) for name in script.COMPONENTS]
    for _ in range(2):
        assert script.infer_one(runners, "image", np.zeros((720, 1280, 3), np.uint8), "encoding") == "final-mask"
    # Neat prepares the same camera image inside the encoder's pipeline on every frame.
    step = ["prompt", ("image_encoder", ["image"]), "handoff", ("prompt_decoder", ["output", "prompt"]), "mask"]
    assert events == step * 2


def test_finish_reads_neats_decoded_mask_without_payload_copy(load_script, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    script = load_script("performance")
    reads, decoded = [], []

    class Tensor:
        def __init__(self, array):
            self.array = array

        def to_numpy(self, copy):
            reads.append(copy)
            self.array.flags.writeable = False
            return self.array

    masks = np.zeros((1, 300, 600), np.uint8)
    masks[0, 100:200] = 255

    def decode_segmentation(tensors):
        decoded.append(tensors)
        return [SimpleNamespace(masks=Tensor(masks))]

    monkeypatch.setattr(script.pyneat, "decode_segmentation", decode_segmentation)
    result = script.finish(["boxdecode"], np.zeros((300, 600, 3), np.uint8))
    assert decoded == [["boxdecode"]] and reads == [False]
    assert result.shape == (300, 600) and result.flags.owndata
    assert result[100:200].all() and not result[:100].any() and not result[200:].any()
    with pytest.raises(ValueError, match="one decoded mask"):
        script.finish(["masks", "iou_predictions"], np.zeros((300, 600, 3), np.uint8))
    with pytest.raises(ValueError, match="expected one"):
        script.finish(["boxdecode"], np.zeros((720, 1280, 3), np.uint8))


def test_neat_prepares_the_frame_and_decodes_the_mask_as_sam_does(load_script):
    import numpy as np

    script = load_script("performance")
    pyneat = script.pyneat

    encoder = script.image_options()
    preprocess = encoder.preprocess
    assert preprocess.kind == pyneat.InputKind.Image
    assert (preprocess.input_max_width, preprocess.input_max_height) == (1280, 720)
    assert preprocess.resize.mode == pyneat.ResizeMode.Letterbox
    assert preprocess.resize.placement == pyneat.LetterboxPlacement.TopLeft
    assert preprocess.resize.pad_value == 116
    assert preprocess.resize.interpolation == pyneat.ResizeInterpolation.BilinearAntialiased
    assert preprocess.rescale_factor == 1.0
    # segment_anything's float32 statistics on 0-255 pixels, which Core keeps as float32.
    assert np.float32(preprocess.normalize.mean).tolist() == np.float32([123.675, 116.28, 103.53]).tolist()
    assert np.float32(preprocess.normalize.stddev).tolist() == np.float32([58.395, 57.12, 57.375]).tolist()
    assert preprocess.normalize.has_explicit_stats
    assert preprocess.floating_conversion == pyneat.PreprocFloatingConversion.NearestEven

    decoder = script.decoder_options()
    assert decoder.decode_type == pyneat.BoxDecodeType.MobileSam
    assert decoder.multimask_output is True  # one click: the best of masks 1-3
    assert decoder.preprocess.kind == pyneat.InputKind.Tensor


def test_handoff_copies_the_embedding_with_its_geometry(load_script):
    from types import SimpleNamespace

    script = load_script("performance")
    outputs = [SimpleNamespace(cvu=lambda: "embedding in EV74 memory")]
    assert script.handoff(outputs) == ["embedding in EV74 memory"]


@pytest.mark.parametrize("failure_index", [0, 1])
def test_partial_build_closes_previous_runners(load_script, monkeypatch, failure_index):
    from types import SimpleNamespace

    script = load_script("performance")
    closed = []
    sentinel = RuntimeError("build-failure")

    def build(index):
        if index == failure_index:
            raise sentinel
        return SimpleNamespace(run=lambda inputs, **kw: ["output"], close=lambda: closed.append(index))

    monkeypatch.setattr(script.pyneat, "NeatError", ValueError)
    monkeypatch.setattr(script, "handoff", lambda outputs: outputs)
    models = {
        name: SimpleNamespace(build=lambda *a, index=index, **kw: build(index))
        for index, name in enumerate(script.COMPONENTS)
    }
    with pytest.raises(RuntimeError, match="build-failure") as failure:
        script.build_runners(models, "image", ["prompt"])
    assert failure.value is sentinel
    assert closed == list(reversed(range(failure_index)))


def test_latency_failure_closes_runners(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    script = load_script("performance")
    closed = []
    runners = [
        SimpleNamespace(close=lambda index=index: closed.append(index), run=lambda inputs, **kw: ["embedding"])
        for index in range(3)
    ]
    monkeypatch.setattr(script, "mla_stats", lambda mpk: ({}, {}))
    monkeypatch.setattr(script.PromptEncoding, "from_package", lambda *args: "encoding")
    monkeypatch.setattr(script, "sample_image", lambda dataset: dataset / "image.jpg")
    monkeypatch.setattr(script, "read_frame", lambda source: "frame")
    monkeypatch.setattr(script, "image_tensor", lambda frame: "image")
    monkeypatch.setattr(script, "prepare_prompt", lambda frame, encoding: ["prompt"])
    monkeypatch.setattr(script, "handoff", lambda tensors: tensors)
    monkeypatch.setattr(script, "build_runners", lambda *args: runners)

    def fail(*args):
        raise ValueError("latency-failure")

    monkeypatch.setattr(script, "measure_latency", fail)
    with pytest.raises(ValueError, match="latency-failure"):
        script.benchmark_pipeline(dict.fromkeys(script.COMPONENTS, tmp_path / "model.tar.gz"), tmp_path, tmp_path)
    assert closed == [2, 1, 0]


def test_the_report_plans_the_decoder_from_the_inputs_it_measured(load_script, monkeypatch, tmp_path):
    # Neat plans a Segment Anything decoder only from inputs: its masks take the image's size from them.
    from types import SimpleNamespace

    script = load_script("performance")
    plans, reported = [], {}
    models = iter([SimpleNamespace(execution_plan=lambda *inputs: plans.append(("encoder", inputs)) or "encoder plan"),
                   SimpleNamespace(execution_plan=lambda *inputs: plans.append(("decoder", inputs)) or "decoder plan")])
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: next(models))
    runners = [SimpleNamespace(close=lambda: None, run=lambda inputs, **kw: ["embedding"]) for _ in range(2)]
    monkeypatch.setattr(script, "mla_stats", lambda mpk: ({}, {}))
    monkeypatch.setattr(script.PromptEncoding, "from_package", lambda *args: "encoding")
    monkeypatch.setattr(script, "sample_image", lambda dataset: dataset / "image.jpg")
    monkeypatch.setattr(script, "read_frame", lambda source: "frame")
    monkeypatch.setattr(script, "image_tensor", lambda frame: "image")
    monkeypatch.setattr(script, "prepare_prompt", lambda frame, encoding: ["coords", "types"])
    monkeypatch.setattr(script, "handoff", lambda tensors: [f"{tensor} in EV74 memory" for tensor in tensors])
    monkeypatch.setattr(script, "build_runners", lambda *args: runners)
    monkeypatch.setattr(script, "measure_latency", lambda *args: [0.1])
    monkeypatch.setattr(script, "measure_throughput", lambda *args: "throughput")
    monkeypatch.setattr(script, "write_report", lambda output, latency, speed, models, *args: reported.update(models))

    script.benchmark_pipeline(dict.fromkeys(script.COMPONENTS, tmp_path / "model.tar.gz"), tmp_path, tmp_path)

    assert list(reported) == list(script.COMPONENTS)
    assert reported["image_encoder"].execution_plan() == "encoder plan"
    assert reported["prompt_decoder"].execution_plan() == "decoder plan"
    assert plans == [("encoder", ()), ("decoder", (["embedding in EV74 memory", "coords", "types"],))]


@pytest.mark.parametrize("fail_finish", [False, True])
def test_threaded_path_counts_final_masks_and_stops_monitor_on_failure(load_script, monkeypatch, fail_finish):
    import queue
    import threading
    from types import SimpleNamespace

    import numpy as np

    script = load_script("performance")
    monkeypatch.setattr(script, "WARMUP_FRAMES", 2)
    monkeypatch.setattr(script, "MEASURED_FRAMES", 3)
    monkeypatch.setattr(script, "MIN_DURATION_S", 0)
    events = {"prompts": 0, "masks": 0, "handoffs": 0}

    def count(name, value):
        events[name] += 1
        return value

    monkeypatch.setattr(script, "prepare_prompt", lambda frame, encoding: count("prompts", ["prompt"]))
    monkeypatch.setattr(script, "handoff", lambda tensors: count("handoffs", tensors))

    def finish(tensors, frame):
        count("masks", None)
        if fail_finish and events["masks"] == 3:
            raise ValueError("final-mask-failure")
        return "mask"

    monkeypatch.setattr(script, "finish", finish)

    class Runner:
        def __init__(self, name):
            self.name = name
            self.queue = queue.Queue(maxsize=2)
            self.closed = threading.Event()
            self.accepted = 0

        def push(self, inputs):
            while not self.closed.is_set():
                try:
                    self.queue.put(inputs, timeout=0.01)
                    self.accepted += 1
                    return True
                except queue.Full:
                    pass
            return False

        def pull(self, timeout_ms):
            while True:
                try:
                    self.queue.get(timeout=0.01)
                    return SimpleNamespace(tensors=[self.name])
                except queue.Empty:
                    if self.closed.is_set():
                        return SimpleNamespace(tensors=[])

        def close_input(self):
            self.closed.set()

    class Monitor:
        def __init__(self):
            self.active = False
            self.starts = self.stops = 0

        def start(self):
            self.active = True
            self.starts += 1

        def stop(self):
            self.active = False
            self.stops += 1
            return "telemetry"

        def running(self):
            return self.active

    monitor = Monitor()
    monkeypatch.setattr(script.pyneat, "TelemetryMonitor", lambda: monitor)
    runners = [Runner(name) for name in script.COMPONENTS]
    if fail_finish:
        with pytest.raises(ValueError, match="final-mask-failure"):
            script.measure_throughput(runners, "image", np.zeros((2, 2, 3), np.uint8), "encoding")
    else:
        result = script.measure_throughput(runners, "image", np.zeros((2, 2, 3), np.uint8), "encoding")
        assert result.frames == 3 and result.telemetry == "telemetry" and result.seconds > 0
        assert events["masks"] == runners[-1].accepted
        assert runners[0].accepted == events["masks"]
        assert events["prompts"] == events["masks"]
        assert events["handoffs"] == events["masks"]
    assert monitor.starts == monitor.stops == 1 and not monitor.running()
    assert all(runner.closed.is_set() for runner in runners)
    assert not any(
        thread.name in {"push_images", "pull_masks", "forward_image_encoder"}
        for thread in threading.enumerate()
    )


def test_close_attempts_every_runner_and_preserves_original_error(load_script):
    from types import SimpleNamespace

    script = load_script("performance")
    calls = []

    def close(index):
        calls.append(index)
        if index == 2:
            raise RuntimeError("cleanup-failure")

    runners = [SimpleNamespace(close=lambda index=index: close(index)) for index in range(3)]
    with pytest.raises(ValueError, match="original-failure"):
        try:
            raise ValueError("original-failure")
        finally:
            script.close_runners(runners)
    assert calls == [2, 1, 0]
    with pytest.raises(RuntimeError, match="cleanup-failure"):
        script.close_runners(runners)


@pytest.mark.parametrize("fault", ["second-start", "third-start", "first-join"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_partial_thread_lifecycle_cancels_and_joins_started_workers(load_script, monkeypatch, fault, close_fails):
    import threading
    from types import SimpleNamespace

    script = load_script("performance")
    cancelled = threading.Event()
    real_thread = threading.Thread
    threads, closed, exited = [], [], []
    primary_error = RuntimeError(fault)

    class Thread(real_thread):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.index = len(threads)
            self.join_calls = 0
            threads.append(self)

        def start(self):
            if (fault, self.index) in (("second-start", 1), ("third-start", 2)):
                raise primary_error
            super().start()

        def join(self, *args, **kwargs):
            self.join_calls += 1
            if fault == "first-join" and self.index == 0 and self.join_calls == 1:
                raise primary_error
            super().join(*args, **kwargs)

    def close(index):
        closed.append(index)
        if close_fails and index == 0:
            raise ValueError("secondary-cleanup-failure")

    def stage(index):
        assert cancelled.wait(2), "cancellation did not reach worker"
        exited.append(index)

    monkeypatch.setattr(script.threading, "Thread", Thread)
    runners = tuple(SimpleNamespace(close_input=lambda index=index: close(index)) for index in range(3))
    stages = [lambda index=index: stage(index) for index in range(3)]
    with pytest.raises(RuntimeError) as failure:
        script.run_together(*stages, runners=runners, cancel=cancelled.set)
    assert failure.value is primary_error
    expected = {"second-start": 1, "third-start": 2, "first-join": 3}[fault]
    assert sorted(exited) == list(range(expected))
    assert cancelled.is_set() and set(closed) == {0, 1, 2}
    assert all(not thread.is_alive() and thread.join_calls >= 1 for thread in threads[:expected])


def test_obsolete_decoder_rejected_before_any_native_profiling(load_script, monkeypatch, tmp_path):
    from types import SimpleNamespace

    script = load_script("performance")
    monkeypatch.setattr(script.pyneat, "Model", lambda *args: SimpleNamespace(metadata=lambda: {}))
    monkeypatch.setattr(script, "mla_stats", lambda *args: pytest.fail("must not profile an obsolete decoder"))
    monkeypatch.setattr(script, "build_runners", lambda *args: pytest.fail("must not build an obsolete decoder"))
    with pytest.raises(ValueError, match="provide --resources or rebuild"):
        script.benchmark_pipeline(dict.fromkeys(script.COMPONENTS, tmp_path / "old.tar.gz"), tmp_path, tmp_path)


def test_prompt_encoding_is_per_frame_not_cached(load_script, monkeypatch):
    import numpy as np

    script = load_script("performance")
    calls = []
    monkeypatch.setattr(script, "hwc", lambda value: value)

    def encode(coords):
        calls.append(coords.copy())
        return np.zeros((1, 256, 1, 6), np.float32)

    for _ in range(2):
        result = script.prepare_prompt(np.zeros((720, 1280, 3), np.uint8), encode)
        assert [item.shape for item in result] == [(1, 256, 1, 6), (1, 5, 1, 6)]
    assert len(calls) == 2


def test_tensors_come_from_writable_arrays(load_script, monkeypatch):
    # Tensor.from_numpy refuses a read-only array, such as an image NumPy reads from Pillow, even when it copies.
    import numpy as np

    performance, accuracy = load_script("performance"), load_script("accuracy")
    given = []
    monkeypatch.setattr(performance.pyneat.Tensor, "from_numpy", lambda array, **kw: given.append(array) or array)
    image = np.zeros((48, 64, 3), np.uint8)
    prompt = np.zeros((1, 5, 1, 6), np.float32)
    for array in (image, prompt):
        array.flags.writeable = False
    for make, value in ((performance.image_tensor, image), (accuracy.rgb, image), (performance.hwc, prompt),
                        (accuracy.hwc, prompt)):
        make(value)
    assert [array.flags.writeable and array.flags.c_contiguous for array in given] == [True] * 4
    np.testing.assert_array_equal(given[2], prompt[0].transpose(1, 2, 0))
