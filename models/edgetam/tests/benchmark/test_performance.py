"""Checks of EdgeTAM's benchmark tracker against the calibration tracker, and of a measured performance report."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls

COMPONENTS = ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"]
# A click on the benchmark frame, as (x, y) frame pixels.
CLICK = (700.0, 400.0)


@pytest.fixture
def script(load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """performance.py, with each Neat tensor standing for its array."""
    script = load_script("performance")
    monkeypatch.setattr(script.pyneat.Tensor, "from_numpy", lambda array, **options: _Tensor(array))
    return script


@pytest.fixture
def calibration(load_script: Callable[[str], ModuleType]) -> ModuleType:
    pytest.importorskip("PIL")
    return load_script("calibration")


class _Tensor:
    """A Neat tensor holding one array."""

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)

    def to_numpy(self, copy: bool) -> np.ndarray:
        return self.array.copy() if copy else self.array


def _channels_last(value: np.ndarray) -> np.ndarray:
    return value.transpose(0, 2, 3, 1) if value.ndim == 4 else value


def _decoded(script: ModuleType, seed: int, object_score: float = 1.0) -> dict[str, np.ndarray]:
    """Seeded mask decoder outputs, channels first as the graphs give them."""
    rng = np.random.default_rng(seed)
    return {
        "mask_logits": rng.normal(size=(1, script.MASKS, script.MASK_SIZE, script.MASK_SIZE)).astype(np.float32),
        "iou_scores": rng.uniform(size=(1, script.MASKS)).astype(np.float32),
        "object_score_logits": np.full((1, 1), object_score, np.float32),
        "object_pointers": rng.normal(size=(1, script.MASKS, script.HIDDEN)).astype(np.float32),
    }


def test_the_tracker_reads_the_graphs_interfaces(script: ModuleType, load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("onnx")
    surgery = load_script("surgery")
    interfaces = surgery.INTERFACES

    def shape(component: str, direction: str, name: str) -> tuple[int, ...]:
        return _channels_last(np.zeros(interfaces[component][direction][name])).shape

    assert script.COMPONENTS == surgery.COMPONENTS == COMPONENTS
    assert shape("image_encoder", "outputs", "image_embeddings") == script.EMBEDDING
    assert tuple(shape("image_encoder", "outputs", f"high_res_features_{index}") for index in (0, 1)) == (
        script.HIGH_RES
    )
    assert shape("mask_decoder", "outputs", "mask_logits") == (1, script.MASK_SIZE, script.MASK_SIZE, script.MASKS)
    assert shape("memory_encoder", "inputs", "mask") == (1, script.IMAGE_SIZE, script.IMAGE_SIZE, 1)
    assert shape("memory_encoder", "outputs", "memory") == (1, script.MEMORY_TOKENS, script.MEMORY_DIM)
    bank = script.MemoryBank()
    bank.add(np.zeros((script.MEMORY_TOKENS, script.MEMORY_DIM), np.float32), np.zeros(script.HIDDEN, np.float32))
    names = ["memories", "memory_mask", "object_pointers", "object_pointer_mask"]
    assert [array.shape for array in bank.arrays()] == [
        tuple(interfaces["memory_attention"]["inputs"][name]) for name in names
    ]
    assert script.POINT_TYPES == surgery.POINT_TYPES and script.PROMPT_POINTS == surgery.PROMPT_POINTS


def test_prompts_match_the_calibration_trackers(script: ModuleType, calibration: ModuleType) -> None:
    x, y = CLICK
    model_pixel = (x * script.IMAGE_SIZE / script.FRAME_WIDTH, y * script.IMAGE_SIZE / script.FRAME_HEIGHT)
    click = calibration.Prompt(points=(model_pixel,), types=("foreground",)).inputs()

    coords, types = script.prompt_arrays([CLICK], ["foreground"])
    no_coords, no_types = script.prompt_arrays([], [])

    np.testing.assert_allclose(coords, _channels_last(click["point_coords"]), atol=1e-6)
    np.testing.assert_array_equal(types, _channels_last(click["point_types"]))
    np.testing.assert_array_equal(no_coords, _channels_last(calibration.NO_PROMPT.inputs()["point_coords"]))
    np.testing.assert_array_equal(no_types, _channels_last(calibration.NO_PROMPT.inputs()["point_types"]))


@pytest.mark.parametrize("object_score", [1.0, 0.0, -3.0])
def test_the_mask_choice_matches_the_calibration_trackers(
    script: ModuleType, calibration: ModuleType, object_score: float
) -> None:
    decoded = _decoded(script, seed=3, object_score=object_score)
    expected_mask, expected_pointer = calibration.select_mask(decoded, multimask=True)

    mask, pointer = script.choose_mask(
        _channels_last(decoded["mask_logits"])[0], decoded["iou_scores"][0], object_score, decoded["object_pointers"][0]
    )

    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(pointer, expected_pointer)


def test_the_memory_mask_matches_the_calibration_trackers(script: ModuleType, calibration: ModuleType) -> None:
    pytest.importorskip("torch")
    mask = np.random.default_rng(5).normal(scale=4, size=(script.MASK_SIZE, script.MASK_SIZE)).astype(np.float32)
    upsampled = calibration.upsample(mask)

    tracked, prompted = script.memory_mask(mask, prompted=False), script.memory_mask(mask, prompted=True)

    assert tracked.shape == prompted.shape == (1, script.IMAGE_SIZE, script.IMAGE_SIZE, 1)
    np.testing.assert_allclose(tracked, _channels_last(calibration.sigmoid(upsampled)), atol=1e-4)
    # Thresholding differs only where the upsampled logit is within rounding of zero.
    assert (prompted != _channels_last((upsampled > 0).astype(np.float32))).mean() < 1e-4


def test_a_frame_without_the_object_remembers_an_empty_mask(script: ModuleType) -> None:
    absent = np.full((script.MASK_SIZE, script.MASK_SIZE), script.NO_OBJECT_LOGIT, np.float32)

    assert not script.memory_mask(absent, prompted=False).any()


def test_the_memory_bank_matches_the_calibration_trackers(script: ModuleType, calibration: ModuleType) -> None:
    rng = np.random.default_rng(7)
    reference, bank = calibration.Tracker(sessions={}), script.MemoryBank()
    for frame in range(20):
        memory = rng.normal(size=(script.MEMORY_TOKENS, script.MEMORY_DIM)).astype(np.float32)
        pointer = rng.normal(size=script.HIDDEN).astype(np.float32)
        reference.memories.append(memory[None])
        reference.pointers.append(pointer)
        bank.add(memory, pointer)
        # Checked while the bank fills and once the oldest tracked frames have left it.
        if frame in (0, 3, 19):
            for array, expected in zip(bank.arrays(), reference.bank().values(), strict=True):
                np.testing.assert_array_equal(array, expected)


def test_the_frame_is_a_seeded_sequences_first_frame_and_the_click_lands_inside_its_object(
    script: ModuleType, tmp_path: Path
) -> None:
    cv2 = pytest.importorskip("cv2")
    for sequence in ("bear", "camel"):
        for frame in ("00000", "00001"):
            (tmp_path / "val" / sequence).mkdir(parents=True, exist_ok=True)
            (tmp_path / "annotations" / sequence).mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(tmp_path / "val" / sequence / f"{frame}.jpg"), np.full((480, 640, 3), 200, np.uint8))
            objects = np.zeros((480, 640), np.uint8)
            # A thin object and a 100x100 object that touches the image's left edge.
            objects[10:14, 300:600] = 38
            objects[200:300, 0:100] = 75
            assert cv2.imwrite(str(tmp_path / "annotations" / sequence / f"{frame}.png"), objects)
    (tmp_path / "val.txt").write_text("bear\ncamel\n", encoding="utf-8")

    image = script.sample_image(tmp_path)
    x, y = script.object_click(image)

    assert image == script.sample_image(tmp_path) and image.name == "00000.jpg"
    # A 640x480 image fills 960x720 of the frame, 160 columns in: the click is the square's centre, scaled by 1.5.
    assert (x, y) == pytest.approx((160 + 50 * 1.5, 250 * 1.5), abs=2)


def test_the_first_frame_is_prompted_and_later_frames_attend_to_the_bank(script: ModuleType) -> None:
    pytest.importorskip("cv2")
    calls: list[tuple[str, list[tuple[int, ...]]]] = []
    decoded = _decoded(script, seed=1)
    outputs = {
        "image_encoder": [np.zeros(script.EMBEDDING), *(np.zeros(shape) for shape in script.HIGH_RES)],
        "memory_attention": [np.ones(script.EMBEDDING)],
        "mask_decoder": [_channels_last(value) for value in decoded.values()],
        "memory_encoder": [np.ones((1, script.MEMORY_TOKENS, script.MEMORY_DIM))],
    }

    def runner(component: str) -> SimpleNamespace:
        def run(inputs: list[_Tensor], timeout_ms: int) -> list[_Tensor]:
            calls.append((component, [tensor.array.shape for tensor in inputs]))
            return [_Tensor(np.asarray(value, np.float32)) for value in outputs[component]]

        return SimpleNamespace(run=run)

    tracker = script.Tracker({component: runner(component) for component in COMPONENTS})
    frame = _Tensor(np.zeros((script.FRAME_HEIGHT, script.FRAME_WIDTH, 3), np.uint8))

    tracker.track(tracker.encode(frame), script.prompt_arrays([CLICK], ["foreground"]))
    mask = tracker.track(tracker.encode(frame))

    assert [component for component, _ in calls] == [
        "image_encoder", "mask_decoder", "memory_encoder", *COMPONENTS
    ]  # fmt: skip
    best = 1 + int(np.argmax(decoded["iou_scores"][0, 1:]))
    np.testing.assert_array_equal(mask, decoded["mask_logits"][0, best])
    assert len(tracker.bank.memories) == 1 and tracker.bank.prompted_memory is not None
    assert dict(calls)["memory_attention"] == [
        script.EMBEDDING,
        (1, script.MEMORY_SLOTS * script.MEMORY_TOKENS, script.MEMORY_DIM),
        (1, script.MEMORY_SLOTS),
        (1, script.OBJECT_POINTERS, script.HIDDEN),
        (1, script.OBJECT_POINTERS),
    ]
    assert set(tracker.seconds) == {*COMPONENTS, "cpu"}


def test_every_component_needs_its_mpk(script: ModuleType) -> None:
    mpks = [f"{component}={component}.tar.gz" for component in COMPONENTS]

    assert list(script.parse_mpks(mpks)) == COMPONENTS
    with pytest.raises(SystemExit, match="memory_encoder"):
        script.parse_mpks(mpks[:-1])


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_each_components_execution_plan(performance: dict[str, Any]) -> None:
    assert list(performance["components"]) == COMPONENTS
    for item in performance["components"].values():
        plan = item["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]


def test_image_encoder_execution_plan_takes_the_benchmark_frame(performance: dict[str, Any]) -> None:
    (frame,) = performance["components"]["image_encoder"]["execution_plan"]["edges"][0]["tensors"]
    assert frame["shape"] == [1, 720, 1280, 3]


def test_mla_stats_collected(performance: dict[str, Any]) -> None:
    for component_id, item in performance["components"].items():
        layer_stats, stage_stats = item["mla_layer_stats"], item["mla_stage_stats"]
        assert (layer_stats["status"], stage_stats["status"]) == ("ok", "ok"), component_id
        assert any(layer["latency_us"] > 0 for layer in layer_stats["layers"]), component_id
        assert stage_stats["stages"], component_id
