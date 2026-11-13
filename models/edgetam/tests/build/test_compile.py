"""Checks of EdgeTAM's compile recipe, its calibration tracker, and the MPKs a build publishes."""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "edgetam"


@pytest.fixture
def compiler(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("compile")


@pytest.fixture
def calibration(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("calibration")


def test_every_build_profile_is_named_after_its_component_precisions(compiler: ModuleType) -> None:
    for profile in load_metadata(MODEL_DIR)["build_profiles"]:
        compiler.check_profile_id(profile["id"], compiler.component_parameters(profile["id"]))


def test_profile_id_must_mark_a_component_with_bf16_regions_mixed(compiler: ModuleType) -> None:
    parameters = {component: {"precision": "w8a8"} for component in compiler.COMPONENTS}
    compiler.check_profile_id("w8a8", parameters)
    parameters["memory_attention"]["bf16_regions"] = ["attention"]
    with pytest.raises(ValueError, match="w8a8_w8a8_mixed_w8a8"):
        compiler.check_profile_id("w8a8", parameters)
    compiler.check_profile_id("w8a8_w8a8_mixed_w8a8", parameters)


def test_bf16_regions_belong_to_a_w8a8_component(compiler: ModuleType) -> None:
    with pytest.raises(ValueError, match="W8A8"):
        compiler.bf16_regions("memory_attention", {"precision": "w16a16", "bf16_regions": ["attention"]})
    with pytest.raises(ValueError, match="unknown"):
        compiler.bf16_regions("image_encoder", {"precision": "w8a8", "bf16_regions": ["attention"]})


def _mpk_manifest(compiler: ModuleType, component: str) -> dict:
    interface = compiler.INTERFACES[component]
    return {
        "input_nodes": [{"name": name, "size": int(np.prod(shape)) * 4} for name, shape in interface["inputs"].items()],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "dequantization_transform"},
                "output_nodes": [{"size": int(np.prod(shape)) * 4} for shape in interface["outputs"].values()],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def _drop_an_input(manifest: dict) -> None:
    manifest["input_nodes"].pop()


def _shrink_an_output(manifest: dict) -> None:
    manifest["plugins"][-1]["output_nodes"][0]["size"] = 4


def _add_an_mla_segment(manifest: dict) -> None:
    manifest["plugins"].insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})


def _add_host_code(manifest: dict) -> None:
    manifest["plugins"].insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})


@pytest.mark.parametrize("component", ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"])
def test_mpk_audit_accepts_one_mla_segment_with_the_declared_interface(
    tmp_path: Path, compiler: ModuleType, component: str
) -> None:
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, _mpk_manifest(compiler, component))
    compiler.audit_mpk(archive, component)


@pytest.mark.parametrize("break_manifest", [_drop_an_input, _shrink_an_output, _add_an_mla_segment, _add_host_code])
@pytest.mark.parametrize("component", ["image_encoder", "memory_attention", "mask_decoder", "memory_encoder"])
def test_mpk_audit_rejects_another_interface_or_placement(
    tmp_path: Path, compiler: ModuleType, component: str, break_manifest: Callable[[dict], None]
) -> None:
    manifest = _mpk_manifest(compiler, component)
    break_manifest(manifest)
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)
    with pytest.raises(RuntimeError):
        compiler.audit_mpk(archive, component)


@pytest.mark.parametrize("precision", ["w8a8", "w8a16", "w16a16"])
def test_every_precision_calibrates_with_histogram_mse(compiler: ModuleType, precision: str) -> None:
    compiler.quantization({"precision": precision, "calibration_bins": 2048})
    arguments = compiler.QuantizationParams.call_args.kwargs
    compiler.HistogramMSEMethod.assert_called_with(2048)
    assert arguments["calibration_method"] is compiler.HistogramMSEMethod.return_value
    # A16 activations are BF16 and W16 weights too; A8 activations are asymmetric INT8 and W8 weights per-channel
    # INT8. Only W8A16 scales each token's activations on their own.
    if precision.endswith("a16"):
        assert arguments["activation_quantization_scheme"] is compiler.bfloat16_scheme.return_value
    else:
        compiler.quantization_scheme.assert_any_call(asymmetric=True, per_channel=False, bits=8)
    if precision.startswith("w8"):
        compiler.quantization_scheme.assert_any_call(asymmetric=False, per_channel=True, bits=8)
    assert arguments.get("per_token_quant", False) == (precision == "w8a16")


def test_quantization_rejects_another_precision(compiler: ModuleType) -> None:
    with pytest.raises(ValueError, match="unsupported precision: w4a8"):
        compiler.quantization({"precision": "w4a8", "calibration_bins": 2048})


def test_calibration_sets_spread_samples_over_earlier_clips_and_check_on_the_last(
    calibration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = [
        {component: [{"clip": clip, "frame": frame} for frame in range(8)] for component in calibration.COMPONENTS}
        for clip in range(3)
    ]
    monkeypatch.setattr(calibration, "tracked_clips", lambda graphs, dataset, count, kept_frames: clips)
    counts = dict.fromkeys(calibration.COMPONENTS, 4)
    sets = calibration.calibration_sets({}, Path("unused"), 3, counts, 4)
    chosen = [(sample["clip"], sample["frame"]) for sample in sets["memory_encoder"].samples]
    assert chosen == [(0, 0), (0, 5), (1, 2), (1, 7)]
    checks = [(check["clip"], check["frame"]) for check in sets["memory_encoder"].checks]
    assert checks == [(2, 0), (2, 2), (2, 5), (2, 7)]
    with pytest.raises(ValueError, match="needs 17"):
        calibration.calibration_sets({}, Path("unused"), 3, dict.fromkeys(calibration.COMPONENTS, 17), 4)


def _write_cal(root: Path) -> np.ndarray:
    """A one-sequence cal split under root/davis; return its first frame's object IDs."""
    from PIL import Image

    object_ids = np.zeros((48, 64), np.uint8)
    object_ids[4:8, 4:8] = 1
    object_ids[20:40, 10:50] = 2
    for frame in ("00000", "00001"):
        for folder in ("cal", "annotations"):
            (root / "davis" / folder / "bear").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((48, 64, 3), 90, np.uint8)).save(root / "davis" / "cal" / "bear" / f"{frame}.jpg")
        annotation = Image.fromarray(object_ids, mode="P")
        annotation.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0])
        annotation.save(root / "davis" / "annotations" / "bear" / f"{frame}.png")
    (root / "davis" / "cal.txt").write_text("bear\n", encoding="utf-8")
    return object_ids


def test_cal_sequences_read_alike_from_the_archive_and_its_extracted_directory(
    calibration: ModuleType, tmp_path: Path
) -> None:
    pytest.importorskip("PIL")
    object_ids = _write_cal(tmp_path)
    archive = tmp_path / "datasets__davis_2017_cal.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(tmp_path / "davis", arcname="davis")

    for dataset in (archive, tmp_path / "davis"):
        with calibration.DavisCal(dataset) as cal:
            assert cal.sequences() == ["bear"] and cal.frames("bear") == ["00000", "00001"]
            assert cal.image("bear", "00000").size == (64, 48)
            np.testing.assert_array_equal(cal.object_ids("bear", "00001"), object_ids)


def test_the_first_frame_is_prompted_on_its_largest_object(calibration: ModuleType, tmp_path: Path) -> None:
    import random

    pytest.importorskip("PIL")
    mask = calibration.largest_object(_write_cal(tmp_path))
    assert mask.sum() == 20 * 40 and mask[20, 10]
    with pytest.raises(ValueError, match="no annotated object"):
        calibration.largest_object(np.zeros((48, 64), np.uint8))

    # Points are scaled from the 64x48 frame to the 1024x1024 model input: 16 times across, 1024 / 48 times down.
    box = calibration.first_frame_prompt(mask, "box", random.Random(0))
    assert box.types == ("box_top_left", "box_bottom_right")
    np.testing.assert_allclose(box.points, [(160, 20 * 1024 / 48), (800, 40 * 1024 / 48)])
    click = calibration.first_frame_prompt(mask, "click_and_background", random.Random(0))
    (inside_x, inside_y), (outside_x, outside_y) = click.points
    assert click.types == ("foreground", "background")
    assert mask[int(inside_y * 48 / 1024), int(inside_x / 16)]
    assert not mask[int(outside_y * 48 / 1024), int(outside_x / 16)]


def test_prompt_inputs_normalize_points_and_pad_the_other_slots(calibration: ModuleType) -> None:
    inputs = calibration.Prompt(((511.5, 0.0),), ("foreground",)).inputs()
    assert inputs["point_coords"].shape == (1, 2, 1, 3)
    assert inputs["point_types"].shape == (1, 5, 1, 3)
    np.testing.assert_allclose(inputs["point_coords"][0, :, 0, 0], [0.0, 1 / 1024 - 1])
    assert inputs["point_types"][0, :, 0].argmax(0).tolist() == [2, 0, 0]
    assert calibration.NO_PROMPT.inputs()["point_types"][0, 0, 0].tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="up to 2 points"):
        calibration.Prompt(((0, 0),) * 3, ("foreground",) * 3)


def test_memory_bank_holds_the_prompted_frame_first_then_the_latest_frames(calibration: ModuleType) -> None:
    tracker = calibration.Tracker(sessions={})
    for frame in range(9):
        tracker.memories.append(np.full((1, 512, 64), frame, np.float32))
        tracker.pointers.append(np.full(256, frame, np.float32))
    bank = tracker.bank()
    slots = bank["memories"][0, ::512, 0]
    assert slots.tolist() == [0, 8, 7, 6, 5, 4, 3]
    assert bank["memory_mask"].tolist() == [[1] * 7]
    assert bank["object_pointers"][0, :9, 0].tolist() == [0, 8, 7, 6, 5, 4, 3, 2, 1]
    assert bank["object_pointer_mask"][0].tolist() == [1] * 9 + [0] * 7


def _decoded(masks: np.ndarray, scores: list[float], object_score: float) -> dict[str, np.ndarray]:
    return {
        "mask_logits": masks[None],
        "iou_scores": np.array([scores], np.float32),
        "object_score_logits": np.array([[object_score]], np.float32),
        "object_pointers": np.arange(4, dtype=np.float32).reshape(1, 4, 1).repeat(256, axis=2),
    }


def test_mask_choice_follows_transformers_multimask_rules(calibration: ModuleType) -> None:
    masks = np.stack([np.full((256, 256), value, np.float32) for value in (1.0, 2.0, 3.0, 4.0)])
    # With one or no point: the best of masks 1-3 and its pointer.
    mask, pointer = calibration.select_mask(_decoded(masks, [0.9, 0.1, 0.8, 0.2], 1.0), multimask=True)
    assert mask[0, 0] == 3.0 and pointer[0] == 2
    # With more points: mask 0 while it is stable, and always pointer 0.
    mask, pointer = calibration.select_mask(_decoded(masks, [0.1, 0.1, 0.8, 0.2], 1.0), multimask=False)
    assert mask[0, 0] == 1.0 and pointer[0] == 0
    unstable = masks.copy()
    unstable[0, :128] = 0.0
    mask, pointer = calibration.select_mask(_decoded(unstable, [0.1, 0.1, 0.8, 0.2], 1.0), multimask=False)
    assert mask[0, 0] == 3.0 and pointer[0] == 0
    # An absent object's mask is all background.
    mask, _ = calibration.select_mask(_decoded(masks, [0.9, 0.1, 0.8, 0.2], -1.0), multimask=True)
    assert (mask == calibration.NO_OBJECT_LOGIT).all()


def test_compiled_build_holds_one_audited_mpk_per_component_in_run_order(compiled: Path, compiler: ModuleType) -> None:
    manifest = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [output["component_id"] for output in manifest["outputs"]] == compiler.COMPONENTS
    assert [graph["component_id"] for graph in manifest["derived_graphs"]] == compiler.COMPONENTS
    for output in manifest["outputs"]:
        compiler.audit_mpk(compiled / output["mpk_path"], output["component_id"])
