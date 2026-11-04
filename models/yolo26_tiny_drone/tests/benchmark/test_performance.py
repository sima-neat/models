from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def script(load_script, monkeypatch):
    module = load_script("performance")
    board = MagicMock()
    # Explicit nested options make assertions independent of a device installation.
    board.ModelOptions.side_effect = lambda: SimpleNamespace(
        preprocess=SimpleNamespace(
            color_convert=SimpleNamespace(), resize=SimpleNamespace(), normalize=SimpleNamespace()
        )
    )
    board.BenchmarkOptions.side_effect = SimpleNamespace
    monkeypatch.setattr(module, "pyneat", board)
    return module


@pytest.fixture
def opencv():
    # Stage-output checks need only the registry's test extras, not image-decoding wheels.
    import cv2

    return cv2


@pytest.fixture
def dataset(tmp_path, opencv):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    images = []
    for index, value in enumerate((11, 43, 79, 137), start=1):
        path = root / "images" / f"frame-{index}.png"
        assert opencv.imwrite(str(path), np.full((512, 640, 3), value, np.uint8))
        payload = path.read_bytes()
        images.append(
            {
                "id": index,
                "file_name": path.relative_to(root).as_posix(),
                "width": 640,
                "height": 512,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "sima.anti-uav-evaluation",
                "schema_version": 1,
                "dataset_id": "anti-uav-track3",
                "variant": "sequence-safe-val-v1",
                "split": "val",
                "counts": {"images": len(images)},
                "images": images,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_sample_is_seeded_over_sorted_manifest_membership(script, dataset, opencv):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    # Seed zero selects the fourth of four sorted files, regardless of manifest order.
    assert script.sample_image(dataset) == dataset / "images/frame-4.png"
    manifest["images"].reverse()
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert opencv.imwrite(str(dataset / "images/unlisted.png"), np.zeros((4, 4, 3), np.uint8))

    assert script.sample_image(dataset) == dataset / "images/frame-4.png"


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size_bytes", 1)])
def test_sample_rejects_a_different_image_identity(script, dataset, field, value):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["images"][-1][field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="image identity differs"):
        script.sample_image(dataset)


def test_sample_rejects_modified_image_bytes(script, dataset):
    path = dataset / "images/frame-4.png"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="image identity differs"):
        script.sample_image(dataset)


@pytest.mark.parametrize("relative", ["../outside.png", "/outside.png"])
def test_sample_rejects_escaping_paths(script, dataset, relative):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["images"] = [{**manifest["images"][-1], "file_name": relative}]
    manifest["counts"]["images"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path escapes"):
        script.sample_image(dataset)


def test_sample_rejects_a_symlink_outside_the_dataset(script, dataset):
    image = dataset / "images/frame-4.png"
    outside = dataset.parent / "outside.png"
    image.rename(outside)
    image.symlink_to(outside)

    with pytest.raises(ValueError, match="path escapes"):
        script.sample_image(dataset)


@pytest.mark.parametrize("malformation", ["schema", "version", "empty", "count", "duplicate"])
def test_sample_rejects_malformed_membership(script, dataset, malformation):
    path = dataset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if malformation == "schema":
        manifest["schema"] = "another-dataset"
    elif malformation == "version":
        manifest["schema_version"] = 2
    elif malformation == "empty":
        manifest["images"] = []
        manifest["counts"]["images"] = 0
    elif malformation == "count":
        manifest["counts"]["images"] += 1
    else:
        manifest["images"][0]["file_name"] = manifest["images"][-1]["file_name"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="versioned|membership|duplicate"):
        script.sample_image(dataset)


def test_image_decode_preserves_rgb_channel_order(script, tmp_path, opencv):
    image = np.array([[[243, 71, 19], [29, 101, 229]]], dtype=np.uint8)
    path = tmp_path / "color.png"
    assert opencv.imwrite(str(path), image[:, :, ::-1])

    np.testing.assert_array_equal(script.read_rgb(path), image)


def test_image_decode_rejects_unreadable_pixels(script, tmp_path):
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an encoded image")

    with pytest.raises(ValueError, match="could not decode"):
        script.read_rgb(path)


@pytest.mark.parametrize(
    "shape,rectangle",
    [
        ((512, 640), (190, 0, 900, 720)),
        ((640, 512), (352, 0, 576, 720)),
        ((360, 1280), (0, 180, 1280, 360)),
        ((720, 1280), (0, 0, 1280, 720)),
    ],
)
def test_frame_has_exact_720p_letterbox_geometry_and_fill(script, shape, rectangle):
    frame = script.letterbox_to_720p(np.full((*shape, 3), 201, np.uint8))
    left, top, width, height = rectangle
    expected = np.full((720, 1280, 3), 114, np.uint8)
    expected[top : top + height, left : left + width] = 201

    np.testing.assert_array_equal(frame, expected)
    assert frame.dtype == np.uint8


def test_frame_resampling_matches_the_bilinear_reference(script, opencv):
    image = np.array([[[0, 41, 251], [255, 211, 7]], [[203, 9, 97], [47, 149, 193]]], dtype=np.uint8)
    expected = np.full((720, 1280, 3), 114, np.uint8)
    expected[:, 280:1000] = opencv.resize(image, (720, 720), interpolation=opencv.INTER_LINEAR)

    np.testing.assert_array_equal(script.letterbox_to_720p(image), expected)


def test_benchmark_keeps_the_app_preprocessing_and_decoder_contract(script):
    options = script.model_options()
    assert options.preprocess.kind == script.pyneat.InputKind.Image
    assert options.preprocess.enable == script.pyneat.AutoFlag.On
    assert options.preprocess.color_convert.input_format == script.pyneat.PreprocessColorFormat.RGB
    assert options.preprocess.resize.enable == script.pyneat.AutoFlag.On
    assert options.preprocess.resize.mode == script.pyneat.ResizeMode.Letterbox
    assert options.preprocess.normalize.enable == script.pyneat.AutoFlag.On
    assert options.preprocess.preset == script.pyneat.NormalizePreset.COCO_YOLO
    assert options.decode_type == script.pyneat.BoxDecodeType.YoloV26
    assert options.num_classes == 1
    assert (options.score_threshold, options.nms_iou_threshold, options.top_k) == (0.05, 0.60, 100)


def test_cli_benchmarks_verified_real_pixels_and_writes_the_standard_report(script, dataset, tmp_path, monkeypatch):
    output = tmp_path / "output"
    model = script.pyneat.Model.return_value
    model.info.return_value.selection.selected_post_kind = "boxdecode"
    report = MagicMock()
    create_report = MagicMock(return_value=report)
    monkeypatch.setattr(script.model_report, "from_performance", create_report)

    assert (
        script.main(
            [
                "--mpk",
                "model.tar.gz",
                "--dataset",
                f"anti-uav-val={dataset}",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    script.pyneat.Model.assert_called_once()
    assert script.pyneat.Model.call_args.args[0] == "model.tar.gz"
    model_options = script.pyneat.Model.call_args.args[1]
    assert (model_options.num_classes, model_options.top_k, model_options.score_threshold) == (1, 100, 0.05)
    assert model_options.preprocess.preset == script.pyneat.NormalizePreset.COCO_YOLO
    options = model.benchmark.call_args.args[0]
    assert (options.num_samples, options.min_duration_ms) == (1000, 60000)
    assert options.include_mla_layer_stats is True and options.include_mla_stage_stats is True
    script.pyneat.Tensor.from_numpy.assert_called_once()
    tensor_call = script.pyneat.Tensor.from_numpy.call_args
    np.testing.assert_array_equal(tensor_call.args[0][:, 190:1090], np.full((720, 900, 3), 137, np.uint8))
    assert tensor_call.kwargs == {"copy": True, "image_format": script.pyneat.PixelFormat.RGB}
    assert options.inputs == [script.pyneat.Tensor.from_numpy.return_value]
    create_report.assert_called_once_with(
        model.benchmark.return_value,
        {"model": model},
        workload=(
            "anti-uav-val frame-4.png in a 1280x720 RGB frame, "
            "Neat COCO_YOLO preprocessing, YoloV26 one-class decoder, top100 above0.05, "
            "NMS0.60, boxdecode post-processing"
        ),
    )
    report.save.assert_called_once_with(str(output))


@pytest.mark.parametrize("argument", ["no-separator", "=directory", "source="])
def test_cli_rejects_an_invalid_dataset_argument_before_model_construction(script, argument, tmp_path):
    with pytest.raises(SystemExit) as error:
        script.main(["--mpk", "model.tar.gz", "--dataset", argument, "--output", str(tmp_path)])
    assert error.value.code == 2
    script.pyneat.Model.assert_not_called()


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
