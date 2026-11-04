from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def dataset(tmp_path):
    from PIL import Image

    root = tmp_path / "dataset"
    root.mkdir()
    rows, images, boxes = [], [], []
    for image_id in (1, 2, 3):
        relative = f"images/sequence-a/{image_id}.jpg"
        image = root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 512), (17, 101, 239)).save(image)
        payload = image.read_bytes()
        row = {
            "id": image_id,
            "sample_id": f"sequence-a_{image_id}",
            "file_name": relative,
            "sequence": "sequence-a",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "width": 640,
            "height": 512,
            "annotation_count": int(image_id != 3),
        }
        rows.append(row)
        images.append({key: row[key] for key in ("id", "file_name", "width", "height")})
        if image_id != 3:
            boxes.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [10, 20, 3, 3],
                    "area": 9,
                    "iscrowd": 0,
                }
            )
    annotations = {"images": images, "annotations": boxes, "categories": [{"id": 1, "name": "drone"}]}
    manifest = {
        "schema": "sima.anti-uav-evaluation",
        "schema_version": 1,
        "dataset_id": "anti-uav-track3",
        "variant": "sequence-safe-val-v1",
        "split": "val",
        "images": rows,
        "counts": {"images": 3, "annotations": 2, "sequences": 1, "negative_images": 1},
        "annotation_file": "annotations/instances_val.json",
        "source_archive_sha256": "a" * 64,
        "source_split_manifest_sha256": "b" * 64,
        "canonicalization_policy": {"unchanged_fixture": True},
        "limitations": ["Synthetic test data; not model qualification"],
    }
    write_dataset(root, manifest, annotations)
    return root, manifest, annotations


def write_dataset(root, manifest, annotations):
    payload = json.dumps(annotations).encode()
    path = root / manifest["annotation_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    manifest["annotation_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))


def box(**changes):
    fields = {"x1": 10, "y1": 20, "x2": 13, "y2": 23, "score": 0.8, "class_id": 0}
    fields.update(changes)
    return SimpleNamespace(**fields)


def neat_stub():
    return SimpleNamespace(
        __version__="test-runtime",
        build_info=lambda: {"version": "test-runtime"},
        ModelOptions=lambda: SimpleNamespace(
            preprocess=SimpleNamespace(resize=SimpleNamespace(), color_convert=SimpleNamespace()),
            processcvu=SimpleNamespace(),
        ),
        InputKind=SimpleNamespace(Image="image"),
        AutoFlag=SimpleNamespace(On="on"),
        PreprocessColorFormat=SimpleNamespace(RGB="rgb"),
        ResizeMode=SimpleNamespace(Letterbox="letterbox"),
        NormalizePreset=SimpleNamespace(COCO_YOLO="yolo"),
        BoxDecodeType=SimpleNamespace(YoloV26="v26"),
        BoxDecodeTypeOption=SimpleNamespace(GroupedByRoleLogit="grouped-logits"),
        PixelFormat=SimpleNamespace(RGB="rgb"),
        Tensor=SimpleNamespace(from_numpy=lambda image, **_: image),
    )


def test_model_options_keep_preproc_and_decode_inside_the_model(load_script, monkeypatch):
    script = load_script("accuracy")
    monkeypatch.setattr(script, "pyneat", neat_stub())
    options = script.model_options()
    assert (options.preprocess.kind, options.preprocess.enable, options.preprocess.preset) == ("image", "on", "yolo")
    assert options.preprocess.color_convert.input_format == options.preprocess.color_convert.output_format == "rgb"
    assert (options.preprocess.resize.mode, options.preprocess.resize.pad_value) == ("letterbox", 114)
    assert options.preprocess.resize.scaling_type == "BILINEAR"
    assert not hasattr(options.preprocess.resize, "width"), "geometry must remain model-derived"
    assert (options.processcvu.pre_run_target, options.processcvu.post_run_target) == ("EV74", "A65")
    assert (options.decode_type, options.decode_type_option, options.num_classes) == ("v26", "grouped-logits", 1)
    assert (options.score_threshold, options.nms_iou_threshold, options.top_k) == (0.001, 0.6, 300)


def test_native_rgb_loading_never_resizes_or_normalizes(dataset, load_script):
    from PIL import Image

    root, manifest, _ = dataset
    script = load_script("accuracy")
    loaded, _ = script.load_dataset(root)
    assert loaded == manifest
    pixels = script.read_rgb(root, manifest["images"][0])
    with Image.open(root / manifest["images"][0]["file_name"]) as image:
        np.testing.assert_array_equal(pixels, np.asarray(image.convert("RGB")))
    assert pixels.shape == (512, 640, 3) and pixels.dtype == np.uint8


@pytest.mark.parametrize("mutation", ["category", "coverage", "duplicate", "bad_box", "checksum"])
def test_dataset_admission_rejects_bad_labels_and_incomplete_coverage(dataset, load_script, mutation):
    root, manifest, annotations = dataset
    if mutation == "category":
        annotations["annotations"][0]["category_id"] = 2
    elif mutation == "coverage":
        annotations["images"].pop()
    elif mutation == "duplicate":
        manifest["images"][1]["id"] = 1
    elif mutation == "bad_box":
        annotations["annotations"][0]["bbox"][2] = -1
    write_dataset(root, manifest, annotations)
    if mutation == "checksum":
        (root / manifest["annotation_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_script("accuracy").load_dataset(root)


def test_image_hash_mismatch_is_not_a_skipped_image(dataset, load_script):
    root, manifest, _ = dataset
    row = manifest["images"][0]
    (root / row["file_name"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum/size"):
        load_script("accuracy").read_rgb(root, row)


@pytest.mark.parametrize("path", ["../escape.jpg", "/absolute.jpg"])
def test_dataset_paths_cannot_escape_the_registered_directory(tmp_path, load_script, path):
    with pytest.raises(ValueError, match="unsafe"):
        load_script("accuracy").dataset_file(tmp_path, path)


def test_public_boxes_are_already_original_coordinates_and_class_zero_is_drone(load_script):
    script = load_script("accuracy")
    rows = script.prediction_rows(SimpleNamespace(boxes=[box()]), 42, 640, 512)
    np.testing.assert_allclose(rows, [[42, 10, 20, 3, 3, 0.8, 1]])
    assert script.prediction_rows(SimpleNamespace(boxes=[]), 42, 640, 512).shape == (0, 7)
    zero = script.prediction_rows(SimpleNamespace(boxes=[box(x2=10)]), 42, 640, 512)
    assert len(zero) == 1 and zero[0, 3] == 0, "do not silently discard false-positive candidates"


@pytest.mark.parametrize(
    "changes",
    [
        {"class_id": 1},
        {"score": math.nan},
        {"x1": math.inf},
        {"x2": 9},
        {"x2": 641},
        {"score": 0.0001},
    ],
)
def test_malformed_public_detections_fail_instead_of_being_filtered(load_script, changes):
    with pytest.raises(ValueError):
        load_script("accuracy").prediction_rows(SimpleNamespace(boxes=[box(**changes)]), 1, 640, 512)


def test_public_capacity_and_score_order_are_checked(load_script):
    script = load_script("accuracy")
    for boxes in ([box()] * 301, [box(score=0.4), box(score=0.8)]):
        with pytest.raises(ValueError):
            script.prediction_rows(SimpleNamespace(boxes=boxes), 1, 640, 512)


def test_real_scorer_distinguishes_perfect_wrong_and_empty_predictions(dataset, load_script):
    _, _, annotations = dataset
    script = load_script("accuracy")
    correct = np.array([[1, 10, 20, 3, 3, 0.9, 1], [2, 10, 20, 3, 3, 0.8, 1]])
    perfect, details = script.score_detections(copy.deepcopy(annotations), correct, [1, 2, 3])
    shifted = correct.copy()
    shifted[:, 1] += 50
    wrong, _ = script.score_detections(copy.deepcopy(annotations), shifted, [1, 2, 3])
    empty, _ = script.score_detections(copy.deepcopy(annotations), np.empty((0, 7)), [1, 2, 3])
    assert perfect["coco_map"] == pytest.approx(1.0)
    assert perfect["coco_ap50"] == pytest.approx(1.0)
    assert wrong["coco_map"] == empty["coco_map"] == 0
    assert details["tiny_ground_truth"] == details["tiny_matched50"] == 2
    assert details["tiny_recall50_max300"] == 1
    false_alarm = np.vstack([correct, [3, 10, 20, 3, 3, 0.99, 1]])
    negatives, _ = script.score_detections(copy.deepcopy(annotations), false_alarm, [1, 2, 3])
    assert negatives["coco_ap50"] < perfect["coco_ap50"], "empty images must count their false alarms"


def test_standard_cap100_and_supplemental_cap300_are_not_confused(dataset, load_script):
    _, _, annotations = dataset
    annotations["annotations"] = [
        {"id": i + 1, "image_id": 1, "category_id": 1, "bbox": [4 * i, 20, 2, 2], "area": 4, "iscrowd": 0}
        for i in range(150)
    ]
    rows = np.array([[1, 4 * i, 20, 2, 2, 1 - i / 1000, 1] for i in range(150)])
    metrics, details = load_script("accuracy").score_detections(annotations, rows, [1])
    assert metrics["coco_ar100"] == pytest.approx(100 / 150)
    assert details["recall50_max300"] == 1
    assert details["tiny_matched50"] == 150


def mock_runner(script, monkeypatch, *, fail=False):
    calls = []

    class Runner:
        closed = False

        def run(self, inputs, **kwargs):
            calls.append(inputs[0])
            if fail:
                raise RuntimeError("actual inference failure")
            return ["public-output"]

        def close(self):
            self.closed = True

    runner = Runner()
    neat = neat_stub()
    neat.detections = SimpleNamespace(decode_bbox_tensor=lambda *_: SimpleNamespace(boxes=[]))
    monkeypatch.setattr(script, "pyneat", neat)
    return SimpleNamespace(build=lambda _: runner), runner, calls, neat


def test_full_coverage_and_honest_partial_report(dataset, load_script, monkeypatch):
    root, _, _ = dataset
    script = load_script("accuracy")
    model, runner, calls, _ = mock_runner(script, monkeypatch)
    full = script.evaluate_dataset(model, root)
    assert runner.closed and len(calls) == full["evaluated_samples"] == full["available_samples"] == 3
    assert full["metrics"]["coco_map"] == 0 and full["details"]["detections"] == 0
    model, runner, calls, _ = mock_runner(script, monkeypatch)
    partial = script.evaluate_dataset(model, root, limit=1)
    assert runner.closed and len(calls) == partial["evaluated_samples"] == 1
    assert partial["available_samples"] == 3
    assert all(image.shape == (512, 640, 3) and image.dtype == np.uint8 for image in calls)


def test_inference_failure_propagates_and_closes_the_runner(dataset, load_script, monkeypatch):
    root, _, _ = dataset
    script = load_script("accuracy")
    model, runner, _, _ = mock_runner(script, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="actual inference failure"):
        script.evaluate_dataset(model, root)
    assert runner.closed


def test_standard_cli_writes_one_report_entry_per_named_dataset(dataset, tmp_path, load_script, monkeypatch):
    root, manifest, annotations = dataset
    script = load_script("accuracy")
    model, _, _, neat = mock_runner(script, monkeypatch)
    neat.Model = lambda *_: model
    monkeypatch.setitem(sys.modules, "pyneat", neat)
    other = tmp_path / "test-dataset"
    shutil.copytree(root, other)
    manifest["split"] = "test"
    manifest["variant"] = "sequence-safe-test-v1"
    manifest["annotation_file"] = "annotations/instances_test.json"
    write_dataset(other, manifest, annotations)
    output = tmp_path / "report"
    assert (
        script.main(
            [
                "--mpk",
                "model=model.tar.gz",
                "--dataset",
                f"anti-uav-val={root}",
                "--dataset",
                f"anti-uav-test={other}",
                "--output",
                str(output),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    report = json.loads((output / "model-report.json").read_text())
    assert (report["schema"], report["schema_version"], report["stage"]) == ("sima.model-report", 1, "accuracy")
    assert set(report["datasets"]) == {"anti-uav-val", "anti-uav-test"}
    assert all(
        result["evaluated_samples"] == 1 and result["available_samples"] == 3 for result in report["datasets"].values()
    )
    assert report["runtime"]["pyneat"] == "test-runtime"


@pytest.mark.parametrize("wrong", ["swapped-source", "variant"])
def test_cli_rejects_swapped_splits_before_model_creation(dataset, tmp_path, load_script, monkeypatch, wrong):
    root, manifest, annotations = dataset
    script = load_script("accuracy")
    neat = neat_stub()
    calls = []
    neat.Model = lambda *_: calls.append("created")
    monkeypatch.setattr(script, "pyneat", neat)
    source = "anti-uav-test"
    if wrong == "variant":
        source = "anti-uav-val"
        manifest["variant"] = "sequence-safe-test-v1"
        write_dataset(root, manifest, annotations)
    with pytest.raises(ValueError, match="manifest|variant"):
        script.main(["--mpk", "model=model.tar.gz", "--dataset", f"{source}={root}", "--output", str(tmp_path / "out")])
    assert not calls


@pytest.mark.parametrize("values", [["missing-separator"], ["=path"], ["name="], ["name=a", "name=b"]])
def test_named_inputs_reject_ambiguous_values(load_script, values):
    with pytest.raises(ValueError):
        load_script("accuracy").parse_named(values, "--dataset")


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []
