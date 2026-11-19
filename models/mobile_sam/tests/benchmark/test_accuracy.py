from __future__ import annotations

import json
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from model_registry.model_tests import measured_shortfalls


@pytest.fixture
def accuracy_script(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("accuracy")


def test_a_box_prompt_has_only_two_active_corners(accuracy_script: ModuleType) -> None:
    prompt = accuracy_script.box_prompt([100.0, 50.0, 200.0, 100.0], (480, 640))

    # 640 wide scales to 1024: corners (100, 50) and (300, 150), shifted to pixel centres and normalized.
    coords = prompt["point_coords"][0, :, 0, :2].T
    np.testing.assert_allclose(coords, (np.array([[100, 50], [300, 150]]) * 1.6 + 0.5) / 1024 * 2 - 1, rtol=1e-6)
    types = prompt["point_types"][0, :, 0, :].argmax(axis=0)
    assert types.tolist() == [3, 4, 0, 0, 0, 0]
    assert (prompt["point_types"][..., 2:] == 0).all()


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_mask_layout_reads_both_layouts(load_script: Callable[[str], ModuleType], layout: str) -> None:
    masks = np.random.default_rng(0).normal(size=(4, 256, 256)).astype(np.float32)
    board = masks[None] if layout == "nchw" else masks.transpose(1, 2, 0)[None]

    read, scores = load_script("pipeline").mask_layout(board, np.arange(4, dtype=np.float32)[None])

    np.testing.assert_array_equal(read, masks)
    assert scores.tolist() == [0, 1, 2, 3]


def test_image_mask_drops_the_padding_and_fits_the_image(load_script: Callable[[str], ModuleType]) -> None:
    # A 1024x512 resized image fills the top half of the padded input; its mask covers that half.
    logits = np.full((256, 256), -5.0, np.float32)
    logits[:128] = 5.0

    mask = load_script("pipeline").image_mask(logits, (300, 600))

    assert mask.shape == (300, 600)
    assert mask.all()


def test_a_result_scores_the_decoded_mask_by_its_predicted_iou(
    accuracy_script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    mask = np.zeros((480, 640), bool)
    mask[256:512, 300:400] = True
    encoded = []
    monkeypatch.setattr(accuracy_script.mask_util, "encode", lambda array: encoded.append(array) or [{"counts": b"x"}])

    item = accuracy_script.result({"image_id": 7, "category_id": 3}, mask, np.float32(0.9))

    (array,) = encoded
    assert array.shape == (480, 640, 1) and array.dtype == np.uint8 and array.flags.f_contiguous
    np.testing.assert_array_equal(array[:, :, 0], mask)
    assert item["segmentation"] == {"counts": "x"}
    assert (item["image_id"], item["category_id"], item["score"]) == (7, 3, pytest.approx(0.9))
    assert type(item["score"]) is float


def test_accuracy_meets_its_expected_values(accuracy: dict[str, Any], model_dir: Path, build_profile: str) -> None:
    assert measured_shortfalls(model_dir, "accuracy", build_profile, accuracy) == []


def _bilinear_reference(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Independent float64 pixel-centre interpolation: align_corners=False, no antialiasing."""
    height, width = size
    rows = np.maximum((np.arange(height) + 0.5) * image.shape[0] / height - 0.5, 0)
    columns = np.maximum((np.arange(width) + 0.5) * image.shape[1] / width - 0.5, 0)
    y0, x0 = rows.astype(int), columns.astype(int)
    y1, x1 = np.minimum(y0 + 1, image.shape[0] - 1), np.minimum(x0 + 1, image.shape[1] - 1)
    dy, dx = rows - y0, columns - x0
    top = image[y0[:, None], x0] * (1 - dx) + image[y0[:, None], x1] * dx
    bottom = image[y1[:, None], x0] * (1 - dx) + image[y1[:, None], x1] * dx
    return top * (1 - dy[:, None]) + bottom * dy[:, None]


@pytest.mark.parametrize("size", [(300, 600), (640, 480), (331, 417), (720, 1280), (1024, 1024), (17, 13)])
def test_mask_geometry_matches_independent_sam_pixel_centres(size, load_script):
    pipeline = load_script("pipeline")
    logits = np.random.default_rng(10).normal(size=(256, 256)).astype(np.float32)
    resized_height, resized_width = pipeline.resized_size(*size)
    upsampled = _bilinear_reference(logits.astype(np.float64), (1024, 1024))
    expected = _bilinear_reference(upsampled[:resized_height, :resized_width], size)
    actual = pipeline.resize_mask_logits(logits, size)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
    assert actual.shape == size
    # Floating-point rounding around zero is not a licence to change interpolation or threshold policy.
    away_from_zero = np.abs(expected) > 2e-5
    np.testing.assert_array_equal((actual > 0)[away_from_zero], (expected > 0)[away_from_zero])


class _Options:
    """A stand-in for pyneat.ModelOptions whose nested fields come into being when read, like the bound ones."""

    def __getattr__(self, name: str) -> _Options:
        value = _Options()
        setattr(self, name, value)
        return value


def test_accuracy_and_performance_share_neats_preparation_and_mask_decoder(load_script, monkeypatch):
    performance, accuracy = load_script("performance"), load_script("accuracy")
    pyneat = performance.pyneat
    monkeypatch.setattr(pyneat, "ModelOptions", _Options)
    encoders = performance.image_options(), accuracy.image_options()
    for name in ("mode", "placement", "pad_value", "interpolation"):
        assert getattr(encoders[0].preprocess.resize, name) == getattr(encoders[1].preprocess.resize, name), name
    for name in ("rescale_factor", "floating_conversion"):
        assert getattr(encoders[0].preprocess, name) == getattr(encoders[1].preprocess, name), name
    for name in ("mean", "stddev", "has_explicit_stats"):
        assert getattr(encoders[0].preprocess.normalize, name) == getattr(encoders[1].preprocess.normalize, name)
    # COCO val2017 images are at most 640 pixels on a side; the benchmark frame is 1280x720.
    assert encoders[1].preprocess.input_max_width == encoders[1].preprocess.input_max_height == 640
    click, box = performance.decoder_options(), accuracy.decoder_options()
    assert click.decode_type == box.decode_type == pyneat.BoxDecodeType.MobileSam
    # segment_anything's multimask_output: True for one click, False for a box prompt (mask 0).
    assert (click.multimask_output, box.multimask_output) == (True, False)


def test_application_helpers_share_one_mask_policy(load_script):
    pipeline = load_script("pipeline")
    masks = np.random.default_rng(10).normal(size=(4, 256, 256)).astype(np.float32)
    size = (300, 600)
    np.testing.assert_array_equal(pipeline.postprocess_masks(masks, size)[0] > 0, pipeline.image_mask(masks[0], size))
    # The single-mask token is not a candidate in multimask click mode, even with its highest score.
    scores = np.array([99, 0.2, 0.8, 0.4], np.float32)
    np.testing.assert_array_equal(
        pipeline.selected_mask(masks, scores, size, multimask=True), pipeline.image_mask(masks[2], size)
    )
    np.testing.assert_array_equal(
        pipeline.selected_mask(masks, scores, size, multimask=False), pipeline.image_mask(masks[0], size)
    )


@pytest.mark.parametrize("field", ["masks", "scores"])
def test_decoder_nonfinite_outputs_fail_before_mask_publication(field, load_script):
    pipeline = load_script("pipeline")
    masks, scores = np.zeros((4, 256, 256), np.float32), np.ones(4, np.float32)
    (masks if field == "masks" else scores).flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pipeline.selected_mask(masks, scores, (300, 600), multimask=True)


def test_accuracy_closes_all_runners_without_masking_inference_error(load_script):
    script = load_script("accuracy")
    calls = []

    class Runner:
        def __init__(self, name, fails=False):
            self.name, self.fails = name, fails

        def close(self):
            calls.append(self.name)
            if self.fails:
                raise RuntimeError("cleanup")

    runners = script.Runners.__new__(script.Runners)
    runners.runners = {"first": Runner("first"), "second": Runner("second", True)}
    with pytest.raises(ValueError, match="inference"):
        with runners:
            raise ValueError("inference")
    assert calls == ["second", "first"]


def test_fp32_runner_feeds_the_real_component_chain(load_script):
    script = load_script("fp32_baseline_accuracy")
    pipeline = load_script("pipeline")
    feeds = []
    runners = script.FloatRunners.__new__(script.FloatRunners)
    embedding = object()
    masks = np.full((1, 4, 256, 256), -5.0, np.float32)
    masks[0, 0, :96] = 5.0  # mask 0 covers the image's top half: a 480x640 image fills rows 0-191 of 256
    scores = np.array([[0.75, 0.9, 0.9, 0.9]], np.float32)

    def run(component, values):
        feeds.append((component, values))
        return {"image_encoder": [embedding], "prompt_decoder": [masks, scores]}[component]

    runners.run = run
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)
    actual = runners.encode(image)
    assert actual == (embedding, (480, 640))
    # segment_anything's CPU preparation, as an NCHW batch of one.
    np.testing.assert_array_equal(feeds[0][1]["image"], pipeline.preprocess(image).transpose(2, 0, 1)[None])
    assert len(feeds) == 1  # One whole encoder invocation, no intermediate feature component.
    mask, score = runners.decode(actual, {"point_coords": "coordinates", "point_types": "types"})
    assert feeds[-1][1] == {"image_embeddings": embedding, "point_coords": "coordinates", "point_types": "types"}
    np.testing.assert_array_equal(mask, pipeline.image_mask(masks[0, 0], (480, 640)))
    assert mask.shape == (480, 640) and mask[:240].all() and not mask[240:].any()
    assert score == pytest.approx(0.75)


@pytest.fixture
def fp32_run(load_script, monkeypatch, tmp_path):
    """Exercise the real baseline CLI/persistence with a tiny fake inference result, not an accuracy claim."""
    script = load_script("fp32_baseline_accuracy")
    source = tmp_path / "source.onnx"
    source.write_bytes(b"test source")
    source.with_name(script.conversion_outputs(script.MODEL_DIR)[1]).write_bytes(b"test weights")
    dataset = tmp_path / "dataset"
    annotations = dataset / script.accuracy.ANNOTATIONS
    annotations.parent.mkdir(parents=True)
    annotations.write_text("{}")
    images = dataset / script.accuracy.IMAGES
    images.mkdir()
    for image_id in (7, 9):
        (images / f"{image_id}.jpg").write_bytes(f"image {image_id}".encode())
    rows = [
        {"image_id": image_id, "category_id": 3, "score": 0.75, "segmentation": {"size": [2, 2], "counts": "4"}}
        for image_id in (7, 9)
    ]

    class GroundTruth:
        def getImgIds(self):
            return [9, 7]

        def getAnnIds(self, imgIds, iscrowd):
            assert iscrowd is False
            return imgIds

        def loadAnns(self, ids):
            return [item for item in rows if item["image_id"] in ids]

        def loadImgs(self, ids):
            return [{"id": image_id, "file_name": f"{image_id}.jpg"} for image_id in ids]

    references = []

    class Runners:
        def __init__(self, graphs):
            references.append(weakref.ref(self))

        def encode(self, value):
            raise AssertionError("fake evaluator does not infer")

        def decode(self, value, prompt):
            raise AssertionError("fake evaluator does not infer")

    monkeypatch.setattr(script, "FloatRunners", Runners)
    monkeypatch.setattr(script, "export_components", lambda *args: {})
    monkeypatch.setattr(script.accuracy, "COCO", lambda *args: GroundTruth())
    monkeypatch.setattr(script.accuracy, "VAL2017_IMAGES", 2)
    monkeypatch.setattr(script.accuracy, "evaluate", lambda *args: ([dict(item) for item in rows], len(rows)))
    monkeypatch.setattr(script.accuracy, "score", lambda *args: {"test_metric": 0.5})
    output = tmp_path / "report.json"
    args = ["--onnx", str(source), "--dataset", f"coco-2017-val={dataset}", "--output", str(output)]
    return script, args, output, output.with_suffix(".predictions.json"), references


def test_fp32_predictions_survive_scoring_failure_and_reuse_the_same_scorer(fp32_run, monkeypatch):
    script, args, output, predictions, runners = fp32_run

    def fail_score(ground_truth, results, image_ids):
        assert predictions.is_file()
        assert runners[0]() is None  # ORT sessions are released before scoring.
        assert image_ids == [7, 9]
        raise RuntimeError("score failed")

    monkeypatch.setattr(script.accuracy, "score", fail_score)
    with pytest.raises(RuntimeError, match="score failed"):
        script.main(args)
    assert not output.exists()
    saved_bytes = predictions.read_bytes()
    saved = json.loads(saved_bytes)
    assert saved["provenance"]["image_ids"] == [7, 9]
    assert saved["prompts"] == len(saved["predictions"]) == 2
    assert len(saved["provenance"]["onnx_sha256"]) == len(saved["provenance"]["weights_sha256"]) == 64
    assert len(saved["provenance"]["annotations_sha256"]) == 64

    def score(ground_truth, results, image_ids):
        assert results == saved["predictions"] and image_ids == [7, 9]
        return {"test_metric": 0.5}

    def cannot_infer(*args):
        raise AssertionError("score-only must not create ORT sessions or run inference")

    monkeypatch.setattr(script, "FloatRunners", cannot_infer)
    monkeypatch.setattr(script, "export_components", cannot_infer)
    monkeypatch.setattr(script.accuracy, "evaluate", cannot_infer)
    monkeypatch.setattr(script.accuracy, "score", score)
    assert script.main([*args, "--score-only"]) == 0
    assert json.loads(output.read_text())["evaluated_samples"] == 2
    assert predictions.read_bytes() == saved_bytes


@pytest.mark.parametrize(
    "field",
    [
        "onnx_sha256",
        "weights_sha256",
        "annotations_sha256",
        "selected_images_sha256",
        "image_ids",
        "scripts_sha256",
        "conditions",
    ],
)
def test_fp32_score_only_rejects_mismatched_provenance(fp32_run, field):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = json.loads(predictions.read_text())
    saved["provenance"][field] = "changed"
    predictions.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="provenance"):
        script.main([*args, "--score-only"])


@pytest.mark.parametrize("damage", ["missing", "duplicate", "prompt_count", "nonfinite"])
def test_fp32_score_only_rejects_incomplete_or_illegal_predictions(fp32_run, damage):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = json.loads(predictions.read_text())
    if damage == "missing":
        saved["predictions"].pop()
    elif damage == "duplicate":
        saved["predictions"][0] = saved["predictions"][1]
    elif damage == "prompt_count":
        saved["prompts"] += 1
    else:
        saved["predictions"][0]["score"] = float("nan")
    predictions.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="counts|finite"):
        script.main([*args, "--score-only"])


def test_fp32_inference_does_not_overwrite_existing_predictions(fp32_run):
    script, args, _, predictions, _ = fp32_run
    script.main(args)
    saved = predictions.read_bytes()
    with pytest.raises(SystemExit):
        script.main(args)
    assert predictions.read_bytes() == saved


def test_fp32_score_only_detects_changed_image_bytes(fp32_run):
    script, args, output, _, _ = fp32_run
    script.main(args)
    (output.parent / "dataset" / script.accuracy.IMAGES / "7.jpg").write_bytes(b"different image")
    with pytest.raises(ValueError, match="provenance"):
        script.main([*args, "--score-only"])


def test_fp32_prediction_publication_does_not_clobber_a_late_writer(fp32_run, monkeypatch):
    script, args, output, predictions, _ = fp32_run
    evaluate = script.accuracy.evaluate

    def infer(*args):
        predictions.write_text("another completed prediction artifact")
        return evaluate(*args)

    monkeypatch.setattr(script.accuracy, "evaluate", infer)
    with pytest.raises(FileExistsError):
        script.main(args)
    assert predictions.read_text() == "another completed prediction artifact"
    assert not output.exists()


def test_fp32_failed_serialization_does_not_publish_a_partial_file(load_script, tmp_path):
    script = load_script("fp32_baseline_accuracy")
    output = tmp_path / "predictions.json"
    with pytest.raises(ValueError, match="JSON compliant"):
        script.write_json(output, {"score": float("nan")})
    assert list(tmp_path.iterdir()) == []


def _decoded(size: tuple[int, int], score: float = 0.5):
    from types import SimpleNamespace

    masks = np.zeros((1, *size), np.uint8)
    masks[0, : size[0] // 2] = 255
    boxes = np.array([[0, 0, size[1], size[0] // 2, score, 0]], np.float32)
    return [SimpleNamespace(boxes=SimpleNamespace(to_numpy=lambda copy: boxes.copy()),
                            masks=SimpleNamespace(to_numpy=lambda copy: masks.copy()))]


def test_native_runner_reuses_one_whole_encoder_for_repeated_prompts(load_script, monkeypatch):
    script = load_script("accuracy")
    runners = script.Runners.__new__(script.Runners)
    runners.runners, runners.size = {}, None
    calls = []
    embedding = object()
    runners.prompt_encoding = lambda coords: "encoding"
    monkeypatch.setattr(script, "hwc", lambda value: value)
    monkeypatch.setattr(script, "rgb", lambda image: "rgb image")
    monkeypatch.setattr(script, "handoff", lambda tensors: tensors)
    monkeypatch.setattr(sys.modules["pyneat"], "decode_segmentation", lambda outputs: _decoded((10, 20), 0.875))

    def run(component, inputs):
        calls.append(component)
        if component == "image_encoder":
            assert inputs == ["rgb image"]
            return [embedding]
        assert inputs == [embedding, "encoding", "types"]
        return ["boxdecode"]

    runners.run = run
    encoded = runners.encode(np.zeros((10, 20, 3), np.uint8))
    for _ in range(3):
        mask, score = runners.decode(encoded, {"point_coords": "coords", "point_types": "types"})
        assert mask.dtype == bool and mask.shape == (10, 20) and mask[:5].all() and not mask[5:].any()
        assert score == 0.875
    assert calls == ["image_encoder", "prompt_decoder", "prompt_decoder", "prompt_decoder"]


def test_a_new_image_size_gets_a_new_decoder_runner(load_script, monkeypatch):
    from types import SimpleNamespace

    script = load_script("accuracy")
    runners = script.Runners.__new__(script.Runners)
    runners.runners, runners.size = {}, None
    built, closed = [], []
    runners.prompt_encoding = lambda coords: "encoding"
    monkeypatch.setattr(script, "hwc", lambda value: value)
    monkeypatch.setattr(script, "rgb", lambda image: image.shape)
    monkeypatch.setattr(script, "handoff", lambda tensors: tensors)

    def model(component):
        def build(inputs):
            built.append((component, runners.size))
            return SimpleNamespace(run=lambda inputs, timeout_ms: ["output"],
                                   close=lambda: closed.append((component, runners.size)))
        return SimpleNamespace(build=build)

    runners.models = {name: model(name) for name in script.COMPONENTS}
    for size in ((480, 640), (480, 640), (427, 640)):
        monkeypatch.setattr(sys.modules["pyneat"], "decode_segmentation", lambda outputs, size=size: _decoded(size))
        embedding = runners.encode(np.zeros((*size, 3), np.uint8))
        for _ in range(2):
            assert runners.decode(embedding, {"point_coords": "coords", "point_types": "types"})[0].shape == size
    # One encoder runner for every size; one decoder runner per size, the previous one closed first.
    assert built == [("image_encoder", (480, 640)), ("prompt_decoder", (480, 640)), ("prompt_decoder", (427, 640))]
    assert closed == [("prompt_decoder", (480, 640))]
    with pytest.raises(ValueError, match="expected one"):
        monkeypatch.setattr(sys.modules["pyneat"], "decode_segmentation", lambda outputs: _decoded((480, 640)))
        runners.decode(embedding, {"point_coords": "coords", "point_types": "types"})


def test_images_run_grouped_by_size_and_keep_every_result(load_script, monkeypatch, tmp_path):
    script = load_script("accuracy")
    records = {
        1: {"id": 1, "file_name": "1.jpg", "height": 480, "width": 640},
        2: {"id": 2, "file_name": "2.jpg", "height": 427, "width": 640},
        3: {"id": 3, "file_name": "3.jpg", "height": 480, "width": 640},
        4: {"id": 4, "file_name": "4.jpg", "height": 427, "width": 640},
    }
    annotations = {image_id: [{"image_id": image_id, "category_id": 1, "bbox": [1, 2, 30, 40]}] for image_id in records}

    class GroundTruth:
        def loadImgs(self, ids):
            return [records[image_id] for image_id in ids]

        def getAnnIds(self, imgIds, iscrowd):
            assert iscrowd is False
            return imgIds

        def loadAnns(self, ids):
            return annotations[ids]

    monkeypatch.setattr(script, "read_rgb", lambda path: path.stem)
    monkeypatch.setattr(script, "box_prompt", lambda bbox, size: size)
    monkeypatch.setattr(script, "result", lambda annotation, mask, score: (annotation["image_id"], mask, score))
    encoded = []

    def encode(image):
        encoded.append(int(image))
        return int(image)

    results, prompts = script.evaluate(
        GroundTruth(), [1, 2, 3, 4], tmp_path, encode, lambda image_id, size: (size, float(image_id))
    )
    assert encoded == [2, 4, 1, 3]  # smallest size first; ids in order within a size
    assert prompts == 4
    assert sorted(results) == [(1, (480, 640), 1.0), (2, (427, 640), 2.0), (3, (480, 640), 3.0), (4, (427, 640), 4.0)]
