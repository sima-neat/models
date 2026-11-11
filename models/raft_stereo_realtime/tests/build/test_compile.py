from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "raft_stereo_realtime"
# Floors under the published w16a16 fidelity on the 8 fidelity pairs: cosine 0.99983, mean absolute error 0.176 px.
FIDELITY_FLOORS = {"w16a16": {"cosine_similarity": 0.999, "mean_absolute_error": 0.25}}


@pytest.fixture
def scene_flow(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("scene_flow")


def _png(value: int, size: tuple[int, int] = (960, 540)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _pfm(disparity: np.ndarray) -> bytes:
    """A little-endian one-channel PFM, which stores its rows from the bottom."""
    height, width = disparity.shape
    return f"Pf\n{width} {height}\n-1.0\n".encode() + np.flipud(disparity).astype("<f4").tobytes()


def _pair(index: int) -> tuple[str, str, str]:
    return (
        f"Monkaa/frames_finalpass/scene/left/{index:04d}.png",
        f"Monkaa/frames_finalpass/scene/right/{index:04d}.png",
        f"Monkaa/disparity/scene/left/{index:04d}.pfm",
    )


def _cal_archive(path: Path, pairs: int) -> Path:
    """The scene-flow cal layout: scene_flow/cal.txt and uniform 960x540 views, grey level = pair (+100 if right),
    each with a disparity map of the pair's index in every pixel."""
    lines, files = [], {}
    for index in range(pairs):
        left, right, disparity = _pair(index)
        lines.append(f"{left} {right} {disparity}\n")
        files.update({left: _png(index), right: _png(index + 100)})
        files[disparity] = _pfm(np.full((540, 960), index, np.float32))
    files["cal.txt"] = "".join(lines).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"scene_flow/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def test_preprocessing_takes_a_native_view_and_normalizes_it(scene_flow: ModuleType) -> None:
    result = scene_flow.preprocess_image(_png(255))

    assert result.shape == (540, 960, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], (1 - scene_flow.MEAN) / scene_flow.STD, rtol=1e-6)
    # The graph pads the rows itself, so a view of any other size is refused rather than resized.
    with pytest.raises(ValueError, match="expected a 960x540 view"):
        scene_flow.preprocess_image(_png(0, (1242, 375)))


def test_pfm_rows_are_read_from_the_top(scene_flow: ModuleType) -> None:
    disparity = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_array_equal(scene_flow.read_pfm(_pfm(disparity)), disparity)
    with pytest.raises(ValueError, match="one-channel"):
        scene_flow.read_pfm(b"PF\n4 3\n-1.0\n" + bytes(144))


def test_calibration_and_fidelity_take_spans_of_one_seeded_order(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 12)

    order = scene_flow.selected_pairs(dataset, seed=42, offset=0, samples=12)

    # Every listed pair once, the same order for the same seed, and later spans of that order for fidelity.
    assert sorted(order) == [_pair(i) for i in range(12)]
    assert scene_flow.selected_pairs(dataset, seed=42, offset=4, samples=6) == order[4:10]
    with pytest.raises(ValueError, match="needs 13 pairs"):
        scene_flow.selected_pairs(dataset, seed=42, offset=1, samples=12)


def test_read_pairs_returns_side_by_side_frames_and_true_disparities(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 4)

    frames, disparities = scene_flow.read_pairs(dataset, [_pair(3), _pair(1)])

    assert frames.shape == (2, 540, 1920, 3) and disparities.shape == (2, 540, 960)
    # The fixture's grey level is the pair index for the left view and 100 more for the right one; the left
    # view fills the first 960 columns of each frame and the right view the rest.
    grey = (np.array([3, 1, 103, 101], np.float32) / 255 - scene_flow.MEAN[0]) / scene_flow.STD[0]
    np.testing.assert_allclose(frames[:, 20, 959, 0], grey[:2], rtol=1e-6)
    np.testing.assert_allclose(frames[:, 20, 960, 0], grey[2:], rtol=1e-6)
    np.testing.assert_array_equal(disparities[:, 7, 7], [3, 1])
    with pytest.raises(ValueError, match="lacks 3 listed files"):
        scene_flow.read_pairs(dataset, [_pair(7)])


def test_end_point_error_scores_pixels_between_0_and_192(scene_flow: ModuleType) -> None:
    truth = np.array([[10.0, 20.0, 0.0], [191.0, 192.0, -3.0]], np.float32)
    disparity = np.array([[[[11.0, 18.0, 5.0], [191.0, 0.0, 9.0]]]], np.float32)

    # |11 - 10|, |18 - 20| and 0 over the three pixels between 0 and 192; the 0, 192 and -3 px pixels are not scored.
    assert scene_flow.end_point_error(disparity, truth) == pytest.approx(1.0)
    # An image with no scored pixel counts as 0, not NaN.
    assert scene_flow.end_point_error(disparity, np.full_like(truth, 300.0)) == 0.0


@pytest.fixture
def surgery(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def test_interface_is_the_native_scene_flow_pair_and_the_left_views_disparity(surgery: ModuleType) -> None:
    # Scene Flow views are 960x540; the graph pads them to 544 rows itself, so Neat's resize leaves a pair as it is.
    assert surgery.INTERFACE == {"stereo": [1, 3, 540, 1920], "disparity": [1, 1, 540, 960]}
    assert surgery.PADDED_HEIGHT % 32 == 0 and surgery.PADDED_HEIGHT - surgery.HEIGHT == 2 * surgery.PAD
    # Upstream's realtime evaluation: 7 iterations over a 4-level correlation pyramid of radius 4.
    assert (surgery.ITERATIONS, surgery.LEVELS, surgery.RADIUS) == (7, 4, 4)


def _export(surgery: ModuleType, case: str) -> onnx.ModelProto:
    """The side-by-side frame sliced into its halves and summed into the disparity through a 1x1 Conv, or one fault
    validate_export must catch."""
    helper = onnx.helper
    shapes = dict(surgery.INTERFACE)
    if case == "shape":
        shapes["stereo"] = [1, 3, 544, 1920]
    inputs = [helper.make_tensor_value_info("stereo", onnx.TensorProto.FLOAT, shapes["stereo"])]
    if case == "two_inputs":
        inputs.append(helper.make_tensor_value_info("right", onnx.TensorProto.FLOAT, [1, 3, 540, 960]))
    output = helper.make_tensor_value_info("disparity", onnx.TensorProto.FLOAT, shapes["disparity"])
    constants = {
        "weight": np.ones((1, 3, 1, 1), np.float32),
        "width": np.array([3]),
        "zero": np.array([0]),
        "half": np.array([960]),
        "end": np.array([1920]),
    }
    if case == "constants":
        constants["table"] = np.zeros(1000, np.float32)
    initializers = [onnx.numpy_helper.from_array(value, name) for name, value in constants.items()]
    nodes = [
        helper.make_node("Slice", ["stereo", "zero", "half", "width"], ["left"]),
        helper.make_node("Slice", ["stereo", "half", "end", "width"], ["right_view"]),
        helper.make_node("Add", ["left", "right_view"], ["sum"]),
        helper.make_node("Conv", ["sum", "weight"], ["disparity"]),
    ]
    if case == "edge_pad":
        nodes.insert(0, helper.make_node("Pad", ["stereo", "pads"], ["padded"], mode="edge"))
        initializers.append(onnx.numpy_helper.from_array(np.array([0, 0, 2, 0, 0, 0, 2, 0]), "pads"))
    if case == "grid_sample":
        nodes.append(helper.make_node("GridSample", ["sum", "grid"], ["sampled"]))
        initializers.append(onnx.numpy_helper.from_array(np.zeros((1, 1, 1, 2), np.float32), "grid"))
    if case == "abs":
        nodes.append(helper.make_node("Abs", ["sum"], ["magnitude"]))
    if case == "shape_arithmetic":
        nodes.append(helper.make_node("Shape", ["sum"], ["unused"]))
    graph = helper.make_graph(nodes, "fixture", inputs, [output], initializers)
    opset = 13 if case == "opset" else surgery.OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("valid", None),
        ("opset", "requires opset"),
        ("two_inputs", "Expected inputs and outputs"),
        ("shape", "Unexpected dtype or shape"),
        ("edge_pad", r"still holds \['Pad'\]"),
        ("grid_sample", r"still holds \['GridSample'\]"),
        ("abs", r"still holds \['Abs'\]"),
        ("shape_arithmetic", r"still holds \['Shape'\]"),
        ("constants", "constant values"),
    ],
)
def test_export_must_keep_the_stereo_interface_and_mla_operators(
    tmp_path: Path, case: str, error: str | None, surgery: ModuleType
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_export(surgery, case), path)

    if error is None:
        surgery.validate_export(path, max_constants=1000)
    else:
        with pytest.raises(ValueError, match=error):
            surgery.validate_export(path, max_constants=1000)


def test_normalize_joins_the_views_and_applies_the_imagenet_mean_and_std(surgery: ModuleType) -> None:
    left = np.zeros((540, 960, 3), np.float32)
    right = np.full((540, 960, 3), 255, np.float32)

    frame = surgery.normalize(left, right)

    assert frame.shape == (1, 3, 540, 1920) and frame.dtype == np.float32
    np.testing.assert_allclose(frame[0, :, 0, 0], -surgery.MEAN / surgery.STD, rtol=1e-6)
    np.testing.assert_allclose(frame[0, :, 0, 960], (1 - surgery.MEAN) / surgery.STD, rtol=1e-6)


def test_disparity_comparison_rejects_shape_nan_and_excess_error(surgery: ModuleType) -> None:
    expected = np.random.RandomState(0).uniform(0, 192, (1, 1, 540, 960)).astype(np.float32)

    assert surgery.compare_disparity(expected, expected + 0.004) == pytest.approx(0.004, abs=1e-4)
    with pytest.raises(ValueError, match="differs from PyTorch by 0.02"):
        surgery.compare_disparity(expected, expected + 0.02)
    with pytest.raises(ValueError, match="non-finite"):
        surgery.compare_disparity(expected, np.full_like(expected, np.nan))
    with pytest.raises(ValueError, match="Unexpected"):
        surgery.compare_disparity(expected, expected[:, :, :-1])


def test_synthetic_pair_shifts_the_right_view_by_a_disparity_growing_down_the_image(surgery: ModuleType) -> None:
    pytest.importorskip("torch")
    left, right = surgery.stereo_pair(123)

    assert left.shape == right.shape == (540, 960, 3)
    # Row r of the right view samples the left view's texture d = 8 + 40 r / 540 columns further right.
    for row, disparity in ((0, 8), (270, 28)):
        np.testing.assert_allclose(right[row, :100], left[row, disparity : disparity + 100], atol=1e-3)
    assert np.array_equal(surgery.stereo_pair(123)[1], right)


def _mpk_manifest(compile_module: ModuleType) -> dict:
    """A plugin chain of the compiled MPK: a cast of the frame, the MLA, then a slice and a cast back."""
    return {
        "input_nodes": [{"name": "stereo", "size": 1 * 3 * 540 * 1920 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "MLA", "config_params": {}},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "slice_transform"}},
            {
                "sequence": 4,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 1 * 1 * 540 * 960 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_inputs", "output", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_stereo_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "two_inputs":
        manifest["input_nodes"].append({"name": "right", "size": 1 * 3 * 540 * 960 * 4})
    elif case == "output":
        plugins[-1]["output_nodes"][0]["size"] = 1 * 1 * 544 * 960 * 4
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65", "config_params": {}})
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    assert compile_module._parameters("w16a16") == {
        "precision": "w16a16",
        "calibration_bins": 2048,
        "calibration_samples": 4,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w32a32")
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_module._quantization({"precision": "w4a16", "calibration_bins": 2048})


def test_model_calibrates_and_compares_disparity_on_scene_flow_cal() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    # model.yaml holds w8a8 back until an INT8 recipe keeps the accuracy.
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16"]
    largest = max(profile["components"][0]["parameters"]["calibration_samples"] for profile in model["build_profiles"])
    fidelity = model["numerical_fidelity"]
    # Calibration takes the first pairs of the seeded order, so the fidelity pairs must start after them.
    assert fidelity["selection"]["offset"] >= largest
    assert fidelity["dataset"]["variant"] == "cal" and fidelity["outputs"] == ["disparity"]
    assert model["license"]["redistribution"] == "allowed"


def test_compiled_build_holds_the_audited_mpk_and_a_disparity_fidelity_report(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    (metric,) = report["metrics"]
    assert metric["name"] == "disparity"
    assert np.isfinite([metric["mean_absolute_error"], metric["cosine_similarity"]]).all()
    floor = FIDELITY_FLOORS[build_profile]
    assert metric["cosine_similarity"] >= floor["cosine_similarity"], metric
    assert metric["mean_absolute_error"] <= floor["mean_absolute_error"], metric
