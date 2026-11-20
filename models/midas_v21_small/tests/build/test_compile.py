from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

# CI on the 50 declared cal images measured cosine 0.99994 (w16a16) and 0.99798 (w8a8) against the converted ONNX.
# Keep margin for compiler rounding.
FIDELITY_MIN_COSINE = {"w16a16": 0.9995, "w8a8": 0.995}


def _image(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    import cv2

    image = np.zeros((height, width, 3), np.uint8)
    image[:] = colour[::-1]  # OpenCV writes BGR.
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_stretches_to_256x256_and_normalizes_with_imagenet_statistics(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")

    result = calibration.preprocess_image(_image(640, 480))

    assert (calibration.HEIGHT, calibration.WIDTH) == (256, 256)
    assert result.shape == (256, 256, 3) and result.dtype == np.float32
    np.testing.assert_allclose(calibration.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(calibration.STD, [0.229, 0.224, 0.225])
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[128, 128], red, atol=1e-5)


def test_preprocessing_does_not_keep_the_aspect_ratio(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    import cv2

    # A 640x480 frame, red above blue: stretched to 256x256, the boundary stays halfway down.
    image = np.zeros((480, 640, 3), np.uint8)
    image[:240] = (0, 0, 255)
    image[240:] = (255, 0, 0)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    result = calibration.preprocess_image(encoded.tobytes())

    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[120, 0], red, atol=1e-5)
    np.testing.assert_allclose(result[136, 255], blue, atol=1e-5)


def test_calibration_takes_a_seeded_sample_of_the_cal_images(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(
        tmp_path / "cal.tgz",
        {
            "nyu_depth_v2/cal/rgb_00009.jpg": _image(64, 48),
            "nyu_depth_v2/cal/rgb_00002.jpg": _image(64, 48, (0, 0, 255)),
            "nyu_depth_v2/annotations/depth_cal/sync_depth_00002.png": b"depth",
            "nyu_depth_v2/cal.txt": b"00009\n00002\n",
        },
    )

    assert calibration.split_names(archive, "cal") == [
        "nyu_depth_v2/cal/rgb_00002.jpg",
        "nyu_depth_v2/cal/rgb_00009.jpg",
    ]
    samples = calibration.calibration_samples(archive, 1)
    assert samples.shape == (1, 256, 256, 3) and samples.dtype == np.float32
    np.testing.assert_array_equal(samples, calibration.calibration_samples(archive, 1))
    with pytest.raises(ValueError, match="calibration requires 3 images"):
        calibration.calibration_samples(archive, 3)


def test_read_images_reports_a_missing_image(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    archive = _write_archive(tmp_path / "cal.tgz", {"nyu_depth_v2/cal/rgb_00001.jpg": b"a"})

    assert calibration.read_images(archive, ["nyu_depth_v2/cal/rgb_00001.jpg"]) == {
        "nyu_depth_v2/cal/rgb_00001.jpg": b"a"
    }
    with pytest.raises(ValueError, match="missing 1 requested images"):
        calibration.read_images(archive, ["nyu_depth_v2/cal/rgb_00001.jpg", "nyu_depth_v2/cal/rgb_00002.jpg"])


def test_fidelity_selection_is_the_declared_seeded_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"nyu_depth_v2/cal/rgb_{index:05d}.jpg" for index in range(10)]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


def test_the_benchmarks_read_the_directory_the_runners_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    """The runners look through the archive's one top-level directory, so the split sits directly under --dataset."""
    arguments = ["--mpk", "model=m.tar.gz", "--dataset", f"nyu-depth-v2-val={tmp_path}", "--output", str(tmp_path)]
    message = f"{tmp_path} has no images under test/"

    with pytest.raises(SystemExit, match=message):
        load_script("accuracy").main(arguments)
    # performance.py parses its arguments and runs as it loads.
    monkeypatch.setattr("sys.argv", ["performance.py", *arguments])
    with pytest.raises(SystemExit, match=message):
        load_script("performance")


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._parameters(profile_id)

    assert parameters["precision"] == profile_id
    assert parameters["input_shape"] == [1, 3, 256, 256]
    if profile_id == "w8a8":
        assert parameters["calibration_samples"] == 50


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="exactly one build profile named 'w32a32'"):
        recipe._parameters("w32a32")


def test_numerical_fidelity_compares_the_depth(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]["outputs"] == ["predicted_depth"]


@pytest.mark.parametrize(("precision", "calibrates"), [("w8a8", True), ("w16a16", False)])
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.CalibrationMethod.from_str.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration


def test_quantization_rejects_an_undeclared_precision(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValueError, match="unsupported precision 'w4a8'"):
        recipe._quantization("w4a8")


def test_bf16_builds_quantize_without_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(
        recipe,
        "_parameters",
        lambda profile_id: {"precision": "w16a16", "input_name": "pixel_values", "input_shape": [1, 3, 256, 256]},
    )
    prepared = tmp_path / "prepared.onnx"
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output: prepared)

    with pytest.raises(RuntimeError, match="expected exactly one MPK archive"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    # The compiler imports the rewritten graph, and BF16 needs no calibration data.
    assert recipe.onnx_source.call_args.args[0] == str(prepared)
    assert recipe.load_model.return_value.quantize.call_args.args[0] is None


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"precision": "fp32"}, "unsupported precision 'fp32'"),
        (
            {"precision": "w16a16", "input_name": "pixel_values", "input_shape": [1, 3, 256, 320]},
            "unsupported MiDaS v2.1 Small input shape",
        ),
    ],
)
def test_compile_rejects_an_undeclared_precision_or_input_shape_before_compiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    parameters: dict,
    message: str,
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)

    with pytest.raises(ValueError, match=message):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "model", "w16a16", None)
    assert not (tmp_path / "model").exists()


def _upsample_then_project(path: Path, kernel: int = 1) -> Path:
    """A RefineNet-style block: a 2x align-corners bilinear upsample, then a projection convolution."""
    helper = onnx.helper
    rng = np.random.default_rng(0)
    initializers = [
        numpy_helper.from_array(np.asarray([1, 1, 2, 2], np.float32), "scales"),
        numpy_helper.from_array(rng.standard_normal((2, 4, kernel, kernel)).astype(np.float32), "weight"),
        numpy_helper.from_array(rng.standard_normal(2).astype(np.float32), "bias"),
    ]
    nodes = [
        helper.make_node(
            "Resize", ["x", "", "scales"], ["up"], mode="linear", coordinate_transformation_mode="align_corners"
        ),
        helper.make_node("Conv", ["up", "weight", "bias"], ["y"], pads=[kernel // 2] * 4),
    ]
    graph = helper.make_graph(
        nodes,
        "block",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 5, 6])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 10, 12])],
        initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_a_projection_moves_ahead_of_its_upsample_without_changing_the_output(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    from onnx.reference import ReferenceEvaluator

    surgery = load_script("surgery")
    model = onnx.load(_upsample_then_project(tmp_path / "block.onnx"))
    x = np.random.default_rng(1).standard_normal((1, 4, 5, 6)).astype(np.float32)
    expected = ReferenceEvaluator(model).run(None, {"x": x})[0]

    assert surgery.commute_projections(model) == 1

    assert [node.op_type for node in model.graph.node] == ["Conv", "Resize"]
    assert model.graph.node[1].output[0] == "y"
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None, {"x": x})[0], expected, rtol=1e-5, atol=1e-5)


def test_a_wider_convolution_stays_after_its_upsample(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    model = onnx.load(_upsample_then_project(tmp_path / "block.onnx", kernel=3))

    assert surgery.commute_projections(model) == 0
    assert [node.op_type for node in model.graph.node] == ["Resize", "Conv"]


def test_the_recipe_moves_exactly_the_four_refinenet_projections(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")

    with pytest.raises(ValueError, match="moved 1 projections, expected 4"):
        surgery.prepare_model(_upsample_then_project(tmp_path / "block.onnx"), tmp_path / "prepared.onnx")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_compiled_build_runs_on_one_mla_stage_and_follows_the_onnx(compiled: Path, build_profile: str) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    with tarfile.open(archive, "r:gz") as stream:
        (manifest,) = [name for name in stream.getnames() if name.endswith("_mpk.json")]
        plugins = json.load(stream.extractfile(manifest))["plugins"]
    processors = [str(plugin.get("processor", "")).upper() for plugin in plugins]
    assert processors.count("MLA") == 1 and not {"A65", "APU"} & set(processors), processors
    (metric,) = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))["metrics"]
    assert metric["name"] == "predicted_depth"
    assert metric["cosine_similarity"] >= FIDELITY_MIN_COSINE[build_profile], metric
