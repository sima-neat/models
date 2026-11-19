from __future__ import annotations

import io
import json
import math
import random
import sys
import tarfile
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call

import numpy as np
import pytest
from PIL import Image


def _png(width: int, height: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 60, 40)).save(stream, format="PNG")
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _mpk(path: Path, plugins: list[str], inputs: dict[str, int]) -> Path:
    manifest = {
        "plugins": [{"processor": processor} for processor in plugins],
        "input_nodes": [{"name": name, "size": size} for name, size in inputs.items()],
    }
    return _write_archive(path, {"model_mpk.json": json.dumps(manifest).encode(), "model.elf": b"elf"})


def test_a_prompt_becomes_normalized_corners_and_one_hot_types(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A box on a 480x640 image: x scales by 1024/640 and y by 1024/480 before the half-pixel shift.
    points = calibration.prompt_inputs([(64.0, 48.0), (320.0, 240.0)], ["box_top_left", "box_bottom_right"],
                                       (480, 640))["points"]

    assert points.shape == (1, 6, 8)
    np.testing.assert_allclose(points[0, 0, :2], [(64 * 1.6 + 0.5) / 1024, (48 * 1024 / 480 + 0.5) / 1024])
    np.testing.assert_allclose(points[0, 2:, :2], -0.5 / 1024)
    # The one-hot types sit in columns 2 to 6: padding, negative, positive, box top-left, box bottom-right.
    assert points[0, :, 2].tolist() == [0, 0, 1, 1, 1, 1]
    assert (points[0, 0, 5], points[0, 1, 6]) == (1, 1)
    assert points[0, :, 2:7].sum() == 6 and not points[0, :, 7].any()
    with pytest.raises(ValueError, match="1 to 6 points"):
        calibration.prompt_inputs([(0.0, 0.0)] * 7, ["positive"] * 7, (10, 10))


def test_the_performance_script_builds_the_same_prompt_inputs(load_script: Callable[[str], ModuleType]) -> None:
    calibration, performance = load_script("calibration"), load_script("performance")
    points, types, size = [(10.0, 20.0), (300.0, 200.0), (5.0, 6.0)], ["positive", "negative", "positive"], (720, 1280)

    expected = calibration.prompt_inputs(points, types, size)
    np.testing.assert_array_equal(performance.prompt_array(points, types, size), expected["points"])
    assert (performance.POINT_TYPES, performance.POINT_COLUMNS) == (calibration.POINT_TYPES, calibration.POINT_COLUMNS)


def test_every_component_input_fills_whole_mla_words(load_script: Callable[[str], ModuleType]) -> None:
    # The MLA reads each input in whole 16-byte words, and the DevKit's Neat refuses an input that leaves one partly
    # filled. A multiple of 16 values fills them at INT8, and so at BF16 too.
    recipe = load_script("compile")
    for component in recipe.COMPONENT_GRAPHS.values():
        for shape in component["inputs"].values():
            assert math.prod(shape) % 16 == 0, shape


def test_samples_use_each_images_largest_object_in_a_seeded_order(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    images = [{"id": index, "file_name": f"{index:012d}.jpg"} for index in range(5)]
    annotations = [
        {"image_id": 0, "area": 10.0, "bbox": [1, 1, 2, 5], "iscrowd": 0},
        {"image_id": 0, "area": 50.0, "bbox": [3, 4, 10, 5], "iscrowd": 0},
        {"image_id": 0, "area": 90.0, "bbox": [0, 0, 9, 10], "iscrowd": 1},
        *({"image_id": index, "area": 5.0, "bbox": [0, 0, 1, 5], "iscrowd": 0} for index in range(1, 4)),
    ]
    files = {f"coco/cal2017/{image['file_name']}": _png(40, 30) for image in images}
    files["coco/annotations/instances_cal2017.json"] = json.dumps(
        {"images": images, "annotations": annotations}
    ).encode()
    archive = _write_archive(tmp_path / "cal.tgz", files)

    boxes = calibration.largest_boxes(archive)
    assert boxes["cal2017/000000000000.jpg"] == [3, 4, 10, 5]  # the crowd region is not an object
    assert "cal2017/000000000004.jpg" not in boxes  # no annotated object, so no prompt
    expected = sorted(boxes)
    random.Random(42).shuffle(expected)
    assert [name for name, _ in calibration.select_samples(archive, 42, 1, 2)] == expected[1:3]
    with pytest.raises(ValueError, match="needs 5 annotated images"):
        calibration.select_samples(archive, 42, 2, 3)
    ((image, prompt),) = calibration.load_samples(archive, [("cal2017/000000000000.jpg", [3, 4, 10, 5])])
    assert (image.shape, image.dtype, float(image.max()) <= 1.0) == ((3, 1024, 1024), np.float32, True)
    assert prompt["points"][0, :2, 2:7].argmax(-1).tolist() == [3, 4]


def test_positional_encoding_is_sine_and_cosine_of_the_projected_point(
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    gaussian = np.asarray([[0.25, 0.0], [0.0, 0.5]], np.float32)
    # c = 0.5 projects to 0: sine 0 and cosine 1. c = 1 projects to 2*pi*G's rows.
    encoding = recipe.positional_encoding(np.asarray([[0.5, 0.5], [1.0, 0.5]], np.float32), gaussian)

    np.testing.assert_allclose(encoding[0], [0, 0, 1, 1], atol=1e-7)
    np.testing.assert_allclose(encoding[1], [1, 0, 0, 1], atol=1e-6)


def test_the_encoder_compiles_without_afes_simulator_check(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    class Config:
        def __init__(self, **options: object) -> None:
            self.options = options

    operations = types.ModuleType("afe.backends.mla.afe_to_n2a_compiler.n2a_compiler_operations")
    networks = types.ModuleType("afe.core.compile_networks")
    operations.MLACompilerConfig = networks.MLACompilerConfig = Config
    monkeypatch.setitem(sys.modules, operations.__name__, operations)
    monkeypatch.setitem(sys.modules, networks.__name__, networks)
    monkeypatch.setattr(sys.modules["afe.core"], "compile_networks", networks, raising=False)

    recipe._skip_simulator_check()
    assert networks.MLACompilerConfig(target="mla").options == {"verify_mlc_files": False, "target": "mla"}
    # A compiler that builds its MLA configuration elsewhere would silently keep the check, so the recipe refuses it.
    networks.MLACompilerConfig = object
    with pytest.raises(RuntimeError, match="revisit the simulator check"):
        recipe._skip_simulator_check()


@pytest.mark.parametrize("precision", ["w8a8", "w16a16"])
def test_the_quantization_recipe_follows_the_precision(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    for name in ("quantization_scheme", "bfloat16_scheme", "HistogramMSEMethod", "default_quantization"):
        monkeypatch.setattr(recipe, name, MagicMock(name=name))
    recipe._quantization_config({"precision": precision, "calibration_bins": 2048})
    if precision == "w8a8":
        assert recipe.quantization_scheme.call_args_list == [
            call(asymmetric=True, per_channel=False, bits=8),
            call(asymmetric=False, per_channel=True, bits=8),
        ]
        recipe.bfloat16_scheme.assert_not_called()
    else:
        recipe.quantization_scheme.assert_not_called()
        recipe.bfloat16_scheme.assert_called_once_with()
    recipe.HistogramMSEMethod.assert_called_once_with(2048)
    with pytest.raises(ValueError, match="unsupported precision"):
        recipe._quantization_config({"precision": "w4a8", "calibration_bins": 2048})


def test_every_build_profile_calibrates_before_the_fidelity_images(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    metadata = recipe.load_metadata(recipe.MODEL_DIR)
    selection = metadata["numerical_fidelity"]["selection"]

    assert metadata["numerical_fidelity"]["outputs"] == ["image_embeddings", "iou_predictions", "low_res_masks"]
    for profile in metadata["build_profiles"]:
        parameters = recipe._component_parameters(metadata, profile["id"])
        assert list(parameters) == list(recipe.COMPONENT_IDS)
        assert recipe._precision_label(parameters) == profile["id"].upper()
        assert int(parameters["image_encoder"]["calibration_samples"]) <= selection["offset"]
    with pytest.raises(ValueError, match="no build profile 'w32a32'"):
        recipe._component_parameters(metadata, "w32a32")
    with pytest.raises(ValueError, match="one precision"):
        recipe._precision_label({"image_encoder": {"precision": "w8a8"}, "prompt_decoder": {"precision": "w16a16"}})


def test_audit_requires_one_mla_plugin_no_cpu_plugin_and_the_inputs(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    plugins = ["EV74", "MLA", "EV74"]
    inputs = {"image_embeddings": 256 * 64 * 64 * 4, "points": 6 * 8 * 4}
    assert recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", plugins, inputs), "prompt_decoder")["mla_plugins"] == 1
    for name, arguments, message in (
        ("cpu", ([*plugins, "A65"], inputs), "host=1"),
        ("input", (plugins, {**inputs, "points": 4}), "input interface"),
    ):
        with pytest.raises(RuntimeError, match=message):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", *arguments), "prompt_decoder")


@pytest.mark.parametrize(
    ("model", "arguments", "message"),
    [
        (
            "efficientsam_ti_encoder.onnx",
            ["--build-profile", "w16a16", "--compile-invocation", "w8a8"],
            "named after it",
        ),
        (
            "model.onnx",
            ["--build-profile", "w16a16", "--compile-invocation", "w16a16"],
            "the pinned efficientsam_ti_encoder",
        ),
        (
            "efficientsam_ti_encoder.onnx",
            ["--build-profile", "w16a16", "--compile-invocation", "w16a16"],
            "must sit beside",
        ),
    ],
)
def test_compile_main_rejects_an_incomplete_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    model: str,
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    (tmp_path / model).write_bytes(b"onnx")
    base = ["compile.py", "--model", str(tmp_path / model), "--dataset", str(tmp_path), "--fidelity-dataset",
            str(tmp_path), "--output-dir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", base + arguments)

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_build_holds_both_components_and_follows_the_onnx(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    outputs = json.loads((compiled / "compile-outputs.json").read_text(encoding="utf-8"))
    assert [item["component_id"] for item in outputs["outputs"]] == list(recipe.COMPONENT_IDS)
    for item in outputs["outputs"]:
        recipe.audit_mpk(compiled / item["mpk_path"], item["component_id"])
    fidelity = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in fidelity["metrics"]] == [
        "image_embeddings",
        "iou_predictions",
        "low_res_masks",
    ]
    # BF16 keeps every output close to the ONNX, and INT8 may drift but not collapse. A broken operator or a collapsed
    # INT8 build shows up as a low cosine, and a failed test leaves the build unpublished.
    floor = 0.99 if build_profile == "w16a16" else 0.95
    assert all(metric["cosine_similarity"] > floor for metric in fidelity["metrics"]), fidelity["metrics"]
