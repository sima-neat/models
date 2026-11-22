from __future__ import annotations

import io
import itertools
import json
import math
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.errors import ValidationError
from model_registry.model import load_model as load_metadata
from model_registry.model_tests import load_model_tests

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "yolo26l_obb"


def test_letterbox_geometry_matches_ultralytics_rounding(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    # An edge tile narrower than the input is padded, not scaled.
    assert calibration.letterbox_geometry(824, 1024) == (1.0, (824, 1024), (100, 0, 100, 0))
    # A whole image smaller than the input scales up.
    assert calibration.letterbox_geometry(800, 600) == (1.28, (1024, 768), (0, 128, 0, 128))
    # An odd padding puts the extra row at the bottom.
    assert calibration.letterbox_geometry(1024, 1023) == (1.0, (1024, 1023), (0, 0, 0, 1))


def _ultralytics_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Ultralytics' get_windows for one 1024-pixel crop with a 200-pixel gap, coverage filter included."""
    size, step = 1024, 824
    starts = []
    for side in (width, height):
        count = 1 if side <= size else math.ceil((side - size) / step + 1)
        values = [step * index for index in range(count)]
        if len(values) > 1 and values[-1] + size > side:
            values[-1] = side - size
        starts.append(values)
    windows = np.array([(x, y, x + size, y + size) for x, y in itertools.product(*starts)], dtype=np.float64)
    inside = windows.copy()
    inside[:, 0::2] = np.clip(inside[:, 0::2], 0, width)
    inside[:, 1::2] = np.clip(inside[:, 1::2], 0, height)
    rates = (inside[:, 2] - inside[:, 0]) * (inside[:, 3] - inside[:, 1]) / (size * size)
    if not (rates > 0.6).any():
        rates[abs(rates - rates.max()) < 0.01] = 1
    return [tuple(int(value) for value in window) for window in windows[rates > 0.6]]


def test_tile_windows_follow_the_ultralytics_dota_split(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    assert calibration.tile_windows(800, 600) == [(0, 0, 1024, 1024)]
    assert calibration.tile_windows(1147, 1023) == [(0, 0, 1024, 1024), (123, 0, 1147, 1024)]
    assert calibration.tile_windows(1490, 2137) == [
        (0, 0, 1024, 1024),
        (0, 824, 1024, 1848),
        (0, 1113, 1024, 2137),
        (466, 0, 1490, 1024),
        (466, 824, 1490, 1848),
        (466, 1113, 1490, 2137),
    ]
    # The coverage filter never drops a window, so tile_windows leaves it out.
    rng = np.random.default_rng(0)
    for width, height in [*rng.integers(200, 8000, (300, 2)), (1024, 1024), (1025, 300), (300, 1848)]:
        assert calibration.tile_windows(int(width), int(height)) == _ultralytics_windows(int(width), int(height))


LABELS = b"""imagesource:GoogleEarth
gsd:0.146
1100 100 1120 100 1120 120 1100 120 plane 0
1150 200 1170 200 1170 220 1150 220 ship 1
100 100 120 100 120 120 100 120 plane 0
"""


def test_label_centres_skip_the_header_lines(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    np.testing.assert_allclose(calibration.label_centres(LABELS), [[1110, 110], [1160, 210], [110, 110]])


def test_busiest_window_holds_the_most_object_centres(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    centres = calibration.label_centres(LABELS)

    assert calibration.busiest_window(2000, 1000, centres) == (824, 0, 1848, 1024)
    # Windows are half-open: a centre on the first window's right edge belongs to the next one.
    assert calibration.busiest_window(2000, 1000, np.array([[1024.0, 10.0]])) == (824, 0, 1848, 1024)
    # A tie goes to the first window.
    assert calibration.busiest_window(2000, 1000, np.array([[10.0, 10.0], [1900.0, 10.0]])) == (0, 0, 1024, 1024)


def _cal_archive(path: Path, images: dict[str, bytes], labels: dict[str, bytes], listing: list[str]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        files = {"dota/cal.txt": "\n".join(listing).encode() + b"\n"}
        files |= {f"dota/cal/images/{name}": content for name, content in images.items()}
        files |= {f"dota/cal/labelTxt/{name}": content for name, content in labels.items()}
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def test_calibration_names_are_the_first_listed_images(
    tmp_path: Path, load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "dota.tgz"
    _cal_archive(archive, {}, {}, ["P0002.jpg", "P0058.jpg", "P0066.jpg"])

    assert calibration.calibration_names(archive, 2) == ["P0002.jpg", "P0058.jpg"]
    with pytest.raises(ValueError, match="cannot select 4"):
        calibration.calibration_names(archive, 4)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no cal.txt"):
        calibration.calibration_names(empty, 1)


def test_calibration_tiles_crop_the_busiest_window(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_script("calibration")
    # The right window of a 2000x1000 image is red and holds two of the three objects.
    image = np.zeros((1000, 2000, 3), np.uint8)
    image[:, 1024:, 2] = 255
    _, content = cv2.imencode(".png", image)
    archive = tmp_path / "dota.tgz"
    _cal_archive(archive, {"P0001.png": content.tobytes()}, {"P0001.txt": LABELS}, ["P0001.png"])

    (tile,) = calibration.calibration_tiles(archive, 1)

    assert tile.shape == (1024, 1024, 3) and tile.dtype == np.float32
    # The 1024x1000 crop is padded by 12 rows above and turns red, in RGB order, from its column 200.
    np.testing.assert_array_equal(tile[11, 1000], [np.float32(114) / np.float32(255)] * 3)
    np.testing.assert_array_equal(tile[500, 1000], [1, 0, 0])
    np.testing.assert_array_equal(tile[500, 100], [0, 0, 0])


def _source_graph(load_script: Callable[[str], ModuleType], constant_weight: bool = False) -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names, plus an unrelated output0 the cut drops."""
    surgery = load_script("surgery")
    helper = onnx.helper
    nodes, initializers = [], []
    for (name, head), shape in zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values()):
        stride = 1024 // shape[2]
        weight = np.zeros((shape[1], 3, stride, stride), np.float32)
        if constant_weight and name == "bbox_0":
            nodes.append(helper.make_node("Constant", [], [f"{name}_w"], value=onnx.numpy_helper.from_array(weight)))
        else:
            initializers.append(onnx.numpy_helper.from_array(weight, f"{name}_w"))
        nodes.append(helper.make_node("Conv", ["images", f"{name}_w"], [head], strides=[stride, stride]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 7), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 1024, 1024])],
        [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 7])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_surgery_cuts_at_the_nine_raw_heads_in_order(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    expected = [f"{name}_{i}" for name in ("bbox", "class_logit", "angle") for i in range(3)]
    assert list(outputs) == surgery.OUTPUTS == expected
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["bbox_0"] == [1, 4, 128, 128] and outputs["class_logit_1"] == [1, 15, 64, 64]
    assert outputs["angle_2"] == [1, 1, 32, 32]
    assert surgery.HEADS["angle_2"] == "/model.23/one2one_cv4.2/one2one_cv4.2.2/Conv_output_0"
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 9


def test_surgery_requires_a_folded_source(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, constant_weight=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    sizes = compile_module.OUTPUT_BYTES
    return {
        "input_nodes": [{"name": "images", "size": 1 * 3 * 1024 * 1024 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": size} for size in sizes],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[3], outputs[6] = outputs[6], outputs[3]
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a16"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    expected = {"precision": profile_id}
    if profile_id == "w8a8":
        expected |= {"calibration_bins": 2048, "calibration_samples": 20}

    recipe = load_script("compile")

    assert recipe.component_parameters(recipe.MODEL_DIR, profile_id, recipe.COMPONENT_ID) == expected


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValidationError, match="unknown build profile: w32a32"):
        recipe.component_parameters(recipe.MODEL_DIR, "w32a32", recipe.COMPONENT_ID)


def test_model_preserves_compilation_and_quality_settings(load_script: Callable[[str], ModuleType]) -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    entry = model["accuracy"]
    for expected in load_model_tests(MODEL_DIR)["accuracy"].get("expected", {}).values():
        assert set(expected["dota-v1-0-val"]) == set(load_script("accuracy").METRICS)
    assert list(entry["datasets"]) == ["dota-v1-0-val"]
    assert model["calibrations"][0]["dataset"]["variant"] == "cal"


@pytest.mark.parametrize(
    ("precision", "calibrates", "per_token"), [("w8a8", True, False), ("w8a16", False, True), ("w16a16", False, False)]
)
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    assert arguments.get("per_token_quant", False) is per_token
    if calibrates:
        recipe.HistogramMSEMethod.assert_called_once_with(2048)


@pytest.mark.parametrize("precision", ["w16a16", "w8a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    precision: str,
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "component_parameters", lambda model_dir, profile_id, component_id: parameters)
    monkeypatch.setattr(
        recipe, "calibration_tiles", lambda dataset, count: np.zeros((count, 1024, 1024, 3), np.float32)
    )
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "cut.onnx")

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "dota.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert data is recipe.convert_data_generator_to_iterable.return_value
    else:
        assert data is None
