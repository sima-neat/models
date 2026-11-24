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
from model_registry.dependencies import onnx_opset_version
from model_registry.errors import ValidationError
from model_registry.model import load_model as load_metadata
from model_registry.model_tests import load_model_tests

OPSET = onnx_opset_version()
ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "yolo26s_pose"


def test_letterbox_geometry_matches_ultralytics_rounding(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")

    # 500x375 scales up by 1.28 to 640x480 and pads 80 rows above and below.
    assert calibration.letterbox_geometry(500, 375) == (1.28, (640, 480), (0, 80, 0, 80))
    # An odd padding puts the extra row at the bottom.
    assert calibration.letterbox_geometry(640, 427) == (1.0, (640, 427), (0, 106, 0, 107))
    assert calibration.letterbox_geometry(1280, 960) == (0.5, (640, 480), (0, 80, 0, 80))


def test_letterbox_centres_the_image_on_114_padding(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    calibration = load_script("calibration")

    result = calibration.letterbox(np.full((427, 640, 3), 200, np.uint8))

    assert result.shape == (640, 640, 3)
    assert (result[:106] == 114).all() and (result[106:533] == 200).all() and (result[533:] == 114).all()


def test_preprocessing_returns_rgb_hwc_in_unit_range(load_script: Callable[[str], ModuleType]) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_script("calibration")
    blue = np.zeros((320, 320, 3), np.uint8)
    blue[..., 0] = 255
    _, content = cv2.imencode(".png", blue)

    result = calibration.preprocess_image(content.tobytes())

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[320, 320], [0, 0, 1])


def test_calibration_draws_a_seeded_sample_of_keypoint_labelled_cal2017_images(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    # Images 0-4 have a person with labelled keypoints; image 5's only person has none.
    keypoints = {
        "images": [{"id": index, "file_name": f"{index:012d}.jpg"} for index in range(6)],
        "annotations": [{"image_id": index, "num_keypoints": 0 if index == 5 else 3} for index in range(6)]
        + [{"image_id": 2, "num_keypoints": 0}],
    }
    files = {f"coco/{split}/{index:012d}.jpg": b"x" for split in ("cal2017", "val2017") for index in range(6)}
    files["coco/annotations/person_keypoints_cal2017.json"] = json.dumps(keypoints).encode()
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "cal2017" and Path(name).stem != f"{5:012d}" for name in names)
    with pytest.raises(ValueError, match="cannot select 6 calibration images from 5 keypoint-labelled"):
        calibration.calibration_names(archive, 6)


def test_calibration_requires_the_cal2017_keypoint_annotations(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    (tmp_path / "cal2017").mkdir()
    (tmp_path / "cal2017" / "000000000001.jpg").write_bytes(b"x")

    with pytest.raises(ValueError, match="person_keypoints_cal2017.json"):
        calibration.calibration_names(tmp_path, 1)


def _source_graph(load_script: Callable[[str], ModuleType], constant_weight: bool = False) -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names, plus an unrelated output0 the cut drops."""
    surgery = load_script("surgery")
    helper = onnx.helper
    nodes, initializers = [], []
    for (name, head), shape in zip(surgery.HEADS.items(), surgery.OUTPUT_SHAPES.values()):
        stride = 640 // shape[2]
        weight = np.zeros((shape[1], 3, stride, stride), np.float32)
        if constant_weight and name == "bbox_0":
            nodes.append(helper.make_node("Constant", [], [f"{name}_w"], value=onnx.numpy_helper.from_array(weight)))
        else:
            initializers.append(onnx.numpy_helper.from_array(weight, f"{name}_w"))
        nodes.append(helper.make_node("Conv", ["images", f"{name}_w"], [head], strides=[stride, stride]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 57), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 57])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def test_surgery_cuts_at_the_nine_raw_heads_in_order(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    expected = [f"{name}_{i}" for name in ("bbox", "class_logit", "keypoint") for i in range(3)]
    assert list(outputs) == surgery.OUTPUTS == expected
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["bbox_0"] == [1, 4, 80, 80] and outputs["class_logit_2"] == [1, 1, 20, 20]
    assert outputs["keypoint_1"] == [1, 51, 40, 40]
    assert surgery.HEADS["keypoint_2"] == "/model.23/one2one_cv4_kpts.2/Conv_output_0"
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
        "input_nodes": [{"name": "images", "size": 1 * 3 * 640 * 640 * 4}],
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
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[3] = outputs[3], outputs[0]
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
    for expected in load_model_tests(MODEL_DIR)["accuracy"]["expected"].values():
        assert set(expected["coco-2017-val"]) == set(load_script("accuracy").METRICS)
    assert list(entry["datasets"]) == ["coco-2017-val"]
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
    monkeypatch.setattr(recipe, "calibration_images", lambda dataset, count: np.zeros((count, 640, 640, 3), np.float32))
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "cut.onnx")

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "coco.tgz")

    data = recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert data is recipe.convert_data_generator_to_iterable.return_value
    else:
        assert data is None
