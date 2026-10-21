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
from model_registry.model_tests import load_model_tests

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "yolo26m_pose"


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
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


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
        "model_sdk_version": compile_module.COMPILER_VERSION,
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


@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host", "compiler"])
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
    elif case == "compiler":
        manifest["model_sdk_version"] = "0.0.0"
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    assert compile_module._parameters("w16a16") == {"precision": "w16a16"}
    assert compile_module._parameters("w8a8") == {
        "precision": "w8a8",
        "calibration_bins": 2048,
        "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w32a32")


def test_model_preserves_compilation_and_quality_settings(load_script: Callable[[str], ModuleType]) -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert {profile["id"] for profile in model["build_profiles"]} == {"w16a16", "w8a8"}
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    entry = model["accuracy"][0]
    for expected in load_model_tests(MODEL_DIR)["accuracy"]["expected"].values():
        assert set(expected) == set(load_script("accuracy").METRICS)
    assert entry["required_metrics"] == ["coco_keypoint_map"]
    assert (entry["dataset"]["id"], entry["dataset"]["variant"]) == ("coco-2017", "val")
    assert model["calibrations"][0]["dataset"]["variant"] == "cal"
