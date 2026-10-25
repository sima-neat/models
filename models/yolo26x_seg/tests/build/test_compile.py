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
MODEL_DIR = ROOT / "models" / "yolo26x_seg"


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


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"mini/{split}/{index:012d}.webp")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _source_graph(
    load_script: Callable[[str], ModuleType],
    constant_weight: bool = False,
    proto_op: str = 'Mul',
) -> onnx.ModelProto:
    """Nine strided Convs at the Ultralytics head names and a SiLU-closed prototype output1, plus an unrelated
    output0 the cut drops."""
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
    initializers.append(onnx.numpy_helper.from_array(np.zeros((32, 3, 4, 4), np.float32), "proto_w"))
    nodes.append(helper.make_node("Conv", ["images", "proto_w"], ["proto_conv"], strides=[4, 4]))
    if proto_op == "Mul":
        nodes.append(helper.make_node("Sigmoid", ["proto_conv"], ["proto_gate"]))
        nodes.append(helper.make_node("Mul", ["proto_conv", "proto_gate"], ["output1"]))
    else:
        nodes.append(helper.make_node(proto_op, ["proto_conv"], ["output1"]))
    nodes.append(
        helper.make_node(
            "Constant", [], ["output0"], value=onnx.numpy_helper.from_array(np.zeros((1, 300, 38), np.float32))
        )
    )
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info("output0", onnx.TensorProto.FLOAT, [1, 300, 38]),
            helper.make_tensor_value_info("output1", onnx.TensorProto.FLOAT, [1, 32, 160, 160]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])


def test_surgery_cuts_at_the_nine_raw_heads_and_the_prototypes_in_order(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script), source)

    cut = onnx.load(surgery.prepare_model(source, tmp_path / "out"))

    outputs = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in cut.graph.output}
    heads = [f"{name}_{level}" for name in ("bbox", "class_logit", "mask_coeff") for level in range(3)]
    assert list(outputs) == surgery.OUTPUTS == [*heads, "mask_proto"]
    assert outputs == surgery.OUTPUT_SHAPES
    assert outputs["mask_coeff_2"] == [1, 32, 20, 20] and outputs["mask_proto"] == [1, 32, 160, 160]
    assert [node.op_type for node in cut.graph.node] == ["Conv"] * 10 + ["Sigmoid", "Mul"]


def test_surgery_requires_a_folded_source(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, constant_weight=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_requires_the_prototype_silu(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(load_script, proto_op="Relu"), source)

    with pytest.raises(ValueError, match="no SiLU Mul produces output1"):
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
        "calibration_bins": 1024,
        "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w32a32")


@pytest.fixture
def accuracy_script(monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]) -> ModuleType:
    module = load_script("accuracy")
    # Uncompressed RLEs stand in for faster_coco_eval's compressed ones.
    monkeypatch.setattr(module.mask_util, "frPyObjects", lambda rles, height, width: rles)
    return module


def test_model_preserves_compilation_and_quality_settings(accuracy_script: ModuleType) -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert {profile["id"] for profile in model["build_profiles"]} == {"w16a16", "w8a8"}
    assert model["license"] == {
        "spdx": "AGPL-3.0-only",
        "redistribution": "restricted",
        "attribution": "YOLO26 is developed by Ultralytics and distributed under AGPL-3.0.",
    }
    declared = model["accuracy"]
    metric_names = {name for metrics in accuracy_script.METRICS.values() for name in metrics}
    for expected in load_model_tests(MODEL_DIR)["accuracy"]["expected"].values():
        assert set(expected["coco-2017-val"]) == metric_names
    assert list(declared["datasets"]) == ["coco-2017-val"]
    assert model["reference_metrics"][0]["metrics"] == {"coco_mask_map": 0.470, "coco_box_map": 0.565}
