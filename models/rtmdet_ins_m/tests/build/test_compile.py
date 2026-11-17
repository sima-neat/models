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

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "rtmdet_ins_m"


@pytest.mark.parametrize(
    ("size", "expected"),
    [((640, 480), (640, 480, 1.0, 1.0)), ((500, 375), (640, 480, 1.28, 1.28)), ((427, 640), (427, 640, 1.0, 1.0)),
     ((333, 500), (426, 640, 426 / 333, 1.28))],
)
def test_letterbox_geometry_rounds_each_side_half_up(
    load_script: Callable[[str], ModuleType], size: tuple[int, int], expected: tuple
) -> None:
    resized_width, resized_height, scale_x, scale_y = load_script("calibration").letterbox_geometry(*size)
    assert (resized_width, resized_height) == expected[:2]
    np.testing.assert_allclose((scale_x, scale_y), expected[2:])


def test_letterbox_pads_the_right_and_bottom_with_114(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    image = np.full((240, 320, 3), 7, np.uint8)

    boxed = calibration.letterbox(image)

    assert boxed.shape == (640, 640, 3)
    assert (boxed[:480, :] == 7).all() and (boxed[480:, :] == 114).all()


def test_preprocessing_normalizes_rgb_with_imagenet_constants(
    monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    image = np.zeros((640, 640, 3), np.uint8)
    image[..., 0] = 255
    monkeypatch.setattr(calibration, "decode_image", lambda content: image)

    tensor = calibration.preprocess_image(b"")

    assert tensor.dtype == np.float32 and tensor.shape == (640, 640, 3)
    np.testing.assert_allclose(tensor[0, 0], (np.array([255, 0, 0]) - calibration.MEAN) / calibration.STD, rtol=1e-6)


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"coco/{split}/{index:012d}.jpg")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4 and all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _image_graph(surgery: ModuleType, shared_weight: bool = False) -> onnx.ModelProto:
    """The image -> one strided convolution producing cls_score_0, with every other output a constant."""
    helper, rng = onnx.helper, np.random.RandomState(0)
    weight = onnx.numpy_helper.from_array(rng.randn(80, 3, 8, 8).astype(np.float32), "weight")
    nodes = [helper.make_node("Conv", ["input", "weight"], ["cls_score_0"], kernel_shape=[8, 8], strides=[8, 8])]
    initializers = [weight]
    for name, shape in surgery.OUTPUT_SHAPES.items():
        if name == "cls_score_0":
            continue
        initializers.append(onnx.numpy_helper.from_array(np.asarray(shape, np.int64), f"{name}_shape"))
        nodes.append(helper.make_node(
            "ConstantOfShape",
            [f"{name}_shape"],
            [name],
            value=helper.make_tensor("zero", onnx.TensorProto.FLOAT, [1], [0.0]),
        ))
    if shared_weight:
        nodes.append(helper.make_node("Identity", ["weight"], ["weight_copy"]))
    outputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.OUTPUT_SHAPES.items()
    ]
    if shared_weight:
        outputs.append(helper.make_tensor_value_info("weight_copy", onnx.TensorProto.FLOAT, [80, 3, 8, 8]))
    inputs = [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, surgery.INPUT_SHAPE)]
    graph = helper.make_graph(nodes, "fixture", inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def test_rgb_input_reorders_the_first_convolution_exactly(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    source = tmp_path / "model.onnx"
    onnx.save(_image_graph(surgery), source)

    graph = surgery.prepare_model(source, tmp_path / "graphs")

    image = np.random.RandomState(1).randn(3, 640, 640).astype(np.float32)
    surgery.validate_rgb_input(source, graph, image)
    original = onnx.numpy_helper.to_array(onnx.load(source).graph.initializer[0])
    prepared = {value.name: value for value in onnx.load(graph).graph.initializer}["weight"]
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(prepared), original[:, ::-1])


def test_rgb_input_rejects_a_shared_first_convolution_weight(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    model = _image_graph(surgery, shared_weight=True)

    with pytest.raises(ValueError, match="output interface|private"):
        surgery.take_rgb_input(model)


def _mpk_manifest(compile_module: ModuleType, kernel: str = "pass_through") -> dict:
    return {
        "input_nodes": [{"name": "input", "size": 1 * 3 * 640 * 640 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": kernel},
                "output_nodes": [{"size": size} for size in compile_module.OUTPUT_BYTES],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize(
    "case", ["valid", "bf16_cast", "int8_dequantization", "reordered", "two_mla", "host", "unknown_kernel"]
)
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    kernel = {"bf16_cast": "cast_transform", "int8_dequantization": "dequantization_transform",
              "unknown_kernel": "slice_transform"}.get(case, "pass_through")
    manifest = _mpk_manifest(compile_module, kernel)
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

    if case in {"valid", "bf16_cast", "int8_dequantization"}:
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a16"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    assert load_script("compile")._parameters(profile_id) == {
        "precision": profile_id, "calibration_bins": 2048, "calibration_samples": 50,
    }


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w32a32")


@pytest.mark.parametrize(("precision", "per_token"), [("w8a8", False), ("w8a16", True), ("w16a16", False)])
def test_every_precision_calibrates_with_histogram_mse(
    load_script: Callable[[str], ModuleType], precision: str, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048)

    arguments = recipe.QuantizationParams.call_args.kwargs
    assert arguments["calibration_method"] is recipe.HistogramMSEMethod.return_value
    assert arguments.get("per_token_quant", False) is per_token
    recipe.HistogramMSEMethod.assert_called_once_with(2048)


def test_compile_requires_a_calibration_dataset(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="requires --dataset"):
        load_script("compile").compile_model(tmp_path / "model.onnx", tmp_path / "out", "w16a16", None)


def test_model_declares_its_license_source_and_reference() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert model["license"]["spdx"] == "Apache-2.0" and model["license"]["redistribution"] == "allowed"
    assert "CC BY 4.0" in model["license"]["attribution"]
    source = model["conversion"]["default"]["source"]
    assert source["type"] == "file" and source["sha256"].startswith("6eba602e")
    assert list(model["accuracy"]["datasets"]) == ["coco-2017-val"]
    assert model["reference_metrics"][0]["metrics"] == {"coco_mask_map": 0.421, "coco_box_map": 0.488}
