"""Small checks for conversion failures that a synthetic smoke input can miss."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/depth_anything_3_small_504x504"
SPEC = importlib.util.spec_from_file_location(f"{MODEL_DIR.name}_convert", MODEL_DIR / "scripts/convert.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _outputs():
    shape = converter.INTERFACE["depth"]
    depth = np.random.RandomState(0).uniform(0.5, 5, shape).astype(np.float32)
    confidence = np.random.RandomState(1).uniform(1, 3, shape).astype(np.float32)
    pose = np.random.RandomState(2).uniform(-1, 1, converter.INTERFACE["pose_enc"]).astype(np.float32)
    return [depth, confidence, pose]


@pytest.mark.parametrize("case", ["valid", "swapped", "shape", "nan"])
def test_output_agreement(case):
    expected, actual = _outputs(), _outputs()
    if case == "swapped":
        actual[:2] = actual[1::-1]
        with pytest.raises(AssertionError, match="depth"):
            converter.compare_outputs(expected, actual)
        return
    if case == "shape":
        actual[1] = actual[1][:, :, :-1]
        match = "Unexpected shape for depth_conf"
    elif case == "nan":
        actual[0][0, 0, 3, 4] = np.nan
        match = "Non-finite output: depth"
    else:
        converter.compare_outputs(expected, [value + 1e-4 for value in actual])
        return
    with pytest.raises(ValueError, match=match):
        converter.compare_outputs(expected, actual)


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape", "unfolded"])
def test_fixed_onnx_interface(tmp_path, case):
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in converter.INTERFACE.items()
    ]
    if case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[2].type.tensor_type.shape.dim[2].dim_value = 378
    inputs, outputs = values[:2], values[2:]
    if case == "order":
        inputs.reverse()
    nodes = [
        helper.make_node("Concat", ["image0", "image1"], ["views"], axis=1),
        helper.make_node("Slice", ["views", "starts", "ends", "axes"], ["depth"]),
        helper.make_node("Exp", ["depth"], ["depth_conf"]),
        helper.make_node("Slice", ["depth", "pose_starts", "pose_ends", "pose_axes"], ["pose_planes"]),
        helper.make_node("Reshape", ["pose_planes", "pose_shape"], ["pose_enc"]),
    ]
    if case == "unfolded":
        nodes.append(helper.make_node("Shape", ["image0"], ["unused"]))
    initializers = [
        onnx.numpy_helper.from_array(np.array(value, np.int64), name)
        for name, value in (
            ("starts", [0]),
            ("ends", [2]),
            ("axes", [1]),
            ("pose_starts", [0, 0]),
            ("pose_ends", [3, 3]),
            ("pose_axes", [2, 3]),
            ("pose_shape", [1, 2, 9]),
        )
    ]
    graph = helper.make_graph(nodes, "fixture", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_pinned_file_set():
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    source = model["conversion"]["default"]["source"]
    converter.validate_source(source)
    with pytest.raises(ValueError, match="model.safetensors"):
        converter.validate_source({**source, "files": source["files"][:2]})


def test_wheel_unpacks_only_the_package_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    wheel = tmp_path / converter.WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("depth_anything_3/cfg.py", "")
        archive.writestr("depth_anything_3-0.1.1.dist-info/METADATA", "")
        archive.writestr("other/module.py", "")

    converter.unpack_package(wheel, tmp_path / "unpacked")

    unpacked = tmp_path / "unpacked"
    assert sorted(path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*")) == [
        "depth_anything_3",
        "depth_anything_3/cfg.py",
    ]
    assert sys.path[0] == str(tmp_path / "unpacked")


@pytest.mark.parametrize("member", ["depth_anything_3/../escape.py", None])
def test_wheel_without_a_clean_package_tree_is_rejected(tmp_path, monkeypatch, member):
    monkeypatch.setattr(sys, "path", list(sys.path))
    wheel = tmp_path / converter.WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("other/module.py", "")
        if member:
            archive.writestr(member, "")

    with pytest.raises(ValueError, match="Unexpected contents"):
        converter.unpack_package(wheel, tmp_path / "unpacked")


def test_synthetic_inputs_are_normalized_nchw():
    samples = converter.synthetic_inputs()

    assert len(samples) == 4
    for pair in samples:
        assert len(pair) == converter.VIEWS
        for view in pair:
            assert view.shape == tuple(converter.INTERFACE["image0"]) and view.dtype == np.float32
            assert view.flags.c_contiguous
    assert not np.array_equal(*samples[0])
    # The gradient's top-left pixel is black before normalization, and its second view is the first shifted.
    reference, shifted = samples[-1]
    np.testing.assert_allclose(reference[0, :, 0, 0], -converter.MEAN / converter.STD, rtol=1e-6)
    np.testing.assert_array_equal(reference[..., converter.PAIR_SHIFT :], shifted[..., : -converter.PAIR_SHIFT])


def test_exported_interface_matches_the_component_declaration():
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    entrypoint = model["components"][0]["interface"]["entrypoints"][0]
    declared = {item["name"]: item["shape"] for item in entrypoint["inputs"] + entrypoint["outputs"]}

    assert declared == converter.INTERFACE
    assert tuple(item["name"] for item in entrypoint["outputs"]) == converter.OUTPUTS
