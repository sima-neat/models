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
MODEL_DIR = ROOT / "models/depth_anything_3_metric_large_280x504"
SPEC = importlib.util.spec_from_file_location(f"{MODEL_DIR.name}_convert", MODEL_DIR / "scripts/convert.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _outputs():
    shape = converter.INTERFACE["depth"]
    depth = np.random.RandomState(0).uniform(0.5, 5, shape).astype(np.float32)
    sky = np.random.RandomState(1).uniform(0, 1, shape).astype(np.float32)
    return [depth, sky]


@pytest.mark.parametrize("case", ["valid", "swapped", "shape", "nan"])
def test_output_agreement(case):
    expected, actual = _outputs(), _outputs()
    if case == "swapped":
        actual.reverse()
        with pytest.raises(AssertionError, match="depth"):
            converter.compare_outputs(expected, actual)
        return
    if case == "shape":
        actual[1] = actual[1][:, :, :-1]
        match = "Unexpected shape for sky"
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
    inputs, outputs = values[:1], values[1:]
    if case == "order":
        outputs.reverse()
    nodes = [
        helper.make_node("Slice", ["image", "starts", "ends", "axes"], ["depth"]),
        helper.make_node("Sigmoid", ["depth"], ["sky"]),
    ]
    if case == "unfolded":
        nodes.append(helper.make_node("Shape", ["image"], ["unused"]))
    initializers = [
        onnx.numpy_helper.from_array(np.array(value, np.int64), name)
        for name, value in (("starts", [0]), ("ends", [1]), ("axes", [1]))
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
    for sample in samples:
        assert sample.shape == tuple(converter.INTERFACE["image"]) and sample.dtype == np.float32
        assert sample.flags.c_contiguous
    # The gradient's top-left pixel is black before normalization.
    np.testing.assert_allclose(samples[-1][0, :, 0, 0], -converter.MEAN / converter.STD, rtol=1e-6)


def test_exported_interface_matches_the_component_declaration():
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    entrypoint = model["components"][0]["interface"]["entrypoints"][0]
    declared = {item["name"]: item["shape"] for item in entrypoint["inputs"] + entrypoint["outputs"]}

    assert declared == converter.INTERFACE
    assert tuple(item["name"] for item in entrypoint["outputs"]) == converter.OUTPUTS
