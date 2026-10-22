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
MODEL_DIR = ROOT / "models" / "superglue_outdoor"


def _mpk_manifest(compile_module: ModuleType) -> dict:
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": name, "size": size} for name, size in compile_module.INPUT_BYTES.items()],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "quantize"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {
                "sequence": 2,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
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


@pytest.mark.parametrize("case", ["valid", "inputs", "outputs", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "inputs":
        manifest["input_nodes"] = manifest["input_nodes"][:-1]
    elif case == "outputs":
        plugins[-1]["output_nodes"][0]["size"] = 4
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
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w8a8")


def test_representative_inputs_are_channels_last_per_input(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    samples = compile_module.representative_inputs()

    for name, shape in compile_module.INPUT_SHAPES.items():
        batch, channels, height, width = shape
        assert samples[name].shape == (batch, height, width, channels)
        assert samples[name].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(samples["descriptors1"][0, 0], axis=1), 1, rtol=1e-5)


@pytest.mark.parametrize("case", ["valid", "shape"])
def test_compile_requires_the_converted_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    helper = onnx.helper
    inputs = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in compile_module.INPUT_SHAPES.items()
    ]
    (name, shape), = compile_module.OUTPUT_SHAPES.items()
    output_shape = [*shape[:-1], 100] if case == "shape" else shape
    output = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, output_shape)
    node = helper.make_node(
        "Constant",
        [],
        [name],
        value=helper.make_tensor("value", onnx.TensorProto.FLOAT, output_shape, np.zeros(int(np.prod(output_shape)))),
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(helper.make_graph([node], "fixture", inputs, [output])), path)

    if case == "valid":
        compile_module.validate_source(path)
    else:
        with pytest.raises(ValueError, match="Unsupported SuperGlue graph"):
            compile_module.validate_source(path)


def test_model_preserves_compilation_and_licence_settings() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16"]
    assert model["license"]["redistribution"] == "restricted"
    assert model["license"]["spdx"] == "LicenseRef-SiMa-Internal"
