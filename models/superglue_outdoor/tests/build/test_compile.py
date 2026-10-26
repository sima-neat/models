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


@pytest.mark.parametrize("case", ["valid", "bf16_cast", "inputs", "outputs", "decoder", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path,
    case: str,
    load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "bf16_cast":
        plugins[-1]["config_params"]["kernel"] = "cast_transform"
    elif case == "inputs":
        manifest["input_nodes"] = manifest["input_nodes"][:-1]
    elif case == "decoder":
        plugins[-1]["config_params"]["kernel"] = "boxdecode"
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

    if case in ("valid", "bf16_cast"):
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


class _Quantized:
    """Stands in for the quantized model: returns given channels-last couplings."""

    def __init__(self, couplings: np.ndarray) -> None:
        self.couplings = couplings

    def execute(self, inputs, fast_mode=False):
        return [self.couplings]


@pytest.mark.parametrize("case", ["exact", "nan", "shuffled"])
def test_quantized_matches_must_follow_fp32(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_module = load_script("compile")
    size = compile_module.KEYPOINTS + 1
    _, shared, target = compile_module.representative_pair()
    # FP32 couplings [1, j, 1, i] in which each re-observed image-0 keypoint scores highest with its partner.
    reference = np.zeros((size, size), np.float32)
    reference[target, shared] = 5.0
    channels_last = reference.T.copy()
    if case == "nan":
        channels_last[0, 0] = np.nan
    elif case == "shuffled":
        channels_last[shared] = channels_last[np.roll(shared, 1)]

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, outputs, feed):
            return [reference.reshape(1, size, 1, size)]

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", type("ort", (), {"InferenceSession": Session}))
    quantized = _Quantized(channels_last.reshape(1, 1, size, size))

    if case == "exact":
        assert compile_module.check_quantized_matches(tmp_path / "model.onnx", quantized) == 1.0
    else:
        with pytest.raises(RuntimeError):
            compile_module.check_quantized_matches(tmp_path / "model.onnx", quantized)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a16"])
def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    assert load_script("compile")._parameters(profile_id) == {"precision": profile_id}


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    with pytest.raises(ValueError, match="no build profile"):
        load_script("compile")._parameters("w8a8")


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
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16", "w8a16"]
    assert model["license"]["redistribution"] == "restricted"
    assert model["license"]["spdx"] == "LicenseRef-SiMa-Internal"
