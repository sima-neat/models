from __future__ import annotations

import io
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "superpoint"


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
        manifest["input_nodes"][0]["size"] //= 3
    elif case == "decoder":
        plugins[-1]["config_params"]["kernel"] = "superpoint_decode"
    elif case == "outputs":
        plugins[-1]["output_nodes"] = plugins[-1]["output_nodes"][:1]
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
    """Stands in for the quantized model: returns given channels-last heads."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs

    def execute(self, inputs, fast_mode=False):
        return self.outputs


def _heads(compile_module: ModuleType, seed: int) -> list[np.ndarray]:
    """Channels-first heads with sharp keypoints in random cells and random descriptors."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(compile_module.OUTPUT_SHAPES["logits"], np.float32)
    logits[0, -1] = 10.0
    cells = rng.choice(60 * 80, 300, replace=False)
    logits[0, rng.integers(0, 64, 300), cells // 80, cells % 80] = 20.0
    descriptors = rng.normal(size=compile_module.OUTPUT_SHAPES["descriptors"]).astype(np.float32)
    return [logits, descriptors]


@pytest.mark.parametrize("case", ["exact", "nan", "moved", "descriptors"])
def test_quantized_keypoints_must_follow_fp32(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    compile_module = load_script("compile")
    reference = _heads(compile_module, 0)
    logits, descriptors = (value.copy() for value in reference)
    if case == "nan":
        descriptors[0, 0, 5, 5] = np.nan
    elif case == "moved":
        logits = np.roll(logits, 3, axis=3)
    elif case == "descriptors":
        descriptors = -descriptors

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, outputs, feed):
            return reference

    monkeypatch.setitem(sys.modules, "onnxruntime", type("ort", (), {"InferenceSession": Session}))
    # The quantized model returns channels-last heads, descriptors first to check they are named by shape.
    quantized = _Quantized([descriptors.transpose(0, 2, 3, 1), logits.transpose(0, 2, 3, 1)])

    if case == "exact":
        assert compile_module.check_quantized_keypoints(tmp_path / "model.onnx", quantized) == (1.0, pytest.approx(1.0))
    else:
        with pytest.raises(RuntimeError):
            compile_module.check_quantized_keypoints(tmp_path / "model.onnx", quantized)


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    assert compile_module._parameters("w16a16") == {"precision": "w16a16"}
    assert compile_module._parameters("w8a8") == {
        "precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w4a8")


@pytest.mark.parametrize("case", ["valid", "shape"])
def test_compile_requires_the_converted_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    helper = onnx.helper
    image = helper.make_tensor_value_info(compile_module.INPUT, onnx.TensorProto.FLOAT, compile_module.INPUT_SHAPE)
    shapes = dict(compile_module.OUTPUT_SHAPES)
    if case == "shape":
        shapes["descriptors"] = (1, 256, 30, 40)
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in shapes.items()]
    nodes = [helper.make_node("Identity", [compile_module.INPUT], [name]) for name in shapes]
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(helper.make_graph(nodes, "fixture", [image], outputs)), path)

    if case == "valid":
        compile_module.validate_source(path)
    else:
        with pytest.raises(ValueError, match="Unsupported SuperPoint graph"):
            compile_module.validate_source(path)


def test_calibration_draws_seeded_cal2017_images(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    for split in ("cal2017", "val2017"):
        (tmp_path / split).mkdir()
        for index in range(6):
            (tmp_path / split / f"{index:012d}.jpg").write_bytes(b"jpeg")

    names = calibration.calibration_names(tmp_path, 4)

    assert names == calibration.calibration_names(tmp_path, 4)
    assert len(set(names)) == 4 and all(name.startswith("cal2017/") for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(tmp_path, 7)


def test_model_preserves_compilation_and_licence_settings() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16", "w8a8"]
    assert model["license"]["redistribution"] == "restricted"
    assert model["license"]["spdx"] == "LicenseRef-SiMa-Internal"
