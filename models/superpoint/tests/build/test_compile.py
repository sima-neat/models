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
    assert compile_module._parameters("w8a16") == {"precision": "w8a16"}
    assert compile_module._parameters("w8a8") == {
        "precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("w4a8")


@pytest.mark.parametrize(
    ("precision", "calibrates", "per_token"), [("w8a8", True, False), ("w8a16", False, True), ("w16a16", False, False)]
)
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool, per_token: bool
) -> None:
    compile_module = load_script("compile")

    compile_module._quantization(precision, 2048 if calibrates else None)

    arguments = compile_module.QuantizationParams.call_args.kwargs
    calibration = (
        compile_module.HistogramMSEMethod.return_value if calibrates else compile_module.SkipCalibration.return_value
    )
    assert arguments["calibration_method"] is calibration
    assert arguments.get("per_token_quant", False) is per_token
    if calibrates:
        compile_module.HistogramMSEMethod.assert_called_once_with(2048)


@pytest.mark.parametrize("precision", ["w16a16", "w8a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_script: Callable[[str], ModuleType],
    precision: str,
) -> None:
    compile_module = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(compile_module, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(compile_module, "validate_source", lambda source: None)
    monkeypatch.setattr(compile_module, "check_quantized_keypoints", lambda source, quantized: (1.0, 1.0))
    monkeypatch.setattr(
        compile_module, "calibration_images", lambda dataset, count: np.zeros((count, 480, 640, 1), np.float32)
    )

    with pytest.raises(RuntimeError, match="expected one MPK"):
        compile_module.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "hpatches.tgz")

    data = compile_module.load_model.return_value.quantize.call_args.kwargs["calibration_data"]
    if precision == "w8a8":
        assert data is compile_module.convert_data_generator_to_iterable.return_value
    else:
        assert data is None


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


def test_calibration_draws_seeded_images_of_the_cal_sequences(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_script("calibration")
    archive = tmp_path / "datasets__hpatches_cal.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for split, value in (("cal", 200), ("val", 0)):
            for sequence in ("i_a", "v_b"):
                for index in range(1, 4):
                    path = tmp_path / split / sequence / f"{index}.ppm"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(path), np.full((300, 400, 3), value, np.uint8))
                    tar.add(path, f"hpatches/{split}/{sequence}/{index}.ppm")
                (tmp_path / split / sequence / "H_1_2").write_text("1 0 0\n0 1 0\n0 0 1\n")
                tar.add(tmp_path / split / sequence / "H_1_2", f"hpatches/{split}/{sequence}/H_1_2")

    images = calibration.calibration_images(archive, 4)

    assert images.shape == (4, 480, 640, 1) and images.dtype == np.float32
    # Every selected image comes from a cal sequence: the val images are black.
    np.testing.assert_allclose(images, 200 / 255, rtol=1e-3)
    np.testing.assert_array_equal(images, calibration.calibration_images(archive, 4))
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_images(archive, 7)


def test_model_preserves_compilation_and_licence_settings() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w16a16", "w8a8", "w8a16"]
    assert model["license"]["redistribution"] == "restricted"
    assert model["license"]["spdx"] == "LicenseRef-SiMa-Internal"
