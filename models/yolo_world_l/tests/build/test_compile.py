from pathlib import Path

import numpy as np
import onnx
import pytest


def test_finite_clip_preserves_ordinary_normalization(load_script):
    surgery = load_script("surgery")
    h = onnx.helper
    graph = h.make_model(h.make_graph([
        h.make_node("Clip", ["norm", "eps", ""], ["bounded"], name="norm_clip")
    ], "normalization", [h.make_tensor_value_info("norm", onnx.TensorProto.FLOAT, [1])],
       [h.make_tensor_value_info("bounded", onnx.TensorProto.FLOAT, [1])],
       [onnx.numpy_helper.from_array(np.array(1e-12, np.float32), "eps")]),
       opset_imports=[h.make_opsetid("", 13)], ir_version=10)
    surgery.finite_clip_bounds(graph)
    bound = onnx.numpy_helper.to_array(graph.graph.initializer[-1])
    assert np.isfinite(bound) and bound > 1e30
    assert graph.graph.node[0].input[2] == "norm_clip/finite_max"
    for value in (0., 1e-15, .9, 1., 1.01):
        assert np.clip(value, 1e-12, bound) == max(value, 1e-12)
    onnx.checker.check_model(graph)


def test_calibration_keeps_image_prompt_pairing_and_sparse_masks(load_script):
    samples = load_script("calibration_samples").SAMPLES
    assert len(samples) == 128
    assert len({row["file_name"] for row in samples}) == 128
    assert {row["active_count"] for row in samples} == {1, 2, 3, 8, 16, 32, 80}
    for row in samples:
        assert len(row["sha256"]) == 64
        assert len(row["prompts"]) == 80
        assert 1 <= row["active_count"] <= 80


def test_supplied_calibration_bytes_are_verified(tmp_path, load_script, monkeypatch):
    calibration = load_script("calibration")
    path = tmp_path / "000000000001.jpg"
    path.write_bytes(b"wrong-image")
    monkeypatch.setattr(calibration, "SAMPLES", [{"file_name": path.name, "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="checksum"):
        calibration.verified_images(tmp_path)


def test_bf16_text_configuration_skips_calibration(load_script, monkeypatch):
    compiler = load_script("compile")
    sentinel = object()
    monkeypatch.setattr(compiler, "QuantizationParams", lambda **values: values)
    monkeypatch.setattr(compiler, "SkipCalibration", lambda: sentinel)
    config = compiler.quantization("text_encoder", {"precision": "w16a16", "context_tokens": 77})
    assert config["calibration_method"] is sentinel
    assert config["per_token_quant"] is False


def test_compiled_components_are_entirely_on_mla(compiled: Path, load_script):
    compiler = load_script("compile")
    manifests = {}
    for component in ("text_encoder", "detector"):
        (archive,) = (compiled / component).glob("*_mpk.tar.gz")
        manifests[component] = compiler.audit_mpk(archive)
    assert [row["name"] for row in manifests["text_encoder"]["input_nodes"]] == ["token_embeddings", "eos_selector"]
    assert [row["name"] for row in manifests["detector"]["input_nodes"]] == ["images", "text_embeddings", "prompt_mask"]
