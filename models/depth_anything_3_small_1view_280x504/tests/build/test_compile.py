"""Static input, quantization, projection equivalence and real compiled-package contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from model_registry.model import load_model


def test_model_uses_a_single_uncalibrated_w8a16_profile(model_dir):
    model = load_model(model_dir)
    assert [profile["id"] for profile in model["build_profiles"]] == ["w8a16"]
    assert not model["calibrations"]
    assert "calibration" not in model["build_profiles"][0]["components"][0]
    assert model["architecture"]["id"] == "depth_anything_3"


def test_preprocessing_preserves_rgb_and_applies_imagenet_normalization(load_script):
    cv2 = pytest.importorskip("cv2")
    preprocessing = load_script("preprocess")
    blue = np.zeros((480, 640, 3), dtype=np.uint8)
    blue[:, :, 0] = 255  # OpenCV's BGR input.
    _, encoded = cv2.imencode(".png", blue)
    image = preprocessing.preprocess_image(encoded.tobytes())
    assert image.shape == (preprocessing.HEIGHT, preprocessing.WIDTH, 3)
    assert image.dtype == np.float32 and image.flags.c_contiguous
    np.testing.assert_allclose(image[0, 0], (np.array([0, 0, 1]) - preprocessing.MEAN) / preprocessing.STD)


def test_w8a16_uses_sima_per_token_quantization_without_calibration(load_script, monkeypatch):
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "QuantizationParams", lambda **kwargs: kwargs)
    monkeypatch.setattr(recipe, "SkipCalibration", lambda: "skip")
    monkeypatch.setattr(recipe, "bfloat16_scheme", lambda: "bf16")
    monkeypatch.setattr(recipe, "quantization_scheme", lambda **kwargs: kwargs)
    parameters = recipe.quantization()
    assert parameters["calibration_method"] == "skip"
    assert parameters["activation_quantization_scheme"] == "bf16"
    assert parameters["weight_quantization_scheme"] == {"asymmetric": False, "per_channel": True, "bits": 8}
    assert parameters["requantization_mode"] is recipe.RequantizationMode.sima
    assert parameters["per_token_quant"] is True


def test_final_projection_split_preserves_both_learned_rows(load_script):
    torch = pytest.importorskip("torch")
    surgery = load_script("surgery")
    torch.manual_seed(123)
    projection = torch.nn.Conv2d(32, 2, 1)
    features = torch.randn(1, 32, 7, 9)
    expected = projection(features)
    head = torch.nn.Module()
    head.scratch = torch.nn.Module()
    head.scratch.output_conv2 = torch.nn.Sequential(projection)
    head.activation, head.conf_activation, head.down_ratio = "exp", "expp1", 1
    surgery.split_head_projections(head)
    actual = torch.cat([head.depth_projection(features), head.confidence_projection(features)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert isinstance(head.scratch.output_conv2[-1], torch.nn.Identity)


def test_explicit_attention_matches_scaled_dot_product_attention(load_script):
    torch = pytest.importorskip("torch")
    surgery = load_script("surgery")
    torch.manual_seed(123)
    attention = SimpleNamespace(
        training=False,
        num_heads=4,
        scale=8**-0.5,
        qkv=torch.nn.Linear(32, 96),
        proj=torch.nn.Linear(32, 32),
        q_norm=torch.nn.Identity(),
        k_norm=torch.nn.Identity(),
        rope=None,
        attn_drop=torch.nn.Identity(),
        proj_drop=torch.nn.Identity(),
    )
    image = torch.randn(1, 9, 32)
    packed = attention.qkv(image).reshape(1, 9, 3, 4, 8).permute(2, 0, 3, 1, 4)
    expected = attention.proj(
        torch.nn.functional.scaled_dot_product_attention(*packed).transpose(1, 2).reshape(1, 9, 32)
    )
    torch.testing.assert_close(surgery.attention_forward(attention, image), expected, rtol=1e-5, atol=1e-6)


def test_compiled_package_keeps_computation_on_one_mla_segment(compiled: Path, model_dir, build_profile, load_script):
    archive = compiled / "model" / f"{model_dir.name}_{build_profile}_mpk.tar.gz"
    load_script("compile").audit_mpk(archive)
