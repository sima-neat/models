"""SESR's collapse surgery, calibration crops, tiling and scoring, and the compiled package contract."""

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
from model_registry.model import load_model
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def test_model_declares_the_profiles_its_recipe_builds(
    model_dir: Path, load_script: Callable[[str], ModuleType]
) -> None:
    sesr = load_script("sesr")
    model = load_model(model_dir)
    assert model_dir.name == f"sesr_m{sesr.NUM_BLOCKS}_{sesr.SCALE}x"
    assert model["architecture"]["id"] == "sesr"
    for profile in model["build_profiles"]:
        (component,) = profile["components"]
        assert component["parameters"]["precision"] in {"w16a16", "w8a8", "w8a16"}


def _block_graph(residual: bool, rng: np.random.Generator) -> onnx.ModelProto:
    """One collapsible linear block: 3x3 expand without bias, 1x1 squeeze, optional skip, then ReLU."""
    channels, wide = 4, 32
    expand = rng.normal(0, 0.3, (wide, channels, 3, 3)).astype(np.float32)
    squeeze = rng.normal(0, 0.3, (channels, wide, 1, 1)).astype(np.float32)
    bias = rng.normal(0, 0.1, channels).astype(np.float32)
    nodes = [
        helper.make_node("Conv", ["x", "expand"], ["wide"], pads=[1, 1, 1, 1]),
        helper.make_node("Conv", ["wide", "squeeze", "bias"], ["narrow"]),
    ]
    last = "narrow"
    if residual:
        nodes.append(helper.make_node("Add", ["x", "narrow"], ["summed"]))
        last = "summed"
    nodes.append(helper.make_node("Relu", [last], ["y"]))
    graph = helper.make_graph(
        nodes,
        "block",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, channels, 9, 9])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, channels, 9, 9])],
        [numpy_helper.from_array(value, name) for name, value in (("expand", expand), ("squeeze", squeeze),
                                                                  ("bias", bias))],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


@pytest.mark.parametrize("residual", [False, True])
def test_collapse_fuses_each_block_into_one_exact_convolution(
    residual: bool, load_script: Callable[[str], ModuleType]
) -> None:
    surgery = load_script("surgery")
    rng = np.random.default_rng(7)
    original = _block_graph(residual, rng)
    sample = rng.normal(0, 1, (1, 4, 9, 9)).astype(np.float32)
    (expected,) = ReferenceEvaluator(original).run(None, {"x": sample})

    collapsed, fused = surgery.collapse(onnx.load_from_string(original.SerializeToString()))

    assert fused == 1
    assert [node.op_type for node in collapsed.graph.node] == ["Conv", "Relu"]
    (actual,) = ReferenceEvaluator(collapsed).run(None, {"x": sample})
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_collapse_leaves_a_lone_convolution_alone(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    weight = numpy_helper.from_array(np.ones((2, 2, 1, 1), np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"])],
        "lone",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2, 4, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 4, 4])],
        [weight],
    )
    model, fused = surgery.collapse(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
    assert fused == 0 and [node.op_type for node in model.graph.node] == ["Conv"]


def test_calibration_takes_one_seeded_crop_per_image_from_the_tarball(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    cv2 = pytest.importorskip("cv2")
    recipe = load_script("compile")
    archive = tmp_path / "div2k.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for index in range(3):
            image = np.full((300, 260, 3), (10 * index, 100, 200), np.uint8)  # BGR
            content = cv2.imencode(".png", image)[1].tobytes()
            member = tarfile.TarInfo(f"./{recipe.CALIBRATION_IMAGES}/{index:04d}x2.png")
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))

    tiles = recipe.calibration_tiles(archive, 2)

    assert [tile.shape for tile in tiles] == [(224, 224, 3)] * 2
    assert all(tile.dtype == np.float32 for tile in tiles)
    # The crops are RGB in [0, 1].
    np.testing.assert_allclose(tiles[1][0, 0], np.array([200, 100, 10]) / 255, rtol=1e-6)
    with pytest.raises(ValueError, match="at least 4"):
        recipe.calibration_tiles(archive, 4)


@pytest.mark.parametrize("length", [192, 300, 384, 678, 1020])
def test_tile_centres_cover_every_pixel(length: int, load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    step = accuracy.TILE - 2 * accuracy.MARGIN
    covered = np.zeros(length, bool)
    for start in accuracy.tile_starts(length):
        assert start + accuracy.TILE <= length + 2 * accuracy.MARGIN
        covered[start : start + step] = True
    assert covered.all()


def test_upscale_stitches_tile_centres_in_place(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (250, 410, 3), dtype=np.uint8)

    def nearest(tile: np.ndarray) -> np.ndarray:
        # A stand-in model that upscales by repeating pixels, in the CHW layout the ONNX returns.
        return tile.repeat(accuracy.SCALE, 0).repeat(accuracy.SCALE, 1).transpose(2, 0, 1)[None]

    upscaled = accuracy.upscale(image, nearest)

    np.testing.assert_array_equal(upscaled, image.repeat(2, 0).repeat(2, 1))


def test_tile_output_accepts_either_layout_and_rejects_others(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    hwc = np.random.default_rng(0).random((448, 448, 3), np.float32)
    np.testing.assert_array_equal(accuracy.tile_output(hwc.transpose(2, 0, 1)[None]), hwc)
    np.testing.assert_array_equal(accuracy.tile_output(hwc[None]), hwc)
    with pytest.raises(ValueError, match="shape"):
        accuracy.tile_output(np.zeros((1, 3, 224, 224), np.float32))
    hwc[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.tile_output(hwc)


def test_luma_follows_bt601(load_script: Callable[[str], ModuleType]) -> None:
    accuracy = load_script("accuracy")
    colours = np.array([[[0, 0, 0], [255, 255, 255], [255, 0, 0]]], np.uint8)
    np.testing.assert_allclose(accuracy.luma(colours), [[16, 235, 16 + 65.481]], atol=1e-9)


def test_scores_reward_exact_matches_and_penalize_noise(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(1)
    # A smooth gradient, so the added noise destroys local structure that SSIM measures.
    rows, columns = np.indices((64, 80))
    original = np.stack([rows * 3, columns * 3, rows + columns], axis=-1).astype(np.uint8)
    noisy = np.clip(original.astype(int) + rng.integers(-20, 21, original.shape), 0, 255).astype(np.uint8)

    exact = accuracy.score(original, original)
    degraded = accuracy.score(noisy, original)

    assert exact["psnr_y"] == float("inf") and exact["ssim_y"] == pytest.approx(1.0)
    assert 20 < degraded["psnr_y"] < 40 and degraded["ssim_y"] < 0.95
    with pytest.raises(ValueError, match="does not match"):
        accuracy.score(original[1:], original)


def test_ssim_matches_scikit_image_gaussian_ssim(load_script: Callable[[str], ModuleType]) -> None:
    pytest.importorskip("cv2")
    metrics = pytest.importorskip("skimage.metrics")
    accuracy = load_script("accuracy")
    rng = np.random.default_rng(2)
    truth = rng.uniform(16, 235, (60, 70))
    predicted = np.clip(truth + rng.normal(0, 8, truth.shape), 16, 235)

    expected = metrics.structural_similarity(
        predicted, truth, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255
    )

    # scikit-image averages over every pixel with reflected borders; this keeps only full windows.
    assert accuracy.ssim(predicted, truth) == pytest.approx(expected, abs=0.02)


def _mpk(path: Path, manifest: dict) -> Path:
    content = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in (("model_mpk.json", content), ("model.elf", b"\0")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_audit_requires_one_mla_segment_and_the_float_interface(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    good = {
        "input_nodes": [{"name": "lr_tile", "size": 224 * 224 * 3 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74"},
            {"sequence": 2, "processor": "MLA"},
            {"sequence": 3, "processor": "EV74", "output_nodes": [{"size": 448 * 448 * 3 * 4}]},
        ],
    }
    recipe.audit_mpk(_mpk(tmp_path / "good.tar.gz", good))
    for name, change in [
        ("two", {"plugins": [*good["plugins"][:2], {"sequence": 3, "processor": "MLA"}]}),
        ("host", {"plugins": [*good["plugins"], {"sequence": 4, "processor": "A65"}]}),
        ("input", {"input_nodes": [{"name": "lr_tile", "size": 1}]}),
    ]:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", {**good, **change}))


def test_compiled_package_keeps_computation_on_one_mla_segment(
    compiled: Path, model_dir: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)
