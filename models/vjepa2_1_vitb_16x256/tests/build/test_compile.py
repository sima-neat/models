from __future__ import annotations

import io
import json
import random
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


def _jpeg(width: int, height: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (200, 100, 50)).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_frames_are_resized_centre_cropped_and_normalized(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    frame = np.zeros((300, 600, 3), np.uint8)
    # Only the centre 256 of the 512 resized columns is kept.
    frame[:, 150:450] = 255

    result = pipeline.preprocess_frame(frame)

    assert (result.shape, result.dtype) == ((256, 256, 3), np.float32)
    np.testing.assert_allclose(result[128, 128], (1 - pipeline.MEAN) / pipeline.STD, atol=1e-5)


def test_clips_stack_each_tubelet_down_the_height(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    # Frame f holds f everywhere, and colour c adds c / 10.
    frames = np.arange(16, dtype=np.float32)[:, None, None, None] + np.asarray([0, 0.1, 0.2], np.float32)
    frames = np.broadcast_to(frames, (16, 256, 256, 3))

    clip = pipeline.stack_clip(frames)

    assert clip.shape == pipeline.INPUT_SHAPE == (1, 6, 2048, 256)
    # Tubelet t fills rows t * 256 onwards; channel (frame in tubelet) * 3 + colour.
    np.testing.assert_allclose(clip[0, :, 3 * 256, 0], [6, 6.1, 6.2, 7, 7.1, 7.2])
    with pytest.raises(ValueError, match="expected"):
        pipeline.stack_clip(frames[:15])


def test_token_features_follow_tubelet_row_column_order(load_script: Callable[[str], ModuleType]) -> None:
    pipeline = load_script("pipeline")
    features = np.zeros(pipeline.OUTPUT_SHAPE, np.float32)
    # Token (1, 2, 3) is 1 * 256 + 2 * 16 + 3.
    features[0, 5, 0, 1 * 256 + 2 * 16 + 3] = 1

    grid = pipeline.token_features(features)

    assert grid.shape == (8, 16, 16, 768)
    assert grid[1, 2, 3, 5] == 1 and grid.sum() == 1


def test_a_pan_clip_moves_across_the_image(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    pipeline = load_script("pipeline")
    image = np.zeros((320, 640, 3), np.uint8)
    image[:, :320] = 255

    clip = calibration.pan_clip(image)

    assert clip.shape == pipeline.INPUT_SHAPE
    # The first frame shows only the bright left half and the last only the dark right half.
    first, last = clip[0, :3, :256], clip[0, 3:, -256:]
    np.testing.assert_allclose(first[:, 0, 0], (1 - pipeline.MEAN) / pipeline.STD, atol=1e-5)
    np.testing.assert_allclose(last[:, 0, -1], -pipeline.MEAN / pipeline.STD, atol=1e-5)


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    files = {f"cal2017/{index:02d}.jpg": _jpeg(64, 48) for index in range(6)}
    files["val2017/x.jpg"] = _jpeg(64, 48)
    archive = _write_archive(tmp_path / "coco.tgz", files)

    clips = calibration.calibration_clips(archive, 2)

    assert [clip.shape for clip in clips] == [(1, 6, 2048, 256)] * 2
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        calibration.calibration_clips(archive, 7)


def test_fidelity_selection_is_the_declared_seeded_val2017_sample(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    names = [f"val2017/{index:02d}.jpg" for index in range(10)]
    for name in [*names, "cal2017/c.jpg"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    expected = sorted(names)
    random.Random(42).shuffle(expected)

    assert calibration.fidelity_names(tmp_path, 42, 2, 5) == expected[2:7]
    with pytest.raises(ValueError, match="needs 11 images"):
        calibration.fidelity_names(tmp_path, 42, 1, 10)


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a16"])
def test_compile_binds_every_part_in_each_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe._component_parameters(profile_id)

    assert list(parameters) == ["encoder_0", "encoder_1", "encoder_2"]
    assert {item["precision"] for item in parameters.values()} == {profile_id}
    with pytest.raises(ValueError, match="no build profile 'w8a8'"):
        recipe._component_parameters("w8a8")


def test_w8a16_calibrates_only_when_it_marks_int8_operations(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    calls = []
    recipe.SkipCalibration = lambda: calls.append("skip") or "skip"
    recipe.HistogramMSEMethod = lambda bins: calls.append(bins) or "mse"

    recipe._quantization({"precision": "w8a16"})
    recipe._quantization({"precision": "w8a16", "int8_operations": ["mlp"], "calibration_bins": 1024})

    assert calls == ["skip", 1024]


@pytest.mark.parametrize(
    ("op_type", "name", "group"),
    [
        ("Conv", "/patch/Conv", "bf16"),
        ("Sqrt", "/blocks.4/norm1/Sqrt", "bf16"),
        ("Einsum", "/blocks.4/Einsum_3", "attention"),
        ("Softmax", "/blocks.4/Softmax_7", "attention"),
        ("Concat", "/blocks.4/Concat", "attention"),
        ("Reshape", "/blocks.4/Reshape_3", "attention"),
        ("Conv", "/blocks.4/query/Conv", "qkv"),
        ("Slice", "/blocks.4/Slice_7", "qkv"),
        ("Mul", "/blocks.4/Mul_3", "qkv"),
        ("Conv", "/blocks.4/fc1/Conv", "mlp"),
        ("Erf", "/blocks.4/Erf", "mlp"),
        ("Mul", "/blocks.4/Mul_13", "mlp"),
        ("Conv", "/blocks.4/output/Conv", "out"),
        ("Add", "/blocks.4/Add_5", "out"),
    ],
)
def test_operations_fall_in_their_precision_groups(
    load_script: Callable[[str], ModuleType], op_type: str, name: str, group: str
) -> None:
    recipe = load_script("compile")
    node = type("Node", (), {"op_type": op_type, "name": name})()

    assert recipe.operation_group(node) == group


def test_precision_marks_follow_the_int8_groups(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="/blocks.0/fc1/Relu"),
        helper.make_node("Softmax", ["a"], ["b"], name="/blocks.0/Softmax"),
    ]
    value = lambda name: helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 4, 2, 2])  # noqa: E731
    onnx.save(helper.make_model(helper.make_graph(nodes, "g", [value("x")], [value("b")])), tmp_path / "part.onnx")
    recipe.onnx = onnx

    marked = onnx.load(recipe.mark_precisions(tmp_path / "part.onnx", tmp_path / "marked.onnx", {"mlp"}))

    annotations = [node for node in marked.graph.node if node.op_type == "AnnotatePrecision"]
    assert [node.domain for node in annotations] == ["ai.sima", "ai.sima"]
    assert [node.attribute[0].s for node in annotations] == [b"int8", b"bfloat16"]
    assert annotations[0].output[0] == "a" and marked.graph.node[0].output[0] == "a/unannotated"
    with pytest.raises(ValueError, match="unsupported int8_operations"):
        recipe.mark_precisions(tmp_path / "part.onnx", tmp_path / "other.onnx", {"attention"})


def test_features_come_back_in_onnx_layout(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    features = np.random.RandomState(0).normal(size=(1, 768, 1, 2048)).astype(np.float32)

    np.testing.assert_array_equal(recipe.channels_first(features.transpose(0, 2, 3, 1), features.shape), features)


def _mpk(path: Path, processors: list[str]) -> Path:
    manifest = {"plugins": [{"processor": p} for p in processors]}
    content = json.dumps(manifest).encode()
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("model_mpk.json")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return path


def test_audit_requires_one_mla_segment_and_no_host_code(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")

    recipe.audit_mpk(_mpk(tmp_path / "ok.tar.gz", ["EV74", "MLA", "EV74"]), "encoder_0")
    for name, processors in [("two", ["MLA", "MLA"]), ("host", ["MLA", "A65"])]:
        with pytest.raises(RuntimeError):
            recipe.audit_mpk(_mpk(tmp_path / f"{name}.tar.gz", processors), "encoder_0")


def test_compile_main_rejects_an_unknown_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile.py",
            "--model",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--build-profile",
            "w16a16",
            "--compile-invocation",
            "model",
        ],
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert "a build profile compiles in one invocation named after it" in capsys.readouterr().err
