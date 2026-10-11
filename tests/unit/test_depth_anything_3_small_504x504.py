from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import yaml
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "depth_anything_3_small_504x504"
MODEL_DIR = ROOT / "models" / MODEL_ID
SCRIPTS_DIR = MODEL_DIR / "scripts"
# Compiler and board modules that compile.py and accuracy.py import but these tests never call.
COMPILER_MODULES = [
    "afe",
    "afe.apis",
    "afe.apis.defines",
    "afe.apis.loaded_net",
    "afe.apis.model",
    "afe.core",
    "afe.core.utils",
    "afe.load",
    "afe.load.importers",
    "afe.load.importers.general_importer",
    "sima_utils",
    "sima_utils.data",
    "sima_utils.data.data_generator",
]
BOARD_MODULES = ["pyneat"]
HELD_OUT = "2011_09_26_drive_0001_sync"


def load_module(name: str, monkeypatch: pytest.MonkeyPatch | None = None, stubs: list[str] | None = None) -> ModuleType:
    for stub in stubs or []:
        monkeypatch.setitem(sys.modules, stub, MagicMock())
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{MODEL_ID}_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Sibling scripts are imported by name, so another model's calibration or surgery must not stand in for ours.
    shadowed = {dependency: sys.modules.pop(dependency, None) for dependency in ("calibration", "surgery")}
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        for dependency, previous in shadowed.items():
            if previous is None:
                sys.modules.pop(dependency, None)
            else:
                sys.modules[dependency] = previous
    return module


def frame(drive: str, index: int) -> str:
    return f"{drive}_{index:010d}.png"


def test_cover_geometry_matches_upstream_resize_and_center_crop() -> None:
    calibration = load_module("calibration")

    # A 1242x375 KITTI frame is enlarged to 1669x504 and loses 582 columns on the left and 583 on the right.
    assert calibration.cover_geometry(1242, 375) == (504 / 375, (1669, 504), (582, 0))
    # A portrait frame is enlarged to cover the width and keeps the middle rows.
    assert calibration.cover_geometry(240, 320) == (2.1, (504, 672), (0, 84))


def test_cover_crop_returns_the_network_view() -> None:
    pytest.importorskip("cv2")
    calibration = load_module("calibration")
    image = np.zeros((375, 1242, 3), np.uint8)
    # Columns 433 to 807 of the frame land in the crop; mark every column outside it.
    image[:, :433] = 255
    image[:, 808:] = 255

    result = calibration.cover_crop(image)

    assert result.shape == (504, 504, 3)
    assert (result[:, 1:-1] == 0).all()
    # Bicubic enlargement reaches one network column past each crop edge.
    assert result[:, 0].all() and result[:, -1].all()


def test_preprocessing_returns_normalized_rgb_hwc() -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    blue = np.zeros((375, 1242, 3), np.uint8)
    blue[..., 0] = 255
    _, content = cv2.imencode(".png", blue)

    result = calibration.preprocess_image(content.tobytes())

    assert result.shape == (504, 504, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[252, 252], (np.array([0, 0, 1]) - calibration.MEAN) / calibration.STD, rtol=1e-6)


def _kitti_archive(path: Path, frames: dict[str, list[int]]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for drive, indices in frames.items():
            for index in indices:
                for folder in ("images", "depth"):
                    info = tarfile.TarInfo(f"kitti/test/{folder}/{frame(drive, index)}")
                    info.size = 1
                    stream.addfile(info, io.BytesIO(b"x"))
        # The grayscale cal folder holds frames of no drive.
        info = tarfile.TarInfo("kitti/cal/000001.png")
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))


def test_calibration_pairs_consecutive_frames_of_the_calibration_drives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = load_module("calibration")
    monkeypatch.setattr(calibration, "CALIBRATION_DRIVES", ("drive_a_drive_0001_sync", "drive_b_drive_0002_sync"))
    archive = tmp_path / "kitti.tgz"
    # Frames 3 and 9 are too far apart to pair; the held-out drive has no frames close enough.
    _kitti_archive(
        archive,
        {"drive_a_drive_0001_sync": [9, 3, 0], "drive_b_drive_0002_sync": [12, 10], HELD_OUT: [0, 20]},
    )

    pairs = calibration.calibration_pairs(archive, 2)

    assert pairs == calibration.calibration_pairs(archive, 2)
    assert sorted((Path(first).name, Path(second).name) for first, second in pairs) == [
        (frame("drive_a_drive_0001_sync", 0), frame("drive_a_drive_0001_sync", 3)),
        (frame("drive_b_drive_0002_sync", 10), frame("drive_b_drive_0002_sync", 12)),
    ]
    assert all(Path(name).parent.name == "images" for pair in pairs for name in pair)
    with pytest.raises(ValueError, match="cannot select 3"):
        calibration.calibration_pairs(archive, 3)
    monkeypatch.setattr(calibration, "CALIBRATION_DRIVES", ("drive_a_drive_0001_sync",))
    with pytest.raises(ValueError, match="expected view pairs only in"):
        calibration.calibration_pairs(archive, 1)


def test_calibration_sample_is_fixed_by_the_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = load_module("calibration")
    drive_a, drive_b = "drive_a_drive_0001_sync", "drive_b_drive_0002_sync"
    monkeypatch.setattr(calibration, "CALIBRATION_DRIVES", (drive_a, drive_b))
    archive = tmp_path / "kitti.tgz"
    _kitti_archive(archive, {drive_a: list(range(10)), drive_b: [0, 1]})

    pairs = calibration.calibration_pairs(archive, 3)

    assert [(Path(first).name, Path(second).name) for first, second in pairs] == [
        (frame(drive_a, 6), frame(drive_a, 7)),
        (frame(drive_b, 0), frame(drive_b, 1)),
        (frame(drive_a, 0), frame(drive_a, 1)),
    ]


def test_calibration_images_return_both_views_as_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    drive = "drive_a_drive_0001_sync"
    monkeypatch.setattr(calibration, "CALIBRATION_DRIVES", (drive,))
    archive = tmp_path / "kitti.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for index, value in ((0, 0), (1, 255)):
            _, content = cv2.imencode(".png", np.full((375, 1242, 3), value, np.uint8))
            info = tarfile.TarInfo(f"kitti/test/images/{frame(drive, index)}")
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content.tobytes()))

    first, second = calibration.calibration_images(archive, 1)

    assert first.shape == second.shape == (1, 504, 504, 3)
    np.testing.assert_allclose(first[0, 0, 0], -calibration.MEAN / calibration.STD, rtol=1e-6)
    np.testing.assert_allclose(second[0, 0, 0], (1 - calibration.MEAN) / calibration.STD, rtol=1e-6)


def _initializer(initializers: list, name: str, value) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def _source_graph(*, unfolded: bool = False) -> onnx.ModelProto:
    """One view-batched attention block, one global step, the head and the pose decoder at toy width."""
    rng = np.random.default_rng(0)
    tokens, heads, width = 4, 2, 6
    channels = heads * width
    inits: list = []

    def constant(name, value):
        return _initializer(inits, name, value)

    def weights(name, *shape):
        # Small weights keep the head's Exp outputs finite.
        return constant(name, (0.3 * rng.normal(size=shape)).astype(np.float32))

    nodes = [
        # The views are stacked to [1, 2, 3, H, W] and flattened to a batch of two.
        helper.make_node("Unsqueeze", ["image0", constant("view_axis", [1])], ["view0"]),
        helper.make_node("Unsqueeze", ["image1", "view_axis"], ["view1"]),
        helper.make_node("Concat", ["view0", "view1"], ["views"], axis=1),
        helper.make_node("Reshape", ["views", constant("batch_shape", [2, 3, 504, 504])], ["batch"]),
        helper.make_node(
            "Slice",
            [
                "batch",
                constant("crop_starts", [0, 0, 0]),
                constant("crop_ends", [1, 4, 12]),
                constant("crop_axes", [1, 2, 3]),
            ],
            ["crop"],
        ),
        helper.make_node("Reshape", ["crop", constant("token_shape", [2, tokens, channels])], ["patches"]),
        # A view-indexed camera token.
        helper.make_node("Add", ["patches", weights("camera_token", 2, 1, channels)], ["tokens"]),
        helper.make_node("MatMul", ["tokens", weights("qkv_w", channels, 3 * channels)], ["qkv"]),
        helper.make_node(
            "Reshape",
            ["qkv", constant("qkv_shape", [2, tokens, 3, heads, width])],
            ["qkv_5d"],
            name="/blocks.0/attn/Reshape",
        ),
        helper.make_node("Transpose", ["qkv_5d"], ["qkv_t"], name="/blocks.0/attn/Transpose", perm=[2, 0, 3, 1, 4]),
    ]
    for index, part in enumerate("qkv"):
        nodes.append(
            helper.make_node(
                "Gather",
                ["qkv_t", constant(f"index_{part}", np.array(index, np.int64))],
                [part],
                name=f"/blocks.0/attn/Gather_{index}",
                axis=0,
            )
        )
    nodes += [
        helper.make_node("Transpose", ["k"], ["k_t"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["q", "k_t"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["attention_weights"], axis=-1),
        helper.make_node("MatMul", ["attention_weights", "v"], ["heads"]),
        helper.make_node("Transpose", ["heads"], ["heads_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["heads_t", constant("merge_shape", [2, tokens, channels])], ["attention"]),
        helper.make_node("Add", ["tokens", "attention"], ["block"]),
        # A global step over both views' tokens, entered again through a Slice of the stacked views.
        helper.make_node("Reshape", ["block", constant("stack_shape", [1, 2, tokens, channels])], ["stacked"]),
        helper.make_node("Reshape", ["stacked", constant("global_shape", [1, 2 * tokens, channels])], ["sequence"]),
        helper.make_node("MatMul", ["sequence", weights("global_w", channels, channels)], ["mixed"]),
        helper.make_node("Reshape", ["mixed", "stack_shape"], ["mixed_views"]),
        helper.make_node("Reshape", ["mixed_views", "merge_shape"], ["features"]),
        # The head emits NCHW depth and confidence channels and reads them through an NHWC transpose.
        helper.make_node("MatMul", ["features", weights("head_w", channels, 2)], ["head_tokens"]),
        helper.make_node("Transpose", ["head_tokens"], ["head_channels"], perm=[0, 2, 1]),
        helper.make_node("Reshape", ["head_channels", constant("map_shape", [2, 2, 2, 2])], ["head_map"]),
        helper.make_node(
            "Resize",
            ["head_map", "", "", constant("head_sizes", [2, 2, 504, 504])],
            ["head_out"],
            mode="nearest",
        ),
        helper.make_node("Transpose", ["head_out"], ["head_nhwc"], name="/head/Transpose", perm=[0, 2, 3, 1]),
        helper.make_node(
            "Slice",
            [
                "head_nhwc",
                constant("depth_starts", [0]),
                constant("depth_ends", [-1]),
                constant("depth_axes", [3]),
            ],
            ["depth_raw"],
        ),
        helper.make_node("Exp", ["depth_raw"], ["depth_exp"]),
        helper.make_node("Squeeze", ["depth_exp", constant("channel_axis", [3])], ["depth_views"]),
        helper.make_node("Reshape", ["depth_views", constant("output_shape", [1, 2, 504, 504])], ["depth"]),
        helper.make_node(
            "Gather", ["head_nhwc", constant("confidence_index", np.array(-1, np.int64))], ["conf_raw"], axis=3
        ),
        helper.make_node("Exp", ["conf_raw"], ["conf_exp"]),
        helper.make_node("Add", ["conf_exp", constant("one", np.array(1, np.float32))], ["conf_views"]),
        helper.make_node("Reshape", ["conf_views", "output_shape"], ["depth_conf"]),
        # The pose decoder reads each view's camera token.
        helper.make_node("Gather", ["features", constant("camera_index", np.array(0, np.int64))], ["camera"], axis=1),
        helper.make_node("Gemm", ["camera", weights("pose_w", channels, 9)], ["pose_views"]),
        helper.make_node("Reshape", ["pose_views", constant("pose_shape", [1, 2, 9])], ["pose_enc"]),
    ]
    if unfolded:
        nodes.insert(0, helper.make_node("Shape", ["image0"], ["unused_shape"]))
    for index, node in enumerate(nodes):
        node.name = node.name or f"/node_{index}"
    graph = helper.make_graph(
        nodes,
        "fixture",
        [
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 3, 504, 504])
            for name in ("image0", "image1")
        ],
        [
            helper.make_tensor_value_info("depth", onnx.TensorProto.FLOAT, [1, 2, 504, 504]),
            helper.make_tensor_value_info("depth_conf", onnx.TensorProto.FLOAT, [1, 2, 504, 504]),
            helper.make_tensor_value_info("pose_enc", onnx.TensorProto.FLOAT, [1, 2, 9]),
        ],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_surgery_computes_each_view_with_batch_one_without_changing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_module("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    path = surgery.prepare_model(source, tmp_path / "out")
    model = onnx.load(path)

    assert path.name == "depth_anything_3_small.onnx"
    operators = [node.op_type for node in model.graph.node]
    assert {"Cast", "Squeeze", "Unsqueeze"}.isdisjoint(operators)
    # One QKV split per view; the head reads its channels with Slices instead of a transpose.
    assert operators.count("Split") == 2
    assert not [node for node in model.graph.node if node.op_type == "Transpose" and "/head/" in node.name]
    shapes = [[d.dim_value for d in value.type.tensor_type.shape.dim] for value in model.graph.value_info]
    assert all(len(shape) <= 4 and shape[0] == 1 for shape in shapes)
    rng = np.random.default_rng(1)
    views = {name: rng.normal(size=(1, 3, 504, 504)).astype(np.float32) for name in ("image0", "image1")}
    expected = ReferenceEvaluator(str(source)).run(None, views)
    actual = ReferenceEvaluator(str(path)).run(None, views)
    for reference, candidate in zip(expected, actual, strict=True):
        np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-6)


def test_surgery_requires_every_attention_block(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    with pytest.raises(ValueError, match="expected 12 QKV blocks, found 1"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_rejects_an_operation_that_mixes_the_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surgery = load_module("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    model = _source_graph()
    softmax = next(node for node in model.graph.node if node.op_type == "Softmax")
    softmax.attribute[0].i = 0
    source = tmp_path / "source.onnx"
    onnx.save(model, source)

    with pytest.raises(ValueError, match="view-axis Softmax"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_requires_unique_node_names(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    model = _source_graph()
    model.graph.node[1].name = model.graph.node[0].name
    source = tmp_path / "source.onnx"
    onnx.save(model, source)

    with pytest.raises(ValueError, match="unique name"):
        surgery.prepare_model(source, tmp_path / "out")


def test_surgery_requires_a_folded_source(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(unfolded=True), source)

    with pytest.raises(ValueError, match="fold the source with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    view = 1 * 3 * 504 * 504 * 4
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "image0", "size": view}, {"name": "image1", "size": view}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "cast"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {"sequence": 2, "processor": "EV74", "config_params": {"kernel": "unpack"}},
            {
                "sequence": 3,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": 2 * 504 * 504 * 4}, {"size": 2 * 504 * 504 * 4}, {"size": 2 * 9 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_outputs", "one_input", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "two_outputs":
        plugins[-1]["output_nodes"].pop()
    elif case == "one_input":
        manifest["input_nodes"].pop()
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


def test_compile_reads_the_declared_build_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)

    assert compile_module._parameters("modalix_bf16") == {
        "precision": "bf16",
        "calibration_bins": 2048,
        "calibration_samples": 50,
    }
    with pytest.raises(ValueError, match="no build profile"):
        compile_module._parameters("modalix_int8")
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_module._quantization("int8", 2048)


def test_compile_recipe_reproduces_the_qualified_compiler_settings() -> None:
    source = (SCRIPTS_DIR / "compile.py").read_text(encoding="utf-8")

    for setting in (
        "with_activation_quantization(bfloat16_scheme())",
        "with_weight_quantization(bfloat16_scheme())",
        "with_requantization_mode(RequantizationMode.sima)",
        "HistogramMSEMethod(calibration_bins)",
        "with_bias_correction(BiasCorrectionType.NONE)",
        "with_channel_equalization(False)",
        "with_smooth_quant(False)",
        "with_per_token_quant(False)",
        'layout="NCHW"',
        "flexible_batch_size=False",
        "compress=True",
        "l2_caching_mode=L2CachingMode.NONE",
        "layer_norm_use_fp32_intermediates=True",
        "rms_norm_use_fp32_intermediates=False",
        "layout_search_effort_level=0",
        "enable_filter_sharing=False",
        "batch_size=1",
        "preserve=True",
        "deployable=True",
        "use_power_limits=False",
        "max_mla_power=None",
        "retained_temporary_directory_name=None",
        'parser.add_argument("--dataset", type=Path, required=True',
    ):
        assert setting in source, setting
    assert "requests" not in source and "urllib" not in source


def _kitti_directory(root: Path, frames: dict[str, list[int]]) -> None:
    for folder in ("images", "depth"):
        (root / "test" / folder).mkdir(parents=True, exist_ok=True)
        for drive, indices in frames.items():
            for index in indices:
                (root / "test" / folder / frame(drive, index)).write_bytes(b"x")


def test_view_pairs_use_the_nearest_frame_of_held_out_drives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    calibration_drive = accuracy.CALIBRATION_DRIVES[0]
    other = "2011_09_30_drive_0016_sync"
    _kitti_directory(tmp_path, {HELD_OUT: [30, 10, 20, 26], other: [5, 9], calibration_drive: [0, 1]})

    pairs = accuracy.view_pairs(tmp_path)

    images = tmp_path / "test" / "images"
    # Frame 20 is 6 from frame 26 and 10 from frame 10; frame 10's two nearest frames are 10 and 16 away.
    assert pairs == [
        (images / frame(HELD_OUT, 10), images / frame(HELD_OUT, 20)),
        (images / frame(HELD_OUT, 20), images / frame(HELD_OUT, 26)),
        (images / frame(HELD_OUT, 26), images / frame(HELD_OUT, 30)),
        (images / frame(HELD_OUT, 30), images / frame(HELD_OUT, 26)),
        (images / frame(other, 5), images / frame(other, 9)),
        (images / frame(other, 9), images / frame(other, 5)),
    ]


def test_view_pairs_prefer_the_later_of_two_equally_near_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    _kitti_directory(tmp_path, {HELD_OUT: [10, 15, 20]})

    pairs = accuracy.view_pairs(tmp_path)

    assert pairs[1][1].name == frame(HELD_OUT, 20)


def test_pairing_scores_only_valid_pixels_in_the_network_view(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    truth = np.full((375, 1242), 20.0)
    truth[100, 500] = 0.0
    truth[200, 600] = 85.0
    depth = np.arange(504 * 504, dtype=np.float32).reshape(504, 504)

    pixels, meters = accuracy.paired(depth, truth)

    # The center crop keeps frame columns 433 to 807 and every row.
    assert len(meters) == 375 * (807 - 433 + 1) - 2
    # Frame pixel (row 0, column 433) lies in network row 0, column floor(433.5 * 1669 / 1242 - 582) = 0.
    assert pixels[0] == depth[0, 0]


def test_alignment_scores_a_prediction_affine_in_disparity_as_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    # Ground truth that is constant over the frame columns each network column covers.
    _, (resized_width, _), (left, _) = accuracy.cover_geometry(1242, 375)
    columns = np.clip(np.floor((np.arange(1242) + 0.5) * resized_width / 1242 - left), 0, 503)
    truth = np.tile(5.0 + 0.1 * columns, (375, 1))
    network_truth = np.tile(5.0 + 0.1 * np.arange(504), (504, 1))
    exact = 1.0 / (2.0 / network_truth + 0.1)

    result = accuracy.score(exact, truth)

    assert result["abs_rel"] == pytest.approx(0.0, abs=1e-9) and result["delta1"] == 1.0
    assert result["pixels"] == 375 * 375
    constant = accuracy.score(np.ones((504, 504)), truth)
    assert constant["abs_rel"] > 0.3 and constant["delta1"] < 0.5


def test_reference_depth_takes_view_zero_of_the_nhwc_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    depth = np.random.default_rng(0).random((1, 504, 504, 2), np.float32)
    confidence = np.ones((1, 504, 504, 2), np.float32)
    pose = np.zeros((1, 2, 9), np.float32)

    np.testing.assert_array_equal(accuracy.reference_depth([depth, confidence, pose]), depth[0, ..., 0])
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.reference_depth([depth, confidence])
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.reference_depth([pose, confidence, depth])
    depth[0, 1, 2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.reference_depth([depth, confidence, pose])


def test_model_declares_the_exported_interface_and_its_preprocessing() -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    surgery = load_module("surgery")
    calibration = load_module("calibration")
    component = model["components"][0]
    entrypoint = component["interface"]["entrypoints"][0]

    assert {item["name"]: item["shape"] for item in entrypoint["inputs"]} == surgery.INPUT_SHAPES
    assert [item["name"] for item in entrypoint["inputs"]] == surgery.INPUTS
    assert {item["name"]: item["shape"] for item in entrypoint["outputs"]} == surgery.OUTPUT_SHAPES
    assert [item["name"] for item in entrypoint["outputs"]] == surgery.OUTPUTS
    assert all(shape[2:] == [calibration.HEIGHT, calibration.WIDTH] for shape in surgery.INPUT_SHAPES.values())
    steps = {step["op"]: step["parameters"] for step in component["preprocessing"]["pipeline"]}
    assert steps["resize"]["height"] == calibration.HEIGHT and steps["resize"]["width"] == calibration.WIDTH
    np.testing.assert_allclose(steps["normalize"]["mean"], calibration.MEAN)
    np.testing.assert_allclose(steps["normalize"]["std"], calibration.STD)
    assert component["surgery"]["entrypoint"] == "scripts/surgery.py:prepare_model"
    assert component["recipe"] == "scripts/compile.py"
    assert [profile["id"] for profile in model["build_profiles"]] == ["modalix_bf16"]
    assert model["calibrations"][0]["dataset"] == model["accuracy"][0]["dataset"]
    assert model["accuracy"][0]["required_metrics"] == ["abs_rel", "delta1"]
    assert "reference" not in model["accuracy"][0]
