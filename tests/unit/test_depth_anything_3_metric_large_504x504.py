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
MODEL_ID = "depth_anything_3_metric_large_504x504"
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


def test_cover_geometry_matches_upstream_resize_and_center_crop() -> None:
    calibration = load_module("calibration")

    # A 640x480 NYU frame is enlarged by 1.05 to 672x504 and loses 84 columns left and right.
    assert calibration.cover_geometry(640, 480) == (1.05, (672, 504), (84, 0))
    # A larger frame shrinks to the same cover.
    assert calibration.cover_geometry(1280, 960) == (0.525, (672, 504), (84, 0))
    # A portrait frame covers the width and keeps the middle rows.
    assert calibration.cover_geometry(480, 640) == (1.05, (504, 672), (0, 84))


def test_cover_crop_returns_the_network_view() -> None:
    pytest.importorskip("cv2")
    calibration = load_module("calibration")
    image = np.zeros((480, 640, 3), np.uint8)
    # Columns 80 to 559 of the frame land in the crop; mark the columns just outside it.
    image[:, :79] = 255
    image[:, 561:] = 255

    result = calibration.cover_crop(image)

    assert result.shape == (504, 504, 3)
    assert (result[:, 1:-1] == 0).all()


def test_cover_crop_enlarges_with_bicubic_sampling() -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), np.uint8)

    result = calibration.cover_crop(image)

    np.testing.assert_array_equal(result, cv2.resize(image, (672, 504), interpolation=cv2.INTER_CUBIC)[:, 84:588])


def test_preprocessing_returns_normalized_rgb_hwc() -> None:
    cv2 = pytest.importorskip("cv2")
    calibration = load_module("calibration")
    blue = np.zeros((480, 640, 3), np.uint8)
    blue[..., 0] = 255
    _, content = cv2.imencode(".png", blue)

    result = calibration.preprocess_image(content.tobytes())

    assert result.shape == (504, 504, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[252, 252], (np.array([0, 0, 1]) - calibration.MEAN) / calibration.STD, rtol=1e-6)


def test_calibration_draws_a_seeded_sample_of_train_frames_only(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    archive = tmp_path / "nyu.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("train", "val", "cal", "test"):
            for index in range(6):
                for name in (f"rgb_{index}.jpg", f"sync_depth_{index}.png"):
                    info = tarfile.TarInfo(f"nyu_depth_v2/{split}/{name}")
                    info.size = 1
                    stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "train" and Path(name).name.startswith("rgb_") for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _initializer(initializers: list, name: str, value) -> str:
    initializers.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def _source_graph(*, unfolded: bool = False) -> onnx.ModelProto:
    """One attention block and one DPT tap in the exported form, at toy width, behind the real interface."""
    rng = np.random.default_rng(0)
    tokens, heads, width = 4, 2, 6
    channels = heads * width
    inits: list = []

    def constant(name, value):
        return _initializer(inits, name, value)

    nodes = [
        helper.make_node(
            "Slice",
            [
                "image",
                constant("crop_starts", [0, 0, 0]),
                constant("crop_ends", [1, 4, 12]),
                constant("crop_axes", [1, 2, 3]),
            ],
            ["crop"],
        ),
        helper.make_node("Reshape", ["crop", constant("token_shape", [1, tokens, channels])], ["tokens"]),
        helper.make_node(
            "MatMul",
            ["tokens", constant("qkv_w", rng.normal(size=(channels, 3 * channels)).astype(np.float32))],
            ["qkv"],
        ),
        helper.make_node(
            "Reshape",
            ["qkv", constant("qkv_shape", [1, tokens, 3, heads, width])],
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
        helper.make_node("Softmax", ["scores"], ["weights"], axis=-1),
        helper.make_node("MatMul", ["weights", "v"], ["heads"]),
        helper.make_node("Transpose", ["heads"], ["heads_t"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["heads_t", constant("merge_shape", [1, tokens, channels])], ["attention"]),
        helper.make_node("Add", ["tokens", "attention"], ["block"]),
        # The DPT tap.
        helper.make_node("Reshape", ["block", constant("tap_shape", [1, 1, tokens, channels])], ["tap_4d"]),
        helper.make_node(
            "LayerNormalization",
            [
                "tap_4d",
                constant("norm_scale", rng.normal(size=channels).astype(np.float32)),
                constant("norm_bias", rng.normal(size=channels).astype(np.float32)),
            ],
            ["tap_norm"],
            name="/head/norm/LayerNormalization",
            axis=-1,
            epsilon=1e-6,
        ),
        helper.make_node(
            "Slice",
            [
                "tap_norm",
                constant("cls_starts", [1]),
                constant("cls_ends", [2**62]),
                constant("cls_axes", [2]),
                constant("cls_steps", [1]),
            ],
            ["patches_4d"],
        ),
        helper.make_node("Reshape", ["patches_4d", constant("patch_shape", [1, tokens - 1, channels])], ["patches"]),
        # Identity data movement that the clean-up removes.
        helper.make_node("Unsqueeze", ["patches", constant("unsqueeze_axes", [0])], ["patches_4d_again"]),
        helper.make_node("Reshape", ["patches_4d_again", "patch_shape"], ["patches_again"]),
        helper.make_node("Cast", ["patches_again"], ["patches_float"], to=onnx.TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["patches_float"], ["mean"], axes=[1, 2], keepdims=1),
        helper.make_node("Reshape", ["mean", constant("scalar_shape", [1, 1, 1, 1])], ["offset"]),
        helper.make_node(
            "Slice",
            ["image", constant("plane_starts", [0]), constant("plane_ends", [1]), constant("plane_axes", [1])],
            ["plane"],
        ),
        helper.make_node("Add", ["plane", "offset"], ["depth"]),
        helper.make_node("Mul", ["plane", "offset"], ["sky"]),
    ]
    if unfolded:
        nodes.insert(0, helper.make_node("Shape", ["image"], ["unused_shape"]))
    for index, node in enumerate(nodes):
        node.name = node.name or f"/node_{index}"
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 504, 504])],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1, 504, 504]) for name in ("depth", "sky")],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_surgery_rewrites_attention_and_taps_without_changing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surgery = load_module("surgery")
    monkeypatch.setattr(surgery, "BLOCKS", 1)
    monkeypatch.setattr(surgery, "TAPS", 1)
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    path = surgery.prepare_model(source, tmp_path / "out")
    model = onnx.load(path)

    assert path.name == "depth_anything_3_metric_large.onnx"
    operators = [node.op_type for node in model.graph.node]
    assert {"Gather", "Cast", "Squeeze", "Unsqueeze"}.isdisjoint(operators)
    assert operators.count("Split") == 1
    shapes = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim] for value in model.graph.value_info}
    assert all(len(shape) <= 4 for shape in shapes.values())
    norm = next(node for node in model.graph.node if node.op_type == "LayerNormalization")
    assert len(shapes[norm.input[0]]) == 3
    image = np.random.default_rng(1).normal(size=(1, 3, 504, 504)).astype(np.float32)
    expected = ReferenceEvaluator(str(source)).run(None, {"image": image})
    actual = ReferenceEvaluator(str(path)).run(None, {"image": image})
    for reference, candidate in zip(expected, actual, strict=True):
        np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-6)


def test_surgery_requires_every_attention_block(tmp_path: Path) -> None:
    surgery = load_module("surgery")
    source = tmp_path / "source.onnx"
    onnx.save(_source_graph(), source)

    with pytest.raises(ValueError, match="expected 24 QKV blocks, found 1"):
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
    return {
        "model_sdk_version": compile_module.COMPILER_VERSION,
        "input_nodes": [{"name": "image", "size": 1 * 3 * 504 * 504 * 4}],
        "plugins": [
            {"sequence": 0, "processor": "EV74", "config_params": {"kernel": "cast"}},
            {"sequence": 1, "processor": "MLA", "config_params": {}},
            {"sequence": 2, "processor": "EV74", "config_params": {"kernel": "unpack"}},
            {
                "sequence": 3,
                "processor": "EV74",
                "config_params": {"kernel": "pass_through"},
                "output_nodes": [{"size": 504 * 504 * 4}, {"size": 504 * 504 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "one_output", "input", "two_mla", "host", "compiler"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    compile_module = load_module("compile", monkeypatch, COMPILER_MODULES)
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "one_output":
        plugins[-1]["output_nodes"].pop()
    elif case == "input":
        manifest["input_nodes"][0]["size"] //= 2
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
        "calibration_samples": 46,
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
        "with_per_token_quant(False)",
        'layout="NCHW", flexible_batch_size=False',
        "compress=True",
        "l2_caching_mode=L2CachingMode.NONE",
        "layer_norm_use_fp32_intermediates=True",
        "layout_search_effort_level=0",
        'parser.add_argument("--dataset", type=Path, required=True',
    ):
        assert setting in source, setting
    assert "requests" not in source and "urllib" not in source


def test_eigen_frames_come_from_the_test_splits_in_frame_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    for split, numbers in {"train": [3], "val": [10, 2], "cal": [7], "test": [1, 100]}.items():
        (tmp_path / split).mkdir()
        for number in numbers:
            (tmp_path / split / f"rgb_{number}.jpg").write_bytes(b"x")
            (tmp_path / split / f"sync_depth_{number}.png").write_bytes(b"x")

    frames = accuracy.frames(tmp_path)

    assert [path.name for path in frames] == ["rgb_1.jpg", "rgb_2.jpg", "rgb_7.jpg", "rgb_10.jpg", "rgb_100.jpg"]
    assert accuracy.ground_truth_path(frames[0]) == tmp_path / "test" / "sync_depth_1.png"


def test_pairing_scores_only_valid_pixels_in_the_eigen_crop_and_the_network_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    truth = np.full((480, 640), 2.0)
    truth[100, 100] = 0.0
    truth[200, 200] = 12.0
    depth = np.arange(504 * 504, dtype=np.float32).reshape(504, 504)

    pixels, meters, scale = accuracy.paired(depth, truth)

    # The Eigen crop keeps rows 45 to 470 and columns 41 to 600; the center crop keeps columns 80 to 559.
    assert len(meters) == (470 - 45 + 1) * (559 - 80 + 1) - 2
    assert scale == 1.05
    # Frame pixel (row 45, column 80) lies in network row floor(45.5 * 1.05) = 47, column floor(80.5 * 1.05 - 84) = 0.
    assert pixels[0] == depth[47, 0]


def test_metric_conversion_scores_a_correct_prediction_as_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    truth = np.full((480, 640), 2.0)
    focal = accuracy.NYU_FOCAL * 1.05
    exact = np.full((504, 504), 2.0 * 300 / focal)

    assert accuracy.score(exact, truth) == pytest.approx({"abs_rel": 0.0, "delta1": 1.0, "pixels": 426 * 480})
    far = accuracy.score(exact * 1.3, truth)
    assert far["abs_rel"] == pytest.approx(0.3) and far["delta1"] == 0.0


def test_depth_output_takes_the_first_map_and_rejects_bad_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    accuracy = load_module("accuracy", monkeypatch, BOARD_MODULES)
    depth = np.random.default_rng(0).random((1, 504, 504, 1), np.float32)
    sky = np.zeros((1, 504, 504, 1), np.float32)

    np.testing.assert_array_equal(accuracy.depth_output([depth, sky]), depth.reshape(504, 504))
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.depth_output([depth])
    with pytest.raises(ValueError, match="unexpected MPK outputs"):
        accuracy.depth_output([depth[:, :252], sky])
    depth[0, 1, 2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        accuracy.depth_output([depth, sky])


def test_model_declares_the_exported_interface_and_its_preprocessing() -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    surgery = load_module("surgery")
    calibration = load_module("calibration")
    component = model["components"][0]
    entrypoint = component["interface"]["entrypoints"][0]

    assert [(item["name"], item["shape"]) for item in entrypoint["inputs"]] == [("image", surgery.INPUT_SHAPE)]
    assert {item["name"]: item["shape"] for item in entrypoint["outputs"]} == surgery.OUTPUT_SHAPES
    assert [item["name"] for item in entrypoint["outputs"]] == surgery.OUTPUTS
    assert surgery.INPUT_SHAPE[2:] == [calibration.HEIGHT, calibration.WIDTH]
    steps = {step["op"]: step["parameters"] for step in component["preprocessing"]["pipeline"]}
    assert steps["resize"]["height"] == calibration.HEIGHT and steps["resize"]["width"] == calibration.WIDTH
    np.testing.assert_allclose(steps["normalize"]["mean"], calibration.MEAN)
    np.testing.assert_allclose(steps["normalize"]["std"], calibration.STD)
    assert component["surgery"]["entrypoint"] == "scripts/surgery.py:prepare_model"
    assert component["recipe"] == "scripts/compile.py"
    assert [profile["id"] for profile in model["build_profiles"]] == ["modalix_bf16"]
    assert model["calibrations"][0]["dataset"] == model["accuracy"][0]["dataset"]
    assert model["accuracy"][0]["required_metrics"] == ["abs_rel", "delta1"]
    assert model["accuracy"][0]["reference"]["metrics"] == {"abs_rel": 0.070, "delta1": 0.963}
