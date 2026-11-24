from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
import pytest
from model_registry.dependencies import onnx_opset_version
from model_registry.errors import ValidationError
from model_registry.model import load_model as load_metadata
from PIL import Image

OPSET = onnx_opset_version()
ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "lightstereo_s"


@pytest.fixture
def scene_flow(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("scene_flow")


def _png(value: int, size: tuple[int, int] = (960, 540)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _pfm(disparity: np.ndarray) -> bytes:
    """A little-endian one-channel PFM, which stores its rows from the bottom."""
    height, width = disparity.shape
    return f"Pf\n{width} {height}\n-1.0\n".encode() + np.flipud(disparity).astype("<f4").tobytes()


def _pair(index: int) -> tuple[str, str, str]:
    return (
        f"Monkaa/frames_finalpass/scene/left/{index:04d}.png",
        f"Monkaa/frames_finalpass/scene/right/{index:04d}.png",
        f"Monkaa/disparity/scene/left/{index:04d}.pfm",
    )


def _cal_archive(path: Path, pairs: int) -> Path:
    """The scene-flow cal layout: scene_flow/cal.txt and uniform 960x540 views, grey level = pair (+100 if right),
    each with a disparity map of the pair's index in every pixel."""
    lines, files = [], {}
    for index in range(pairs):
        left, right, disparity = _pair(index)
        lines.append(f"{left} {right} {disparity}\n")
        files.update({left: _png(index), right: _png(index + 100)})
        files[disparity] = _pfm(np.full((540, 960), index, np.float32))
    files["cal.txt"] = "".join(lines).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(f"scene_flow/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def test_preprocessing_takes_a_native_view_and_normalizes_it(scene_flow: ModuleType) -> None:
    result = scene_flow.preprocess_image(_png(255))

    assert result.shape == (540, 960, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result[0, 0], (1 - scene_flow.MEAN) / scene_flow.STD, rtol=1e-6)
    # The graph pads the rows itself, so a view of any other size is refused rather than resized.
    with pytest.raises(ValueError, match="expected a 960x540 view"):
        scene_flow.preprocess_image(_png(0, (1242, 375)))


def test_pfm_rows_are_read_from_the_top(scene_flow: ModuleType) -> None:
    disparity = np.arange(12, dtype=np.float32).reshape(3, 4)

    np.testing.assert_array_equal(scene_flow.read_pfm(_pfm(disparity)), disparity)
    with pytest.raises(ValueError, match="one-channel"):
        scene_flow.read_pfm(b"PF\n4 3\n-1.0\n" + bytes(144))


def test_calibration_and_fidelity_take_spans_of_one_seeded_order(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 12)

    order = scene_flow.selected_pairs(dataset, seed=42, offset=0, samples=12)

    # Every listed pair once, the same order for the same seed, and later spans of that order for fidelity.
    assert sorted(order) == [_pair(i) for i in range(12)]
    assert scene_flow.selected_pairs(dataset, seed=42, offset=4, samples=6) == order[4:10]
    with pytest.raises(ValueError, match="needs 13 pairs"):
        scene_flow.selected_pairs(dataset, seed=42, offset=1, samples=12)


def test_read_pairs_returns_side_by_side_frames_and_true_disparities(tmp_path: Path, scene_flow: ModuleType) -> None:
    dataset = _cal_archive(tmp_path / "cal.tgz", 4)

    frames, disparities = scene_flow.read_pairs(dataset, [_pair(3), _pair(1)])

    assert frames.shape == (2, 540, 1920, 3) and disparities.shape == (2, 540, 960)
    # The fixture's grey level is the pair index for the left view and 100 more for the right one; the left
    # view fills the first 960 columns of each frame and the right view the rest.
    grey = (np.array([3, 1, 103, 101], np.float32) / 255 - scene_flow.MEAN[0]) / scene_flow.STD[0]
    np.testing.assert_allclose(frames[:, 20, 959, 0], grey[:2], rtol=1e-6)
    np.testing.assert_allclose(frames[:, 20, 960, 0], grey[2:], rtol=1e-6)
    np.testing.assert_array_equal(disparities[:, 7, 7], [3, 1])
    with pytest.raises(ValueError, match="lacks 3 listed files"):
        scene_flow.read_pairs(dataset, [_pair(7)])


def test_end_point_error_scores_pixels_between_0_and_192(scene_flow: ModuleType) -> None:
    truth = np.array([[10.0, 20.0, 0.0], [191.0, 192.0, -3.0]], np.float32)
    disparity = np.array([[[[11.0, 18.0, 5.0], [191.0, 0.0, 9.0]]]], np.float32)

    # |11 - 10|, |18 - 20| and 0 over the three pixels between 0 and 192; the 0, 192 and -3 px pixels are not scored.
    assert scene_flow.end_point_error(disparity, truth) == pytest.approx(1.0)
    # An image with no scored pixel counts as 0, not NaN.
    assert scene_flow.end_point_error(disparity, np.full_like(truth, 300.0)) == 0.0


@pytest.fixture
def surgery(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def test_interface_is_one_side_by_side_frame_and_the_left_views_disparity(surgery: ModuleType) -> None:
    # The export takes both 960x540 Scene Flow views as one [left | right] frame, pads each to OpenStereo's 544x960
    # evaluation size itself, and returns the left view's disparity at the view's own size.
    assert surgery.INTERFACE == {"stereo": [1, 3, 540, 1920], "disparity": [1, 1, 540, 960]}
    assert surgery.TOP_PAD == 4
    # MAX_DISP 192 at quarter resolution: 48 cost-volume bins.
    assert surgery.BINS == 48


def _export(surgery: ModuleType, case: str) -> onnx.ModelProto:
    """The side-by-side frame sliced into its halves, summed into the disparity through a 1x1 Conv, or one fault
    validate_export must catch."""
    helper = onnx.helper
    shapes = dict(surgery.INTERFACE)
    if case == "shape":
        shapes["stereo"] = [1, 3, 544, 1920]
    inputs = [helper.make_tensor_value_info("stereo", onnx.TensorProto.FLOAT, shapes["stereo"])]
    if case == "two_inputs":
        inputs.append(helper.make_tensor_value_info("right", onnx.TensorProto.FLOAT, [1, 3, 540, 960]))
    output = helper.make_tensor_value_info("disparity", onnx.TensorProto.FLOAT, shapes["disparity"])
    constants = {
        "weight": np.ones((1, 3, 1, 1), np.float32),
        "axes": np.array([3]),
        "zero": np.array([0]),
        "half": np.array([960]),
        "end": np.array([1920]),
    }
    initializers = [onnx.numpy_helper.from_array(value, name) for name, value in constants.items()]
    nodes = [
        helper.make_node("Slice", ["stereo", "zero", "half", "axes"], ["left"]),
        helper.make_node("Slice", ["stereo", "half", "end", "axes"], ["right_view"]),
        helper.make_node("Add", ["left", "right_view"], ["sum"]),
        helper.make_node("Conv", ["sum", "weight"], ["disparity"]),
    ]
    if case == "shape_arithmetic":
        nodes.append(helper.make_node("Shape", ["sum"], ["unused"]))
    graph = helper.make_graph(nodes, "fixture", inputs, [output], initializers)
    opset = OPSET + 1 if case == "opset" else OPSET
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "two_inputs", "shape", "shape_arithmetic"])
def test_export_must_keep_the_stereo_interface_without_index_arithmetic(
    tmp_path: Path, case: str, surgery: ModuleType
) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_export(surgery, case), path)

    if case == "valid":
        surgery.validate_export(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_export(path)


def test_disparity_comparison_rejects_shape_nan_and_excess_error(surgery: ModuleType) -> None:
    expected = np.random.RandomState(0).uniform(0, 192, (1, 1, 540, 960)).astype(np.float32)

    assert surgery.compare_disparity(expected, expected + 0.004) == pytest.approx(0.004, abs=1e-5)
    with pytest.raises(ValueError, match="differs from PyTorch by 0.02"):
        surgery.compare_disparity(expected, expected + 0.02)
    with pytest.raises(ValueError, match="non-finite"):
        surgery.compare_disparity(expected, np.full_like(expected, np.nan))
    with pytest.raises(ValueError, match="Unexpected"):
        surgery.compare_disparity(expected, expected[:, :, :-1])


def test_synthetic_frame_shifts_the_right_view_by_a_seeded_disparity(surgery: ModuleType) -> None:
    stereo = surgery.synthetic_frame(123)
    left, right = stereo[..., :960], stereo[..., 960:]

    shifts = [d for d in range(surgery.MAX_DISP) if np.array_equal(np.roll(left, -d, axis=3), right)]
    assert stereo.shape == (1, 3, 540, 1920)
    # Exactly one shift in 1..191 maps the left view onto the right, and the same seed gives the same frame.
    assert len(shifts) == 1 and 1 <= shifts[0] < 192
    assert np.array_equal(surgery.synthetic_frame(123), stereo)


def _readout_export(path: Path) -> Path:
    """A convolution, then the read-out's shape: a softmax regressed by a convolution, times a second softmax."""
    helper = onnx.helper
    nodes = [
        helper.make_node("Conv", ["stereo", "kernel"], ["features"], name="features"),
        helper.make_node("Softmax", ["features"], ["bins"], name="bins"),
        helper.make_node("Conv", ["bins", "kernel"], ["regressed"], name="regression"),
        helper.make_node("Softmax", ["features"], ["blend"], name="weights"),
        helper.make_node("Mul", ["regressed", "blend"], ["disparity"], name="product"),
    ]
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 1, 2, 2]) for name in ("stereo", "disparity")
    ]
    kernel = onnx.numpy_helper.from_array(np.ones((1, 1, 1, 1), np.float32), "kernel")
    onnx.save(helper.make_model(helper.make_graph(nodes, "fixture", values[:1], values[1:], [kernel])), path)
    return path


def test_readout_region_marks_the_softmaxes_and_everything_after_them_bf16(tmp_path: Path, surgery: ModuleType) -> None:
    source = _readout_export(tmp_path / "model.onnx")

    assert surgery.annotate_bf16_regions(source, tmp_path / "marked.onnx", ["readout"]) == 4

    marked = onnx.load(tmp_path / "marked.onnx")
    marks = [node for node in marked.graph.node if node.op_type == "AnnotatePrecision"]
    # The convolution before the softmaxes stays INT8; each read-out operation is followed by its mark, which
    # keeps the operation's value name, so the graph still returns "disparity".
    assert [node.name for node in marks] == [
        f"{name}/AnnotatePrecision" for name in ("bins", "regression", "weights", "product")
    ]
    assert {(node.domain, onnx.helper.get_attribute_value(node.attribute[0])) for node in marks} == {
        ("ai.sima", b"bfloat16")
    }
    assert marks[-1].output == ["disparity"] and [output.name for output in marked.graph.output] == ["disparity"]
    assert ("ai.sima", 1) in {(item.domain, item.version) for item in marked.opset_import}
    # The source stays unmarked: numerical fidelity runs it in ONNX Runtime.
    assert not [node for node in onnx.load(source).graph.node if node.op_type == "AnnotatePrecision"]
    with pytest.raises(ValueError, match="unknown BF16 regions"):
        surgery.annotate_bf16_regions(source, tmp_path / "marked.onnx", ["backbone"])


@pytest.mark.parametrize(
    ("build_id", "parameters", "error"),
    [
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["readout"]}, None),
        ("w8a8", {"precision": "w8a8"}, None),
        ("w8a8", {"precision": "w8a8", "bf16_regions": ["readout"]}, "mixed token"),
        ("w8a8_mixed", {"precision": "w8a8"}, "mixed token"),
        ("w16a16_mixed", {"precision": "w16a16", "bf16_regions": ["readout"]}, "W8A8 build"),
    ],
)
def test_bf16_regions_belong_to_a_w8a8_profile_named_mixed(
    build_id: str,
    parameters: dict,
    error: str | None,
    load_script: Callable[[str], ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_module = load_script("compile")
    monkeypatch.setattr(compile_module, "component_parameters", lambda model_dir, profile_id, component_id: parameters)

    if error is None:
        assert compile_module._bf16_regions(build_id) == parameters.get("bf16_regions", [])
    else:
        with pytest.raises(ValueError, match=error):
            compile_module._bf16_regions(build_id)


@pytest.mark.parametrize(("bf16_softmaxes", "accepted"), [(2, True), (1, False), (0, False)])
def test_mixed_build_fails_when_the_compiler_ignores_the_bf16_marks(
    bf16_softmaxes: int, accepted: bool, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    bf16 = compile_module.QuantizationPrecision.BFLOAT_16

    def node(operation: str, precision: object) -> SimpleNamespace:
        config = SimpleNamespace(quantization_precision=SimpleNamespace(value=precision))
        return SimpleNamespace(ir=SimpleNamespace(operation=type(operation, (), {})(), quant_config=config))

    nodes = [node("SoftmaxOp", bf16 if index < bf16_softmaxes else "int8") for index in range(2)]
    nodes += [node("ConvOp", bf16), node("ConvOp", "int8"), SimpleNamespace(ir=SimpleNamespace(operation=object()))]
    quantized = SimpleNamespace(_net=SimpleNamespace(iter_nodes_recursive=lambda: nodes))

    if accepted:
        compile_module._require_bf16_readout(quantized)
    else:
        with pytest.raises(RuntimeError, match="both read-out softmaxes"):
            compile_module._require_bf16_readout(quantized)


def _mpk_manifest(compile_module: ModuleType) -> dict:
    """A plugin chain of the compiled MPK: a cast of the frame, the MLA, then a slice and a cast back."""
    return {
        "input_nodes": [{"name": "stereo", "size": 1 * 3 * 540 * 1920 * 4}],
        "plugins": [
            {"sequence": 1, "processor": "EV74", "config_params": {"kernel": "cast_transform"}},
            {"sequence": 2, "processor": "MLA", "config_params": {}},
            {"sequence": 3, "processor": "EV74", "config_params": {"kernel": "slice_transform"}},
            {
                "sequence": 4,
                "processor": "EV74",
                "config_params": {"kernel": "cast_transform"},
                "output_nodes": [{"size": 1 * 1 * 540 * 960 * 4}],
            },
        ],
    }


def _write_mpk(path: Path, manifest: dict) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, content in (("model.elf", b"elf"), ("model_mpk.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("case", ["valid", "two_inputs", "output", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_stereo_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "two_inputs":
        manifest["input_nodes"].append({"name": "right", "size": 1 * 3 * 540 * 960 * 4})
    elif case == "output":
        plugins[-1]["output_nodes"][0]["size"] = 1 * 1 * 544 * 960 * 4
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 2, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 2, "processor": "A65", "config_params": {}})
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


def test_fidelity_compares_the_quantized_disparity_with_the_onnx_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    load_script: Callable[[str], ModuleType],
) -> None:
    compile_module = load_script("compile")
    fidelity = load_metadata(MODEL_DIR)["numerical_fidelity"]
    pairs = [("a/left.png", "a/right.png", "a/disparity.pfm"), ("b/left.png", "b/right.png", "b/disparity.pfm")]
    # Two side-by-side frames of 2x2 views, valued 1 and 2, whose left views' true disparity is 4 px everywhere.
    frames = np.stack([np.full((2, 4, 3), value, np.float32) for value in (1.0, 2.0)])
    truths = np.full((2, 2, 2), 4.0, np.float32)
    onnx_inputs = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, outputs: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            onnx_inputs.append(feeds["stereo"].shape)
            return [np.full((1, 1, 2, 2), feeds["stereo"][0, 0, 0, 0], np.float32)]

    def execute(inputs: dict, use_jax: bool) -> list[np.ndarray]:
        # The quantized graph returns the disparity channels-last, with its last pixel 1 px above the ONNX's.
        (frame,) = inputs.values()
        disparity = np.full((1, 2, 2, 1), frame[0, 0, 0, 0], np.float32)
        disparity[0, 1, 1, 0] += 1
        return [disparity]

    monkeypatch.setattr(compile_module, "selected_pairs", lambda dataset, seed, offset, samples: pairs)
    monkeypatch.setattr(compile_module, "read_pairs", lambda dataset, selected: (frames, truths))
    monkeypatch.setattr(compile_module, "ort", SimpleNamespace(InferenceSession=Session))

    report = compile_module.measure_fidelity(
        tmp_path / "model.onnx", SimpleNamespace(execute=execute), tmp_path / "cal.tgz", fidelity, "w16a16"
    )

    assert onnx_inputs == [(1, 3, 2, 4)] * 2
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("lightstereo_s", "w16a16", 2)
    (metric,) = report["metrics"]
    assert metric["name"] == "disparity"
    assert metric["max_absolute_error"] == 1.0
    assert metric["mean_absolute_error"] == 0.25
    # Each pair's EPE against the true 4 px: 3 and 2 px for the ONNX, 2.75 and 1.75 px for the quantized graph.
    assert "Scene Flow EPE on the 2 fidelity pairs: 2.500 px ONNX, 2.250 px quantized (-10.0%)" in caplog.text


def test_compile_reads_the_declared_build_parameters(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")
    model_dir, component_id = compile_module.MODEL_DIR, compile_module.COMPONENT_ID

    assert compile_module.component_parameters(model_dir, "w16a16", component_id) == {
        "precision": "w16a16",
        "calibration_bins": 2048,
        "calibration_samples": 16,
    }
    with pytest.raises(ValidationError, match="unknown build profile: w8a16"):
        compile_module.component_parameters(model_dir, "w8a16", component_id)
    with pytest.raises(ValueError, match="unsupported precision"):
        compile_module._quantization({"precision": "w4a16", "calibration_bins": 2048})


@pytest.mark.parametrize(
    ("precision", "activations", "weights", "per_token"),
    [
        ("w8a8", "quantization_scheme", "quantization_scheme", False),
        ("w8a16", "bfloat16_scheme", "quantization_scheme", True),
        ("w16a16", "bfloat16_scheme", "bfloat16_scheme", False),
    ],
)
def test_every_precision_calibrates_its_histogram_ranges(
    precision: str, activations: str, weights: str, per_token: bool, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")

    compile_module._quantization({"precision": precision, "calibration_bins": 2048})

    # The instance normalizations' square roots run as INT8 lookup tables even with BF16 activations, so each
    # precision takes its ranges from the calibration pairs rather than skipping calibration.
    compile_module.HistogramMSEMethod.assert_called_once_with(2048)
    params = compile_module.QuantizationParams.call_args.kwargs
    assert params["calibration_method"] is compile_module.HistogramMSEMethod.return_value
    assert params["activation_quantization_scheme"] is getattr(compile_module, activations).return_value
    assert params["weight_quantization_scheme"] is getattr(compile_module, weights).return_value
    assert params.get("per_token_quant", False) is per_token


def test_model_calibrates_every_profile_and_compares_disparity_on_scene_flow() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert [profile["id"] for profile in model["build_profiles"]] == ["w8a8_mixed", "w16a16"]
    assert {profile["components"][0]["calibration"] for profile in model["build_profiles"]} == {"scene-flow-cal"}
    largest = max(profile["components"][0]["parameters"]["calibration_samples"] for profile in model["build_profiles"])
    fidelity = model["numerical_fidelity"]
    # Calibration takes the first pairs of the seeded order, so the fidelity pairs must start after them.
    assert fidelity["selection"]["offset"] >= largest
    assert fidelity["dataset"]["variant"] == "cal" and fidelity["outputs"] == ["disparity"]


def test_compiled_build_holds_the_audited_mpk_and_a_disparity_fidelity_report(
    compiled: Path, build_profile: str, load_script: Callable[[str], ModuleType]
) -> None:
    (archive,) = (compiled / "model").glob("*_mpk.tar.gz")
    load_script("compile").audit_mpk(archive)
    report = json.loads((compiled / "fidelity.json").read_text(encoding="utf-8"))
    (metric,) = report["metrics"]
    assert metric["name"] == "disparity"
    assert np.isfinite([metric["mean_absolute_error"], metric["cosine_similarity"]]).all()
