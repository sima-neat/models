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
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / "models" / "superglue_outdoor"


@pytest.fixture
def surgery(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    return {
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


@pytest.mark.parametrize(
    "case", ["valid", "bf16_cast", "int8_dequantization", "inputs", "outputs", "decoder", "two_mla", "host"]
)
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
    elif case == "int8_dequantization":
        plugins[-1]["config_params"]["kernel"] = "dequantization_transform"
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
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case in ("valid", "bf16_cast", "int8_dequantization"):
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
    pair = compile_module.synthetic_pair()
    shared, target = pair.shared, pair.target
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
        load_script("compile")._parameters("w4a16")


@pytest.mark.parametrize(("precision", "per_token"), [("w16a16", False), ("w8a16", True)])
def test_quantization_skips_calibration_for_bf16_activations(
    load_script: Callable[[str], ModuleType], precision: str, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization({"precision": precision})

    arguments = recipe.QuantizationParams.call_args.kwargs
    assert arguments["calibration_method"] is recipe.SkipCalibration.return_value
    assert arguments["activation_quantization_scheme"] is recipe.bfloat16_scheme.return_value
    assert arguments.get("per_token_quant", False) is per_token


def test_w8a8_calibrates_int8_activations_by_histogram_mse(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    recipe._quantization({"precision": "w8a8", "calibration_bins": 2048})

    arguments = recipe.QuantizationParams.call_args.kwargs
    recipe.HistogramMSEMethod.assert_called_once_with(2048)
    assert arguments["calibration_method"] is recipe.HistogramMSEMethod.return_value
    assert recipe.quantization_scheme.call_args_list[0].kwargs == {"asymmetric": True, "per_channel": False, "bits": 8}


def test_every_w8a8_profile_calibrates_and_names_its_bf16_regions(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    mixed = {}
    for profile in load_metadata(MODEL_DIR)["build_profiles"]:
        parameters = recipe._parameters(profile["id"])
        regions = recipe._bf16_regions(profile["id"], parameters)
        if parameters["precision"] == "w8a8":
            assert parameters["calibration_samples"] == 32 and parameters["calibration_bins"] == 2048
        if regions:
            mixed[profile["id"]] = regions
    assert mixed == {"w8a8_mixed": list(recipe.BF16_REGIONS)}


@pytest.mark.parametrize(
    ("profile", "parameters", "message"),
    [
        ("w8a8", {"precision": "w8a8", "bf16_regions": ["attention"]}, "mixed token"),
        ("w8a8_mixed", {"precision": "w8a8"}, "mixed token"),
        ("w16a16_mixed", {"precision": "w16a16", "bf16_regions": ["attention"]}, "W8A8 build"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["attention", "attention"]}, "distinct"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["head"]}, "unknown BF16 regions"),
    ],
)
def test_bf16_regions_must_be_known_regions_of_a_mixed_w8a8_profile(
    load_script: Callable[[str], ModuleType], profile: str, parameters: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_script("compile")._bf16_regions(profile, parameters)


@pytest.mark.parametrize(("softmaxes", "accepted"), [(36, True), (0, False), (35, False)])
def test_a_mixed_build_must_hold_the_bf16_softmaxes(
    load_script: Callable[[str], ModuleType], softmaxes: int, accepted: bool
) -> None:
    recipe = load_script("compile")
    recipe.QuantizationPrecision.BFLOAT_16 = "bfloat16"

    class SoftmaxOp:
        pass

    def node(precision: str) -> SimpleNamespace:
        config = SimpleNamespace(quantization_precision=SimpleNamespace(value=precision))
        return SimpleNamespace(ir=SimpleNamespace(operation=SoftmaxOp(), quant_config=config))

    nodes = [node("bfloat16")] * softmaxes + [node("int8")] * (36 - softmaxes)
    quantized = SimpleNamespace(_net=SimpleNamespace(iter_nodes_recursive=lambda: nodes))

    if accepted:
        recipe._require_bf16_regions(quantized, ["attention"])
    else:
        with pytest.raises(RuntimeError, match="36 softmaxes"):
            recipe._require_bf16_regions(quantized, ["attention"])


@pytest.mark.parametrize("precision", ["w16a16", "w8a16"])
def test_compile_quantizes_without_calibration_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: {"precision": precision})
    monkeypatch.setattr(recipe, "check_quantized_matches", lambda source, quantized, minimum: 1.0)

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, None)

    assert recipe.load_model.return_value.quantize.call_args.kwargs["calibration_data"] is None


def _write_cal_dataset(root: Path, counts: dict[str, int]) -> None:
    """A yfcc100m-keypoints cal directory with one pair per two consecutive images."""
    rng = np.random.default_rng(0)
    (root / "annotations").mkdir(parents=True)
    names = list(counts)
    for name, count in counts.items():
        directory = root / "cal" / Path(name).with_suffix("")
        directory.mkdir(parents=True)
        np.save(directory / "keypoints.npy", rng.uniform(0, 480, (count, 2)).astype(np.float32))
        np.save(directory / "scores.npy", rng.uniform(0, 1, count).astype(np.float32))
        np.save(directory / "descriptors.npy", rng.normal(0, 1, (count, 256)).astype(np.float32))
    images = [f"{name} 640 480 0 {count}" for name, count in counts.items()]
    (root / "annotations" / "images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
    pairs = [f"{first} {second} 0 0" for first, second in zip(names, names[1:])]
    (root / "annotations" / "pairs_cal.txt").write_text("\n".join(pairs) + "\n", encoding="utf-8")


def test_calibration_stacks_the_pairs_whose_images_have_512_keypoints(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    _write_cal_dataset(tmp_path, {"a/1.jpg": 512, "a/2.jpg": 512, "a/3.jpg": 512, "a/4.jpg": 300})

    samples = calibration.calibration_inputs(tmp_path, 2)

    assert {name: value.shape for name, value in samples.items()} == {
        name: (2, *shape[1:]) for name, shape in calibration.INPUT_SHAPES.items()
    }
    assert all(value.dtype == np.float32 for value in samples.values())
    first, _ = calibration.seeded_pairs(tmp_path)[0]
    keypoints = np.load(tmp_path / "cal" / Path(first).with_suffix("") / "keypoints.npy")
    np.testing.assert_allclose(samples["keypoints0"][0, 0], (keypoints - [320, 240]) / 448, rtol=1e-5)
    # The pair with the 300-keypoint image is left out, so a third pair does not exist.
    with pytest.raises(ValueError, match="requires 3 pairs"):
        calibration.calibration_inputs(tmp_path, 3)


def test_synthetic_pair_is_channels_last_per_input(load_script: Callable[[str], ModuleType]) -> None:
    compile_module = load_script("compile")

    samples = compile_module.synthetic_pair().inputs

    for name, shape in compile_module.INPUT_SHAPES.items():
        batch, channels, height, width = shape
        assert samples[name].shape == (batch, height, width, channels)
        assert samples[name].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(samples["descriptors1"][0, 0], axis=1), 1, rtol=1e-5)


def test_model_preserves_compilation_and_licence_settings() -> None:
    model = load_metadata(MODEL_DIR)
    assert model["components"][0]["interface"]["format"] == "onnx"
    assert "w16a16" in [profile["id"] for profile in model["build_profiles"]]
    assert model["license"]["redistribution"] == "restricted"
    assert model["license"]["spdx"] == "LicenseRef-MagicLeap-Noncommercial-Research"


def test_sinkhorn_and_extraction_find_clear_matches(surgery: ModuleType) -> None:
    size = 6
    couplings = np.full((size + 1, size + 1), -2.0)
    permutation = np.array([3, 0, 5, 1, 4, 2])
    couplings[np.arange(size), permutation] = 8.0
    couplings[4] = -2.0  # keypoint 4 has no partner: its mass goes to the dustbin
    couplings[:, -1] = couplings[-1] = 1.0
    log_transport = surgery.log_sinkhorn(couplings[None], surgery.SINKHORN_ITERATIONS)

    matches, scores = surgery.extract_matches(log_transport)
    # Extraction compares keypoints only, so the two leftovers still pair up, with a near-zero score...
    assert matches.tolist() == [3, 0, 5, 1, 4, 2]
    assert scores[4] < 0.1 and (np.delete(scores, 4) > 0.9).all()
    # ...and a match threshold drops them.
    assert surgery.extract_matches(log_transport, threshold=0.2)[0].tolist() == [3, 0, 5, 1, -1, 2]


def test_extraction_requires_mutual_best_matches(surgery: ModuleType) -> None:
    log_transport = np.log(np.array([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))

    matches, _ = surgery.extract_matches(log_transport)

    # Keypoint 0 prefers 0, but image-1 keypoint 0 prefers keypoint 1: only 1 -> 0 is mutual.
    assert matches.tolist() == [-1, 0]


def test_exported_layout_round_trips(surgery: ModuleType) -> None:
    pair = surgery.synthetic_pair(7)
    inputs = surgery.to_inputs(pair)

    assert {name: list(value.shape) for name, value in inputs.items()} == surgery.INPUTS
    np.testing.assert_array_equal(inputs["descriptors0"][0, :, 0, :].T, pair["descriptors0"])
    np.testing.assert_array_equal(inputs["scores1"][0, 0, 0], pair["scores1"])
    couplings = np.arange((surgery.KEYPOINTS + 1) ** 2, dtype=np.float32).reshape(surgery.OUTPUT["couplings"])
    matrix = surgery.couplings_matrix(couplings)
    # couplings[0, j, 0, i] is row i (image 0), column j (image 1).
    assert matrix[2, 5] == couplings[0, 5, 0, 2]


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path: Path, case: str, surgery: ModuleType) -> None:
    helper = onnx.helper
    values = [
        helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in surgery.INTERFACE.items()
    ]
    if case == "order":
        values[0], values[1] = values[1], values[0]
    elif case == "dtype":
        values[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        values[-1].type.tensor_type.shape.dim[1].dim_value = 100
    inputs, outputs = values[:-1], values[-1:]
    output_shape = surgery.OUTPUT["couplings"]
    node = helper.make_node(
        "Constant",
        [],
        [outputs[0].name],
        value=helper.make_tensor("value", onnx.TensorProto.FLOAT, output_shape, np.zeros(int(np.prod(output_shape)))),
    )
    graph = helper.make_graph([node], "fixture", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13 if case == "opset" else 17)])
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    if case == "valid":
        surgery.validate_export(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_export(path)


def _attention_graph() -> onnx.ModelProto:
    """A query and key convolution, one attention, and the convolution with a Relu that reads it through a Split."""
    helper = onnx.helper
    weight = onnx.numpy_helper.from_array(np.ones((4, 4, 1, 1), np.float32), "w")
    sizes = onnx.numpy_helper.from_array(np.array([2, 2], np.int64), "sizes")
    attention = "/layers.0/attention"
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["q"], name="/q/Conv"),
        helper.make_node("Conv", ["x", "w"], ["k"], name="/k/Conv"),
        helper.make_node("Einsum", ["q", "k"], ["scores"], name=f"{attention}/Einsum", equation="nchw,nchq->nqhw"),
        helper.make_node("Softmax", ["scores"], ["probabilities"], name=f"{attention}/Softmax", axis=1),
        helper.make_node(
            "Einsum", ["probabilities", "k"], ["context"], name=f"{attention}/Einsum_1", equation="nchw,nqhc->nqhw"
        ),
        helper.make_node("Split", ["context", "sizes"], ["half0", "half1"], name=f"{attention}/Split", axis=2),
        helper.make_node("Concat", ["half0", "half1"], ["merged"], name=f"{attention}/Concat", axis=1),
        helper.make_node("Conv", ["merged", "w"], ["projected"], name="/output/Conv"),
        helper.make_node("Relu", ["projected"], ["y"], name="/output/Relu"),
    ]
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 4, 4, 4]) for name in ("x", "y")]
    graph = helper.make_graph(nodes, "fixture", values[:1], values[1:], [weight, sizes])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_a_bf16_region_ends_at_the_first_operations_that_read_it(
    tmp_path: Path, surgery: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(surgery.BF16_REGIONS, "attention", (surgery.BF16_REGIONS["attention"][0], 3))
    source, annotated = tmp_path / "model.onnx", tmp_path / "annotated.onnx"
    onnx.save(_attention_graph(), source)

    counts = surgery.annotate_bf16_regions(source, annotated, ["attention"])

    assert counts == {"bfloat16": 3, "int8": 1}
    nodes = onnx.load(annotated).graph.node
    marks = {node.input[0]: node.attribute[0].s.decode() for node in nodes if node.op_type == "AnnotatePrecision"}
    assert marks == {
        "scores/unannotated": "bfloat16",
        "probabilities/unannotated": "bfloat16",
        "context/unannotated": "bfloat16",
        # The compiler fuses the Relu into its convolution, so the mark follows the Relu.
        "y/unannotated": "int8",
    }
    assert [value.name for value in onnx.load(annotated).graph.output] == ["y"]


def test_a_region_must_match_its_operation_count(tmp_path: Path, surgery: ModuleType) -> None:
    with pytest.raises(ValueError, match="expected 108 attention operations"):
        surgery.precision_marks(_attention_graph(), ["attention"])


def test_initializer_sharing_merges_identical_weights(surgery: ModuleType) -> None:
    helper = onnx.helper
    weight = onnx.numpy_helper.from_array(np.ones((2, 2), np.float32), "w0")
    copy = onnx.numpy_helper.from_array(np.ones((2, 2), np.float32), "w1")
    nodes = [helper.make_node("MatMul", ["x", "w0"], ["y0"]), helper.make_node("MatMul", ["x", "w1"], ["y1"])]
    values = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [2, 2]) for name in ("x", "y0", "y1")]
    model = helper.make_model(helper.make_graph(nodes, "fixture", values[:1], values[1:], [weight, copy]))

    shared = surgery.share_initializers(model)

    assert [item.name for item in shared.graph.initializer] == ["w0"]
    assert [node.input[1] for node in shared.graph.node] == ["w0", "w0"]


def test_mla_layout_matcher_equals_the_transformers_layers(surgery: ModuleType) -> None:
    """The 4D rewrite with random weights must give the transformers layers' couplings."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.SuperGlueForKeypointMatching(transformers.SuperGlueConfig(**surgery.CONFIG)).eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean.uniform_(-0.2, 0.2)
            module.running_var.uniform_(0.5, 1.5)
    pair = surgery.synthetic_pair(11)
    with torch.inference_mode():
        reference = surgery.reference_matcher(model)
        expected = reference(*(torch.from_numpy(pair[name])[None] for name in surgery.INPUTS))
        actual = surgery.build_matcher(model)(
            *(torch.from_numpy(value) for value in surgery.to_inputs(pair).values())
        )

    np.testing.assert_allclose(surgery.couplings_matrix(actual.numpy()), expected[0].numpy(), atol=1e-4)
