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
from PIL import Image


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), color).save(stream, format="PNG")
    return stream.getvalue()


def test_preprocessing_stretches_to_the_input_and_normalizes_with_imagenet_statistics(
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")

    result = calibration.preprocess_image(_png(500, 375, (255, 0, 0)))

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[320, 320], (np.array([1.0, 0.0, 0.0]) - calibration.MEAN) / calibration.STD)


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path,
    load_script: Callable[[str], ModuleType],
) -> None:
    calibration = load_script("calibration")
    archive = tmp_path / "coco.tgz"
    with tarfile.open(archive, "w:gz") as stream:
        for split in ("cal2017", "val2017"):
            for index in range(6):
                info = tarfile.TarInfo(f"mini/{split}/{index:012d}.webp")
                info.size = 1
                stream.addfile(info, io.BytesIO(b"x"))

    names = calibration.calibration_names(archive, 4)

    assert names == calibration.calibration_names(archive, 4)
    assert len(set(names)) == 4
    assert all(Path(name).parent.name == "cal2017" for name in names)
    with pytest.raises(ValueError, match="cannot select 7"):
        calibration.calibration_names(archive, 7)


def _model(nodes, inputs, outputs, initializers) -> onnx.ModelProto:
    helper = onnx.helper
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in inputs.items()],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in outputs.items()],
        [onnx.numpy_helper.from_array(value, name) for name, value in initializers.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _run(model: onnx.ModelProto, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    ort = pytest.importorskip("onnxruntime")
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)


def _batch_norm_graph(scale_shape=(1, 4, 1, 1)) -> onnx.ModelProto:
    """A bias-free Conv followed by a frozen batch norm's Mul and Add, then a Relu."""
    rng = np.random.default_rng(0)
    helper = onnx.helper
    nodes = [
        helper.make_node("Conv", ["x", "w"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Mul", ["conv", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "shift"], ["shifted"]),
        helper.make_node("Relu", ["shifted"], ["y"]),
    ]
    initializers = {
        "w": rng.standard_normal((4, 3, 3, 3), dtype=np.float32),
        "scale": rng.uniform(0.5, 2.0, scale_shape).astype(np.float32),
        "shift": rng.standard_normal(scale_shape, dtype=np.float32),
    }
    return _model(nodes, {"x": [1, 3, 8, 8]}, {"y": [1, 4, 8, 8]}, initializers)


def test_batch_norms_fold_into_the_convolution_without_changing_outputs(
    load_script: Callable[[str], ModuleType],
) -> None:
    surgery = load_script("surgery")
    source = _batch_norm_graph()
    feeds = {"x": np.random.default_rng(1).standard_normal((1, 3, 8, 8), dtype=np.float32)}

    folded = surgery._fold_frozen_batch_norms(onnx.ModelProto.FromString(source.SerializeToString()), 1)

    assert [node.op_type for node in folded.graph.node] == ["Conv", "Relu"]
    np.testing.assert_allclose(_run(folded, feeds)[0], _run(source, feeds)[0], rtol=1e-5, atol=1e-5)


def test_batch_norm_folding_rejects_an_unexpected_count_or_shape(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    with pytest.raises(ValueError, match="expected 2 frozen batch norms, found 1"):
        surgery._fold_frozen_batch_norms(_batch_norm_graph(), 2)
    with pytest.raises(ValueError, match="per-channel batch norm"):
        surgery._fold_frozen_batch_norms(_batch_norm_graph(scale_shape=(1, 1, 1, 1)), 1)


def _attention_graph(queries: int, keys: int, mask: np.ndarray | None = None) -> onnx.ModelProto:
    """One exported DETR attention: per-head Q/K/V, scaled scores, an optional additive mask, softmax, projection."""
    heads, dim = 2, 4
    channels = heads * dim
    rng = np.random.default_rng(2)
    helper = onnx.helper
    initializers = {
        **{name: rng.standard_normal((channels, channels), dtype=np.float32) for name in ("wq", "wk", "wv", "wo")},
        "bo": rng.standard_normal(channels, dtype=np.float32),
        "scale": np.array(dim**-0.5, np.float32),
        "heads_q": np.array([1, queries, heads, dim], np.int64),
        "heads_k": np.array([1, keys, heads, dim], np.int64),
        "merged": np.array([1, queries, channels], np.int64),
    }
    nodes = [
        helper.make_node("MatMul", ["q_in", "wq"], ["q2"]),
        helper.make_node("Reshape", ["q2", "heads_q"], ["q3"]),
        helper.make_node("Transpose", ["q3"], ["q"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["kv_in", "wk"], ["k2"]),
        helper.make_node("Reshape", ["k2", "heads_k"], ["k3"]),
        helper.make_node("Transpose", ["k3"], ["k"], perm=[0, 2, 3, 1]),
        helper.make_node("MatMul", ["kv_in", "wv"], ["v2"]),
        helper.make_node("Reshape", ["v2", "heads_k"], ["v3"]),
        helper.make_node("Transpose", ["v3"], ["v"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q", "k"], ["scores"]),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
    ]
    logits = "scaled"
    if mask is not None:
        initializers["mask"] = mask.astype(np.float32)
        nodes.append(helper.make_node("Add", ["scaled", "mask"], ["masked"]))
        logits = "masked"
    nodes += [
        helper.make_node("Softmax", [logits], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "v"], ["values"]),
        helper.make_node("Transpose", ["values"], ["merge"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["merge", "merged"], ["flat"]),
        helper.make_node("MatMul", ["flat", "wo"], ["projected"]),
        helper.make_node("Add", ["bo", "projected"], ["y"]),
    ]
    inputs = {"q_in": [1, queries, channels], "kv_in": [1, keys, channels]}
    return _model(nodes, inputs, {"y": [1, queries, channels]}, initializers)


@pytest.mark.parametrize(("queries", "keys"), [(5, 5), (3, 7)], ids=["self", "cross"])
def test_attention_rewrite_keeps_outputs_with_the_softmax_over_channels(
    load_script: Callable[[str], ModuleType], queries: int, keys: int
) -> None:
    surgery = load_script("surgery")
    source = _attention_graph(queries, keys)
    rng = np.random.default_rng(3)
    feeds = {
        "q_in": rng.standard_normal((1, queries, 8), dtype=np.float32),
        "kv_in": rng.standard_normal((1, keys, 8), dtype=np.float32),
    }

    rewritten = surgery._optimize_attention(onnx.ModelProto.FromString(source.SerializeToString()), 1)

    operations = [node.op_type for node in rewritten.graph.node]
    assert operations.count("Einsum") == 2 and "Conv" in operations and operations.count("MatMul") == 3
    (softmax,) = [node for node in rewritten.graph.node if node.op_type == "Softmax"]
    assert onnx.helper.get_attribute_value(softmax.attribute[0]) == 1
    np.testing.assert_allclose(_run(rewritten, feeds)[0], _run(source, feeds)[0], rtol=1e-4, atol=1e-4)


def test_only_an_all_zero_attention_mask_is_removed(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    removed = surgery._remove_zero_masks(_attention_graph(3, 7, np.zeros((1, 1, 3, 7))), 1)
    assert "mask" not in {node.input[1] for node in removed.graph.node if node.op_type == "Add"}

    excluding = np.zeros((1, 1, 3, 7))
    excluding[..., -1] = -np.inf
    with pytest.raises(ValueError, match="excludes positions"):
        surgery._remove_zero_masks(_attention_graph(3, 7, excluding), 1)


def test_surgery_requires_the_folded_interface(tmp_path: Path, load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")
    helper = onnx.helper
    outputs = {name: shape for name, shape in surgery.OUTPUT_SHAPES.items()}
    nodes = [
        helper.make_node("Constant", [], [name], value=onnx.numpy_helper.from_array(np.zeros(shape, np.float32)))
        for name, shape in outputs.items()
    ]
    source = tmp_path / "source.onnx"
    onnx.save(_model(nodes, {surgery.INPUT: surgery.INPUT_SHAPE}, outputs, {}), source)
    with pytest.raises(ValueError, match="fold it with onnxsim first"):
        surgery.prepare_model(source, tmp_path / "out")

    onnx.save(_model(nodes, {surgery.INPUT: [1, 3, 800, 800]}, outputs, {}), source)
    with pytest.raises(ValueError, match="unexpected input interface"):
        surgery.prepare_model(source, tmp_path / "out")


def _mpk_manifest(compile_module: ModuleType) -> dict:
    return {
        "input_nodes": [{"name": "input", "size": 1 * 3 * 640 * 640 * 4}],
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


@pytest.mark.parametrize("case", ["valid", "reordered", "two_mla", "host"])
def test_mpk_audit_requires_one_mla_segment_and_the_declared_interface(
    tmp_path: Path, case: str, load_script: Callable[[str], ModuleType]
) -> None:
    compile_module = load_script("compile")
    manifest = _mpk_manifest(compile_module)
    plugins = manifest["plugins"]
    if case == "reordered":
        outputs = plugins[-1]["output_nodes"]
        outputs[0], outputs[1] = outputs[1], outputs[0]
    elif case == "two_mla":
        plugins.insert(2, {"sequence": 1, "processor": "MLA", "config_params": {}})
    elif case == "host":
        plugins.insert(2, {"sequence": 1, "processor": "A65", "config_params": {}})
    archive = tmp_path / "model_mpk.tar.gz"
    _write_mpk(archive, manifest)

    if case == "valid":
        compile_module.audit_mpk(archive)
    else:
        with pytest.raises(RuntimeError):
            compile_module.audit_mpk(archive)


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        ("w16a16", {"precision": "w16a16"}),
        ("w8a16", {"precision": "w8a16", "per_token": True}),
        ("w8a8", {"precision": "w8a8", "calibration_bins": 2048, "calibration_samples": 50}),
    ],
)
def test_compile_reads_the_declared_build_parameters(
    load_script: Callable[[str], ModuleType], profile_id: str, expected: dict
) -> None:
    assert load_script("compile")._parameters(profile_id) == expected


@pytest.mark.parametrize(
    ("precision", "calibrates", "per_token"), [("w8a8", True, False), ("w8a16", False, True), ("w16a16", False, False)]
)
def test_quantization_calibrates_only_int8_activations(
    load_script: Callable[[str], ModuleType], precision: str, calibrates: bool, per_token: bool
) -> None:
    recipe = load_script("compile")

    recipe._quantization(precision, 2048 if calibrates else None, per_token)

    arguments = recipe.QuantizationParams.call_args.kwargs
    calibration = recipe.HistogramMSEMethod.return_value if calibrates else recipe.SkipCalibration.return_value
    assert arguments["calibration_method"] is calibration
    assert arguments.get("per_token_quant", False) is per_token


@pytest.mark.parametrize("precision", ["w16a16", "w8a16", "w8a8"])
def test_only_the_int8_build_quantizes_with_calibration_data_and_only_bf16_keeps_fp32_layer_norms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType], precision: str
) -> None:
    recipe = load_script("compile")
    parameters = {"precision": precision}
    if precision == "w8a8":
        parameters |= {"calibration_bins": 2048, "calibration_samples": 2}
    monkeypatch.setattr(recipe, "_parameters", lambda profile_id: parameters)
    monkeypatch.setattr(recipe, "calibration_images", lambda dataset, count: np.zeros((count, 640, 640, 3), np.float32))
    monkeypatch.setattr(recipe, "prepare_model", lambda source, output_dir: output_dir / "rewritten.onnx")

    with pytest.raises(RuntimeError, match="expected one MPK"):
        recipe.compile_model(tmp_path / "model.onnx", tmp_path / "out", precision, tmp_path / "coco.tgz")

    quantize = recipe.load_model.return_value.quantize
    data = quantize.call_args.kwargs["calibration_data"]
    assert (data is recipe.convert_data_generator_to_iterable.return_value) is (precision == "w8a8")
    compile_options = quantize.return_value.compile.call_args.kwargs
    assert compile_options["layer_norm_use_fp32_intermediates"] is (precision != "w8a8")
