from __future__ import annotations

import io
import random
import sys
import tarfile
from collections import Counter
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from model_registry.errors import ValidationError
from onnx import TensorProto, helper, numpy_helper
from PIL import Image


def _jpeg(width: int, height: int, colour: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(stream, format="JPEG", quality=100)
    return stream.getvalue()


def _write_archive(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(f"./{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def test_preprocessing_stretches_to_512_and_normalizes(load_script: Callable[[str], ModuleType]) -> None:
    calibration = load_script("calibration")
    # A portrait image, red above blue, stretches without cropping: each colour keeps half the rows.
    image = Image.new("RGB", (100, 300), (255, 0, 0))
    image.paste((0, 0, 255), (0, 150, 100, 300))
    stream = io.BytesIO()
    image.save(stream, format="PNG")

    result = calibration.preprocess_image(stream.getvalue())

    assert result.shape == (512, 512, 3)
    assert result.dtype == np.float32
    red = (np.asarray([1, 0, 0], np.float32) - calibration.MEAN) / calibration.STD
    blue = (np.asarray([0, 0, 1], np.float32) - calibration.MEAN) / calibration.STD
    np.testing.assert_allclose(result[200, 256], red, atol=1e-5)
    np.testing.assert_allclose(result[312, 256], blue, atol=1e-5)


def test_calibration_draws_a_seeded_sample_from_cal2017_only(
    tmp_path: Path, load_script: Callable[[str], ModuleType]
) -> None:
    calibration = load_script("calibration")
    files = {f"cal2017/{index:02d}.jpg": _jpeg(64, 48) for index in range(6)}
    files["val2017/x.jpg"] = _jpeg(64, 48)
    archive = _write_archive(tmp_path / "coco.tgz", files)

    samples = calibration.calibration_samples(archive, 4)

    assert samples.shape == (4, 512, 512, 3)
    with pytest.raises(ValueError, match="cannot select 7 calibration images from 6"):
        calibration.calibration_samples(archive, 7)


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


@pytest.mark.parametrize("profile_id", ["w16a16", "w8a8", "w8a8_mixed"])
def test_compile_reads_each_declared_build_profile(load_script: Callable[[str], ModuleType], profile_id: str) -> None:
    recipe = load_script("compile")
    parameters = recipe.component_parameters(recipe.MODEL_DIR, profile_id, recipe.COMPONENT_ID)

    assert parameters["precision"] == profile_id.removesuffix("_mixed")
    if profile_id.startswith("w8a8"):
        assert (parameters["calibration_bins"], parameters["calibration_samples"]) == (2048, 20)
    expected_regions = ["attention_core", "norms", "gelu"] if profile_id == "w8a8_mixed" else []
    assert recipe._bf16_regions(profile_id, parameters) == expected_regions


def test_bf16_regions_belong_to_a_w8a8_profile_marked_mixed(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    regions = ["attention_core", "norms"]
    assert recipe._bf16_regions("w8a8_mixed", {"precision": "w8a8", "bf16_regions": regions}) == regions
    assert recipe._bf16_regions("w8a8", {"precision": "w8a8"}) == []
    for profile_id, parameters, message in [
        ("w8a8", {"precision": "w8a8", "bf16_regions": ["norms"]}, "mixed token"),
        ("w8a8_mixed", {"precision": "w8a8"}, "mixed token"),
        ("w16a16_mixed", {"precision": "w16a16", "bf16_regions": ["norms"]}, "W8A8 build"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["norms", "norms"]}, "distinct"),
        ("w8a8_mixed", {"precision": "w8a8", "bf16_regions": ["decoder"]}, "unknown BF16 regions"),
    ]:
        with pytest.raises(ValueError, match=message):
            recipe._bf16_regions(profile_id, parameters)


def test_bf16_check_requires_each_marked_region_to_quantize_bf16(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    regions = ["attention_core", "norms", "gelu"]
    # Twenty LayerNorm normalizations and the 24 marked LayerNorms' scale and shift convolutions.
    norms = {"LayerNormOp": 20, "ConvAddActivationOp": 24}
    recipe._require_bf16_regions(Counter({"SoftmaxOp": 8, **norms, "GeluOp": 8, "AddOp": 16}), regions)
    # A GELU that AFE does not fuse is identified by its Erf.
    recipe._require_bf16_regions(Counter({"SoftmaxOp": 8, **norms, "ErfOp": 8, "DivideOp": 8}), regions)
    recipe._require_bf16_regions(Counter(norms), ["norms"])
    # AFE ignored the softmax marks, left some residual-stream LayerNorms INT8, ran a Linear layer BF16, or ran an
    # unmarked region BF16.
    for bf16, marked in [
        (Counter({**norms, "GeluOp": 8}), regions),
        (Counter({"SoftmaxOp": 8, "LayerNormOp": 16, "ConvAddActivationOp": 24, "GeluOp": 8}), regions),
        (Counter({"SoftmaxOp": 8, "LayerNormOp": 20, "ConvAddActivationOp": 25, "GeluOp": 8}), regions),
        (Counter({**norms, "SoftmaxOp": 8}), ["norms"]),
        (Counter(), regions),
    ]:
        with pytest.raises(RuntimeError, match="expected BF16 operations"):
            recipe._require_bf16_regions(bf16, marked)


def _quantized_node(name: str, kind: str, precision: Any, sources: list[str]) -> Any:
    config = SimpleNamespace(quantization_precision=SimpleNamespace(value=precision))
    return SimpleNamespace(
        name=name,
        source_node_names=sources,
        ir=SimpleNamespace(operation=type(kind, (), {})(), quant_config=config),
    )


class _Precision(Enum):
    # AFE's QuantizationPrecision values, for checks that run without the Model Compiler.
    INT_8 = "int8"
    BFLOAT_16 = "bfloat16"


def test_marked_precision_check_names_each_operation_off_its_mark(
    load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = load_script("compile")
    monkeypatch.setattr(recipe, "QuantizationPrecision", _Precision)
    int8, bf16 = _Precision.INT_8, _Precision.BFLOAT_16
    # A marked LayerNorm lowered as its normalization, which takes its INT8 input's precision, and its BF16 scale and
    # shift, which AFE names no source for; AFE then ran the Linear layer reading it BF16 despite its INT8 mark, and the
    # softmax INT8 despite its BF16 mark. The softmax's reader has no precision of its own (AFE's EmptyValue).
    nodes = [
        _quantized_node("conv", "ConvAddActivationOp", int8, ["/model/embed/Conv"]),
        _quantized_node("norm", "LayerNormOp", int8, ["/model/embed/LayerNormalization"]),
        _quantized_node("affine", "ConvAddActivationOp", bf16, ["Not available"]),
        _quantized_node("query", "ConvAddActivationOp", bf16, ["/model/query/MatMul", "/model/query/Add"]),
        _quantized_node("softmax", "SoftmaxOp", int8, ["/model/Softmax"]),
        _quantized_node("context", "BatchMatmulOp", SimpleNamespace(empty=""), ["/model/MatMul_1"]),
        _quantized_node("scale", "ConstantOp", bf16, []),
    ]
    segment = SimpleNamespace(name="MLA_0", ir=SimpleNamespace())
    quantized = SimpleNamespace(_net=SimpleNamespace(iter_nodes_recursive=lambda: iter([segment, *nodes])))
    marks = {"/model/embed/LayerNormalization": "bfloat16", "/model/query/Add": "int8", "/model/Softmax": "bfloat16"}

    with pytest.raises(RuntimeError, match="marked precision") as failed:
        recipe._check_precision(quantized, marks, ["attention_core", "norms", "gelu"])

    assert str(failed.value).splitlines()[1:] == [
        "query (ConvAddActivationOp): bfloat16, expected ['int8'] from ['/model/query/MatMul', '/model/query/Add']",
        "softmax (SoftmaxOp): int8, expected ['bfloat16'] from ['/model/Softmax']",
    ]


def test_compile_rejects_an_undeclared_build_profile(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    with pytest.raises(ValidationError, match="unknown build profile: w8a16"):
        recipe.component_parameters(recipe.MODEL_DIR, "w8a16", recipe.COMPONENT_ID)


def test_logits_come_back_in_onnx_order(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")
    logits = np.random.RandomState(0).normal(size=recipe.LOGITS_SHAPE)

    np.testing.assert_array_equal(recipe._logits([logits.transpose(0, 2, 3, 1)]), logits)
    np.testing.assert_array_equal(recipe._logits([logits]), logits)
    with pytest.raises(RuntimeError, match="unexpected logits shape"):
        recipe._logits([logits[:, :, :64]])


def test_fidelity_compares_the_quantized_logits_with_the_onnx_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_script: Callable[[str], ModuleType]
) -> None:
    recipe = load_script("compile")
    fidelity = recipe.load_metadata(recipe.MODEL_DIR)["numerical_fidelity"]
    images = {"val2017/a.jpg": 1.0, "val2017/b.jpg": 2.0}
    onnx_inputs = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            onnx_inputs.append(feeds["pixel_values"].shape)
            return [np.full(recipe.LOGITS_SHAPE, feeds["pixel_values"][0, 0, 0, 0], np.float32)]

    def channels_last_logits(value: float) -> np.ndarray:
        logits = np.full((1, 128, 128, 150), value, np.float32)
        logits[..., -1] += 1
        return logits

    candidate = MagicMock()
    # The compiler returns the logits channels-last. Each image's last class comes back 1 too high.
    candidate.execute.side_effect = [[channels_last_logits(value)] for value in images.values()]
    monkeypatch.setattr(recipe, "fidelity_names", lambda dataset, seed, offset, samples: list(images))
    monkeypatch.setattr(recipe, "read_images", lambda dataset, names: dict(images))
    monkeypatch.setattr(recipe, "preprocess_image", lambda value: np.full((512, 512, 3), value, np.float32))
    monkeypatch.setattr(recipe, "ort", MagicMock(InferenceSession=Session))

    report = recipe.measure_fidelity(tmp_path / "model.onnx", candidate, tmp_path / "val", fidelity, "w8a8")

    assert onnx_inputs == [(1, 3, 512, 512)] * 2
    assert (report["model_id"], report["build_profile_id"], report["samples"]) == ("segformer_b0_ade_512", "w8a8", 2)
    (metric,) = report["metrics"]
    assert metric["name"] == "logits"
    assert metric["max_absolute_error"] == 1.0
    assert metric["mean_absolute_error"] == pytest.approx(1 / 150)


def test_mpk_interface_is_the_float32_onnx_interface(load_script: Callable[[str], ModuleType]) -> None:
    recipe = load_script("compile")

    assert recipe.INPUT_BYTES == {"pixel_values": 3 * 512 * 512 * 4}
    assert recipe.OUTPUT_BYTES == [150 * 128 * 128 * 4]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--compile-invocation", "model"], "a build profile compiles in one invocation named after it"),
        (["--compile-invocation", "w16a16"], "pass --fidelity-dataset"),
    ],
)
def test_compile_main_rejects_an_unknown_invocation_or_missing_fidelity_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    load_script: Callable[[str], ModuleType],
    arguments: list[str],
    message: str,
) -> None:
    recipe = load_script("compile")
    source = tmp_path / "model.onnx"
    source.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compile.py", "--model", str(source), "--output-dir", str(tmp_path / "out"), "--build-profile", "w16a16"]
        + arguments,
    )

    with pytest.raises(SystemExit) as stopped:
        recipe.main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


# One SegFormer stage with one block, built with the converted graph's operations in the same structure: an
# overlapping patch embedding, efficient self-attention with a spatially reduced key/value path, a Mix-FFN with a
# depthwise convolution and the exported GELU (Div, Erf, Add, Mul, Mul), and the stage's closing LayerNorm.
BLOCK_REGIONS = {"attention_core": 4, "norms": 6, "gelu": 1}


def _segformer_block_graph() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    nodes, inits = [], []

    def const(name: str, value) -> str:
        inits.append(
            numpy_helper.from_array(
                np.asarray(value, np.float32 if np.asarray(value).dtype.kind == "f" else np.int64), name
            )
        )
        return name

    def op(kind: str, inputs: list[str], name: str, **attributes) -> str:
        nodes.append(helper.make_node(kind, inputs, [name + "_out"], name=name, **attributes))
        return name + "_out"

    def weight(name: str, *shape: int) -> str:
        return const(name, rng.standard_normal(shape).astype(np.float32) * 0.1)

    def shape(name: str, *dims: int) -> str:
        return const(name, np.asarray(dims, np.int64))

    def linear(prefix: str, x: str, cin: int, cout: int) -> str:
        y = op("MatMul", [x, weight(prefix + ".w", cin, cout)], prefix + "/MatMul")
        return op("Add", [const(prefix + ".b", np.zeros(cout, np.float32)), y], prefix + "/Add")

    def layer_norm(prefix: str, x: str, c: int) -> str:
        return op(
            "LayerNormalization",
            [x, const(prefix + ".g", np.ones(c, np.float32)), const(prefix + ".beta", np.zeros(c, np.float32))],
            prefix + "/LayerNormalization",
            axis=-1,
        )

    x = op(
        "Conv",
        ["pixel_values", weight("pe.w", 8, 3, 4, 4), const("pe.b", np.zeros(8, np.float32))],
        "/patch_embeddings.0/proj/Conv",
        strides=[4, 4],
        kernel_shape=[4, 4],
    )
    x = op("Reshape", [x, shape("s0", 1, 8, 16)], "/patch_embeddings.0/Reshape")
    x = op("Transpose", [x], "/patch_embeddings.0/Transpose", perm=[0, 2, 1])
    embedded = layer_norm("/patch_embeddings.0/layer_norm", x, 8)
    normed = layer_norm("/block.0.0/layer_norm_1", embedded, 8)
    q = linear("/block.0.0/attention/self/query", normed, 8, 8)
    q = op("Reshape", [q, shape("s1", 1, 16, 1, 8)], "/block.0.0/attention/self/Reshape")
    q = op("Transpose", [q], "/block.0.0/attention/self/Transpose", perm=[0, 2, 1, 3])
    kv = op("Transpose", [normed], "/block.0.0/attention/self/Transpose_1", perm=[0, 2, 1])
    kv = op("Reshape", [kv, shape("s2", 1, 8, 4, 4)], "/block.0.0/attention/self/Reshape_1")
    kv = op(
        "Conv",
        [kv, weight("sr.w", 8, 8, 2, 2), const("sr.b", np.zeros(8, np.float32))],
        "/block.0.0/attention/self/sr/Conv",
        strides=[2, 2],
        kernel_shape=[2, 2],
    )
    kv = op("Reshape", [kv, shape("s3", 1, 8, 4)], "/block.0.0/attention/self/Reshape_2")
    kv = op("Transpose", [kv], "/block.0.0/attention/self/Transpose_2", perm=[0, 2, 1])
    kv = layer_norm("/block.0.0/attention/self/layer_norm", kv, 8)
    k = linear("/block.0.0/attention/self/key", kv, 8, 8)
    k = op("Reshape", [k, shape("s4", 1, 4, 1, 8)], "/block.0.0/attention/self/Reshape_3")
    k = op("Transpose", [k], "/block.0.0/attention/self/Transpose_4", perm=[0, 2, 3, 1])
    v = linear("/block.0.0/attention/self/value", kv, 8, 8)
    v = op("Reshape", [v, shape("s5", 1, 4, 1, 8)], "/block.0.0/attention/self/Reshape_4")
    v = op("Transpose", [v], "/block.0.0/attention/self/Transpose_3", perm=[0, 2, 1, 3])
    scores = op("MatMul", [q, k], "/block.0.0/attention/self/MatMul")
    scores = op("Div", [scores, const("scale", np.float32(np.sqrt(8)))], "/block.0.0/attention/self/Div")
    probabilities = op("Softmax", [scores], "/block.0.0/attention/self/Softmax", axis=-1)
    context = op("MatMul", [probabilities, v], "/block.0.0/attention/self/MatMul_1")
    context = op("Transpose", [context], "/block.0.0/attention/self/Transpose_5", perm=[0, 2, 1, 3])
    context = op("Reshape", [context, shape("s6", 1, 16, 8)], "/block.0.0/attention/self/Reshape_5")
    attended = linear("/block.0.0/attention/output/dense", context, 8, 8)
    residual = op("Add", [attended, embedded], "/block.0.0/Add")
    hidden = linear("/block.0.0/mlp/dense1", layer_norm("/block.0.0/layer_norm_2", residual, 8), 8, 32)
    hidden = op("Transpose", [hidden], "/block.0.0/mlp/dwconv/Transpose", perm=[0, 2, 1])
    hidden = op("Reshape", [hidden, shape("s7", 1, 32, 4, 4)], "/block.0.0/mlp/dwconv/Reshape")
    hidden = op(
        "Conv",
        [hidden, weight("dw.w", 32, 1, 3, 3), const("dw.b", np.zeros(32, np.float32))],
        "/block.0.0/mlp/dwconv/dwconv/Conv",
        group=32,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    hidden = op("Reshape", [hidden, shape("s8", 1, 32, 16)], "/block.0.0/mlp/dwconv/Reshape_1")
    hidden = op("Transpose", [hidden], "/block.0.0/mlp/dwconv/Transpose_1", perm=[0, 2, 1])
    gelu = op("Div", [hidden, const("sqrt2", np.float32(np.sqrt(2)))], "/block.0.0/mlp/intermediate_act_fn/Div")
    gelu = op("Erf", [gelu], "/block.0.0/mlp/intermediate_act_fn/Erf")
    gelu = op("Add", [gelu, const("one", np.float32(1))], "/block.0.0/mlp/intermediate_act_fn/Add")
    gelu = op("Mul", [hidden, gelu], "/block.0.0/mlp/intermediate_act_fn/Mul")
    gelu = op("Mul", [gelu, const("half", np.float32(0.5))], "/block.0.0/mlp/intermediate_act_fn/Mul_1")
    output = op("Add", [linear("/block.0.0/mlp/dense2", gelu, 32, 8), residual], "/block.0.0/Add_1")
    output = layer_norm("/layer_norm.0", output, 8)
    output = op("Transpose", [output], "/Transpose", perm=[0, 2, 1])
    output = op("Reshape", [output, shape("s9", 1, 8, 4, 4)], "/Reshape")
    classifier = [output, weight("cls.w", 5, 8, 1, 1), const("cls.b", np.zeros(5, np.float32))]
    nodes.append(helper.make_node("Conv", classifier, ["logits"], name="/decode_head/classifier/Conv"))
    graph = helper.make_graph(
        nodes,
        "segformer_block",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 5, 4, 4])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def block_surgery(load_script: Callable[[str], ModuleType], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    surgery = load_script("surgery")
    monkeypatch.setattr(surgery, "BF16_REGIONS", BLOCK_REGIONS)
    return surgery


def test_each_region_finds_its_operations_in_a_block(block_surgery: ModuleType) -> None:
    graph = block_surgery._index(_segformer_block_graph())
    found = {
        region: sorted(node.name for node in find(graph)) for region, find in block_surgery.REGION_OPERATIONS.items()
    }

    assert found["attention_core"] == sorted(
        f"/block.0.0/attention/self/{name}" for name in ("MatMul", "Div", "Softmax", "MatMul_1")
    )
    # The residual stream's LayerNorms, including the patch embedding's but not the spatial reduction's, and both
    # residual sums.
    assert found["norms"] == sorted(
        [
            "/patch_embeddings.0/layer_norm/LayerNormalization",
            "/block.0.0/layer_norm_1/LayerNormalization",
            "/block.0.0/layer_norm_2/LayerNormalization",
            "/layer_norm.0/LayerNormalization",
            "/block.0.0/Add",
            "/block.0.0/Add_1",
        ]
    )
    # The GELU's five operations fuse into one, marked at its final multiply.
    assert found["gelu"] == ["/block.0.0/mlp/intermediate_act_fn/Mul_1"]


def test_attention_and_norm_marks_end_at_their_first_int8_readers(block_surgery: ModuleType) -> None:
    marks = block_surgery.precision_marks(_segformer_block_graph(), ["attention_core", "norms"])

    assert sum(precision == "bfloat16" for precision in marks.values()) == 10
    # Readers looked up through layout operations: every Linear layer and convolution after a region stays INT8, a
    # Linear layer marked after its bias. The key and value layers read the INT8 spatial reduction and need no mark.
    assert {name for name, precision in marks.items() if precision == "int8"} == {
        "/block.0.0/attention/self/query/Add",
        "/block.0.0/attention/self/sr/Conv",
        "/block.0.0/attention/output/dense/Add",
        "/block.0.0/mlp/dense1/Add",
        "/decode_head/classifier/Conv",
    }


def test_gelu_marks_end_at_the_second_linear_layer(block_surgery: ModuleType) -> None:
    marks = block_surgery.precision_marks(_segformer_block_graph(), ["gelu"])

    # The depthwise convolution and both Linear layers of the Mix-FFN stay INT8; no mark splits the GELU pattern.
    assert [name for name, precision in marks.items() if precision == "bfloat16"] == [
        "/block.0.0/mlp/intermediate_act_fn/Mul_1"
    ]
    assert {name for name, precision in marks.items() if precision == "int8"} == {"/block.0.0/mlp/dense2/Add"}


def test_precision_marks_reject_an_unknown_region_or_count(block_surgery: ModuleType, monkeypatch) -> None:
    model = _segformer_block_graph()
    with pytest.raises(ValueError, match="unknown BF16 regions"):
        block_surgery.precision_marks(model, ["decoder"])
    monkeypatch.setitem(block_surgery.BF16_REGIONS, "gelu", 2)
    with pytest.raises(ValueError, match="expected 2 gelu operations, found 1"):
        block_surgery.precision_marks(model, ["gelu"])


def test_annotation_writes_the_marks_and_keeps_the_interface(tmp_path: Path, block_surgery: ModuleType) -> None:
    path = tmp_path / "model.onnx"
    onnx.save(_segformer_block_graph(), path)

    written = block_surgery.annotate_bf16_regions(path, ["attention_core", "norms"])

    model = onnx.load(path)
    annotations = [node for node in model.graph.node if node.op_type == "AnnotatePrecision"]
    assert written == block_surgery.precision_marks(_segformer_block_graph(), ["attention_core", "norms"])
    assert Counter(written.values()) == {"bfloat16": 10, "int8": 5}
    assert len(annotations) == 15 and all(node.domain == "ai.sima" for node in annotations)
    assert [value.name for value in model.graph.output] == ["logits"]
    assert any(item.domain == "ai.sima" for item in model.opset_import)


def test_the_full_graph_regions_have_the_declared_operation_counts(load_script: Callable[[str], ModuleType]) -> None:
    surgery = load_script("surgery")

    assert surgery.BF16_REGIONS == {"attention_core": 32, "norms": 40, "gelu": 8}
