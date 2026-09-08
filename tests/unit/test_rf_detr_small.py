from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "rf_detr_small"


def load_module(name: str) -> ModuleType:
    path = MODEL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rf_detr_small_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
    return module


def test_preprocessing_produces_normalized_nchw_image() -> None:
    calibration = load_module("calibration")
    image = Image.new("RGB", (16, 8), color=(255, 128, 0))
    content = io.BytesIO()
    image.save(content, format="PNG")

    result = calibration.preprocess_image(content.getvalue())

    assert result.shape == (3, 512, 512)
    assert result.dtype == np.float32
    expected = (np.asarray([1.0, 128 / 255, 0.0], dtype=np.float32) - calibration.MEAN[:, 0, 0]) / calibration.STD[
        :, 0, 0
    ]
    np.testing.assert_allclose(result[:, 0, 0], expected)


def test_surgery_import_does_not_require_onnxruntime(monkeypatch) -> None:
    original_import = builtins.__import__

    def without_runtime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_runtime)
    assert callable(load_module("surgery").prepare_model)


def test_image_selection_is_seeded_and_disjoint(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    for index in range(8):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(tmp_path / f"{index}.jpg")

    first = calibration.select_images(tmp_path, calibration_count=3, fidelity_count=3, seed=42)
    second = calibration.select_images(tmp_path, calibration_count=3, fidelity_count=3, seed=42)

    assert first == second
    calibration_names, fidelity_names = first
    assert len(calibration_names) == 3
    assert len(fidelity_names) == 3
    assert set(calibration_names).isdisjoint(fidelity_names)


def test_afe_layout_converts_only_rank_four_tensors() -> None:
    calibration = load_module("calibration")
    images = np.zeros((2, 3, 5, 7), dtype=np.float32)
    proposals = np.zeros((2, 300, 4), dtype=np.float32)

    converted_images = calibration.to_afe_layout(images)
    converted_proposals = calibration.to_afe_layout(proposals)

    assert converted_images.shape == (2, 5, 7, 3)
    assert converted_images.flags.c_contiguous
    assert converted_proposals.shape == proposals.shape
    assert converted_proposals.flags.c_contiguous


def test_coco_archive_selection_and_webp_decoding(tmp_path: Path) -> None:
    calibration = load_module("calibration")
    image = io.BytesIO()
    Image.new("RGB", (2, 2), color=(64, 128, 192)).save(image, format="WEBP")
    names = [f"val2017/{index:012d}.webp" for index in range(8)]
    selections = []
    for index, members in enumerate((names, list(reversed(names)))):
        dataset = tmp_path / f"coco-{index}.tgz"
        with tarfile.open(dataset, "w:gz") as archive:
            for name in members:
                member = tarfile.TarInfo(name)
                member.size = len(image.getvalue())
                archive.addfile(member, io.BytesIO(image.getvalue()))
        selected = calibration.select_images(dataset, calibration_count=3, fidelity_count=3, seed=42)
        selections.append(selected)
        assert calibration.load_images(dataset, selected[0]).shape == (3, 3, 512, 512)
        assert set(selected[0]).isdisjoint(selected[1])
    assert selections[0] == selections[1]


def test_conversion_validation_images_are_deterministic(tmp_path: Path) -> None:
    converter = load_module("convert")
    source = tmp_path / "source"
    source.mkdir()
    real_image = source / "validation.png"
    Image.new("RGB", (16, 16), color=(100, 150, 200)).save(real_image)
    runs = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        paths = converter._images(source, output)
        assert paths[-1][1] == real_image
        arrays = []
        for _, path in paths:
            with Image.open(path) as image:
                arrays.append(np.asarray(image).copy())
        runs.append(arrays)

    assert runs[0][0].shape == (512, 512, 3)
    assert runs[0][1].shape == (375, 640, 3)
    for first, second in zip(*runs, strict=True):
        assert first.dtype == np.uint8
        np.testing.assert_array_equal(first, second)


def test_topk_gather_uses_stable_score_order() -> None:
    calibration = load_module("calibration")
    scores = np.asarray([[0.5, 0.9, 0.9, 0.1]], dtype=np.float32)
    boxes = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    result = calibration.stable_topk_gather(scores, boxes, count=3)

    np.testing.assert_array_equal(result, boxes[:, [1, 2, 0]])


def test_surgery_moves_box_decode_before_topk() -> None:
    surgery = load_module("surgery")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["normalized"], name="normalized"),
            helper.make_node(
                "MatMul",
                ["old_topk_features", "weights"],
                ["box_mlp"],
                name="/transformer/enc_out_bbox_embed.0/layers.0/MatMul",
            ),
            helper.make_node("Slice", ["old_anchors"], ["left"], name="/transformer/Slice_2"),
            helper.make_node("Slice", ["old_anchors"], ["right"], name="/transformer/Slice_3"),
            helper.make_node("Concat", ["left", "right"], [surgery._BOXES], name="boxes"),
            helper.make_node(
                "GatherElements",
                ["old_boxes", "indices"],
                [surgery._PROPOSALS],
                name="/transformer/GatherElements_1",
            ),
            helper.make_node("Concat", ["old_proposals", "feature"], ["decoder"], name="/transformer/Concat_6"),
        ],
        "fixture",
        [],
        [],
    )
    model = helper.make_model(graph)

    surgery._rewrite_proposals(model)

    nodes = {node.name: node for node in model.graph.node}
    assert nodes["/transformer/enc_out_bbox_embed.0/layers.0/MatMul"].input[0] == (
        "/transformer/enc_output_norm.0/LayerNormalization_output_0"
    )
    assert nodes["/transformer/Slice_2"].input[0] == "/transformer/Cast_4_output_0"
    assert nodes["/transformer/Slice_3"].input[0] == "/transformer/Cast_4_output_0"
    assert nodes["/transformer/GatherElements_1"].input[0] == surgery._BOXES
    assert nodes["/transformer/Concat_6"].input[0] == surgery._PROPOSALS
    names = [node.name for node in model.graph.node]
    assert names.index("/transformer/GatherElements_1") < names.index("/transformer/Concat_6")


def test_export_interface_is_static_opset_17(tmp_path: Path) -> None:
    exporter = load_module("convert")
    helper = onnx.helper
    path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("ConstantOfShape", ["dets_shape"], ["dets"], name="boxes"),
            helper.make_node("ConstantOfShape", ["labels_shape"], ["labels"], name="classes"),
        ],
        "fixture",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 512, 512])],
        [
            helper.make_tensor_value_info("dets", onnx.TensorProto.FLOAT, [1, 300, 4]),
            helper.make_tensor_value_info("labels", onnx.TensorProto.FLOAT, [1, 300, 91]),
        ],
        initializer=[
            helper.make_tensor("dets_shape", onnx.TensorProto.INT64, [3], [1, 300, 4]),
            helper.make_tensor("labels_shape", onnx.TensorProto.INT64, [3], [1, 300, 91]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)

    exporter.validate_export(path)


def test_compile_recipe_uses_one_grouped_entrypoint_and_direct_afe_api() -> None:
    source = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--compile-invocation", required=True)' in source
    assert 'parser.add_argument("--correctness-dataset", type=Path, required=True)' in source
    assert "prepare_model(source" in source
    assert "transformer_inputs(" in source
    assert "loaded.quantize(" in source
    assert "any_shape_on_mla=True" in source
    assert "automatic_layout_conversion=True" in source
    assert "with_per_token_quant(False)" in source
    assert "flexible_batch_size=True" in source
    assert 'layer_norm_use_fp32_intermediates=precision == "bf16"' in source
    assert "quantized.compile(" in source
    assert "compile-outputs.json" in source
    assert '"schema_version": 2' in source
    assert '"derived_graphs"' in source
    assert '"correctness"' in source
    assert "shutil.copyfile" in source
    assert "requests" not in source
    assert "urllib" not in source
    assert 'get_context("spawn")' in source
    assert source.index('"RF-DETR backbone compilation"') < source.index(
        '"RF-DETR transformer compilation and fidelity"'
    )


def test_surgery_simplifies_only_the_transformer_partition() -> None:
    source = (MODEL_DIR / "surgery.py").read_text(encoding="utf-8")

    assert "transformer, valid = simplify(" in source
    assert "full, valid = simplify(" not in source


def test_model_declares_grouped_components_and_full_pipeline_correctness() -> None:
    model = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    invocation = model["build_profiles"][0]["compile_invocations"][0]
    correctness = model["tests"]["correctness"][0]
    performance = yaml.safe_load((MODEL_DIR / "tests" / "performance.yaml").read_text(encoding="utf-8"))

    assert invocation["components"] == ["backbone", "transformer"]
    assert "python_packages" not in invocation
    assert model["components"][0]["source"]["type"] == "onnx_bundle"
    assert model["components"][0]["source"] == model["components"][1]["source"]
    assert model["calibrations"][0]["dataset"]["id"] == "coco-2017"
    assert model["build_profiles"][0]["components"][0]["parameters"]["calibration_samples"] == 50
    assert model["build_profiles"][0]["components"][1]["parameters"]["calibration_samples"] == 50
    assert correctness["scope"] == {"type": "package", "package": "default"}
    assert performance["required_metrics"] == ["avg_power_watts", "energy_per_inference_joules"]


def test_graph_preparation_consumes_supplied_onnx_without_exporter(tmp_path: Path) -> None:
    source_text = (MODEL_DIR / "compile.py").read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] not in {"export", "convert", "rfdetr", "torch", "transformers"}
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"export", "convert", "rfdetr", "torch", "transformers"}
    preparation = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_graphs_and_reference"
    )
    source = tmp_path / "downloaded.onnx"
    source.write_bytes(b"verified ONNX")
    backbone, transformer = tmp_path / "backbone.onnx", tmp_path / "transformer.onnx"
    calls = []
    images = [np.full((3, 2, 2), index, dtype=np.float32) for index in range(2)]

    def prepare(graph, directory):
        assert graph == source
        calls.append("surgery")
        return source, backbone, transformer

    def validate_split(original, actual_backbone, actual_transformer, image):
        assert (original, actual_backbone, actual_transformer) == (source, backbone, transformer)
        np.testing.assert_array_equal(image, images[0])
        calls.append("split_equivalence")

    class Session:
        def __init__(self, path, providers):
            assert path == str(source)
            assert providers == ["CPUExecutionProvider"]

        def run(self, names, inputs):
            assert names is None
            value = float(inputs["input"][0, 0, 0, 0])
            return [np.full((1, 300, 4), value), np.full((1, 300, 91), value + 1)]

    namespace = {
        "Path": Path,
        "np": np,
        "prepare_model": prepare,
        "validate_float_split": validate_split,
        "load_images": lambda dataset, names: images[: len(names)],
        "ort": SimpleNamespace(InferenceSession=Session),
        "INPUT": "input",
        "OUTPUTS": ["dets", "labels"],
    }
    exec(compile(ast.Module(body=[preparation], type_ignores=[]), str(MODEL_DIR / "compile.py"), "exec"), namespace)
    namespace["_prepare_graphs_and_reference"](source, tmp_path, ("first", "second"), tmp_path)

    assert calls == ["surgery", "split_equivalence"]
    with np.load(tmp_path / "reference_outputs.npz") as reference:
        assert reference["dets"].shape == (2, 300, 4)
        assert reference["labels"].shape == (2, 300, 91)
        np.testing.assert_array_equal(reference["dets"][:, 0, 0], [0, 1])
        np.testing.assert_array_equal(reference["labels"][:, 0, 0], [1, 2])
