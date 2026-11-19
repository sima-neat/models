"""Small checks for conversion failures that a synthetic smoke input can miss."""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


@pytest.fixture
def surgery(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("surgery")


def _interface(pipeline, components, opset, case):
    helper = onnx.helper
    inputs = [
        helper.make_tensor_value_info(pipeline.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in components
        for n, s in pipeline.INPUT_SHAPES[c].items()
    ]
    outputs = [
        helper.make_tensor_value_info(pipeline.tensor_name(c, n), onnx.TensorProto.FLOAT, s)
        for c in components
        for n, s in pipeline.OUTPUT_SHAPES[c].items()
    ]
    if case == "dtype":
        inputs[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        inputs[-1].type.tensor_type.shape.dim[3].dim_value = 7
    elif case == "missing":
        outputs.pop()
    nodes = []
    for value in outputs:
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        tensor = helper.make_tensor(value.name, onnx.TensorProto.FLOAT, shape, np.zeros(int(np.prod(shape))))
        nodes.append(helper.make_node("Constant", [], [value.name], value=tensor))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset - 1 if case == "opset" else opset)])


@pytest.mark.parametrize("case", ["valid", "opset", "dtype", "shape", "missing"])
def test_image_encoder_onnx_interface(tmp_path, case, converter: ModuleType, load_script):
    path = tmp_path / "image_encoder.onnx"
    onnx.save(_interface(load_script("pipeline"), ["image_encoder"], converter.OPSET, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "opset", "dtype", "shape", "missing"])
def test_side_by_side_onnx_interface(tmp_path, case, surgery: ModuleType, load_script):
    path = tmp_path / "side_by_side.onnx"
    pipeline = load_script("pipeline")
    onnx.save(_interface(pipeline, pipeline.COMPONENTS, surgery.OPSET, case), path)
    if case == "valid":
        surgery.validate_export(path)
    else:
        with pytest.raises(ValueError):
            surgery.validate_export(path)


@pytest.mark.parametrize("case", ["valid", "shape", "nan", "drift"])
def test_onnx_outputs_must_match_pytorch(case, converter: ModuleType):
    expected = np.random.RandomState(0).normal(0, 4, (1, 4, 256, 256)).astype(np.float32)
    actual = expected + 1e-5
    error, match = ValueError, "Unexpected or non-finite"
    if case == "shape":
        actual = actual[:, :3]
    elif case == "nan":
        actual[0, 0, 0, 0] = np.nan
    elif case == "drift":
        actual = expected + 0.01
        error, match = AssertionError, "masks"
    else:
        converter.compare(expected, actual, "masks")
        return
    with pytest.raises(error, match=match):
        converter.compare(expected, actual, "masks")


def test_source_must_be_the_pinned_checkpoint(converter: ModuleType):
    model = load_metadata(ROOT / "models/mobile_sam")
    converter.validate_source(model["conversion"]["default"]["source"])
    with pytest.raises(ValueError, match="mobile_sam.pt"):
        converter.validate_source({"type": "file", "uri": "hf://models/dhkim2810/MobileSAM@main/sam_vit_b.pth"})


def test_check_prompts_fit_the_decoder_and_cover_every_point_type(
    surgery: ModuleType, load_script: Callable[[str], ModuleType]
):
    kinds = {kind for _, types in surgery.PROMPTS for kind in types}

    assert kinds == set(load_script("pipeline").POINT_TYPES[1:])
    assert max(len(points) for points, _ in surgery.PROMPTS) == surgery.MAX_POINTS


def test_actual_export_is_the_image_encoder_with_weights_beside_it(converted, converter):
    converter.validate_export(converted)
    assert converted.name == "image_encoder.onnx" and converted.with_name("weights.pth").is_file()


def test_surgery_rebuilds_the_converted_model_from_its_weights(converted, converter, surgery, monkeypatch):
    """The vendored modules surgery.py builds from weights.pth are segment-anything's, number for number."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("segment_anything")
    monkeypatch.syspath_prepend(str(converter.MODEL_DIR / "scripts"))
    weights = converted.with_name("weights.pth")
    upstream, rebuilt = converter.build(weights), surgery.build(weights)
    state = upstream.state_dict()
    assert {key: value.shape for key, value in rebuilt.state_dict().items()} == {
        key: value.shape for key, value in state.items()
    }
    generator = torch.Generator().manual_seed(5)
    embeddings = torch.randn(1, 256, 64, 64, generator=generator)
    for points, types in surgery.PROMPTS:
        with torch.inference_mode():
            expected = surgery.reference_decoder(upstream, embeddings, points, types)
            actual = surgery.reference_decoder(rebuilt, embeddings, points, types)
        for got, want in zip(actual, expected):
            assert torch.equal(got, want), types


def test_masked_softmax_ignores_extreme_inactive_logits(load_script):
    torch = pytest.importorskip("torch")
    decoder = load_script("sam_decoder")
    logits = torch.tensor([[[[-2.0, 10000.0, -4.0, -10000.0, 5.0]]]])
    valid = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    result = decoder.masked_softmax(logits, valid)
    expected = torch.softmax(logits[..., [0, 2, 4]], dim=-1)
    torch.testing.assert_close(result[..., [0, 2, 4]], expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(result[..., [1, 3]], torch.zeros_like(result[..., [1, 3]]))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("active", [2, 3, 4, 5, 6])
def test_masked_decoder_matches_upstream_with_inactive_slot_perturbations(load_script, active):
    torch = pytest.importorskip("torch")
    upstream = pytest.importorskip("segment_anything.modeling")
    torch.manual_seed(13)
    decoder = upstream.MaskDecoder(
        transformer_dim=32,
        transformer=upstream.TwoWayTransformer(depth=2, embedding_dim=32, num_heads=4, mlp_dim=64),
        iou_head_hidden_dim=32,
    ).eval()
    candidate = load_script("sam_decoder")
    image, position, dense = [torch.randn(1, 32, 4, 4) for _ in range(3)]
    sparse = torch.randn(1, active, 32)
    valid = torch.tensor([[1.0] * active + [0.0] * (6 - active)])
    with torch.inference_mode():
        expected = decoder.predict_masks(image, position, sparse, dense)
        for scale in (0.0, 10000.0):
            padded = torch.cat([sparse, torch.randn(1, 6 - active, 32) * scale], dim=1)
            actual = candidate.predict_masks(decoder, image, position, padded, dense, valid)
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "dependency,conversion_changes",
    [
        ("pipeline.py", True),
        ("tiny_vit_sam.py", True),
        ("surgery.py", False),
        ("sam_decoder.py", False),
        ("sam_modules.py", False),
        ("tiny_vit_4d.py", False),
        ("calibration.py", False),
    ],
)
def test_prompt_dependencies_invalidate_their_stage_fingerprints(tmp_path, dependency, conversion_changes):
    import shutil

    from model_registry.conversion import conversion_fingerprint
    from model_registry.errors import ValidationError
    from model_registry.fingerprints import COMPILE_TOOLING, compile_fingerprint
    from model_registry.model import load_and_validate_model

    for directory in ("schemas", "taxonomies", "datasets", "models/mobile_sam"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in COMPILE_TOOLING:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    model = load_and_validate_model(tmp_path / "models/mobile_sam")
    assert {item["source"]["type"] for item in model["components"]} == {"conversion"}
    before = conversion_fingerprint(tmp_path, model)
    selected = {"mobile_sam": {"conversion_fingerprint": before, "sha256": "a" * 64}}
    first = compile_fingerprint(tmp_path, "mobile_sam", "w16a16", "w16a16", "3.0.0.1", conversion_sources=selected)
    path = tmp_path / "models/mobile_sam/scripts" / dependency
    path.write_text(path.read_text() + "\n# A semantic dependency changed.\n")
    after = conversion_fingerprint(tmp_path, model)
    assert (after != before) == conversion_changes
    if conversion_changes:
        with pytest.raises(ValidationError, match="does not match"):
            compile_fingerprint(tmp_path, "mobile_sam", "w16a16", "w16a16", "3.0.0.1", conversion_sources=selected)
    selected["mobile_sam"]["conversion_fingerprint"] = after
    assert (
        compile_fingerprint(tmp_path, "mobile_sam", "w16a16", "w16a16", "3.0.0.1", conversion_sources=selected) != first
    )


@pytest.mark.parametrize("parity_fails", [False, True])
def test_export_is_same_filesystem_and_published_only_after_parity(converter, monkeypatch, tmp_path, parity_fails):
    import sys
    from types import SimpleNamespace

    output = tmp_path / "converted"
    checked = []

    def export(module, inputs, destination, **kwargs):
        assert Path(destination).parent.parent == output
        Path(destination).write_bytes(b"ONNX fixture")

    def save(state, destination):
        assert Path(destination).parent.parent == output and not (output / "weights.pth").exists()
        Path(destination).write_bytes(b"weights fixture")

    def parity(module, path):
        assert Path(path).is_file() and not (output / "image_encoder.onnx").exists()
        checked.append(path)
        if parity_fails:
            raise RuntimeError("independent-parity-failed")

    torch = SimpleNamespace(
        set_num_threads=lambda count: None, zeros=lambda shape: None, onnx=SimpleNamespace(export=export), save=save
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(converter, "build", lambda path: SimpleNamespace(image_encoder=None, state_dict=dict))
    for name in ("fold_constants", "validate_export"):
        monkeypatch.setattr(converter, name, lambda *args: None)
    monkeypatch.setattr(converter, "check_onnx", parity)
    if parity_fails:
        with pytest.raises(RuntimeError, match="independent-parity-failed"):
            converter.convert(tmp_path / "checkpoint.pt", output)
        assert list(output.iterdir()) == []
    else:
        assert converter.convert(tmp_path / "checkpoint.pt", output) == output / "image_encoder.onnx"
        assert sorted(path.name for path in output.iterdir()) == ["image_encoder.onnx", "weights.pth"]
    assert len(checked) == 1


def test_real_image_split_and_fourier_boundary_parity(
    converted, coco_source_images, converter, surgery, load_script, tmp_path, monkeypatch
):
    """Pinned upstream, the converted encoder and surgery's graphs agree on real photos and point/box/mixed prompts."""
    import hashlib
    import json

    import onnxruntime as ort
    import torch
    from model_registry.fetch import Fetcher
    from PIL import Image
    from segment_anything import SamPredictor

    # Use the declared source resolver/cache, just as conversion does; no separate download route.
    source = load_metadata(converter.MODEL_DIR)["conversion"]["default"]["source"]
    converter.validate_source(source)
    checkpoint, *_ = Fetcher().fetch_source(source)
    monkeypatch.syspath_prepend(str(converter.MODEL_DIR / "scripts"))
    predictor = SamPredictor(converter.build(checkpoint))
    pipeline = load_script("pipeline")
    weights = converted.with_name("weights.pth")
    paths = surgery.export_components(converted, weights, tmp_path / "components")
    decoder = tmp_path / "decoder.onnx"
    metadata = surgery.compile_decoder(
        paths["prompt_decoder"], decoder, hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    encoding = pipeline.PromptEncoding.from_metadata({key: json.dumps(value) for key, value in metadata.items()})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2

    def session(path):
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    upstream_encoder = session(converted)
    parts = {key: session(path) for key, path in paths.items()}
    cut = session(decoder)
    name = pipeline.tensor_name
    outputs = [name("prompt_decoder", key) for key in pipeline.OUTPUT_SHAPES["prompt_decoder"]]
    for path in coco_source_images:
        pixels = np.asarray(Image.open(path).convert("RGB"))
        height, width = pixels.shape[:2]
        image = np.ascontiguousarray(pipeline.preprocess(pixels).transpose(2, 0, 1)[None])
        embeddings = parts["image_encoder"].run(None, {name("image_encoder", "image"): image})[0]
        expected = upstream_encoder.run(None, {name("image_encoder", "image"): image})[0]
        np.testing.assert_allclose(embeddings, expected, rtol=1e-3, atol=1e-3, err_msg=path.name)
        with torch.inference_mode():
            predictor.set_image(pixels)
        np.testing.assert_allclose(embeddings, predictor.features.numpy(), rtol=1e-3, atol=1e-3, err_msg=path.name)
        prompts = [
            ([(width / 2, height / 2)], ["foreground"]),
            ([(width / 4, height / 4), (width * 3 / 4, height * 3 / 4)], ["box_top_left", "box_bottom_right"]),
            (
                [(width / 4, height / 4), (width * 3 / 4, height * 3 / 4), (width / 2, height / 2)],
                ["box_top_left", "box_bottom_right", "background"],
            ),
        ]
        for points, types in prompts:
            prompt = pipeline.prompt_inputs(points, types, (height, width))
            indices = [i for i, kind in enumerate(types) if kind in ("foreground", "background")]
            clicks = np.asarray(points)[indices] if indices else None
            labels = np.asarray([int(types[i] == "foreground") for i in indices]) if indices else None
            box = None
            if "box_top_left" in types:
                box = np.asarray(points)[[types.index("box_top_left"), types.index("box_bottom_right")]].reshape(4)
            with torch.inference_mode():
                _, score0, mask0 = predictor.predict(
                    clicks, labels, box=box, multimask_output=False, return_logits=True
                )
                _, scores, masks = predictor.predict(clicks, labels, box=box, multimask_output=True, return_logits=True)
            expected = (np.concatenate([mask0, masks])[None], np.concatenate([score0, scores])[None])
            extracted = parts["prompt_decoder"].run(
                outputs,
                {
                    name("prompt_decoder", "image_embeddings"): embeddings,
                    **{name("prompt_decoder", key): value for key, value in prompt.items()},
                },
            )
            actual = cut.run(
                outputs,
                {
                    name("prompt_decoder", "image_embeddings"): embeddings,
                    name("prompt_decoder", "point_encoding"): encoding(prompt["point_coords"]),
                    name("prompt_decoder", "point_types"): prompt["point_types"],
                },
            )
            for output, want, split, got in zip(outputs, expected, extracted, actual):
                assert np.isfinite(got).all()
                np.testing.assert_allclose(split, want, rtol=1e-3, atol=1e-3, err_msg=f"upstream {path.name}: {output}")
                np.testing.assert_allclose(got, split, rtol=1e-3, atol=1e-3, err_msg=f"{path.name}: {output}")
