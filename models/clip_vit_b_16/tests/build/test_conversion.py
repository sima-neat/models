"""Checks of the conversion's fixed interface, its source and prompt tables, and the converted ONNX."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
from onnx import numpy_helper
import pytest
from model_registry.model import load_model as load_metadata

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_prompt_tables_are_openais_imagenet_classes_and_templates(converter: ModuleType):
    assert len(converter.CLASS_NAMES) == converter.CLASS_COUNT == 1000
    assert (converter.CLASS_NAMES[0], converter.CLASS_NAMES[-1]) == ("tench", "toilet paper")
    assert len(converter.TEMPLATES) == 80
    assert all(template.count("{}") == 1 for template in converter.TEMPLATES)


def _text_probe() -> dict[str, np.ndarray]:
    with np.load(ROOT / "models/clip_vit_b_16/scripts/text_probe.npz") as saved:
        return {name: saved[name] for name in saved.files}


def test_the_text_probe_holds_the_prompts_convert_writes(converter: ModuleType):
    assert "a photo of a {}." in converter.TEMPLATES
    assert _text_probe()["prompts"].tolist() == list(converter.TEXT_PROBE_PROMPTS)


def test_the_text_probe_matches_the_converted_onnx(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    probe = _text_probe()
    table = dict(zip(probe["row_ids"].tolist(), probe["rows"]))
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros(converter.INTERFACE["pixel_values"], dtype=np.float32)
    for ids, expected in zip(probe["token_ids"], probe["text_embeds"]):
        token_embeds = np.stack([table[token] for token in ids.tolist()])[None]
        (text_embeds,) = session.run(["text_embeds"], {"pixel_values": pixel_values, "token_embeds": token_embeds})
        np.testing.assert_allclose(text_embeds[0, ids.argmax()], expected, rtol=1e-4, atol=1e-4)


def test_the_text_probe_takes_any_onnx_and_fetches_only_the_checkpoint(tmp_path, monkeypatch, converter: ModuleType):
    onnx_path, checkpoint, output = tmp_path / "model.onnx", tmp_path / "pytorch_model.bin", tmp_path / "probe"
    # Any ONNX is taken: the conversion tests, not a compilation.source pin, check the probe against each conversion.
    onnx_path.write_bytes(b"a converted ONNX")
    fetched, written = [], []

    def fetch_source(source):
        fetched.append(source)
        return (checkpoint,)

    # Nothing downloads, and the probe writer only records what it is given.
    monkeypatch.setattr(converter, "Fetcher", lambda: SimpleNamespace(fetch_source=fetch_source))
    monkeypatch.setattr(converter, "write_text_probe", lambda *arguments: written.append(arguments))
    monkeypatch.setattr("sys.argv", ["convert.py", "--output-dir", str(output), "--text-probe", str(onnx_path)])

    converter.main()

    assert fetched == [load_metadata(ROOT / "models/clip_vit_b_16")["conversion"]["default"]["source"]]
    assert written == [(checkpoint, onnx_path.resolve(), output.resolve())]


def _interface_model(converter: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper

    def value(name: str) -> onnx.ValueInfoProto:
        dtype = onnx.TensorProto.INT64 if name == "calibration_ends" else onnx.TensorProto.FLOAT
        return helper.make_tensor_value_info(name, dtype, converter.INTERFACE[name])

    inputs = [value(name) for name in converter.INPUTS]
    outputs = [value(name) for name in converter.OUTPUTS]
    if case == "order":
        outputs[:2] = outputs[1::-1]
    elif case == "dtype":
        outputs[-1].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif case == "shape":
        outputs[0].type.tensor_type.shape.dim[0].dim_value = 2
    nodes = []
    for output in outputs:
        dtype = output.type.tensor_type.elem_type
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        array = np.zeros(shape, dtype=np.int64 if dtype == onnx.TensorProto.INT64 else np.float32)
        nodes.append(helper.make_node("Constant", [], [output.name], value=numpy_helper.from_array(array)))
    graph = helper.make_graph(nodes, "fixture", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if case == "opset" else 17)])


@pytest.mark.parametrize("case", ["valid", "opset", "order", "dtype", "shape"])
def test_fixed_onnx_interface(tmp_path, case, converter: ModuleType):
    path = tmp_path / "model.onnx"
    onnx.save(_interface_model(converter, case), path)
    if case == "valid":
        converter.validate_export(path)
    else:
        with pytest.raises(ValueError):
            converter.validate_export(path)


def test_source_must_be_the_hugging_face_file_set(converter: ModuleType):
    model = load_metadata(ROOT / "models/clip_vit_b_16")
    converter.validate_source(model["conversion"]["default"]["source"])
    pth = {"type": "file", "uri": "https://example.com/checkpoint.pth"}
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source(pth)


@pytest.mark.parametrize("model_type", ["clip", "siglip"])
def test_config_must_name_clip(tmp_path, model_type, converter: ModuleType):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if model_type == "clip":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="'clip'"):
            converter.validate_config(tmp_path)


def test_converted_model_scores_images_and_embeds_its_prompts(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    inputs = {name: np.zeros(converter.INTERFACE[name], dtype=np.float32) for name in converter.INPUTS}
    image_embeds, logits, text_embeds, prompts, ends = session.run(None, inputs)
    assert np.isfinite(image_embeds).all() and np.isfinite(logits).all() and np.isfinite(text_embeds).all()
    assert prompts.shape == (converter.CALIBRATION_PROMPTS, converter.TEXT_LENGTH, converter.EMBED_DIM)
    assert np.isfinite(prompts).all()
    # Each prompt has a start token and at least one word before its end-of-text token, and fits 77 tokens.
    assert ((ends >= 2) & (ends < converter.TEXT_LENGTH)).all()
    # The logits are CLIP's scaled cosine similarities, so they stay within the logit scale of 100.
    assert np.abs(logits).max() <= 100.0 + 1e-3
