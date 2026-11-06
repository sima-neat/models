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
TEXT_CHECK = ROOT / "models/siglip2_vit_b_16/scripts/text_encoder_check.npz"


@pytest.fixture
def converter(load_script: Callable[[str], ModuleType]) -> ModuleType:
    return load_script("convert")


def test_prompt_tables_are_openais_imagenet_classes_and_templates(converter: ModuleType):
    assert len(converter.CLASS_NAMES) == converter.CLASS_COUNT == 1000
    assert (converter.CLASS_NAMES[0], converter.CLASS_NAMES[-1]) == ("tench", "toilet paper")
    assert converter.TEMPLATES[0] == "itap of a {}." and len(converter.TEMPLATES) == 7
    assert all(template.count("{}") == 1 for template in converter.TEMPLATES)


def test_prompts_are_lowercased_class_by_class(converter: ModuleType):
    assert converter.prompts(["American robin", "tench"], ["a photo of a {}.", "art of the {}."]) == [
        "a photo of a american robin.", "art of the american robin.", "a photo of a tench.", "art of the tench."
    ]


def test_the_text_check_prompts_use_each_template_on_classes_the_embedded_prompts_leave_out(converter: ModuleType):
    embedded = converter.calibration_prompts()
    assert len(embedded) == converter.CALIBRATION_PROMPTS
    assert embedded[:2] == ["itap of a tench.", "a bad photo of the american robin."]
    check = converter.check_prompts()
    assert len(check) == len(converter.TEMPLATES) and check[0] == "itap of a goldfish."
    assert not set(check) & set(embedded)
    embedded_classes = {index * converter.CLASS_COUNT // len(embedded) for index in range(len(embedded))}
    assert not embedded_classes & set(converter.CHECK_CLASSES)
    with np.load(TEXT_CHECK, allow_pickle=False) as saved:
        assert saved["prompts"].tolist() == check, "the stored check holds these prompts"


def test_the_text_check_matches_the_converted_onnx(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    with np.load(TEXT_CHECK, allow_pickle=False) as saved:
        check = {name: saved[name] for name in saved.files}
    rows = dict(zip(check["row_ids"].tolist(), check["rows"]))
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros(converter.INTERFACE["pixel_values"], dtype=np.float32)
    for ids, expected in zip(check["token_ids"], check["text_embeds"], strict=True):
        token_embeds = np.stack([rows[token] for token in ids.tolist()])[None]
        (text_embeds,) = session.run(["text_embeds"], {"pixel_values": pixel_values, "token_embeds": token_embeds})
        np.testing.assert_allclose(text_embeds[0], expected, rtol=1e-4, atol=1e-4)


def test_the_text_check_embeds_with_the_given_onnx_and_downloads_only_the_checkpoint(
    tmp_path, monkeypatch, capsys, converter: ModuleType
):
    onnx_path, checkpoint, output = tmp_path / "model.onnx", tmp_path / "model.safetensors", tmp_path / "check.npz"
    fetched, written = [], []

    def fetch_source(source):
        fetched.append(source)
        return (checkpoint,)

    def write_text_check(*arguments):
        written.append(arguments)
        output.write_bytes(b"check")

    # Nothing downloads, and the check writer records what it is given and leaves a file for main() to hash.
    monkeypatch.setattr(converter, "Fetcher", lambda: SimpleNamespace(fetch_source=fetch_source))
    monkeypatch.setattr(converter, "write_text_check", write_text_check)
    for arguments in (["--text-check", str(output)], ["--output-dir", str(tmp_path), "--onnx", str(onnx_path)]):
        monkeypatch.setattr("sys.argv", ["convert.py", *arguments])
        with pytest.raises(SystemExit) as stopped:
            converter.main()
        assert stopped.value.code == 2 and "--text-check and --onnx go together" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["convert.py", "--text-check", str(output), "--onnx", str(onnx_path)])
    converter.main()
    # Only the checkpoint downloads: the ONNX comes from --onnx, because compilation.source names no file once it is
    # a conversion.
    assert fetched == [load_metadata(ROOT / "models/siglip2_vit_b_16")["conversion"]["default"]["source"]]
    assert written == [(checkpoint, onnx_path.resolve(), output.resolve())]


def _interface_model(converter: ModuleType, case: str) -> onnx.ModelProto:
    helper = onnx.helper

    def value(name: str) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, converter.INTERFACE[name])

    inputs = [value(name) for name in converter.INPUTS]
    outputs = [value(name) for name in converter.OUTPUTS]
    if case == "order":
        outputs[:2] = outputs[1::-1]
    elif case == "dtype":
        outputs[-1].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif case == "shape":
        outputs[0].type.tensor_type.shape.dim[0].dim_value = 2
    nodes = []
    for output in outputs:
        dtype = output.type.tensor_type.elem_type
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        array = np.zeros(shape, dtype=np.float64 if dtype == onnx.TensorProto.DOUBLE else np.float32)
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
    model = load_metadata(ROOT / "models/siglip2_vit_b_16")
    converter.validate_source(model["conversion"]["default"]["source"])
    pth = {"type": "file", "uri": "https://example.com/checkpoint.pth"}
    with pytest.raises(ValueError, match="Hugging Face file set"):
        converter.validate_source(pth)


@pytest.mark.parametrize("model_type", ["siglip", "siglip2"])
def test_config_must_name_siglip(tmp_path, model_type, converter: ModuleType):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if model_type == "siglip":
        converter.validate_config(tmp_path)
    else:
        with pytest.raises(ValueError, match="'siglip'"):
            converter.validate_config(tmp_path)


def test_converted_model_scores_images_and_embeds_its_prompts(converted: Path, converter: ModuleType):
    import onnxruntime as ort

    converter.validate_export(converted)
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    inputs = {name: np.zeros(converter.INTERFACE[name], dtype=np.float32) for name in converter.INPUTS}
    image_embeds, logits, text_embeds, prompts = session.run(None, inputs)
    assert np.isfinite(image_embeds).all() and np.isfinite(logits).all() and np.isfinite(text_embeds).all()
    assert prompts.shape == (converter.CALIBRATION_PROMPTS, converter.TEXT_LENGTH, converter.EMBED_DIM)
    assert np.isfinite(prompts).all() and np.abs(prompts).sum(-1).all(), "every embedded token has an embedding"
    # Every class has its own embedding, so an image does not score every class the same.
    assert np.ptp(logits) > 0
