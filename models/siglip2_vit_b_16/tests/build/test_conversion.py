"""Checks of the conversion's fixed interface, its source and prompt tables, and the converted ONNX."""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

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
    assert converter.TEMPLATES[0] == "itap of a {}." and len(converter.TEMPLATES) == 7
    assert all(template.count("{}") == 1 for template in converter.TEMPLATES)


def test_prompts_are_lowercased_class_by_class(converter: ModuleType):
    assert converter.prompts(["American robin", "tench"], ["a photo of a {}.", "art of the {}."]) == [
        "a photo of a american robin.", "art of the american robin.", "a photo of a tench.", "art of the tench."
    ]


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
