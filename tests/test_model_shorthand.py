from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from model_registry.benchmark_declarations import benchmarked_profiles
from model_registry.errors import ValidationError
from model_registry.model import normalize_model, validate_model
from model_registry.source import source_identity

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/model.schema.json"
MODELS = ROOT / "models"


def _authored(name: str) -> dict:
    return yaml.safe_load((MODELS / name / "model.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,entrypoint", [("qwen3_5_0_8b", "generate"), ("whisper_small", "transcribe")])
def test_authored_short_form_keeps_the_public_package_entrypoint(name: str, entrypoint: str) -> None:
    authored = _authored(name)
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(authored)

    model = normalize_model(authored, MODELS / name)

    assert model["id"] == name
    assert model["compositions"] == [{"id": "default", "type": "single", "component": "model",
                                      "entrypoint": entrypoint}]
    assert model["packages"][0]["components"] == ["model"]
    assert model["packages"][0]["composition"] == "default"
    assert model["packages"][0]["build_profiles"] == [profile["id"] for profile in model["build_profiles"]]


def test_normalization_does_not_modify_authored_settings() -> None:
    authored = _authored("resnet_50")
    original = copy.deepcopy(authored)

    model = normalize_model(authored, MODELS / "resnet_50")

    assert authored == original
    assert model["architecture"] == {"id": "resnet"}
    assert all(profile["target"] == "modalix" for profile in model["build_profiles"])
    assert [profile["compile_invocations"] for profile in model["build_profiles"]] == [
        [{"id": "w16a16", "components": ["model"]}],
        [{"id": "w8a8", "components": ["model"]}],
        [{"id": "w8a16", "components": ["model"]}],
    ]
    assert model["performance"] == authored["performance"]
    assert model["compositions"][0]["entrypoint"] == "default"
    assert model["components"][0]["interface"] == {"format": "onnx"}


def test_multi_component_model_compiles_and_packages_its_components_together() -> None:
    authored = _authored("rf_detr_nano")

    model = normalize_model(authored, MODELS / "rf_detr_nano")

    assert [component["id"] for component in model["components"]] == ["backbone", "decoder"]
    assert {component["source"]["sha256"] for component in model["components"]} == {
        authored["compilation"]["source"]["sha256"]
    }
    for profile in model["build_profiles"]:
        assert profile["compile_invocations"] == [{"id": profile["id"], "components": ["backbone", "decoder"]}]
        # The recipe calibrates the backbone; the decoder calibrates on the backbone's outputs.
        assert [binding.get("calibration") for binding in profile["components"]] == ["coco-2017-cal", None]
    assert model["compositions"] == [{"id": "default", "type": "declarative"}]
    assert model["packages"][0]["components"] == ["backbone", "decoder"]


def test_hugging_face_mapping_preserves_layout_media_and_content_identity() -> None:
    authored = _authored("qwen3_5_0_8b")
    revision = "a" * 40
    authored["compilation"]["source"] = {
        "repository": "example/model", "revision": revision, "main_file": "target/config.json",
        "files": {
            "target/config.json": {"sha256": "b" * 64, "size_bytes": 10, "media_type": "application/json"},
            "draft/weights.safetensors": {"sha256": "c" * 64, "size_bytes": 20,
                                          "uri": f"hf://models/example/draft@{revision}/weights.safetensors"},
        },
    }
    expected = {"type": "file_set", "entrypoint": "target/config.json", "files": [
        {"path": "target/config.json", "uri": f"hf://models/example/model@{revision}/target/config.json",
         "sha256": "b" * 64, "size_bytes": 10, "media_type": "application/json"},
        {"path": "draft/weights.safetensors", "uri": f"hf://models/example/draft@{revision}/weights.safetensors",
         "sha256": "c" * 64, "size_bytes": 20},
    ]}

    expanded = normalize_model(authored, MODELS / "qwen3_5_0_8b")["components"][0]["source"]

    assert expanded == expected
    assert source_identity(expanded) == source_identity(expected)


@pytest.mark.parametrize("exception_first", [True, False])
@pytest.mark.parametrize("uri", [
    f"hf://models/example/tokenizer@{'a' * 40}/chat_template.jinja",
    f"https://huggingface.co/example/tokenizer/resolve/{'b' * 40}/chat_template.jinja",
])
def test_upstream_defaults_to_the_declared_checkpoint_with_exceptional_members(uri: str, exception_first: bool) -> None:
    authored = _authored("qwen3_5_0_8b")
    del authored["upstream"]
    source = authored["compilation"]["source"]
    exception = {"chat_template.jinja": {"uri": uri, "sha256": "b" * 64, "size_bytes": 1}}
    ordinary = {key: value for key, value in source["files"].items() if key != "chat_template.jinja"}
    source["files"] = {**exception, **ordinary} if exception_first else {**ordinary, **exception}

    model = normalize_model(authored, MODELS / "qwen3_5_0_8b")

    repository = f"https://huggingface.co/{source['repository']}"
    assert model["upstream"] == {"repository": repository, "revision": source["revision"], "model_card": repository}
    validate_model(MODELS / "qwen3_5_0_8b", authored, SCHEMA)


def test_upstream_is_required_when_the_source_names_no_hugging_face_repository() -> None:
    authored = _authored("resnet_50")
    del authored["upstream"]

    assert "upstream" not in normalize_model(authored, MODELS / "resnet_50")
    with pytest.raises(ValidationError, match="upstream is required"):
        validate_model(MODELS / "resnet_50", authored, SCHEMA)


def test_normalization_preserves_existing_explicit_upstream_overrides() -> None:
    for path in sorted(MODELS.glob("*/model.yaml")):
        authored = _authored(path.parent.name)
        if "upstream" not in authored:
            continue
        upstream = authored["upstream"]
        model = normalize_model(authored, path.parent)
        assert model["upstream"] == {"model_card": upstream["repository"], **upstream}, path.parent.name


def test_accuracy_names_each_dataset_by_its_registry_source_and_keeps_measured_build_profiles() -> None:
    authored = _authored("resnet_50")
    model = normalize_model(authored, MODELS / "resnet_50")
    [pin] = authored["accuracy"]["datasets"]

    assert model["accuracy"] == {**authored["accuracy"], "datasets": {"imagenet-2012-mini": pin}}
    assert benchmarked_profiles(model, "accuracy") == ["w16a16", "w8a8", "w8a16"]
    assert benchmarked_profiles(model, "performance") == ["w16a16", "w8a8", "w8a16"]
    authored["accuracy"]["build_profiles"] = ["w8a8", "w8a16"]
    assert benchmarked_profiles(normalize_model(authored, MODELS / "resnet_50"), "accuracy") == ["w8a8", "w8a16"]
    authored = _authored("yolo26n")
    del authored["accuracy"]
    without_accuracy = normalize_model(authored, MODELS / "yolo26n")
    assert "accuracy" not in without_accuracy
    assert benchmarked_profiles(without_accuracy, "accuracy") == []


def test_hugging_face_shorthand_rejects_a_mutable_revision() -> None:
    authored = _authored("qwen3_5_0_8b")
    authored["compilation"]["source"]["revision"] = "main"
    with pytest.raises(ValidationError, match="compilation.source"):
        validate_model(MODELS / "qwen3_5_0_8b", authored, SCHEMA)
