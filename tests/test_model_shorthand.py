from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from model_registry.accuracy_config import find_devkit_accuracy_config
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


def test_normalization_is_idempotent_and_does_not_modify_authored_settings() -> None:
    authored = _authored("resnet_50")
    original = copy.deepcopy(authored)

    model = normalize_model(authored, MODELS / "resnet_50")

    assert authored == original
    assert normalize_model(model, MODELS / "resnet_50") == model
    assert model["architecture"] == {"id": "resnet"}
    assert all(profile["target"] == "modalix" for profile in model["build_profiles"])
    assert model["benchmark"][0]["id"] == "modalix_latency"
    assert model["benchmark"][0]["scope"] == {"type": "component", "component": "model"}
    assert model["compositions"][0]["entrypoint"] == "default"
    assert "entrypoints" not in model["components"][0]["interface"]


def test_hugging_face_mapping_preserves_layout_media_and_content_identity() -> None:
    authored = _authored("qwen3_5_0_8b")
    revision = "a" * 40
    authored["components"][0]["source"] = {
        "repository": "example/model", "revision": revision, "entrypoint": "target/config.json",
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
    "https://github.com/example/tokenizer/releases/download/v1/chat_template.jinja",
])
def test_upstream_defaults_to_the_declared_checkpoint_with_exceptional_members(uri: str, exception_first: bool) -> None:
    authored = _authored("qwen3_5_0_8b")
    del authored["upstream"]
    source = authored["components"][0]["source"]
    exception = {"chat_template.jinja": {"uri": uri, "sha256": "b" * 64, "size_bytes": 1}}
    ordinary = {key: value for key, value in source["files"].items() if key != "chat_template.jinja"}
    source["files"] = {**exception, **ordinary} if exception_first else {**ordinary, **exception}

    model = normalize_model(authored, MODELS / "qwen3_5_0_8b")

    repository = f"https://huggingface.co/{source['repository']}"
    assert model["upstream"] == {"repository": repository, "revision": source["revision"], "model_card": repository}
    validate_model(MODELS / "qwen3_5_0_8b", authored, SCHEMA)


@pytest.mark.parametrize("file_set", [False, True])
def test_upstream_retains_the_uri_fallback_for_explicit_sources(file_set: bool) -> None:
    authored = _authored("resnet_50")
    del authored["upstream"]
    source = authored["components"][0]["source"]
    if file_set:
        member = {key: value for key, value in source.items() if key != "type"}
        authored["components"][0]["source"] = {
            "type": "file_set", "entrypoint": "model.onnx", "files": [{"path": "model.onnx", **member}],
        }

    model = normalize_model(authored, MODELS / "resnet_50")

    checkpoint, revision = source["uri"].removeprefix("hf://models/").split("@", 1)
    assert model["upstream"]["repository"] == f"https://huggingface.co/{checkpoint}"
    assert model["upstream"]["revision"] == revision.split("/", 1)[0]


def test_normalization_preserves_existing_explicit_upstream_overrides() -> None:
    for path in sorted(MODELS.glob("*/model.yaml")):
        authored = _authored(path.parent.name)
        if "upstream" not in authored:
            continue
        upstream = authored["upstream"]
        model = normalize_model(authored, path.parent)
        assert model["upstream"] == {"model_card": upstream["repository"], **upstream}, path.parent.name


def test_dataset_aliases_keep_pins_scopes_and_recorded_accuracy_behavior() -> None:
    authored = _authored("resnet_50")
    model = normalize_model(authored, MODELS / "resnet_50")
    pin = authored["datasets"]["default"]

    assert model["calibrations"][0] == {"id": "default", "status": "ready", "dataset": pin}
    for stage in ("fidelity", "accuracy"):
        assert model[stage][0]["dataset"] == pin
        assert model[stage][0]["scope"] == {"type": "package", "package": "default"}
        assert model[stage][0]["result_component"] == "model"
    assert find_devkit_accuracy_config(model, "w16a16") is None
    assert find_devkit_accuracy_config(model, "w8a8")["id"] == "imagenet_accuracy"
    recorded = _authored("ssd_mobilenet_v1")
    normalized = normalize_model(recorded, MODELS / "ssd_mobilenet_v1")
    assert normalized["accuracy"] == recorded["accuracy"]
    assert find_devkit_accuracy_config(normalized, "w8a8") is None


def test_hugging_face_shorthand_rejects_a_mutable_revision() -> None:
    authored = _authored("qwen3_5_0_8b")
    authored["components"][0]["source"]["revision"] = "main"
    with pytest.raises(ValidationError, match="immutable 40-character commit"):
        normalize_model(authored, MODELS / "qwen3_5_0_8b")


def test_unknown_dataset_alias_has_an_actionable_error() -> None:
    authored = _authored("resnet_50")
    authored["accuracy"][0]["dataset"] = "missing"
    with pytest.raises(ValidationError, match="accuracy 'imagenet_accuracy': unknown dataset 'missing'"):
        normalize_model(authored, MODELS / "resnet_50")


def test_multi_component_benchmark_does_not_guess_its_scope() -> None:
    authored = _authored("rf_detr_nano")
    authored["benchmark"][0].pop("scope")
    with pytest.raises(ValidationError, match="multi-component scope must be explicit"):
        normalize_model(authored, MODELS / "rf_detr_nano")


def test_unregistered_required_accuracy_metric_fails_even_when_tests_are_skipped(tmp_path: Path) -> None:
    directory = tmp_path / "resnet_50"
    shutil.copytree(MODELS / "resnet_50", directory)
    tests_file = directory / "tests/tests.yaml"
    tests = yaml.safe_load(tests_file.read_text())
    tests["accuracy"] = {"run_tests": False, "skip_reason": "No DevKit available for this fixture."}
    tests_file.write_text(yaml.safe_dump(tests))
    authored = _authored("resnet_50")
    authored["accuracy"][0]["required_metrics"] = ["top1_acuracy"]

    with pytest.raises(ValidationError, match="unregistered metrics: top1_acuracy"):
        validate_model(directory, authored, SCHEMA, ROOT / "taxonomies/model-taxonomy.yaml")
