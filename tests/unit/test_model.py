from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model, load_model, validate_model
from model_registry.taxonomy import load_taxonomy, validate_model_categorization

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy-v2.yaml"


def test_valid_fixture_passes() -> None:
    model = load_and_validate_model(FIXTURES / "fixture_model", SCHEMA)
    assert model["id"] == "fixture_model"


def test_invalid_fixture_reports_field_paths() -> None:
    with pytest.raises(ValidationError) as error:
        load_and_validate_model(FIXTURES / "invalid_model", SCHEMA)
    message = str(error.value)
    assert "components[0].source" in message
    assert "not valid under any of the given schemas" in message


def test_directory_id_must_match(tmp_path: Path) -> None:
    model = load_model(FIXTURES / "fixture_model")
    directory = tmp_path / "wrong_directory"
    directory.mkdir()
    (directory / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ValidationError, match="id must match directory name"):
        load_and_validate_model(directory, SCHEMA)


def test_duplicate_build_profile_ids_fail() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["build_profiles"].append(copy.deepcopy(model["build_profiles"][0]))
    with pytest.raises(ValidationError, match="duplicate build profile ids"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_unsupported_component_build_requires_reason() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["build_profiles"][0]["components"][0]["status"] = "unsupported"
    with pytest.raises(ValidationError, match="components.*reason.*required"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_runner_class_is_optional_and_validated() -> None:
    model = load_model(FIXTURES / "fixture_model")
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)
    model["build_profiles"][0]["runner_class"] = "expensive"
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)
    model["build_profiles"][0]["runner_class"] = "arbitrary-instance"
    with pytest.raises(ValidationError, match="runner_class"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_declarative_composition_validates_ports_and_package_contents() -> None:
    model = load_model(FIXTURES / "fixture_model")
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    model["compositions"] = [
        {"id": "pipeline", "type": "declarative", "schema_version": 1, "file": "composition.yaml"}
    ]
    model["packages"][0]["components"] = ["model", "second_model"]
    model["packages"][0]["composition"] = "pipeline"
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)

    model["packages"][0]["components"] = ["model"]
    with pytest.raises(ValidationError, match="omits components required by composition"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_unapproved_https_host_is_a_validation_error() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["components"][0]["source"]["uri"] = "https://unapproved.example/model.onnx"
    with pytest.raises(ValidationError, match="source policy.*not approved"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_missing_model_local_test_config_fails() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["tests"]["correctness"] = [
        {
            "id": "missing",
            "config": "tests/missing.yaml",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
        }
    ]
    with pytest.raises(ValidationError, match="referenced file does not exist"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_performance_configuration_is_validated_during_catalog_validation() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["tests"]["performance"] = [
        {
            "id": "latency",
            "config": "invalid-performance.yaml",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
        }
    ]
    with pytest.raises(ValidationError, match="unsupported_setting.*unexpected"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_yaml_uses_safe_loader(tmp_path: Path) -> None:
    directory = tmp_path / "unsafe"
    directory.mkdir()
    (directory / "model.yaml").write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="could not load YAML"):
        load_model(directory)


def test_catalog_model_is_valid() -> None:
    model = load_and_validate_model(ROOT / "models" / "resnet_50", SCHEMA)
    assert model["schema_version"] == 2
    assert model["tasks"] == ["image_classification"]
    categorization = validate_model_categorization(model, load_taxonomy(TAXONOMY))
    assert categorization is not None
    assert categorization["architecture"] == {
        "id": "resnet_50",
        "paradigm": "cnn",
        "variant": "imagenet_1k",
    }
    assert model["components"][0]["source"]["sha256"] == (
        "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
    )
    profiles = {profile["id"]: profile for profile in model["build_profiles"]}
    assert set(profiles) == {"modalix_bf16", "modalix_int8"}
    assert profiles["modalix_bf16"]["components"][0]["parameters"]["precision"] == "bf16"
    assert profiles["modalix_int8"]["components"][0]["parameters"]["precision"] == "int8"
    assert profiles["modalix_int8"]["components"][0]["parameters"]["calibration_samples"] == 500
    assert model["calibrations"][0]["dataset"] == {
        "id": "imagenet-2012",
        "variant": "mini",
        "sha256": "d769e759abba1fe6e03db704020dcff3f1e8309a8cac8885ab60f5a0e512c239",
        "size_bytes": 356052889,
    }


def test_unknown_task_fails_semantic_validation() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["tasks"] = ["unknown_task"]
    with pytest.raises(ValidationError, match="tasks references unknown taxonomy ID"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_incompatible_model_type_and_architecture_fail() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["model_type"] = "llm"
    model["tasks"] = ["text_generation"]
    model["modalities"] = {"inputs": ["text"], "outputs": ["text"]}
    with pytest.raises(ValidationError, match="architecture 'resnet_50' is not compatible"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_model_without_required_taxonomy_version_fails() -> None:
    model = load_model(FIXTURES / "fixture_model")
    del model["taxonomy_version"]
    with pytest.raises(ValidationError, match="taxonomy_version.*required"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_legacy_schema_is_rejected_even_without_category_fields() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["schema_version"] = 1
    for field in ("taxonomy_version", "model_type", "tasks", "modalities", "architecture"):
        model.pop(field)
    with pytest.raises(ValidationError, match="schema_version.*2 was expected"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_duplicate_component_roles_fail() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["architecture"]["components"] = [
        {"role": "encoder", "architecture_id": "clip_vit_l_14"},
        {"role": "encoder", "architecture_id": "llama_2"},
    ]
    with pytest.raises(ValidationError, match=r"architecture.components\[1\].role duplicates"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)
