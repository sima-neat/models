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
    assert "source.sha256" in message
    assert "source.uri" in message


def test_directory_id_must_match(tmp_path: Path) -> None:
    model = load_model(FIXTURES / "fixture_model")
    directory = tmp_path / "wrong_directory"
    directory.mkdir()
    (directory / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ValidationError, match="id must match directory name"):
        load_and_validate_model(directory, SCHEMA)


def test_duplicate_variant_ids_fail() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["compile_variants"].append(copy.deepcopy(model["compile_variants"][0]))
    with pytest.raises(ValidationError, match="duplicate compile variant ids"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_unsupported_variant_requires_reason() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["compile_variants"][0]["status"] = "unsupported"
    with pytest.raises(ValidationError, match="compile_variants.*reason.*required"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_unapproved_https_host_is_a_validation_error() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["source"]["uri"] = "https://unapproved.example/model.onnx"
    with pytest.raises(ValidationError, match="source policy.*not approved"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_missing_model_local_test_config_fails() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["tests"]["correctness"] = [{"id": "missing", "config": "tests/missing.yaml"}]
    with pytest.raises(ValidationError, match="referenced file does not exist"):
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
    assert model["source"]["sha256"] == "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50"
    variants = {variant["id"]: variant for variant in model["compile_variants"]}
    assert set(variants) == {"modalix_bf16", "modalix_int8"}
    assert variants["modalix_bf16"]["parameters"]["precision"] == "bf16"
    assert variants["modalix_int8"]["parameters"]["precision"] == "int8"
    assert variants["modalix_int8"]["parameters"]["calibration_samples"] == 500
    assert model["calibration"]["dataset"] == {
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
