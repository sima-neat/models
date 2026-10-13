from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model, load_model, validate_model
from model_registry.taxonomy import load_taxonomy, validate_model_categorization

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"


def _two_component_model() -> dict:
    model = load_model(FIXTURES / "fixture_model")
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    return model


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


def test_compile_invocation_can_group_components_with_shared_inputs() -> None:
    model = _two_component_model()
    model["build_profiles"][0]["compile_invocations"] = [{"id": "combined", "components": ["model", "second_model"]}]

    validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize(
    ("invocations", "expected"),
    [
        ([{"id": "combined", "components": ["model", "missing"]}], "unknown components: missing"),
        (
            [
                {"id": "first", "components": ["model"]},
                {"id": "second", "components": ["model", "second_model"]},
            ],
            "duplicate components: model",
        ),
        ([{"id": "first", "components": ["model"]}], "omit supported components: second_model"),
    ],
)
def test_compile_invocations_partition_supported_components(invocations: list[dict], expected: str) -> None:
    model = _two_component_model()
    model["build_profiles"][0]["compile_invocations"] = invocations

    with pytest.raises(ValidationError, match=expected):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_compile_invocation_rejects_unsupported_component() -> None:
    model = _two_component_model()
    model["build_profiles"][0]["components"][1].update({"status": "unsupported", "reason": "compiler limitation"})
    model["build_profiles"][0]["compile_invocations"] = [{"id": "combined", "components": ["model", "second_model"]}]

    with pytest.raises(ValidationError, match="include unsupported components: second_model"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda model: model["components"][1]["source"].update({"sha256": "b" * 64}),
            "share one source content identity",
        ),
        (
            lambda model: model["build_profiles"][0]["components"][1].update({"runner_class": "expensive"}),
            "share one effective runner class",
        ),
    ],
)
def test_compile_invocation_components_must_share_compile_contract(change, expected: str) -> None:
    model = _two_component_model()
    model["build_profiles"][0]["compile_invocations"] = [{"id": "combined", "components": ["model", "second_model"]}]
    change(model)

    with pytest.raises(ValidationError, match=expected):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_compile_invocation_uses_the_first_components_shared_calibration() -> None:
    model = _two_component_model()
    model["calibrations"] = [
        {"id": "first", "status": "pending", "reason": "fixture"},
        {"id": "second", "status": "pending", "reason": "fixture"},
    ]
    model["build_profiles"][0]["compile_invocations"] = [{"id": "combined", "components": ["model", "second_model"]}]
    bindings = model["build_profiles"][0]["components"]
    bindings[1]["calibration"] = "first"

    with pytest.raises(ValidationError, match="shared calibration on its first component"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)

    bindings[0]["calibration"] = "first"
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)

    bindings[1]["calibration"] = "second"
    with pytest.raises(ValidationError, match="shared calibration on its first component"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize(
    "invocation",
    [
        {"id": "Not Valid", "components": ["model"]},
        {"id": "combined", "components": []},
        {"id": "combined", "components": ["model", "model"]},
    ],
)
def test_compile_invocation_schema_requires_normal_id_and_unique_non_empty_components(
    invocation: dict,
) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["build_profiles"][0]["compile_invocations"] = [invocation]

    with pytest.raises(ValidationError, match="compile_invocations"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_declarative_composition_validates_ports_and_package_contents() -> None:
    model = load_model(FIXTURES / "fixture_model")
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    model["compositions"] = [
        {
            "id": "pipeline",
            "type": "declarative",
            "components": [
                {"id": "first", "component": "model", "entrypoint": "default"},
                {"id": "second", "component": "second_model", "entrypoint": "default"},
            ],
            "connections": [
                {"from": {"component": "first", "output": "output"}, "to": {"component": "second", "input": "input"}}
            ],
            "inputs": [{"name": "input", "to": {"component": "first", "input": "input"}}],
            "outputs": [{"name": "output", "from": {"component": "second", "output": "output"}}],
        }
    ]
    model["build_profiles"][0]["compile_invocations"] = [{"id": "combined", "components": ["model", "second_model"]}]
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


@pytest.mark.parametrize("model_card", (
    "https://huggingface.co/example/model",
    "https://github.com/example/model",
    "https://docs.example.org/model.html",
))
def test_model_card_accepts_any_https_page(model_card: str) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["upstream"]["model_card"] = model_card
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize("model_card", ("http://example.org/model", "javascript:alert(1)", "hf://models/a/b"))
def test_model_card_must_be_https(model_card: str) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["upstream"]["model_card"] = model_card
    with pytest.raises(ValidationError, match="upstream.model_card"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_missing_compile_script_fails(tmp_path: Path) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURES / "fixture_model", directory)
    (directory / "scripts" / "compile.py").unlink()
    with pytest.raises(ValidationError, match="referenced file does not exist"):
        validate_model(directory, load_model(directory), SCHEMA)


def test_package_components_must_compile_in_one_invocation() -> None:
    model = _two_component_model()
    model["build_profiles"][0]["compile_invocations"] = [
        {"id": "first", "components": ["model"]},
        {"id": "second", "components": ["second_model"]},
    ]
    model["packages"][0]["components"] = ["model", "second_model"]
    with pytest.raises(ValidationError, match="in one compile invocation"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_recipe_must_live_in_scripts() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["components"][0]["recipe"] = "compile.py"
    with pytest.raises(ValidationError, match="recipe"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def _benchmark_entry(**settings) -> dict:
    return {
        "id": "latency",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
        "status": "provisional",
        "metrics": ["latency_avg_ms"],
        **settings,
    }


def test_benchmark_settings_are_validated() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["benchmark"] = [_benchmark_entry(unsupported_setting=True)]
    with pytest.raises(ValidationError, match="unsupported_setting.*unexpected"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_required_benchmark_metrics_must_be_declared() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["benchmark"] = [_benchmark_entry(required_metrics=["avg_power_watts"])]
    with pytest.raises(ValidationError, match="required_metrics are not declared"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_recorded_accuracy_entry_needs_no_engine() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["accuracy"] = [
        {
            "id": "owned",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
            "thresholds": {"minimum_cosine_similarity": 0.98},
        }
    ]
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def _fidelity_entry(**settings) -> dict:
    return {
        "id": "output_fidelity",
        "scope": {"type": "package", "package": "default"},
        "build_profiles": ["fixture"],
        "status": "diagnostic",
        "result_component": "model",
        "dataset": {"id": "fixture-set", "variant": "mini", "sha256": "a" * 64, "size_bytes": 1},
        "selection": {"method": "seeded_random_without_replacement", "seed": 1, "offset": 0, "samples": 2},
        "outputs": ["output"],
        **settings,
    }


def test_fidelity_entry_is_valid() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["fidelity"] = [_fidelity_entry()]
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"scope": {"type": "component", "component": "model"}}, "fidelity requires package scope"),
        ({"outputs": ["missing"]}, "fidelity outputs are not exposed by 'model': missing"),
        ({"engine": "model_recipe"}, "'engine' was unexpected"),
    ],
)
def test_fidelity_entry_is_validated(settings: dict, expected: str) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["fidelity"] = [_fidelity_entry(**settings)]
    with pytest.raises(ValidationError, match=expected):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


RUN_TESTS = {"run_tests": True}
EXPECTED_SCORE = {"run_tests": True, "allowed_drop": "5%", "expected": {"fixture": {"score": 0.5}}}
EXPECTED_THROUGHPUT = {
    "run_tests": True,
    "allowed_drop": "10%",
    "expected": {"fixture": {"model": {"throughput_fps": 9.0}}},
}


def _write_model_tests(directory: Path, tests: dict) -> None:
    (directory / "tests" / "tests.yaml").write_text(yaml.safe_dump(tests), encoding="utf-8")


def _devkit_accuracy_entry(**settings) -> dict:
    return {
        "id": "devkit_accuracy",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
        "engine": "devkit_script",
        "status": "ready",
        "result_component": "model",
        "dataset": {"id": "fixture-set", "variant": "mini", "sha256": "a" * 64, "size_bytes": 1},
        "metrics": ["score"],
        **settings,
    }


def _model_with_accuracy_script(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURES / "fixture_model", directory)
    (directory / "scripts" / "accuracy.py").write_text("score = 1.0\n", encoding="utf-8")
    _write_model_tests(directory, {"compile": RUN_TESTS, "accuracy": EXPECTED_SCORE})
    (directory / "tests" / "measure").mkdir()
    (directory / "tests" / "measure" / "test_accuracy.py").write_text("", encoding="utf-8")
    return directory, load_model(directory)


def test_devkit_accuracy_requires_its_script() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["accuracy"] = [_devkit_accuracy_entry()]
    with pytest.raises(ValidationError, match="referenced file does not exist.*accuracy.py"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


def test_devkit_accuracy_needs_no_benchmark_entry(tmp_path: Path) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    model["accuracy"] = [_devkit_accuracy_entry()]
    validate_model(directory, model, SCHEMA)


def test_devkit_accuracy_accepts_a_published_reference(tmp_path: Path) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    reference = {"source": "https://example.com/card", "metrics": {"score": 0.5}}
    model["accuracy"] = [_devkit_accuracy_entry(reference=reference)]
    validate_model(directory, model, SCHEMA)


def test_devkit_accuracy_may_share_scope_with_a_recorded_entry(tmp_path: Path) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    model["accuracy"] = [
        _devkit_accuracy_entry(),
        {"id": "owned", "scope": {"type": "component", "component": "model"}, "build_profiles": ["fixture"]},
    ]
    validate_model(directory, model, SCHEMA)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([_devkit_accuracy_entry(result_component="second_model")], "result_component 'second_model'"),
        ([_devkit_accuracy_entry(required_metrics=["missing"])], "required_metrics are not declared"),
        ([_devkit_accuracy_entry(metrics=["Score"])], "not valid under any of the given schemas"),
        (
            [_devkit_accuracy_entry(reference={"source": "https://example.com/card", "metrics": {"other": 0.5}})],
            "reference metrics are not declared in metrics: other",
        ),
        (
            [_devkit_accuracy_entry(reference={"source": "http://example.com/card", "metrics": {"score": 0.5}})],
            "not valid under any of the given schemas",
        ),
        (
            [
                _devkit_accuracy_entry(),
                _devkit_accuracy_entry(id="second", scope={"type": "package", "package": "default"}),
            ],
            "more than one devkit_script entry covers build profile 'fixture'",
        ),
    ],
)
def test_devkit_accuracy_entry_is_validated(tmp_path: Path, entries: list[dict], expected: str) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    model["accuracy"] = entries
    with pytest.raises(ValidationError, match=expected):
        validate_model(directory, model, SCHEMA)


def test_devkit_accuracy_package_must_be_compiled_in_one_invocation() -> None:
    directory = ROOT / "models" / "rf_detr_small"
    model = load_model(directory)
    # Without grouping, each component compiles in its own job, so no job holds every MPK the script needs.
    del model["build_profiles"][0]["compile_invocations"]
    with pytest.raises(ValidationError, match="must compile backbone, transformer in one compile invocation"):
        validate_model(directory, model, SCHEMA)


def _model_with_conversion(tmp_path: Path) -> tuple[Path, dict]:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURES / "fixture_model", directory)
    for relative in ("scripts/convert.py", "scripts/requirements-convert.txt"):
        (directory / relative).write_text("", encoding="utf-8")
    _write_model_tests(directory, {"conversion": RUN_TESTS, "compile": RUN_TESTS})
    (directory / "tests" / "build" / "test_conversion.py").write_text("", encoding="utf-8")
    model = load_model(directory)
    model["conversion"] = {"default": {"source": copy.deepcopy(model["components"][0]["source"])}}
    return directory, model


def test_conversion_sources_are_validated(tmp_path: Path) -> None:
    directory, model = _model_with_conversion(tmp_path)
    validate_model(directory, model, SCHEMA)
    model["conversion"]["default"]["config"] = "convert.yaml"
    with pytest.raises(ValidationError, match="conversion"):
        validate_model(directory, model, SCHEMA)


def test_conversion_sources_follow_source_policy(tmp_path: Path) -> None:
    directory, model = _model_with_conversion(tmp_path)
    model["conversion"]["default"]["source"]["uri"] = "https://unapproved.example/model.pth"
    with pytest.raises(ValidationError, match="source policy.*not approved"):
        validate_model(directory, model, SCHEMA)


@pytest.mark.parametrize(
    ("entrypoint", "paths", "expected"),
    [
        ("model.safetensors", ["model.safetensors", "model.safetensors"], "contains duplicate paths"),
        ("missing.safetensors", ["model.safetensors"], "entrypoint is not a declared file"),
    ],
)
def test_conversion_file_sets_are_checked(tmp_path: Path, entrypoint: str, paths: list[str], expected: str) -> None:
    directory, model = _model_with_conversion(tmp_path)
    member = {key: value for key, value in model["components"][0]["source"].items() if key != "type"}
    model["conversion"]["default"]["source"] = {
        "type": "file_set",
        "entrypoint": entrypoint,
        "files": [{**member, "path": path} for path in paths],
    }
    with pytest.raises(ValidationError, match=f"conversion 'default' source {expected}"):
        validate_model(directory, model, SCHEMA)


@pytest.mark.parametrize("missing", ["scripts/convert.py", "scripts/requirements-convert.txt"])
def test_declared_conversion_requires_its_workflow_files(tmp_path: Path, missing: str) -> None:
    directory, model = _model_with_conversion(tmp_path)
    (directory / missing).unlink()
    with pytest.raises(ValidationError, match=f"referenced file does not exist inside model directory: {missing}"):
        validate_model(directory, model, SCHEMA)


@pytest.mark.parametrize("stage", ["accuracy", "benchmark"])
def test_stage_entry_ids_must_be_unique(stage: str) -> None:
    model = load_model(FIXTURES / "fixture_model")
    first = _benchmark_entry() if stage == "benchmark" else {
        "id": "owned",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
    }
    second = {**copy.deepcopy(first), "build_profiles": ["fixture_bf16"]}
    model["build_profiles"].append({**copy.deepcopy(model["build_profiles"][0]), "id": "fixture_bf16"})
    model[stage] = [first, second]
    with pytest.raises(ValidationError, match=f"duplicate {stage} entry ids: {first['id']}"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize("entry", ["compile.py", "convert.yaml"])
def test_legacy_model_folder_entries_are_rejected(tmp_path: Path, entry: str) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURES / "fixture_model", directory)
    (directory / entry).write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match=f"unexpected model folder entries: {entry}.*scripts/"):
        load_and_validate_model(directory, SCHEMA, TAXONOMY)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda tests: (tests / "tests.yaml").unlink(), "tests/tests.yaml is required"),
        (
            lambda tests: _write_model_tests(tests.parent, {"conversion": RUN_TESTS, "compile": RUN_TESTS}),
            "set up exactly the stages this model has: unknown conversion",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": False}}),
            "compile: 'skip_reason' is a required property",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": False, "skip_reason": ""}}),
            "compile.skip_reason",
        ),
        (
            lambda tests: _write_model_tests(tests.parent, {"compile": {"run_tests": True, "skip_reason": "stale"}}),
            "compile: .* should not be valid",
        ),
        (lambda tests: (tests / "build" / "test_compile.py").unlink(), "tests/build/test_compile.py is missing"),
        (
            lambda tests: (tests / "build" / "test_helpers.py").write_text("", encoding="utf-8"),
            "unexpected test files: tests/build/test_helpers.py",
        ),
        # The model-level tests/correctness.yaml these folders once held is not a test file.
        (
            lambda tests: (tests / "correctness.yaml").write_text("", encoding="utf-8"),
            "unexpected test files: tests/correctness.yaml",
        ),
    ],
    ids=[
        "missing tests file",
        "unknown entry",
        "skipped without reason",
        "skipped with empty reason",
        "running with a skip reason",
        "running without its file",
        "unexpected build file",
        "legacy correctness file",
    ],
)
def test_model_tests_follow_their_tests_file(tmp_path: Path, change, expected: str) -> None:
    directory = tmp_path / "fixture_model"
    shutil.copytree(FIXTURES / "fixture_model", directory)
    change(directory / "tests")
    with pytest.raises(ValidationError, match=expected):
        load_and_validate_model(directory, SCHEMA, TAXONOMY)


def test_model_tests_set_up_every_stage_the_model_has(tmp_path: Path) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    model["accuracy"] = [_devkit_accuracy_entry()]
    model["benchmark"] = [_benchmark_entry()]
    with pytest.raises(ValidationError, match="set up exactly the stages this model has: missing performance"):
        validate_model(directory, model, SCHEMA)

    _write_model_tests(
        directory,
        {
            "compile": RUN_TESTS,
            "accuracy": EXPECTED_SCORE,
            "performance": {"run_tests": False, "skip_reason": "no published result"},
        },
    )
    validate_model(directory, model, SCHEMA)


@pytest.mark.parametrize(
    ("stage", "entry", "expected"),
    [
        ("accuracy", {**EXPECTED_SCORE, "expected": {"other": {"score": 0.5}}}, "expected.fixture lacks score"),
        ("accuracy", {**EXPECTED_SCORE, "expected": {"fixture": {"top1": 0.5}}}, "expected.fixture lacks score"),
        ("accuracy", {**EXPECTED_SCORE, "expected": {"fixture": {}}}, "accuracy.expected.fixture"),
        ("accuracy", {"run_tests": True, "expected": {"fixture": {"score": 0.5}}}, "'allowed_drop' is a required"),
        ("accuracy", {**EXPECTED_SCORE, "allowed_drop": "0.05"}, "accuracy.allowed_drop"),
        ("accuracy", {**EXPECTED_SCORE, "unknown": 1}, "accuracy: Unevaluated properties"),
        ("performance", {**EXPECTED_THROUGHPUT, "expected": {"fixture": {}}}, "expected.fixture.model lacks"),
        (
            "performance",
            {**EXPECTED_THROUGHPUT, "expected": {"fixture": {"model": {"latency_avg_ms": 1.0}}}},
            "performance.expected.fixture.model",
        ),
    ],
    ids=[
        "accuracy without its build profile",
        "accuracy without a declared metric",
        "accuracy without values",
        "missing allowed drop",
        "allowed drop not a percentage",
        "unknown setting",
        "performance without its component",
        "performance without its throughput",
    ],
)
def test_measured_stages_state_what_the_model_must_deliver(
    tmp_path: Path, stage: str, entry: dict, expected: str
) -> None:
    directory, model = _model_with_accuracy_script(tmp_path)
    model["accuracy"] = [_devkit_accuracy_entry()]
    model["benchmark"] = [_benchmark_entry()]
    (directory / "tests" / "measure" / "test_performance.py").write_text("", encoding="utf-8")
    tests = {"compile": RUN_TESTS, "accuracy": EXPECTED_SCORE, "performance": EXPECTED_THROUGHPUT}
    _write_model_tests(directory, tests)
    validate_model(directory, model, SCHEMA)

    _write_model_tests(directory, {**tests, stage: entry})
    with pytest.raises(ValidationError, match=expected):
        validate_model(directory, model, SCHEMA)


def test_yaml_uses_safe_loader(tmp_path: Path) -> None:
    directory = tmp_path / "unsafe"
    directory.mkdir()
    (directory / "model.yaml").write_text("!!python/object/apply:os.system ['echo unsafe']\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="could not load YAML"):
        load_model(directory)


def test_catalog_model_is_valid() -> None:
    model = load_and_validate_model(ROOT / "models" / "resnet_50", SCHEMA)
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


def test_model_rejects_obsolete_taxonomy_version() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["taxonomy_version"] = 4
    with pytest.raises(ValidationError, match="taxonomy_version.*unexpected"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_category_fields_are_required() -> None:
    model = load_model(FIXTURES / "fixture_model")
    for field in ("model_type", "tasks", "modalities", "architecture"):
        model.pop(field)
    with pytest.raises(ValidationError, match="'model_type' is a required property"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


def test_duplicate_component_roles_fail() -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["architecture"]["components"] = [
        {"role": "encoder", "architecture_id": "clip_vit_l_14"},
        {"role": "encoder", "architecture_id": "llama_2"},
    ]
    with pytest.raises(ValidationError, match=r"architecture.components\[1\].role duplicates"):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA, TAXONOMY)


@pytest.mark.parametrize("suffix", ["backbone_int8", "transformer_bf16", "model_preview"])
def test_component_binding_accepts_artifact_suffix(suffix: str) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["build_profiles"][0]["components"][0]["artifact_suffix"] = suffix
    validate_model(FIXTURES / "fixture_model", model, SCHEMA)


@pytest.mark.parametrize("suffix", [None, "", "../model", "Backbone", "backbone-int8", "model.tar.gz", "model\n"])
def test_component_binding_rejects_invalid_artifact_suffix(suffix) -> None:
    model = load_model(FIXTURES / "fixture_model")
    model["build_profiles"][0]["components"][0]["artifact_suffix"] = suffix
    with pytest.raises(ValidationError):
        validate_model(FIXTURES / "fixture_model", model, SCHEMA)
