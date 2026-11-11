from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from model_registry.compile_matrix import build_matrix
from model_registry.conversion import conversion_fingerprint
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.fingerprints import COMPILE_TOOLING, benchmark_fingerprint, compile_fingerprint
from model_registry.model import load_and_validate_model, load_models
from model_registry.registry import RegistryPublicationError, publish_catalog

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
MODEL_TESTS_SCHEMA = ROOT / "schemas" / "model-tests.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]


def _fingerprint(repository: Path, model_id: str, compiler_version: str) -> str:
    model = yaml.safe_load((repository / "models" / model_id / "model.yaml").read_text(encoding="utf-8"))
    profile_id = next(iter(model["build_profiles"]))
    return compile_fingerprint(repository, model_id, profile_id, profile_id, compiler_version)


def _write_model(repository: Path, model_id: str) -> None:
    model_dir = repository / "models" / model_id
    model_dir.mkdir(parents=True)
    shutil.copyfile(FIXTURE / "model.yaml", model_dir / "model.yaml")
    (model_dir / "scripts").mkdir()
    (model_dir / "scripts" / "compile.py").write_text("# fixture\n", encoding="utf-8")
    shutil.copytree(FIXTURE / "tests", model_dir / "tests")


def _declare_grouped_invocation(model: dict) -> None:
    """Make the loaded fixture compile two components in its profile's one invocation."""
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    model["build_profiles"][0]["compile_invocations"] = [
        {"id": "fixture", "components": ["model", "second_model"]}
    ]


def _declare_components(model: dict) -> None:
    """Make the authored fixture compile two components."""
    model["compilation"]["components"] = ["backbone", "decoder"]
    model["build_profiles"]["fixture"]["components"] = {
        "backbone": {"parameters": {}}, "decoder": {"parameters": {}},
    }


DATASET = {"id": "imagenet-2012", "variant": "mini", "sha256": "d" * 64, "size_bytes": 1}


@pytest.fixture
def model_repository(tmp_path: Path) -> Path:
    (tmp_path / "schemas").mkdir()
    for schema in (SCHEMA, MODEL_TESTS_SCHEMA, ROOT / "schemas" / "dataset-requirements.schema.json"):
        (tmp_path / "schemas" / schema.name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "taxonomies").mkdir()
    (tmp_path / "taxonomies" / "model-taxonomy.yaml").write_text(
        TAXONOMY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copytree(ROOT / "datasets", tmp_path / "datasets")
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps" / "manifest.json").write_text(
        (ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for relative in COMPILE_TOOLING:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, tmp_path / relative)
    _write_model(tmp_path, "alpha")
    _write_model(tmp_path, "bravo")
    return tmp_path


def test_discovery_and_matrix_are_stable(model_repository: Path) -> None:
    models = load_models(model_repository)
    matrix = build_matrix(models, ["bravo", "alpha", "alpha"])
    assert [entry["model_id"] for entry in matrix["include"]] == ["alpha", "bravo"]
    assert "compiler_label" not in matrix["include"][0]
    assert "compiler_version" not in matrix["include"][0]
    assert matrix["include"][0]["requested_runner_class"] == "default"
    # The one compile invocation of a build profile is named after the profile.
    assert matrix["include"][0]["compile_invocation_id"] == "fixture"
    assert matrix["include"][0]["component_ids"] == ["model"]
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["stages"] == []
    assert "compile_fingerprint" not in matrix["include"][0]


def test_matrix_entries_carry_the_stages_each_profile_benchmarks() -> None:
    models = {model_id: load_and_validate_model(ROOT / "models" / model_id, SCHEMA)
              for model_id in ("resnet_50", "rf_detr_small", "qwen3_5_0_8b")}

    matrix = build_matrix(models, sorted(models))

    assert [(entry["model_id"], entry["compile_invocation_id"], entry["component_ids"], entry["stages"])
            for entry in matrix["include"]] == [
        ("qwen3_5_0_8b", "w4a16", ["model"], []),
        ("resnet_50", "w16a16", ["model"], ["performance", "accuracy"]),
        ("resnet_50", "w8a16", ["model"], ["performance", "accuracy"]),
        ("resnet_50", "w8a8", ["model"], ["performance", "accuracy"]),
        ("rf_detr_small", "w8a16_w16a16", ["backbone", "decoder"], ["performance", "accuracy"]),
        ("rf_detr_small", "w8a8_mixed_w16a16", ["backbone", "decoder"], ["performance", "accuracy"]),
        ("rf_detr_small", "w8a8_w16a16", ["backbone", "decoder"], ["performance", "accuracy"]),
    ]


def test_missing_taxonomy_blocks_models_before_matrix(model_repository: Path) -> None:
    path = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    del model["tasks"]
    path.write_text(yaml.safe_dump(model), encoding="utf-8")

    with pytest.raises(ValidationError, match="tasks.*required"):
        load_models(model_repository)


def test_unknown_requested_model_fails(model_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown model id"):
        build_matrix(load_models(model_repository), ["missing"])


def test_matrix_can_select_requested_build_profiles(model_repository: Path) -> None:
    models = load_models(model_repository)
    second = copy.deepcopy(models["alpha"]["build_profiles"][0])
    second["id"] = "second"
    models["alpha"]["build_profiles"].append(second)
    matrix = build_matrix(models, ["alpha"], ["second"])
    assert [entry["build_profile_id"] for entry in matrix["include"]] == ["second"]


def test_matrix_carries_explicit_per_component_runner_class(model_repository: Path) -> None:
    models = load_models(model_repository)
    models["alpha"]["build_profiles"][0]["components"][0]["runner_class"] = "expensive"
    matrix = build_matrix(models, ["alpha"])
    assert matrix["include"][0]["requested_runner_class"] == "expensive"


def test_matrix_emits_one_row_per_declared_compile_invocation(model_repository: Path) -> None:
    models = load_models(model_repository)
    _declare_grouped_invocation(models["alpha"])

    matrix = build_matrix(models, ["alpha"])

    assert len(matrix["include"]) == 1
    assert matrix["include"][0]["compile_invocation_id"] == "fixture"
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["component_ids"] == ["model", "second_model"]


@pytest.mark.parametrize("component_id", ["model", "second_model"])
def test_component_filter_selects_the_full_compile_invocation(
    model_repository: Path, component_id: str
) -> None:
    models = load_models(model_repository)
    _declare_grouped_invocation(models["alpha"])

    matrix = build_matrix(models, ["alpha"], component_ids=[component_id])

    assert [entry["component_ids"] for entry in matrix["include"]] == [["model", "second_model"]]


def test_unknown_requested_build_profile_fails(model_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown build profile id"):
        build_matrix(load_models(model_repository), ["alpha"], ["missing"])


def test_compile_fingerprint_invalidates_model_dependency_and_compiler_changes(model_repository: Path) -> None:
    manifest = model_repository / "deps" / "manifest.json"
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    (model_repository / "models" / "alpha" / "scripts" / "compile.py").write_text("# changed\n", encoding="utf-8")
    model_changed = _fingerprint(model_repository, "alpha", "2.1.3")
    assert model_changed != first

    dependency = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    dependency["unrelated-publication-revision"] = 2
    manifest.write_text(json.dumps(dependency), encoding="utf-8")
    dependency_changed = _fingerprint(model_repository, "alpha", "2.1.3")
    assert dependency_changed == model_changed
    assert _fingerprint(model_repository, "alpha", "2.1.4") != dependency_changed


def test_calibrated_build_tracks_its_calibration_dataset_pin(model_repository: Path) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["calibrations"] = [{"id": "imagenet-2012-mini", "dataset": dict(DATASET)}]
    model["build_profiles"]["fixture"]["calibration_dataset"] = "imagenet-2012-mini"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    model["calibrations"][0]["dataset"]["sha256"] = "e" * 64
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") != first


def test_onnx_loader_changes_invalidate_the_compile_fingerprint(model_repository: Path) -> None:
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    loader = model_repository / "tools/model_registry/model.py"
    loader.write_text(loader.read_text(encoding="utf-8") + "\n# loader change\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") != first


def test_compile_invocation_fingerprint_includes_every_component_binding(
    model_repository: Path,
) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_components(model)
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3")

    model["build_profiles"]["fixture"]["components"]["decoder"]["parameters"]["output_name"] = "second"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3") != first


def test_numerical_fidelity_settings_invalidate_the_compile_fingerprint(model_repository: Path) -> None:
    """scripts/compile.py measures numerical fidelity, so its settings are compile inputs."""
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_components(model)
    model["numerical_fidelity"] = {
        "status": "diagnostic",
        "dataset": dict(DATASET),
        "selection": {"method": "seeded_random_without_replacement", "seed": 1, "offset": 0, "samples": 1},
        "outputs": ["output"],
    }
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3")

    model["numerical_fidelity"]["selection"]["samples"] = 2
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3") != first


def test_package_resources_invalidate_only_their_models_compile_fingerprint(model_repository: Path) -> None:
    """A package is assembled when its profile compiles, so a changed resource recompiles that profile."""
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    without = compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3")
    model["resources"] = {"runtime": {
        "source": {"uri": "hf://models/example/runtime@" + "0" * 40 + "/voices.json", "sha256": "a" * 64,
                   "size_bytes": 4},
        "install_path": "runtime/voices.json",
    }}
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3")

    model["resources"]["runtime"]["source"]["sha256"] = "b" * 64
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert first != without
    assert compile_fingerprint(model_repository, "alpha", "fixture", "fixture", "2.1.3") not in {first, without}


def test_compile_fingerprint_includes_model_local_recipe_helpers(model_repository: Path) -> None:
    helper = model_repository / "models" / "alpha" / "scripts" / "calibration.py"
    helper.write_text("# calibration helper\n", encoding="utf-8")
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    helper.write_text("# changed calibration helper\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") != first


def test_compile_fingerprint_ignores_non_transforming_registry_tooling_changes(
    model_repository: Path,
) -> None:
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    ignored = [
        model_repository / ".github" / "workflows" / "vulcan-model-orchestrator.yml",
        model_repository / "schemas" / "publication.schema.json",
        model_repository / "tools" / "model_registry" / "__init__.py",
        model_repository / "tools" / "model_registry" / "errors.py",
        model_repository / "tools" / "model_registry" / "fetch.py",
        model_repository / "tools" / "model_registry" / "registry.py",
        model_repository / "tools" / "model_registry" / "source.py",
        model_repository / "tools" / "runtime" / "execute.py",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# delivery-only change\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


def test_compile_fingerprint_ignores_model_documentation_and_board_stage_scripts(model_repository: Path) -> None:
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    model_dir = model_repository / "models" / "alpha"
    (model_dir / "README.md").write_text("documentation only\n", encoding="utf-8")
    (model_dir / "hero.png").write_bytes(b"documentation image only")
    (model_dir / "scripts" / "performance.py").write_text("report = model.benchmark()\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


def test_compile_fingerprint_ignores_display_name_and_description(model_repository: Path) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["display_name"] = "Renamed Fixture"
    model["description"] = "A rewritten one-line catalog description."
    manifest.write_text(yaml.safe_dump(model), encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


RELICENSED = {
    "spdx": "LicenseRef-MagicLeap-Noncommercial-Research",
    "redistribution": "restricted",
    "attribution": "Relicensed fixture.",
}


def test_compile_fingerprint_ignores_the_license(model_repository: Path) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert model["license"] != RELICENSED
    model["license"] = RELICENSED
    manifest.write_text(yaml.safe_dump(model), encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


RF_DETR_BUILDS = {
    component_id: {"run_id": f"run-{component_id}", "artifact_sha256": "a" * 64}
    for component_id in ("backbone", "decoder")
}


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
def test_benchmark_fingerprints_ignore_display_name_and_description(stage: str) -> None:
    model = load_and_validate_model(ROOT / "models" / "rf_detr_small", SCHEMA)
    renamed = {**model, "display_name": "Renamed", "description": "A rewritten one-line catalog description."}

    assert benchmark_fingerprint(ROOT, renamed, "w8a8_w16a16", RF_DETR_BUILDS, stage) == benchmark_fingerprint(
        ROOT, model, "w8a8_w16a16", RF_DETR_BUILDS, stage
    )


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
def test_benchmark_fingerprints_ignore_the_license(stage: str) -> None:
    model = load_and_validate_model(ROOT / "models" / "rf_detr_small", SCHEMA)
    relicensed = {**model, "license": RELICENSED}

    assert benchmark_fingerprint(ROOT, relicensed, "w8a8_w16a16", RF_DETR_BUILDS, stage) == benchmark_fingerprint(
        ROOT, model, "w8a8_w16a16", RF_DETR_BUILDS, stage
    )


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
def test_a_whole_model_benchmark_fingerprint_covers_every_component_build(stage: str) -> None:
    model = load_and_validate_model(ROOT / "models" / "rf_detr_small", SCHEMA)
    rebuilt = {**RF_DETR_BUILDS, "backbone": {"run_id": "run-backbone-2", "artifact_sha256": "b" * 64}}
    fingerprint = benchmark_fingerprint(ROOT, model, "w8a8_w16a16", RF_DETR_BUILDS, stage)

    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", rebuilt, stage) != fingerprint
    assert benchmark_fingerprint(ROOT, model, "w8a16_w16a16", RF_DETR_BUILDS, stage) != fingerprint
    with pytest.raises(ValidationError, match="needs compiled component 'decoder'"):
        benchmark_fingerprint(ROOT, model, "w8a8_w16a16", {"backbone": RF_DETR_BUILDS["backbone"]}, stage)


def test_benchmark_fingerprint_requires_a_declared_stage() -> None:
    model = load_and_validate_model(ROOT / "models" / "yolo26n", SCHEMA)
    del model["accuracy"]
    with pytest.raises(ValidationError, match="yolo26n declares no accuracy"):
        benchmark_fingerprint(ROOT, model, "w8a8", {"model": RF_DETR_BUILDS["backbone"]}, "accuracy")


def _copy_repository(repository: Path, model_id: str) -> Path:
    """Copy what the model's fingerprints read into ``repository``; return the copied model directory."""
    for relative in ("schemas", "taxonomies", "datasets", "deps", "scripts", "tools", ".github/workflows",
                     f"models/{model_id}"):
        shutil.copytree(ROOT / relative, repository / relative, ignore=shutil.ignore_patterns("__pycache__"))
    return repository / "models" / model_id


def test_accuracy_fingerprint_covers_the_model_datasets_and_their_requirements(tmp_path: Path) -> None:
    """Accuracy reruns when what the model must report on its datasets changes, and for nothing else there."""
    model_dir = _copy_repository(tmp_path, "rf_detr_small")
    coco = tmp_path / "datasets" / "coco-2017.yaml"
    manifest = model_dir / "model.yaml"
    authored = {path: path.read_text(encoding="utf-8") for path in (coco, manifest)}
    # The tests' expectations name the datasets; skipping them lets the model declare another one.
    tests = model_dir / "tests" / "tests.yaml"
    document = yaml.safe_load(tests.read_text(encoding="utf-8"))
    document["accuracy"] = {"run_tests": False, "skip_reason": "fixture"}
    tests.write_text(yaml.safe_dump(document), encoding="utf-8")

    def fingerprint() -> str:
        model = load_and_validate_model(model_dir, tmp_path / "schemas" / "model.schema.json")
        return benchmark_fingerprint(tmp_path, model, "w8a8_w16a16", RF_DETR_BUILDS, "accuracy")

    def edit(path: Path, change: Callable[[dict], None]) -> str:
        for original, text in authored.items():
            original.write_text(text, encoding="utf-8")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        change(document)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return fingerprint()

    def second_dataset(model: dict) -> None:
        model["accuracy"]["datasets"].append({**model["accuracy"]["datasets"][0], "variant": "mini"})

    first = fingerprint()
    rerun = {
        "the model's task requirement": edit(
            coco, lambda d: d["required_metrics"]["object_detection"].append("coco_ap50")
        ),
        "the dataset pin": edit(manifest, lambda d: d["accuracy"]["datasets"][0].update(sha256="e" * 64)),
        "the dataset list": edit(manifest, second_dataset),
    }
    unchanged = {
        "another task's requirement": edit(
            coco, lambda d: d["required_metrics"]["pose_estimation"].append("coco_keypoint_ap50")
        ),
        "another dataset file": edit(tmp_path / "datasets" / "imagenet-2012.yaml",
                                     lambda d: d["required_metrics"]["image_classification"].append("top5_accuracy")),
        "the build profiles": edit(manifest, lambda d: d["accuracy"].update(build_profiles=["w8a8_w16a16"])),
    }

    assert [change for change, value in rerun.items() if value == first] == []
    assert [change for change, value in unchanged.items() if value != first] == []


def test_fingerprints_ignore_the_models_tests(tmp_path: Path) -> None:
    """A changed test applies the next time its stage runs; it reruns no stage by itself."""
    repository = tmp_path / "repository"
    model_dir = _copy_repository(repository, "rf_detr_small")

    def fingerprints() -> list[str]:
        model = load_and_validate_model(model_dir, repository / "schemas" / "model.schema.json")
        selected = {"rf_detr_small": {
            "sha256": "a" * 64, "conversion_fingerprint": conversion_fingerprint(repository, model),
        }}
        return [
            compile_fingerprint(
                repository, "rf_detr_small", "w8a8_w16a16", "w8a8_w16a16", COMPILER_VERSION,
                conversion_sources=selected,
            ),
            benchmark_fingerprint(repository, model, "w8a8_w16a16", RF_DETR_BUILDS, "performance"),
            benchmark_fingerprint(repository, model, "w8a8_w16a16", RF_DETR_BUILDS, "accuracy"),
        ]

    first = fingerprints()
    for path in sorted((model_dir / "tests").rglob("*")):
        if path.is_file():
            path.write_text(path.read_text(encoding="utf-8") + "\n# revised\n", encoding="utf-8")
    (model_dir / "tests" / "fixtures").mkdir()
    (model_dir / "tests" / "fixtures" / "detections.json").write_text("[]\n", encoding="utf-8")

    assert fingerprints() == first


@pytest.mark.parametrize("stage", ["performance", "accuracy"])
def test_benchmark_requirements_rerun_only_their_stage(tmp_path: Path, stage: str) -> None:
    model_dir = _copy_repository(tmp_path, "rf_detr_small")
    model = load_and_validate_model(model_dir)
    selected = {"rf_detr_small": {
        "sha256": "a" * 64, "conversion_fingerprint": conversion_fingerprint(tmp_path, model),
    }}

    def fingerprints() -> dict[str, str]:
        return {
            "conversion": conversion_fingerprint(tmp_path, model),
            "compile": compile_fingerprint(
                tmp_path, "rf_detr_small", "w8a8_w16a16", "w8a8_w16a16", COMPILER_VERSION,
                conversion_sources=selected,
            ),
            **{name: benchmark_fingerprint(tmp_path, model, "w8a8_w16a16", RF_DETR_BUILDS, name)
               for name in ("performance", "accuracy")},
        }

    requirements = model_dir / "scripts" / f"requirements-{stage}.txt"
    requirements.unlink(missing_ok=True)
    before = fingerprints()
    for content in ("benchmark-wheel==1.0\n", "benchmark-wheel==2.0\n"):
        requirements.write_text(content)
        after = fingerprints()
        assert [name for name in before if before[name] != after[name]] == [stage]
    requirements.unlink()
    assert fingerprints() == before


@pytest.mark.parametrize("through_helper", [False, True])
def test_accuracy_fingerprint_tracks_imported_performance_code(tmp_path, through_helper):
    model_dir = _copy_repository(tmp_path, "rf_detr_small")
    scripts = model_dir / "scripts"
    accuracy = scripts / "accuracy.py"
    if through_helper:
        (scripts / "shared.py").write_text("from performance import options\n")
        accuracy.write_text("from shared import options\n")
    else:
        accuracy.write_text("from performance import options\n")
    model = load_and_validate_model(model_dir)
    selected = {"rf_detr_small": {"sha256": "a" * 64,
                                 "conversion_fingerprint": conversion_fingerprint(tmp_path, model)}}

    def fingerprints():
        return {"compile": compile_fingerprint(tmp_path, model["id"], "w8a8_w16a16", "w8a8_w16a16",
                                                COMPILER_VERSION, conversion_sources=selected),
                **{stage: benchmark_fingerprint(tmp_path, model, "w8a8_w16a16", RF_DETR_BUILDS, stage)
                   for stage in ("performance", "accuracy")}}

    before = fingerprints()
    performance = scripts / "performance.py"
    performance.write_text(performance.read_text() + "\n# changed prompt preparation\n")
    after = fingerprints()
    assert after["compile"] == before["compile"]
    assert after["performance"] != before["performance"]
    assert after["accuracy"] != before["accuracy"]
    accuracy.write_text("# independently implemented accuracy\n")
    before = fingerprints()
    performance.write_text(performance.read_text() + "\n# another change\n")
    assert fingerprints()["accuracy"] == before["accuracy"]


def test_matrix_records_fingerprints_without_catalog_carry_forward(model_repository: Path) -> None:
    models = load_models(model_repository)
    matrix = build_matrix(
        models,
        ["alpha", "bravo"],
        repository=model_repository,
        compiler_version="2.1.3",
    )

    assert {entry["model_id"] for entry in matrix["include"]} == {"alpha", "bravo"}
    assert all(len(entry["compile_fingerprint"]) == 64 for entry in matrix["include"])

@pytest.mark.parametrize("source", ["base", "target", "stale", "missing", "forbidden"])
def test_publish_fresh_branch_reuses_only_compatible_catalog_runs(model_repository: Path, source: str) -> None:
    models = load_models(model_repository)
    matrix = build_matrix(
        models,
        sorted(models),
        repository=model_repository,
        compiler_version=COMPILER_VERSION,
    )
    prior = {"models": [
        {"id": model_id, "builds": [
            {
                "build_profile_id": entry["build_profile_id"], "component_id": component_id,
                "compile_fingerprint": entry["compile_fingerprint"], "artifact_sha256": "a" * 64,
                "run_id": f"run-{model_id}-{component_id}",
            }
            for entry in matrix["include"] if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in models
    ]}
    if source == "stale":
        prior["models"][0]["builds"][0]["compile_fingerprint"] = "0" * 64
    reads, writes = [], []

    def read(url):
        reads.append(url)
        if source == "forbidden":
            raise RegistryPublicationError("public artifact registry HTTP 403 forbidden")
        if source != "target" and (len(reads) == 1 or source == "missing"):
            raise RegistryPublicationError("public artifact registry HTTP 404 catalog_not_found")
        return {"catalog": {"document": prior}}

    def write(method, path, body, key):
        writes.append(body)
        return {"stored": True}

    def publish():
        return publish_catalog(
            repository=model_repository, api_url="https://writer.example.invalid",
            read_api_url="https://reader.example.invalid", aws_region="us-west-2",
            branch="feat/names", base_branches=["develop"], git_sha="b" * 40,
            request=write, read_request=read,
        )

    if source == "forbidden":
        with pytest.raises(RegistryPublicationError, match="403 forbidden"):
            publish()
        assert writes == []
    else:
        result = publish()
        assert result["catalog"]["git_ref"] == "refs/heads/feat/names"
        assert result["catalog"]["catalog_revision"] == "b" * 40
        if source in {"stale", "missing"}:
            expected_pending = {"alpha"} if source == "stale" else set(models)
            assert {pending["model_id"] for pending in result["catalog"]["pending_builds"]} == expected_pending
        else:
            assert result["catalog"]["pending_builds"] == []
            for current, previous in zip(result["catalog"]["models"], prior["models"], strict=True):
                for build, old_build in zip(current["builds"], previous["builds"], strict=True):
                    assert build["run_id"] == old_build["run_id"]
                    assert build["artifact_sha256"] == old_build["artifact_sha256"]
                    assert build["compile_fingerprint"] == old_build["compile_fingerprint"]
        assert writes == [{
            "initialize": True,
            "previous_snapshot": None,
            "catalog": result["catalog"],
        }]
    assert reads[0].endswith("git_ref=refs%2Fheads%2Ffeat%2Fnames")
    if source == "forbidden":
        assert len(reads) == 1
    else:
        # A feature branch also reads its base branches, even with a catalog of its own.
        assert len(reads) == 2
        assert reads[1].endswith("git_ref=refs%2Fheads%2Fdevelop")


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/performance.py",
        "scripts/requirements-performance.txt",
        "scripts/accuracy.py",
        "scripts/fp32_baseline_accuracy.py",
        "scripts/requirements-accuracy.txt",
        "scripts/requirements-performance.txt",
        "scripts/convert.py",
        "scripts/requirements-convert.txt",
    ],
)
def test_compile_fingerprint_ignores_board_stage_and_conversion_files(model_repository: Path, relative: str) -> None:
    path = model_repository / "models" / "alpha" / relative
    path.write_text("# initial\n", encoding="utf-8")
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    path.write_text("# revised\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


@pytest.mark.parametrize("scope", [None, "components"])
def test_compile_fingerprint_ignores_performance_and_accuracy_declarations(
    model_repository: Path, scope: str | None,
) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["performance"] = {"scope": scope} if scope else {}
    model["accuracy"] = {"datasets": [dict(DATASET)]}
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    (manifest.parent / "scripts" / "performance.py").write_text("report = model.benchmark()\n", encoding="utf-8")
    (manifest.parent / "scripts" / "accuracy.py").write_text("score = 0.5\n", encoding="utf-8")
    skipped = {"run_tests": False, "skip_reason": "fixture"}
    (manifest.parent / "tests" / "tests.yaml").write_text(
        yaml.safe_dump({"compile": {"run_tests": True}, "accuracy": skipped, "performance": skipped}), encoding="utf-8"
    )

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


@pytest.mark.parametrize(
    ("relative", "onnx_changed"),
    [
        ("scripts/run_llima_compilation.sh", False),
        ("tools/model_registry/artifact.py", False),
        ("tools/model_registry/cli.py", False),
        ("tools/model_registry/model.py", True),
        ("tools/model_registry/mla_stage_simulation.py", True),
        ("tools/model_registry/compile_matrix.py", False),
        ("models/qwen3_5_0_8b/scripts/compile.sh", False),
        ("scripts/run_model_compilation.sh", True),
    ],
)
def test_compile_tooling_is_scoped_to_backend(tmp_path, relative, onnx_changed):
    native_id = "qwen3_5_0_8b"
    native = load_and_validate_model(ROOT / "models" / native_id)
    entry = build_matrix({native_id: native}, [native_id])["include"][0]
    assert entry["runner_script"] == "scripts/run_llima_compilation.sh"
    assert entry["stages"] == []
    repository = tmp_path / "repository"
    for folder in ("schemas", "taxonomies", "datasets", "deps", "scripts", "tools"):
        shutil.copytree(ROOT / folder, repository / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for name in (native_id, "resnet_50"):
        shutil.copytree(ROOT / "models" / name, repository / "models" / name)

    def fingerprints():
        return [
            compile_fingerprint(repository, name, profile, profile, "compiler")
            for name, profile in ((native_id, "w4a16"), ("resnet_50", "w16a16"))
        ]

    before = fingerprints()
    path = repository / relative
    path.write_text(path.read_text() + "\n# tooling change\n")
    after = fingerprints()
    assert before[0] == after[0]
    assert (before[1] != after[1]) == onnx_changed


@pytest.mark.parametrize(("model_id", "profile_id"), [
    ("qwen3_5_0_8b", "w4a16"), ("resnet_50", "w16a16"),
])
def test_checkpoint_repository_name_does_not_change_compile_fingerprint(tmp_path, model_id, profile_id):
    for folder in ("schemas", "taxonomies", "datasets", "deps", "scripts", "tools", f"models/{model_id}"):
        shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = tmp_path / "models" / model_id / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    before = compile_fingerprint(tmp_path, model_id, profile_id, profile_id, "compiler")
    # A Hugging Face checkpoint names its repository in its source; an ONNX model names it as upstream.
    repository = model["upstream"] if "upstream" in model else model["compilation"]["source"]
    repository["repository"] += "-renamed"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False))

    assert compile_fingerprint(tmp_path, model_id, profile_id, profile_id, "compiler") == before


@pytest.mark.parametrize(("setting", "value"), [
    ("source", "a" * 64),
    ("compiler", "3.0.0+develop.111111111111"),
    ("precision", "a16w8"),
    ("max_num_tokens", 8192),
    ("language_group_size", 64),
    ("input_height", 512),
    ("input_width", 640),
])
def test_llima_compile_fingerprint_tracks_source_compiler_and_graph_settings(tmp_path, setting, value):
    model_id = "qwen3_5_0_8b"
    for folder in ("schemas", "taxonomies", "datasets", f"models/{model_id}"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    manifest = tmp_path / "models" / model_id / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    compiler = "3.0.0+develop.000000000000"
    before = compile_fingerprint(tmp_path, model_id, "w4a16", "w4a16", compiler)
    if setting == "source":
        next(iter(model["compilation"]["source"]["files"].values()))["sha256"] = value
    elif setting == "compiler":
        compiler = value
    else:
        model["build_profiles"]["w4a16"]["components"]["model"]["parameters"][setting] = value
    manifest.write_text(yaml.safe_dump(model, sort_keys=False))

    assert compile_fingerprint(tmp_path, model_id, "w4a16", "w4a16", compiler) != before


def test_market_inheritance_and_fingerprint_stability(model_repository):
    path = model_repository / "models/alpha/model.yaml"
    authored = yaml.safe_load(path.read_text())
    before = _fingerprint(model_repository, "alpha", COMPILER_VERSION)
    profile = authored["build_profiles"]["fixture"]
    profile["runner_market"] = "on-demand"
    path.write_text(yaml.safe_dump(authored))
    model = load_and_validate_model(path.parent)
    assert build_matrix({"alpha": model}, ["alpha"])["include"][0]["requested_runner_market"] == "on-demand"
    assert _fingerprint(model_repository, "alpha", COMPILER_VERSION) == before
    profile["components"]["model"]["runner_market"] = "spot"
    path.write_text(yaml.safe_dump(authored))
    model = load_and_validate_model(path.parent)
    assert build_matrix({"alpha": model}, ["alpha"])["include"][0]["requested_runner_market"] == "spot"
    assert _fingerprint(model_repository, "alpha", COMPILER_VERSION) == before


def test_grouped_compile_rejects_conflicting_markets(model_repository):
    path = model_repository / "models/alpha/model.yaml"
    authored = yaml.safe_load(path.read_text())
    _declare_components(authored)
    authored["build_profiles"]["fixture"]["components"]["decoder"]["runner_market"] = "on-demand"
    path.write_text(yaml.safe_dump(authored))
    with pytest.raises(ValidationError, match="conflicting runner_market"):
        load_and_validate_model(path.parent)


def test_speculative_compile_fingerprint_tracks_weights_not_benchmark_baseline(tmp_path):
    model_id = "gemma_4_e2b_it_textonly_mtp"
    for folder in ("schemas", "taxonomies", f"models/{model_id}", "models/gemma_4_e2b_it_textonly"):
        shutil.copytree(ROOT / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = tmp_path / "models" / model_id / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    compiler = "3.0.0+develop.000000000000"
    before = compile_fingerprint(tmp_path, model_id, "w4a16", "w4a16", compiler)
    model["build_profiles"]["w4a16"].pop("benchmark_baseline")
    manifest.write_text(yaml.safe_dump(model, sort_keys=False))
    assert compile_fingerprint(tmp_path, model_id, "w4a16", "w4a16", compiler) == before
    draft = model["build_profiles"]["w4a16"]["components"]["model"]["parameters"]["draft_model_path"]
    model["compilation"]["source"]["files"][f"{draft}/model.safetensors"]["sha256"] = "0" * 64
    manifest.write_text(yaml.safe_dump(model, sort_keys=False))
    assert compile_fingerprint(tmp_path, model_id, "w4a16", "w4a16", compiler) != before


def test_benchmark_baseline_changes_only_its_profiles_performance_fingerprint():
    model = load_and_validate_model(ROOT / "models/rf_detr_small")
    builds = {name: {"run_id": name, "artifact_sha256": "a" * 64} for name in ("backbone", "decoder")}
    before = benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "performance")
    other = benchmark_fingerprint(ROOT, model, "w8a16_w16a16", builds, "performance")
    accuracy = benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "accuracy")
    model["build_profiles"][0]["benchmark_baseline"] = {"model_id": "resnet_50", "build_profile": "w16a16"}
    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "performance") != before
    assert benchmark_fingerprint(ROOT, model, "w8a16_w16a16", builds, "performance") == other
    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "accuracy") == accuracy
    model["build_profiles"][0].pop("benchmark_baseline")
    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "performance") == before


def test_performance_scope_changes_performance_fingerprint_without_changing_accuracy():
    model = load_and_validate_model(ROOT / "models/rf_detr_small")
    builds = {name: {"run_id": name, "artifact_sha256": "a" * 64} for name in ("backbone", "decoder")}
    before = benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "performance")
    accuracy = benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "accuracy")

    model["performance"]["scope"] = "components"

    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "performance") != before
    assert benchmark_fingerprint(ROOT, model, "w8a8_w16a16", builds, "accuracy") == accuracy


def test_yolo26x_pose_requests_moderate_on_demand():
    model = load_and_validate_model(ROOT / "models/yolo26x_pose", SCHEMA)
    entries = build_matrix({"yolo26x_pose": model}, ["yolo26x_pose"])["include"]
    assert {entry["build_profile_id"]: (
        entry["requested_runner_class"], entry["requested_runner_market"]
    ) for entry in entries} == {
        "w8a16": ("moderate", "on-demand"),
        "w8a8": ("default", "spot"),
        "w16a16": ("default", "spot"),
    }


@pytest.mark.parametrize("runner_class", ["moderate"])
def test_new_runner_classes_do_not_change_compile_fingerprint(model_repository, runner_class):
    path = model_repository / "models/alpha/model.yaml"
    before = _fingerprint(model_repository, "alpha", COMPILER_VERSION)
    authored = yaml.safe_load(path.read_text())
    profile = next(iter(authored["build_profiles"].values()))
    profile["runner_class"] = runner_class
    path.write_text(yaml.safe_dump(authored, sort_keys=False))
    load_and_validate_model(path.parent, model_repository / "schemas/model.schema.json")
    assert _fingerprint(model_repository, "alpha", COMPILER_VERSION) == before
