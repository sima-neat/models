from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import (
    benchmark_fingerprint,
    build_matrix,
    compile_invocation_fingerprint,
    component_build_fingerprint,
    discover_catalog,
    git_changed_paths,
    resolve_base_branch,
    select_model_ids,
    semantic_change_plan,
    validation_fingerprint,
)
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.registry import RegistryPublicationError, build_catalog_snapshot, publish_catalog

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
PERFORMANCE_SCHEMA = ROOT / "schemas" / "performance-config-v1.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]


def _fingerprint(repository: Path, model_id: str, compiler_version: str) -> str:
    model = yaml.safe_load((repository / "models" / model_id / "model.yaml").read_text(encoding="utf-8"))
    return component_build_fingerprint(
        repository,
        model_id,
        "model",
        model["build_profiles"][0]["id"],
        compiler_version,
    )


def _write_model(repository: Path, model_id: str, *, declared_id: str | None = None) -> None:
    model_dir = repository / "models" / model_id
    model_dir.mkdir(parents=True)
    model = yaml.safe_load((FIXTURE / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = declared_id or model_id
    (model_dir / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    (model_dir / "compile.py").write_text("# fixture\n", encoding="utf-8")


def _declare_grouped_invocation(model: dict) -> None:
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    model["build_profiles"][0]["compile_invocations"] = [
        {"id": "combined", "components": ["model", "second_model"]}
    ]


def _commit(repository: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repository, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _initialize_repository(repository: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=repository, check=True)
    return _commit(repository, "base")


@pytest.fixture
def catalog_repository(tmp_path: Path) -> Path:
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "model.schema.json").write_text(SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "schemas" / "performance-config-v1.schema.json").write_text(
        PERFORMANCE_SCHEMA.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "schemas" / "recipe-fidelity-v1.schema.json").write_text(
        (ROOT / "schemas" / "recipe-fidelity-v1.schema.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "taxonomies").mkdir()
    (tmp_path / "taxonomies" / "model-taxonomy.yaml").write_text(
        TAXONOMY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps" / "manifest.json").write_text(
        (ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "tools" / "model_registry").mkdir(parents=True)
    (tmp_path / "tools" / "model_registry" / "catalog.py").write_text(
        "COMPILE_FINGERPRINT_SCHEMA = 3\n"
        "VALIDATION_FINGERPRINT_SCHEMA = 1\n"
        "BENCHMARK_FINGERPRINT_SCHEMA = 1\n",
        encoding="utf-8",
    )
    _write_model(tmp_path, "alpha")
    _write_model(tmp_path, "bravo")
    return tmp_path


def test_discovery_and_matrix_are_stable(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    matrix = build_matrix(catalog, ["bravo", "alpha", "alpha"])
    assert [entry["model_id"] for entry in matrix["include"]] == ["alpha", "bravo"]
    assert "compiler_label" not in matrix["include"][0]
    assert "compiler_version" not in matrix["include"][0]
    assert matrix["include"][0]["recipe"] == "compile.py"
    assert matrix["include"][0]["requested_runner_class"] == "default"
    assert matrix["include"][0]["compile_invocation_id"] == "model"
    assert matrix["include"][0]["component_ids"] == ["model"]
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["benchmark_component_ids"] == []
    assert matrix["include"][0]["explicit_invocation"] is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("models/alpha/model.yaml", ["alpha"]),
        ("models/alpha/compile.py", ["alpha"]),
        ("models/deleted/model.yaml", []),
        ("design-doc/model-registry-architecture.md", []),
    ],
)
def test_model_local_added_modified_and_deleted_paths(catalog_repository: Path, path: str, expected: list[str]) -> None:
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == expected


@pytest.mark.parametrize(
    "path",
    [
        "training/yolo26_tiny_drone/compile.py",
        "training/yolo26_tiny_drone/contract.py",
        "training/yolo26_tiny_drone/dataset.py",
        "training/yolo26_tiny_drone/onnx_audit.py",
    ],
)
def test_yolo26_training_inputs_select_only_the_tiny_drone_model(catalog_repository: Path, path: str) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")

    assert select_model_ids(discover_catalog(catalog_repository), [path]) == ["yolo26_tiny_drone"]


@pytest.mark.parametrize(
    "path",
    [
        "training/yolo26_tiny_drone/evaluate.py",
        "training/yolo26_tiny_drone/train_qat.py",
        "training/yolo26_tiny_drone_extra/compile.py",
        "training/unrelated_model/compile.py",
    ],
)
def test_unrelated_training_inputs_do_not_select_tiny_drone(catalog_repository: Path, path: str) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")

    assert select_model_ids(discover_catalog(catalog_repository), [path]) == []


@pytest.mark.parametrize(
    "path",
    [
        "deps/manifest.json",
    ],
)
def test_global_compile_inputs_select_every_model(catalog_repository: Path, path: str) -> None:
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == ["alpha", "bravo"]


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/Vulcan-Model-Compile.yml",
        "pyproject.toml",
        "schemas/model.schema.json",
        "tools/model_registry/__init__.py",
        "tools/model_registry/artifact.py",
        "tools/model_registry/catalog.py",
        "tools/model_registry/cli.py",
        "tools/model_registry/errors.py",
        "tools/model_registry/fetch.py",
        "tools/model_registry/model.py",
        "tools/model_registry/registry.py",
        "tools/model_registry/source.py",
        "tools/runtime/execute.py",
    ],
)
def test_orchestration_and_publication_paths_do_not_select_models(catalog_repository: Path, path: str) -> None:
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == []


@pytest.mark.parametrize(
    "path",
    [
        "tools/model_registry/__init__.py",
        "tools/model_registry/errors.py",
        "tools/model_registry/fetch.py",
        "tools/model_registry/model.py",
        "tools/model_registry/source.py",
    ],
)
def test_registry_control_plane_changes_do_not_select_models(catalog_repository: Path, path: str) -> None:
    # These modules validate, locate, or transport immutable inputs; they do not
    # transform the size- and SHA-256-pinned bytes passed to Model Compiler.
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == []


def test_mismatched_directory_id_fails(catalog_repository: Path) -> None:
    model = yaml.safe_load((catalog_repository / "models" / "alpha" / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = "bravo"
    (catalog_repository / "models" / "alpha" / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ValidationError, match="id must match directory name"):
        discover_catalog(catalog_repository)


def test_missing_taxonomy_blocks_catalog_before_matrix(catalog_repository: Path) -> None:
    path = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    del model["tasks"]
    path.write_text(yaml.safe_dump(model), encoding="utf-8")

    with pytest.raises(ValidationError, match="tasks.*required"):
        discover_catalog(catalog_repository)


def test_duplicate_catalog_ids_fail_before_matrix(catalog_repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from model_registry import catalog as catalog_module

    original = catalog_module.load_and_validate_model

    def duplicate_id(directory: Path, schema: Path):
        model = copy.deepcopy(original(directory, schema))
        model["id"] = "duplicate"
        return model

    monkeypatch.setattr(catalog_module, "load_and_validate_model", duplicate_id)
    with pytest.raises(ValidationError, match="duplicate model id"):
        discover_catalog(catalog_repository)


def test_unknown_requested_model_fails(catalog_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown model id"):
        build_matrix(discover_catalog(catalog_repository), ["missing"])


def test_matrix_can_select_requested_build_profiles(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    second = copy.deepcopy(catalog["alpha"]["build_profiles"][0])
    second["id"] = "second"
    catalog["alpha"]["build_profiles"].append(second)
    matrix = build_matrix(catalog, ["alpha"], ["second"])
    assert [entry["build_profile_id"] for entry in matrix["include"]] == ["second"]


def test_matrix_carries_explicit_per_component_runner_class(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["build_profiles"][0]["components"][0]["runner_class"] = "expensive"
    matrix = build_matrix(catalog, ["alpha"])
    assert matrix["include"][0]["requested_runner_class"] == "expensive"


def test_matrix_emits_one_row_per_declared_compile_invocation(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    _declare_grouped_invocation(catalog["alpha"])

    matrix = build_matrix(catalog, ["alpha"])

    assert len(matrix["include"]) == 1
    assert matrix["include"][0]["compile_invocation_id"] == "combined"
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["component_ids"] == ["model", "second_model"]
    assert matrix["include"][0]["explicit_invocation"] is True


def test_explicit_singleton_uses_invocation_recipe_contract(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["build_profiles"][0]["compile_invocations"] = [
        {"id": "export_and_compile", "components": ["model"]}
    ]

    entry = build_matrix(catalog, ["alpha"])["include"][0]

    assert entry["explicit_invocation"] is True
    assert entry["compile_invocation_id"] == "export_and_compile"
    assert entry["component_ids"] == ["model"]


def test_rf_detr_compile_matrix_needs_no_exporter_packages() -> None:
    catalog = discover_catalog(ROOT)
    matrix = build_matrix(catalog, ["rf_detr_small"])
    assert matrix["include"][0]["python_packages"] == []


@pytest.mark.parametrize("component_id", ["model", "second_model"])
def test_component_filter_selects_the_full_compile_invocation(
    catalog_repository: Path, component_id: str
) -> None:
    catalog = discover_catalog(catalog_repository)
    _declare_grouped_invocation(catalog["alpha"])

    matrix = build_matrix(catalog, ["alpha"], component_ids=[component_id])

    assert [entry["component_ids"] for entry in matrix["include"]] == [["model", "second_model"]]


def test_matrix_excludes_explicitly_unsupported_component_builds(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    unsupported = copy.deepcopy(catalog["alpha"]["build_profiles"][0])
    unsupported["id"] = "unsupported"
    unsupported["components"][0].update({"status": "unsupported", "reason": "compiler limitation"})
    catalog["alpha"]["build_profiles"].append(unsupported)
    matrix = build_matrix(catalog, ["alpha"])
    assert [entry["build_profile_id"] for entry in matrix["include"]] == ["fixture"]


def test_explicit_unsupported_component_is_excluded(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["build_profiles"][0]["components"][0].update(
        {"status": "unsupported", "reason": "compiler zeroed weights"}
    )
    assert build_matrix(catalog, ["alpha"], ["fixture"])["include"] == []


def test_requested_profile_builds_supported_models_and_skips_unsupported_components(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["build_profiles"][0]["components"][0].update(
        {"status": "unsupported", "reason": "compiler limitation"}
    )
    matrix = build_matrix(catalog, ["alpha", "bravo"], ["fixture"])
    assert [(entry["model_id"], entry["build_profile_id"]) for entry in matrix["include"]] == [("bravo", "fixture")]


def test_unknown_requested_build_profile_fails(catalog_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown build profile id"):
        build_matrix(discover_catalog(catalog_repository), ["alpha"], ["missing"])


def test_git_diff_reports_added_modified_and_deleted_paths(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=tmp_path, check=True)
    for name in ("modified.txt", "deleted.txt"):
        (tmp_path / name).write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    (tmp_path / "modified.txt").write_text("after\n", encoding="utf-8")
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / "added.txt").write_text("added\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "changes"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    assert git_changed_paths(tmp_path, base, head) == ["added.txt", "deleted.txt", "modified.txt"]


def test_build_fingerprint_invalidates_model_dependency_and_compiler_changes(catalog_repository: Path) -> None:
    manifest = catalog_repository / "deps" / "manifest.json"
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")

    (catalog_repository / "models" / "alpha" / "compile.py").write_text("# changed\n", encoding="utf-8")
    model_changed = _fingerprint(catalog_repository, "alpha", "2.1.3")
    assert model_changed != first

    dependency = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    dependency["unrelated-publication-revision"] = 2
    manifest.write_text(json.dumps(dependency), encoding="utf-8")
    dependency_changed = _fingerprint(catalog_repository, "alpha", "2.1.3")
    assert dependency_changed == model_changed
    assert _fingerprint(catalog_repository, "alpha", "2.1.4") != dependency_changed


def test_calibrated_build_fingerprint_includes_preprocessing_and_shared_loader(
    catalog_repository: Path,
) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    uncalibrated = _fingerprint(catalog_repository, "alpha", "2.1.3")
    model["components"][0]["preprocessing"]["pipeline"].append({
        "op": "resize",
        "parameters": {"height": 128, "width": 128},
    })
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == uncalibrated

    model["calibrations"] = [{
        "id": "default",
        "status": "ready",
        "dataset": {
            "id": "imagenet-2012",
            "variant": "mini",
            "sha256": "d" * 64,
            "size_bytes": 1,
        },
    }]
    model["build_profiles"][0]["components"][0]["calibration"] = "default"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    for relative in (
        "schemas/accuracy.schema.json",
        "tools/model_registry/accuracy.py",
        "tools/model_registry/artifact_metadata.py",
        "tools/model_registry/accuracy_config.py",
        "tools/model_registry/model.py",
    ):
        destination = catalog_repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")

    first = _fingerprint(catalog_repository, "alpha", "2.1.3")
    model["components"][0]["preprocessing"]["pipeline"].append({
        "op": "normalize",
        "parameters": {"mean": [0.5], "std": [0.25]},
    })
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    preprocessing_changed = _fingerprint(catalog_repository, "alpha", "2.1.3")
    assert preprocessing_changed != first

    metadata = catalog_repository / "tools" / "model_registry" / "artifact_metadata.py"
    metadata.write_text(metadata.read_text(encoding="utf-8") + "\n# metadata validation change\n", encoding="utf-8")
    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == preprocessing_changed

    loader = catalog_repository / "tools" / "model_registry" / "accuracy.py"
    loader.write_text(loader.read_text(encoding="utf-8") + "\n# calibration loader change\n", encoding="utf-8")
    assert _fingerprint(catalog_repository, "alpha", "2.1.3") != preprocessing_changed


def test_compile_invocation_fingerprint_includes_every_component_binding(
    catalog_repository: Path,
) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_grouped_invocation(model)
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_invocation_fingerprint(
        catalog_repository, "alpha", "combined", "fixture", "2.1.3"
    )

    model["build_profiles"][0]["components"][1]["parameters"]["output_name"] = "second"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert (
        compile_invocation_fingerprint(
            catalog_repository, "alpha", "combined", "fixture", "2.1.3"
        )
        != first
    )


def test_recipe_owned_correctness_inputs_invalidate_grouped_compile(
    catalog_repository: Path,
) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_grouped_invocation(model)
    model["tests"]["correctness"] = [{
        "id": "recipe_fidelity",
        "config": "tests/correctness.yaml",
        "scope": {"type": "package", "package": "default"},
        "build_profiles": ["fixture"],
    }]
    tests_dir = manifest.parent / "tests"
    tests_dir.mkdir()
    correctness = tests_dir / "correctness.yaml"
    correctness.write_text(
        "schema_version: 1\nengine: model_recipe\nstatus: diagnostic\n"
        "result_component: model\n"
        "dataset:\n  id: fixture-data\n  variant: mini\n  sha256: " + "d" * 64 + "\n  size_bytes: 1\n"
        "selection:\n  method: seeded_random_without_replacement\n  seed: 1\n  offset: 0\n  samples: 1\n"
        "outputs: [output]\n",
        encoding="utf-8",
    )
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_invocation_fingerprint(catalog_repository, "alpha", "combined", "fixture", "2.1.3")

    correctness.write_text(
        correctness.read_text(encoding="utf-8").replace("samples: 1", "samples: 2"),
        encoding="utf-8",
    )

    assert compile_invocation_fingerprint(
        catalog_repository, "alpha", "combined", "fixture", "2.1.3"
    ) != first

def test_build_fingerprint_includes_model_local_recipe_helpers(catalog_repository: Path) -> None:
    helper = catalog_repository / "models" / "alpha" / "calibration.py"
    helper.write_text("# calibration helper\n", encoding="utf-8")
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")

    helper.write_text("# changed calibration helper\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") != first


def test_yolo26_build_fingerprint_includes_its_training_subtree(
    catalog_repository: Path,
) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")
    training_dir = catalog_repository / "training" / "yolo26_tiny_drone"
    training_dir.mkdir(parents=True)
    for name in ("compile.py", "contract.py", "dataset.py", "onnx_audit.py", "paths.py"):
        (training_dir / name).write_text("# qualified helper\n", encoding="utf-8")
    helper = training_dir / "compile.py"
    first = _fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3")

    helper.write_text("# changed helper\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3") != first


def test_yolo26_build_fingerprint_requires_its_training_subtree(
    catalog_repository: Path,
) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")

    with pytest.raises(ValidationError, match="model compile input not found"):
        _fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3")


def test_build_fingerprint_ignores_non_transforming_registry_tooling_changes(
    catalog_repository: Path,
) -> None:
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")
    ignored = [
        catalog_repository / ".github" / "workflows" / "Vulcan-Model-Compile.yml",
        catalog_repository / "schemas" / "publication.schema.json",
        catalog_repository / "tools" / "model_registry" / "__init__.py",
        catalog_repository / "tools" / "model_registry" / "errors.py",
        catalog_repository / "tools" / "model_registry" / "fetch.py",
        catalog_repository / "tools" / "model_registry" / "model.py",
        catalog_repository / "tools" / "model_registry" / "registry.py",
        catalog_repository / "tools" / "model_registry" / "source.py",
        catalog_repository / "tools" / "runtime" / "execute.py",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# delivery-only change\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == first


def test_build_fingerprint_ignores_model_documentation_and_non_benchmark_tests(catalog_repository: Path) -> None:
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "README.md").write_text("documentation only\n", encoding="utf-8")
    (model_dir / "tests").mkdir()
    (model_dir / "tests" / "correctness.yaml").write_text("samples: 100\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == first


def test_benchmark_configuration_has_an_independent_fingerprint(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    benchmark = model_dir / "tests" / "performance.yaml"
    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: initial metric profile\n",
        encoding="utf-8",
    )
    manifest = model_dir / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["tests"]["performance"] = [
        {
            "id": "latency",
            "config": "tests/performance.yaml",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
        }
    ]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    compile_before = _fingerprint(catalog_repository, "alpha", "2.1.3")
    benchmark_before = benchmark_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3")

    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: revised metric profile\n",
        encoding="utf-8",
    )

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == compile_before
    assert benchmark_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3") != benchmark_before
    before_script = benchmark_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3")
    (model_dir / "tests/benchmark.py").write_text("report = model.benchmark()\n")
    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == compile_before
    assert benchmark_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3") != before_script



def test_correctness_configuration_has_an_independent_fingerprint(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    correctness = model_dir / "tests" / "correctness.yaml"
    correctness.write_text("schema_version: 1\nstatus: diagnostic\n", encoding="utf-8")
    manifest = model_dir / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["tests"]["correctness"] = [
        {
            "id": "fidelity",
            "config": "tests/correctness.yaml",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
        }
    ]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    compile_before = _fingerprint(catalog_repository, "alpha", "2.1.3")
    validation_before = validation_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3")

    correctness.write_text("schema_version: 1\nstatus: provisional\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == compile_before
    assert validation_fingerprint(catalog_repository, "alpha", "model", "fixture", "2.1.3") != validation_before


def test_matrix_records_changed_benchmark_configuration_fingerprint(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    benchmark = model_dir / "tests" / "performance.yaml"
    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: initial metric profile\n",
        encoding="utf-8",
    )
    manifest = model_dir / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["tests"]["performance"] = [
        {
            "id": "latency",
            "config": "tests/performance.yaml",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
        }
    ]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: revised metric profile\n",
        encoding="utf-8",
    )

    matrix = build_matrix(
        discover_catalog(catalog_repository),
        ["alpha"],
        repository=catalog_repository,
        compiler_version="2.1.3",
    )

    assert [(entry["model_id"], entry["build_profile_id"]) for entry in matrix["include"]] == [("alpha", "fixture")]
    assert set(matrix["include"][0]["benchmark_fingerprints"]) == {"model"}


def test_matrix_records_fingerprints_without_catalog_carry_forward(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    matrix = build_matrix(
        catalog,
        ["alpha", "bravo"],
        repository=catalog_repository,
        compiler_version="2.1.3",
    )

    assert {entry["model_id"] for entry in matrix["include"]} == {"alpha", "bravo"}
    assert all(len(entry["build_fingerprint"]) == 64 for entry in matrix["include"])


def test_semantic_plan_treats_display_metadata_as_catalog_only(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["display_name"] = "A clearer Alpha label"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    head = _commit(catalog_repository, "metadata")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert plan["include"] == []
    assert plan["validation_include"] == []
    assert plan["benchmark_include"] == []
    assert plan["metadata_changed"] is True


def test_semantic_plan_rebuilds_every_model_for_a_new_develop_revision(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    manifest_path = catalog_repository / "deps" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model-compiler"]["revision"] = "f94f03a8f59c"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    head = _commit(catalog_repository, "compiler revision")
    compiler_identity = resolve_model_compiler(catalog_repository)["identity"]

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=compiler_identity)

    assert {entry["model_id"] for entry in plan["include"]} == {"alpha", "bravo"}
    assert all(entry["build_fingerprint"] for entry in plan["include"])


def test_semantic_plan_schedules_runner_class_changes(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["build_profiles"][0]["runner_class"] = "expensive"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    head = _commit(catalog_repository, "use expensive runner")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert [entry["model_id"] for entry in plan["include"]] == ["alpha"]
    assert plan["include"][0]["requested_runner_class"] == "expensive"


def test_semantic_plan_selects_only_changed_compile_invocation(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    (catalog_repository / "models" / "alpha" / "compile.py").write_text("# changed\n", encoding="utf-8")
    head = _commit(catalog_repository, "recipe")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert [entry["model_id"] for entry in plan["include"]] == ["alpha"]
    assert plan["metadata_changed"] is False


def test_semantic_plan_selects_benchmark_without_compile(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    benchmark = model_dir / "tests" / "performance.yaml"
    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: initial metric profile\n",
        encoding="utf-8",
    )
    manifest = model_dir / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["tests"]["performance"] = [{
        "id": "latency",
        "config": "tests/performance.yaml",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
    }]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    base = _initialize_repository(catalog_repository)
    benchmark.write_text(
        "schema_version: 1\nstatus: qualified_reference\nmetrics: [latency_avg_ms]\n"
        "note: revised metric profile\n",
        encoding="utf-8",
    )
    head = _commit(catalog_repository, "benchmark")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert plan["include"] == []
    assert [(item["model_id"], item["component_ids"]) for item in plan["benchmark_include"]] == [
        ("alpha", ["model"])
    ]


def test_semantic_plan_selects_validation_without_compile(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    correctness = model_dir / "tests" / "correctness.yaml"
    correctness.write_text("schema_version: 1\nstatus: diagnostic\n", encoding="utf-8")
    manifest = model_dir / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["tests"]["correctness"] = [{
        "id": "fidelity",
        "config": "tests/correctness.yaml",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
    }]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    base = _initialize_repository(catalog_repository)
    correctness.write_text("schema_version: 1\nstatus: provisional\n", encoding="utf-8")
    head = _commit(catalog_repository, "validation")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert plan["include"] == []
    assert [(item["model_id"], item["component_id"]) for item in plan["validation_include"]] == [
        ("alpha", "model")
    ]


def test_semantic_plan_reports_removed_invocation_without_rebuild(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    for path in (catalog_repository / "models" / "bravo").iterdir():
        path.unlink()
    (catalog_repository / "models" / "bravo").rmdir()
    head = _commit(catalog_repository, "remove model")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert plan["include"] == []
    assert plan["deleted_invocations"] == [{
        "model_id": "bravo",
        "build_profile_id": "fixture",
        "compile_invocation_id": "model",
    }]
    assert plan["metadata_changed"] is True


def test_semantic_plan_taxonomy_label_change_validates_without_rebuild(catalog_repository: Path) -> None:
    base = _initialize_repository(catalog_repository)
    taxonomy_path = catalog_repository / "taxonomies" / "model-taxonomy.yaml"
    taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["tasks"]["image_classification"]["display_name"] = "Image categorization"
    taxonomy_path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
    head = _commit(catalog_repository, "taxonomy label")

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert plan["include"] == []
    assert plan["taxonomy_changed"] is True
    assert plan["metadata_changed"] is False


def test_compile_fingerprint_schema_bump_forces_all_invocations(
    catalog_repository: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _initialize_repository(catalog_repository)
    marker = catalog_repository / "schema-bump.txt"
    marker.write_text("changed\n", encoding="utf-8")
    head = _commit(catalog_repository, "schema bump")
    monkeypatch.setattr("model_registry.catalog.COMPILE_FINGERPRINT_SCHEMA", 4)

    plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)

    assert {entry["model_id"] for entry in plan["include"]} == {"alpha", "bravo"}


def test_resolve_base_prefers_explicit_branch_and_returns_merge_base(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(repository)], check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=repository, check=True)
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    subprocess.run(["git", "branch", "-M", "develop"], cwd=repository, check=True)
    subprocess.run(["git", "push", "-q", "origin", "develop"], cwd=repository, check=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "switch", "-qc", "feature/model"], cwd=repository, check=True)
    (repository / "model.txt").write_text("model\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "model"], cwd=repository, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()

    result = resolve_base_branch(
        repository,
        current_branch="feature/model",
        head=head,
        preferred_branch="develop",
    )

    assert result == {"base_branch": "develop", "base_sha": base_sha, "merge_base": base_sha}
    assert resolve_base_branch(repository, current_branch="feature/model", head=head) == result


def test_resolve_base_preserves_explicit_base_with_same_branch_name(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(repository)], check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=repository, check=True)
    (repository / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    subprocess.run(["git", "branch", "-M", "develop"], cwd=repository, check=True)
    subprocess.run(["git", "push", "-q", "origin", "develop"], cwd=repository, check=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()

    result = resolve_base_branch(
        repository,
        current_branch="develop",
        head=base_sha,
        preferred_branch="develop",
    )

    assert result == {"base_branch": "develop", "base_sha": base_sha, "merge_base": base_sha}


@pytest.mark.parametrize("grouped", [False, True])
def test_artifact_suffix_updates_catalog_without_rebuilding(catalog_repository: Path, grouped: bool) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    if grouped:
        _declare_grouped_invocation(model)
        manifest.write_text(yaml.safe_dump(model))
    base = _initialize_repository(catalog_repository)
    catalog = discover_catalog(catalog_repository)
    matrix = build_matrix(
        catalog,
        sorted(catalog),
        repository=catalog_repository,
        compiler_version=COMPILER_VERSION,
    )
    prior = {"models": [
        {"id": model_id, "builds": [
            {
                "build_profile_id": entry["build_profile_id"], "component_id": component_id,
                "compile_fingerprint": entry["build_fingerprint"], "artifact_sha256": "a" * 64,
                "run_id": f"run-{model_id}-{component_id}",
            }
            for entry in matrix["include"] if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in catalog
    ]}
    previous = build_catalog_snapshot(repository=catalog_repository, branch="develop", git_sha=base, previous=prior)
    for suffix in ("model_int8", "model_preview", None):
        binding = model["build_profiles"][0]["components"][0]
        if suffix is None:
            binding.pop("artifact_suffix")
        else:
            binding["artifact_suffix"] = suffix
        manifest.write_text(yaml.safe_dump(model))
        head = _commit(catalog_repository, "rename download")
        plan = semantic_change_plan(catalog_repository, base, head, compiler_version=COMPILER_VERSION)
        assert plan["include"] == []
        assert plan["validation_include"] == []
        assert plan["benchmark_include"] == []
        assert plan["metadata_changed"] is True
        current = build_catalog_snapshot(
            repository=catalog_repository, branch="develop", git_sha=head, previous=previous,
        )
        alpha = next(item for item in current["models"] if item["id"] == "alpha")
        build = next(item for item in alpha["builds"] if item["component_id"] == "model")
        assert build.get("artifact_suffix") == suffix
        if suffix is None:
            assert "artifact_suffix" not in build
        normalized = [copy.deepcopy(previous["models"]), copy.deepcopy(current["models"])]
        for models in normalized:
            for item in models:
                for entry in item["builds"]:
                    entry.pop("artifact_suffix", None)
        # All identities, hashes and stage fingerprints are retained, including siblings.
        assert normalized[0] == normalized[1]
        previous, base = current, head

    binding["artifact_suffix"] = "a" * 256
    manifest.write_text(yaml.safe_dump(model))
    with pytest.raises(ValidationError, match="filename limit"):
        build_catalog_snapshot(repository=catalog_repository, branch="develop", git_sha=head, previous=previous)


@pytest.mark.parametrize("source", ["base", "target", "stale", "missing", "forbidden"])
def test_publish_fresh_branch_reuses_only_compatible_catalog_runs(catalog_repository: Path, source: str) -> None:
    catalog = discover_catalog(catalog_repository)
    matrix = build_matrix(
        catalog,
        sorted(catalog),
        repository=catalog_repository,
        compiler_version=COMPILER_VERSION,
    )
    prior = {"models": [
        {"id": model_id, "builds": [
            {
                "build_profile_id": entry["build_profile_id"], "component_id": component_id,
                "compile_fingerprint": entry["build_fingerprint"], "artifact_sha256": "a" * 64,
                "run_id": f"run-{model_id}-{component_id}",
            }
            for entry in matrix["include"] if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in catalog
    ]}
    if source == "stale":
        prior["models"][0]["builds"][0]["compile_fingerprint"] = "0" * 64
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    model["build_profiles"][0]["components"][0]["artifact_suffix"] = "model_int8"
    manifest.write_text(yaml.safe_dump(model))
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
            repository=catalog_repository, api_url="https://writer.example.invalid",
            read_api_url="https://reader.example.invalid", aws_region="us-west-2",
            branch="feat/names", base_branch="develop", git_sha="b" * 40,
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
            expected_pending = {"alpha"} if source == "stale" else set(catalog)
            assert {pending["model_id"] for pending in result["catalog"]["pending_builds"]} == expected_pending
        else:
            assert result["catalog"]["pending_builds"] == []
            for current, previous in zip(result["catalog"]["models"], prior["models"], strict=True):
                for build, old_build in zip(current["builds"], previous["builds"], strict=True):
                    assert build["run_id"] == old_build["run_id"]
                    assert build["artifact_sha256"] == old_build["artifact_sha256"]
                    assert build["compile_fingerprint"] == old_build["compile_fingerprint"]
            assert result["catalog"]["models"][0]["builds"][0]["artifact_suffix"] == "model_int8"
        assert writes == [{
            "initialize": True,
            "previous_snapshot": None,
            "catalog": result["catalog"],
        }]
    assert reads[0].endswith("git_ref=refs%2Fheads%2Ffeat%2Fnames")
    if source in {"target", "forbidden"}:
        assert len(reads) == 1
    else:
        assert len(reads) == 2
        assert reads[1].endswith("git_ref=refs%2Fheads%2Fdevelop")
