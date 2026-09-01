from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import (
    build_matrix,
    discover_catalog,
    git_changed_paths,
    model_build_fingerprint,
    resolve_base_branch,
    select_model_ids,
)
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy-v2.yaml"


def _write_model(repository: Path, model_id: str, *, declared_id: str | None = None) -> None:
    model_dir = repository / "models" / model_id
    model_dir.mkdir(parents=True)
    model = yaml.safe_load((FIXTURE / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = declared_id or model_id
    (model_dir / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    (model_dir / "compile.py").write_text("# fixture\n", encoding="utf-8")


@pytest.fixture
def catalog_repository(tmp_path: Path) -> Path:
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "model.schema.json").write_text(SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "taxonomies").mkdir()
    (tmp_path / "taxonomies" / "model-taxonomy-v2.yaml").write_text(
        TAXONOMY.read_text(encoding="utf-8"), encoding="utf-8"
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


def test_matrix_can_select_requested_variants(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    second = copy.deepcopy(catalog["alpha"]["compile_variants"][0])
    second["id"] = "second"
    catalog["alpha"]["compile_variants"].append(second)
    matrix = build_matrix(catalog, ["alpha"], ["second"])
    assert [entry["variant_id"] for entry in matrix["include"]] == ["second"]


def test_matrix_excludes_explicitly_unsupported_variants(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    unsupported = copy.deepcopy(catalog["alpha"]["compile_variants"][0])
    unsupported.update({"id": "unsupported", "status": "unsupported", "reason": "compiler limitation"})
    catalog["alpha"]["compile_variants"].append(unsupported)
    matrix = build_matrix(catalog, ["alpha"])
    assert [entry["variant_id"] for entry in matrix["include"]] == ["fixture"]


def test_explicit_unsupported_variant_fails_with_evidence(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["compile_variants"][0].update({"status": "unsupported", "reason": "compiler zeroed weights"})
    with pytest.raises(
        ValidationError,
        match=r"requested variant id\(s\) are unsupported: fixture \(alpha: compiler zeroed weights\)",
    ):
        build_matrix(catalog, ["alpha"], ["fixture"])


def test_requested_variant_builds_supported_models_and_skips_unsupported_models(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["compile_variants"][0].update({"status": "unsupported", "reason": "compiler limitation"})
    matrix = build_matrix(catalog, ["alpha", "bravo"], ["fixture"])
    assert [(entry["model_id"], entry["variant_id"]) for entry in matrix["include"]] == [("bravo", "fixture")]


def test_unknown_requested_variant_fails(catalog_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown variant id"):
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
    (catalog_repository / "deps").mkdir()
    manifest = catalog_repository / "deps" / "manifest.json"
    manifest.write_text('{"model-compiler":"2.1.3"}\n', encoding="utf-8")
    first = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")

    (catalog_repository / "models" / "alpha" / "compile.py").write_text("# changed\n", encoding="utf-8")
    model_changed = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")
    assert model_changed != first

    manifest.write_text('{"model-compiler":"2.1.3", "revision":2}\n', encoding="utf-8")
    dependency_changed = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")
    assert dependency_changed != model_changed
    assert model_build_fingerprint(catalog_repository, "alpha", "2.1.4") != dependency_changed


def test_yolo26_build_fingerprint_includes_its_training_subtree(
    catalog_repository: Path,
) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")
    training_dir = catalog_repository / "training" / "yolo26_tiny_drone"
    training_dir.mkdir(parents=True)
    helper = training_dir / "compile.py"
    helper.write_text("# qualified helper\n", encoding="utf-8")
    first = model_build_fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3")

    helper.write_text("# changed helper\n", encoding="utf-8")

    assert model_build_fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3") != first


def test_yolo26_build_fingerprint_requires_its_training_subtree(
    catalog_repository: Path,
) -> None:
    _write_model(catalog_repository, "yolo26_tiny_drone")

    with pytest.raises(ValidationError, match="model compile input not found"):
        model_build_fingerprint(catalog_repository, "yolo26_tiny_drone", "2.1.3")


def test_build_fingerprint_ignores_non_transforming_registry_tooling_changes(
    catalog_repository: Path,
) -> None:
    first = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")
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

    assert model_build_fingerprint(catalog_repository, "alpha", "2.1.3") == first


def test_build_fingerprint_ignores_model_documentation_and_non_benchmark_tests(catalog_repository: Path) -> None:
    first = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "README.md").write_text("documentation only\n", encoding="utf-8")
    (model_dir / "tests").mkdir()
    (model_dir / "tests" / "correctness.yaml").write_text("samples: 100\n", encoding="utf-8")

    assert model_build_fingerprint(catalog_repository, "alpha", "2.1.3") == first


def test_build_fingerprint_invalidates_benchmark_configuration_changes(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    benchmark = model_dir / "tests" / "performance.yaml"
    benchmark.write_text("benchmark:\n  samples: 100\n", encoding="utf-8")
    first = model_build_fingerprint(catalog_repository, "alpha", "2.1.3")

    benchmark.write_text("benchmark:\n  samples: 5\n", encoding="utf-8")

    assert model_build_fingerprint(catalog_repository, "alpha", "2.1.3") != first


def test_matrix_records_changed_benchmark_configuration_fingerprint(catalog_repository: Path) -> None:
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "tests").mkdir()
    benchmark = model_dir / "tests" / "performance.yaml"
    benchmark.write_text("benchmark:\n  samples: 100\n", encoding="utf-8")
    benchmark.write_text("benchmark:\n  samples: 5\n", encoding="utf-8")

    matrix = build_matrix(
        discover_catalog(catalog_repository),
        ["alpha"],
        repository=catalog_repository,
        compiler_version="2.1.3",
    )

    assert [(entry["model_id"], entry["variant_id"]) for entry in matrix["include"]] == [("alpha", "fixture")]


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
