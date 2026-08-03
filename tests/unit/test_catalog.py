from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import yaml
from model_registry.catalog import build_matrix, discover_catalog, git_changed_paths, select_model_ids
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"


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
        ("docs/architecture.md", []),
    ],
)
def test_model_local_added_modified_and_deleted_paths(catalog_repository: Path, path: str, expected: list[str]) -> None:
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == expected


@pytest.mark.parametrize(
    "path",
    [
        "schemas/model.schema.json",
        "tools/model_registry/catalog.py",
        "tools/runtime/execute.py",
        "pyproject.toml",
        ".github/workflows/Vulcan-Model-Compile.yml",
        "deps/manifest.json",
    ],
)
def test_shared_build_paths_select_every_model(catalog_repository: Path, path: str) -> None:
    assert select_model_ids(discover_catalog(catalog_repository), [path]) == ["alpha", "bravo"]


def test_mismatched_directory_id_fails(catalog_repository: Path) -> None:
    model = yaml.safe_load((catalog_repository / "models" / "alpha" / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = "bravo"
    (catalog_repository / "models" / "alpha" / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ValidationError, match="id must match directory name"):
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
