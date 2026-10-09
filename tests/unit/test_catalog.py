from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from model_registry.catalog import (
    COMPILE_TOOLING,
    _compile_consumed_fidelity,
    build_matrix,
    compile_fingerprint,
    discover_catalog,
    resolve_base_branch,
)
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model
from model_registry.registry import RegistryPublicationError, build_catalog_snapshot, publish_catalog
from model_registry.stages import (
    ACCURACY_TOOLING,
    BOARD_STAGE_TOOLING,
    PERFORMANCE_TOOLING,
    applicable_stages,
    plan_stages,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]


def _fingerprint(repository: Path, model_id: str, compiler_version: str) -> str:
    model = yaml.safe_load((repository / "models" / model_id / "model.yaml").read_text(encoding="utf-8"))
    return compile_fingerprint(repository, model_id, model["build_profiles"][0]["id"], "model", compiler_version)


def _write_model(repository: Path, model_id: str, *, declared_id: str | None = None) -> None:
    model_dir = repository / "models" / model_id
    model_dir.mkdir(parents=True)
    model = yaml.safe_load((FIXTURE / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = declared_id or model_id
    (model_dir / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    (model_dir / "scripts").mkdir()
    (model_dir / "scripts" / "compile.py").write_text("# fixture\n", encoding="utf-8")


def _declare_grouped_invocation(model: dict) -> None:
    second = copy.deepcopy(model["components"][0])
    second["id"] = "second_model"
    model["components"].append(second)
    model["build_profiles"][0]["components"].append({"id": "second_model", "parameters": {}})
    model["build_profiles"][0]["compile_invocations"] = [
        {"id": "combined", "components": ["model", "second_model"]}
    ]


@pytest.fixture
def catalog_repository(tmp_path: Path) -> Path:
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "model.schema.json").write_text(SCHEMA.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "taxonomies").mkdir()
    (tmp_path / "taxonomies" / "model-taxonomy.yaml").write_text(
        TAXONOMY.read_text(encoding="utf-8"), encoding="utf-8"
    )
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


def test_discovery_and_matrix_are_stable(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    matrix = build_matrix(catalog, ["bravo", "alpha", "alpha"])
    assert [entry["model_id"] for entry in matrix["include"]] == ["alpha", "bravo"]
    assert "compiler_label" not in matrix["include"][0]
    assert "compiler_version" not in matrix["include"][0]
    assert matrix["include"][0]["requested_runner_class"] == "default"
    assert matrix["include"][0]["compile_invocation_id"] == "model"
    assert matrix["include"][0]["component_ids"] == ["model"]
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["benchmark_component_ids"] == []
    assert matrix["include"][0]["accuracy_id"] is None
    assert "build_fingerprint" not in matrix["include"][0]


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


def test_explicit_singleton_uses_invocation_recipe_contract(catalog_repository: Path) -> None:
    catalog = discover_catalog(catalog_repository)
    catalog["alpha"]["build_profiles"][0]["compile_invocations"] = [
        {"id": "export_and_compile", "components": ["model"]}
    ]

    entry = build_matrix(catalog, ["alpha"])["include"][0]

    assert entry["compile_invocation_id"] == "export_and_compile"
    assert entry["component_ids"] == ["model"]


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


def test_build_fingerprint_invalidates_model_dependency_and_compiler_changes(catalog_repository: Path) -> None:
    manifest = catalog_repository / "deps" / "manifest.json"
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")

    (catalog_repository / "models" / "alpha" / "scripts" / "compile.py").write_text("# changed\n", encoding="utf-8")
    model_changed = _fingerprint(catalog_repository, "alpha", "2.1.3")
    assert model_changed != first

    dependency = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    dependency["unrelated-publication-revision"] = 2
    manifest.write_text(json.dumps(dependency), encoding="utf-8")
    dependency_changed = _fingerprint(catalog_repository, "alpha", "2.1.3")
    assert dependency_changed == model_changed
    assert _fingerprint(catalog_repository, "alpha", "2.1.4") != dependency_changed


def test_calibrated_build_fingerprint_includes_preprocessing_but_not_registry_tooling(
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

    for relative in ("tools/model_registry/artifact_metadata.py", "tools/model_registry/model.py"):
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

    # Model-owned scripts calibrate; registry tooling changes must not rebuild MPKs.
    for relative in ("artifact_metadata.py", "model.py"):
        tool = catalog_repository / "tools" / "model_registry" / relative
        tool.write_text(tool.read_text(encoding="utf-8") + "\n# registry tooling change\n", encoding="utf-8")
    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == preprocessing_changed


def test_compile_invocation_fingerprint_includes_every_component_binding(
    catalog_repository: Path,
) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_grouped_invocation(model)
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_fingerprint(catalog_repository, "alpha", "fixture", "combined", "2.1.3")

    model["build_profiles"][0]["components"][1]["parameters"]["output_name"] = "second"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert (
        compile_fingerprint(catalog_repository, "alpha", "fixture", "combined", "2.1.3")
        != first
    )


def test_recipe_owned_correctness_inputs_invalidate_grouped_compile(
    catalog_repository: Path,
) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_grouped_invocation(model)
    model["fidelity"] = [{
        "id": "recipe_fidelity",
        "scope": {"type": "package", "package": "default"},
        "build_profiles": ["fixture"],
        "status": "diagnostic",
        "result_component": "model",
        "dataset": {"id": "fixture-data", "variant": "mini", "sha256": "d" * 64, "size_bytes": 1},
        "selection": {"method": "seeded_random_without_replacement", "seed": 1, "offset": 0, "samples": 1},
        "outputs": ["output"],
    }]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_fingerprint(catalog_repository, "alpha", "fixture", "combined", "2.1.3")

    model["fidelity"][0]["selection"]["samples"] = 2
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert compile_fingerprint(catalog_repository, "alpha", "fixture", "combined", "2.1.3") != first


def _consumed_ids(model: dict, profile_id: str) -> list[str]:
    consumed = _compile_consumed_fidelity(model, profile_id)
    assert all("scope" not in entry and "build_profiles" not in entry for entry in consumed)
    return [entry["id"] for entry in consumed]


def test_compile_consumes_only_fidelity_settings_of_its_build_profile() -> None:
    resnet = load_and_validate_model(ROOT / "models" / "resnet_50", SCHEMA)
    assert _consumed_ids(resnet, "modalix_int8") == ["logits_fidelity"]
    assert _consumed_ids(resnet, "modalix_bf16") == ["logits_fidelity"]

    rf_detr = load_and_validate_model(ROOT / "models" / "rf_detr_small", SCHEMA)
    assert _consumed_ids(rf_detr, rf_detr["build_profiles"][0]["id"]) == ["full_pipeline_fidelity"]

    ssd = load_and_validate_model(ROOT / "models" / "ssd_mobilenet_v1", SCHEMA)
    for profile in ssd["build_profiles"]:
        assert _consumed_ids(ssd, profile["id"]) == []


def test_build_fingerprint_includes_model_local_recipe_helpers(catalog_repository: Path) -> None:
    helper = catalog_repository / "models" / "alpha" / "scripts" / "calibration.py"
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


def test_build_fingerprint_ignores_model_documentation_and_board_stage_scripts(catalog_repository: Path) -> None:
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")
    model_dir = catalog_repository / "models" / "alpha"
    (model_dir / "README.md").write_text("documentation only\n", encoding="utf-8")
    (model_dir / "hero.png").write_bytes(b"documentation image only")
    (model_dir / "scripts" / "performance.py").write_text("report = model.benchmark()\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == first


def _declare_benchmark(manifest: Path, note: str) -> None:
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["benchmark"] = [
        {
            "id": "latency",
            "scope": {"type": "component", "component": "model"},
            "build_profiles": ["fixture"],
            "status": "qualified_reference",
            "metrics": ["latency_avg_ms"],
            "note": note,
        }
    ]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")


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


@pytest.mark.parametrize("grouped", [False, True])
def test_artifact_suffix_updates_catalog_without_rebuilding(catalog_repository: Path, grouped: bool) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    if grouped:
        _declare_grouped_invocation(model)
        manifest.write_text(yaml.safe_dump(model))
    base = "1" * 40
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
        head = f"{int(base[0]) + 1}" * 40
        # The download name is catalog metadata: it never changes a compile key.
        assert [
            entry["build_fingerprint"]
            for entry in build_matrix(
                discover_catalog(catalog_repository),
                sorted(catalog),
                repository=catalog_repository,
                compiler_version=COMPILER_VERSION,
            )["include"]
        ] == [entry["build_fingerprint"] for entry in matrix["include"]]
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


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/performance.py",
        "scripts/accuracy.py",
        "scripts/requirements-accuracy.txt",
        "scripts/convert.py",
        "scripts/requirements-convert.txt",
    ],
)
def test_compile_key_ignores_board_stage_and_conversion_files(catalog_repository: Path, relative: str) -> None:
    path = catalog_repository / "models" / "alpha" / relative
    path.write_text("# initial\n", encoding="utf-8")
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")

    path.write_text("# revised\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == first


def test_compile_key_ignores_benchmark_and_accuracy_declarations(catalog_repository: Path) -> None:
    manifest = catalog_repository / "models" / "alpha" / "model.yaml"
    first = _fingerprint(catalog_repository, "alpha", "2.1.3")

    _declare_benchmark(manifest, "revised metric profile")
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    model["accuracy"] = [{
        "id": "devkit_accuracy",
        "scope": {"type": "component", "component": "model"},
        "build_profiles": ["fixture"],
        "engine": "devkit_script",
        "status": "provisional",
        "result_component": "model",
        "dataset": {"id": "fixture-set", "variant": "mini", "sha256": "a" * 64, "size_bytes": 1},
        "metrics": ["score"],
    }]
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    (manifest.parent / "scripts" / "accuracy.py").write_text("score = 0.5\n", encoding="utf-8")

    assert _fingerprint(catalog_repository, "alpha", "2.1.3") == first


# Stage planning runs against a copy of resnet_50: modalix_bf16 declares only performance,
# modalix_int8 declares performance and DevKit accuracy over the default package.
PLAN_MODEL = "resnet_50"
BRANCH = "feat/stages"
READ_API = "https://reader.example.invalid"


@pytest.fixture
def stage_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    for relative in ("schemas", "taxonomies", f"models/{PLAN_MODEL}"):
        shutil.copytree(ROOT / relative, repository / relative)
    for relative in ("deps/manifest.json", *COMPILE_TOOLING, *PERFORMANCE_TOOLING, *ACCURACY_TOOLING):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return repository


def _published_catalog(repository: Path) -> dict:
    """The catalog a publisher writes after compiling every invocation of the repository."""
    catalog = discover_catalog(repository)
    matrix = build_matrix(catalog, sorted(catalog), repository=repository, compiler_version=COMPILER_VERSION)
    prior = {"models": [{"id": model_id, "builds": [
        {
            "build_profile_id": entry["build_profile_id"],
            "component_id": component_id,
            "compile_fingerprint": entry["build_fingerprint"],
            "artifact_sha256": f"{index:x}" * 64,
            "run_id": f"00000000-0000-4000-8000-{index:012d}",
        }
        for index, entry in enumerate(matrix["include"], 1)
        if entry["model_id"] == model_id
        for component_id in entry["component_ids"]
    ]} for model_id in catalog]}
    return build_catalog_snapshot(repository=repository, branch="develop", git_sha="a" * 40, previous=prior)


def _published_runs(document: dict) -> set[tuple[str, str, int]]:
    """Registry runs that carry the stage keys the catalog records."""
    runs: set[tuple[str, str, int]] = set()
    for model in document["models"]:
        for build in model["builds"]:
            if build["benchmark_fingerprint"]:
                runs.add(("model_benchmark", build["benchmark_fingerprint"], 2))
            if build["validation_fingerprint"]:
                runs.add(("model_accuracy", build["validation_fingerprint"], 1))
    return runs


class FakeRegistry:
    def __init__(self, catalogs: dict[str, dict], runs: set[tuple[str, str, int]]) -> None:
        self.catalogs = catalogs
        self.runs = runs
        self.urls: list[str] = []

    def read(self, url: str) -> dict:
        self.urls.append(url)
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path.endswith("/catalogs/current"):
            branch = query["git_ref"][0].removeprefix("refs/heads/")
            if branch not in self.catalogs:
                raise RegistryPublicationError("public artifact registry HTTP 404 catalog_not_found")
            return {"catalog": {"canonical_sha256": "c" * 64, "document": self.catalogs[branch]}}
        if parsed.path.endswith("/resolve"):
            run = (query["run_type"][0], query["fingerprint"][0], int(query["fingerprint_version"][0]))
            if run not in self.runs:
                raise RegistryPublicationError("public artifact registry HTTP 404 fingerprint_not_found")
            return {"run": {"run_id": "00000000-0000-4000-8000-000000000099"}}
        raise AssertionError(url)

    def resolved(self) -> list[tuple[str, str]]:
        return [
            (query["run_type"][0], query["fingerprint_version"][0])
            for query in (parse_qs(urlsplit(url).query) for url in self.urls if "/resolve?" in url)
        ]


def _plan(repository: Path, registry: FakeRegistry | None, **kwargs) -> dict:
    return plan_stages(
        repository,
        compiler_version=COMPILER_VERSION,
        read=registry.read if registry is not None else None,
        read_api_url=READ_API,
        branch=BRANCH,
        **kwargs,
    )


def _selected(plan: dict, stage: str) -> list[tuple[str, list[str], list[str]]]:
    return [
        (entry["build_profile_id"], entry["stages"], entry["benchmark_component_ids"])
        for entry in plan[stage]["include"]
    ]


def _edit(repository: Path, relative: str) -> None:
    path = repository / "models" / PLAN_MODEL / relative
    path.write_text(path.read_text(encoding="utf-8") + "\n# revised\n", encoding="utf-8")


def test_stage_repository_declares_both_stage_shapes(stage_repository: Path) -> None:
    entries = build_matrix(discover_catalog(stage_repository), [PLAN_MODEL])["include"]
    assert [(entry["build_profile_id"], applicable_stages(entry)) for entry in entries] == [
        ("modalix_bf16", ["performance"]),
        ("modalix_int8", ["performance", "accuracy"]),
    ]


@pytest.mark.parametrize(
    ("relative", "changes_compile_key"),
    [("scripts/performance.py", False), ("scripts/accuracy.py", False), ("scripts/compile.py", True)],
)
def test_only_compile_inputs_change_the_compile_key(
    stage_repository: Path, relative: str, changes_compile_key: bool
) -> None:
    def keys() -> list[str]:
        return [
            compile_fingerprint(stage_repository, PLAN_MODEL, profile, "model", COMPILER_VERSION)
            for profile in ("modalix_bf16", "modalix_int8")
        ]

    before = keys()
    _edit(stage_repository, relative)
    after = keys()

    assert all((first != second) is changes_compile_key for first, second in zip(before, after, strict=True))


def test_everything_published_plans_nothing(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))

    plan = _plan(stage_repository, registry)

    assert plan == {"compile": {"include": []}, "measure": {"include": []}, "catalog_found": True}


def test_resolve_requests_carry_each_stage_fingerprint_version(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))

    _plan(stage_repository, registry)

    # One performance lookup per profile, one accuracy lookup for the profile that declares it.
    assert sorted(registry.resolved()) == [
        ("model_accuracy", "1"),
        ("model_benchmark", "2"),
        ("model_benchmark", "2"),
    ]
    builds = {build["build_profile_id"]: build for build in document["models"][0]["builds"]}
    assert builds["modalix_bf16"]["validation_fingerprint"] is None
    assert all(
        any(f"fingerprint={build['benchmark_fingerprint']}&" in url for url in registry.urls)
        for build in builds.values()
    )
    assert any(f"fingerprint={builds['modalix_int8']['validation_fingerprint']}&" in url for url in registry.urls)


def test_compile_key_change_compiles_with_every_applicable_stage(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "scripts/compile.py")

    plan = _plan(stage_repository, registry)

    assert _selected(plan, "compile") == [
        ("modalix_bf16", ["performance"], ["model"]),
        ("modalix_int8", ["performance", "accuracy"], ["model"]),
    ]
    assert plan["measure"]["include"] == []
    assert all(len(entry["build_fingerprint"]) == 64 for entry in plan["compile"]["include"])
    # A compiled build is measured by its own workflow; no published stage run is consulted.
    assert registry.resolved() == []


def test_performance_script_change_measures_performance_only(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "scripts/performance.py")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [
        ("modalix_bf16", ["performance"], ["model"]),
        ("modalix_int8", ["performance"], ["model"]),
    ]


def test_accuracy_script_change_measures_accuracy_only(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "scripts/accuracy.py")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [("modalix_int8", ["accuracy"], [])]


def test_shared_compile_runner_change_compiles_every_build(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "../../scripts/run_model_compilation.sh")

    plan = _plan(stage_repository, registry)

    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["modalix_bf16", "modalix_int8"]


@pytest.mark.parametrize("relative", BOARD_STAGE_TOOLING)
def test_board_stage_tooling_change_measures_both_stages(stage_repository: Path, relative: str) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, f"../../{relative}")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [
        ("modalix_bf16", ["performance"], ["model"]),
        ("modalix_int8", ["performance", "accuracy"], ["model"]),
    ]


def test_pending_or_missing_build_compiles_its_invocation(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    runs = _published_runs(document)
    model = document["models"][0]
    int8 = next(build for build in model["builds"] if build["build_profile_id"] == "modalix_int8")
    # A cancelled run leaves its invocation in pending_builds rather than builds.
    model["builds"].remove(int8)
    document["pending_builds"] = [{
        "model_id": PLAN_MODEL,
        "build_profile_id": "modalix_int8",
        "compile_invocation_id": "model",
        "component_ids": ["model"],
        "compile_fingerprint": int8["compile_fingerprint"],
    }]
    registry = FakeRegistry({BRANCH: document}, runs)

    plan = _plan(stage_repository, registry)

    assert _selected(plan, "compile") == [("modalix_int8", ["performance", "accuracy"], ["model"])]
    assert plan["measure"]["include"] == []


def test_force_rebuild_compiles_published_invocations(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))

    plan = _plan(stage_repository, registry, force_rebuild_all=True)

    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["modalix_bf16", "modalix_int8"]
    assert plan["measure"]["include"] == []


def test_branch_without_catalog_plans_against_base_branch_catalog(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({"develop": document}, _published_runs(document))
    _edit(stage_repository, "scripts/accuracy.py")

    plan = _plan(stage_repository, registry, base_branch="develop")

    assert plan["catalog_found"] is True
    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [("modalix_int8", ["accuracy"], [])]
    catalog_reads = [url for url in registry.urls if "/catalogs/current?" in url]
    assert [url.rsplit("git_ref=", 1)[1] for url in catalog_reads] == [
        "refs%2Fheads%2Ffeat%2Fstages",
        "refs%2Fheads%2Fdevelop",
    ]
    assert all(url.startswith(f"{READ_API}/v1/repositories/sima-neat/models/") for url in registry.urls)


@pytest.mark.parametrize("base_branch", [None, "develop"])
def test_no_catalog_compiles_everything(stage_repository: Path, base_branch: str | None) -> None:
    registry = FakeRegistry({}, set())

    plan = _plan(stage_repository, registry, base_branch=base_branch)

    assert plan["catalog_found"] is False
    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["modalix_bf16", "modalix_int8"]
    assert plan["measure"]["include"] == []


def test_offline_plan_compiles_everything(stage_repository: Path) -> None:
    plan = _plan(stage_repository, None)

    assert plan["catalog_found"] is False
    assert _selected(plan, "compile") == [
        ("modalix_bf16", ["performance"], ["model"]),
        ("modalix_int8", ["performance", "accuracy"], ["model"]),
    ]


@pytest.mark.parametrize("failing", ["/catalogs/current?", "/resolve?"])
def test_registry_failures_other_than_not_found_propagate(stage_repository: Path, failing: str) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    original = registry.read

    def read(url: str) -> dict:
        if failing in url:
            raise RegistryPublicationError("public artifact registry HTTP 403 forbidden")
        return original(url)

    registry.read = read
    with pytest.raises(RegistryPublicationError, match="403 forbidden"):
        _plan(stage_repository, registry, base_branch="develop")
