from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.compile_matrix import build_matrix
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model, load_models
from model_registry.registry import RegistryPublicationError, build_catalog_snapshot, publish_catalog
from model_registry.stage_keys import (
    COMPILE_TOOLING,
    _compile_consumed_fidelity,
    compile_key,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]


def _fingerprint(repository: Path, model_id: str, compiler_version: str) -> str:
    model = yaml.safe_load((repository / "models" / model_id / "model.yaml").read_text(encoding="utf-8"))
    return compile_key(repository, model_id, model["build_profiles"][0]["id"], "model", compiler_version)


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
def model_repository(tmp_path: Path) -> Path:
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


def test_discovery_and_matrix_are_stable(model_repository: Path) -> None:
    models = load_models(model_repository)
    matrix = build_matrix(models, ["bravo", "alpha", "alpha"])
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


def test_mismatched_directory_id_fails(model_repository: Path) -> None:
    model = yaml.safe_load((model_repository / "models" / "alpha" / "model.yaml").read_text(encoding="utf-8"))
    model["id"] = "bravo"
    (model_repository / "models" / "alpha" / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    with pytest.raises(ValidationError, match="id must match directory name"):
        load_models(model_repository)


def test_missing_taxonomy_blocks_models_before_matrix(model_repository: Path) -> None:
    path = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    del model["tasks"]
    path.write_text(yaml.safe_dump(model), encoding="utf-8")

    with pytest.raises(ValidationError, match="tasks.*required"):
        load_models(model_repository)


def test_duplicate_model_ids_fail_before_matrix(model_repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from model_registry import model as model_module

    original = model_module.load_and_validate_model

    def duplicate_id(directory: Path, schema: Path):
        model = copy.deepcopy(original(directory, schema))
        model["id"] = "duplicate"
        return model

    monkeypatch.setattr(model_module, "load_and_validate_model", duplicate_id)
    with pytest.raises(ValidationError, match="duplicate model id"):
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
    assert matrix["include"][0]["compile_invocation_id"] == "combined"
    assert matrix["include"][0]["component_id"] == "model"
    assert matrix["include"][0]["component_ids"] == ["model", "second_model"]


def test_explicit_singleton_uses_invocation_recipe_contract(model_repository: Path) -> None:
    models = load_models(model_repository)
    models["alpha"]["build_profiles"][0]["compile_invocations"] = [
        {"id": "export_and_compile", "components": ["model"]}
    ]

    entry = build_matrix(models, ["alpha"])["include"][0]

    assert entry["compile_invocation_id"] == "export_and_compile"
    assert entry["component_ids"] == ["model"]


@pytest.mark.parametrize("component_id", ["model", "second_model"])
def test_component_filter_selects_the_full_compile_invocation(
    model_repository: Path, component_id: str
) -> None:
    models = load_models(model_repository)
    _declare_grouped_invocation(models["alpha"])

    matrix = build_matrix(models, ["alpha"], component_ids=[component_id])

    assert [entry["component_ids"] for entry in matrix["include"]] == [["model", "second_model"]]


def test_matrix_excludes_explicitly_unsupported_component_builds(model_repository: Path) -> None:
    models = load_models(model_repository)
    unsupported = copy.deepcopy(models["alpha"]["build_profiles"][0])
    unsupported["id"] = "unsupported"
    unsupported["components"][0].update({"status": "unsupported", "reason": "compiler limitation"})
    models["alpha"]["build_profiles"].append(unsupported)
    matrix = build_matrix(models, ["alpha"])
    assert [entry["build_profile_id"] for entry in matrix["include"]] == ["fixture"]


def test_explicit_unsupported_component_is_excluded(model_repository: Path) -> None:
    models = load_models(model_repository)
    models["alpha"]["build_profiles"][0]["components"][0].update(
        {"status": "unsupported", "reason": "compiler zeroed weights"}
    )
    assert build_matrix(models, ["alpha"], ["fixture"])["include"] == []


def test_requested_profile_builds_supported_models_and_skips_unsupported_components(model_repository: Path) -> None:
    models = load_models(model_repository)
    models["alpha"]["build_profiles"][0]["components"][0].update(
        {"status": "unsupported", "reason": "compiler limitation"}
    )
    matrix = build_matrix(models, ["alpha", "bravo"], ["fixture"])
    assert [(entry["model_id"], entry["build_profile_id"]) for entry in matrix["include"]] == [("bravo", "fixture")]


def test_unknown_requested_build_profile_fails(model_repository: Path) -> None:
    with pytest.raises(ValidationError, match="unknown build profile id"):
        build_matrix(load_models(model_repository), ["alpha"], ["missing"])


def test_build_fingerprint_invalidates_model_dependency_and_compiler_changes(model_repository: Path) -> None:
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


def test_calibrated_build_fingerprint_includes_preprocessing_but_not_registry_tooling(
    model_repository: Path,
) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    uncalibrated = _fingerprint(model_repository, "alpha", "2.1.3")
    model["components"][0]["preprocessing"]["pipeline"].append({
        "op": "resize",
        "parameters": {"height": 128, "width": 128},
    })
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    assert _fingerprint(model_repository, "alpha", "2.1.3") == uncalibrated

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
        destination = model_repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")

    first = _fingerprint(model_repository, "alpha", "2.1.3")
    model["components"][0]["preprocessing"]["pipeline"].append({
        "op": "normalize",
        "parameters": {"mean": [0.5], "std": [0.25]},
    })
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    preprocessing_changed = _fingerprint(model_repository, "alpha", "2.1.3")
    assert preprocessing_changed != first

    # Model-owned scripts calibrate; registry tooling changes must not rebuild MPKs.
    for relative in ("artifact_metadata.py", "model.py"):
        tool = model_repository / "tools" / "model_registry" / relative
        tool.write_text(tool.read_text(encoding="utf-8") + "\n# registry tooling change\n", encoding="utf-8")
    assert _fingerprint(model_repository, "alpha", "2.1.3") == preprocessing_changed


def test_compile_invocation_fingerprint_includes_every_component_binding(
    model_repository: Path,
) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    _declare_grouped_invocation(model)
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    first = compile_key(model_repository, "alpha", "fixture", "combined", "2.1.3")

    model["build_profiles"][0]["components"][1]["parameters"]["output_name"] = "second"
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert (
        compile_key(model_repository, "alpha", "fixture", "combined", "2.1.3")
        != first
    )


def test_recipe_owned_correctness_inputs_invalidate_grouped_compile(
    model_repository: Path,
) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
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
    first = compile_key(model_repository, "alpha", "fixture", "combined", "2.1.3")

    model["fidelity"][0]["selection"]["samples"] = 2
    manifest.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    assert compile_key(model_repository, "alpha", "fixture", "combined", "2.1.3") != first


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


def test_build_fingerprint_includes_model_local_recipe_helpers(model_repository: Path) -> None:
    helper = model_repository / "models" / "alpha" / "scripts" / "calibration.py"
    helper.write_text("# calibration helper\n", encoding="utf-8")
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    helper.write_text("# changed calibration helper\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") != first


def test_yolo26_build_fingerprint_includes_its_training_subtree(
    model_repository: Path,
) -> None:
    _write_model(model_repository, "yolo26_tiny_drone")
    training_dir = model_repository / "training" / "yolo26_tiny_drone"
    training_dir.mkdir(parents=True)
    for name in ("compile.py", "contract.py", "dataset.py", "onnx_audit.py", "paths.py"):
        (training_dir / name).write_text("# qualified helper\n", encoding="utf-8")
    helper = training_dir / "compile.py"
    first = _fingerprint(model_repository, "yolo26_tiny_drone", "2.1.3")

    helper.write_text("# changed helper\n", encoding="utf-8")

    assert _fingerprint(model_repository, "yolo26_tiny_drone", "2.1.3") != first


def test_yolo26_build_fingerprint_requires_its_training_subtree(
    model_repository: Path,
) -> None:
    _write_model(model_repository, "yolo26_tiny_drone")

    with pytest.raises(ValidationError, match="model compile input not found"):
        _fingerprint(model_repository, "yolo26_tiny_drone", "2.1.3")


def test_build_fingerprint_ignores_non_transforming_registry_tooling_changes(
    model_repository: Path,
) -> None:
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    ignored = [
        model_repository / ".github" / "workflows" / "vulcan-model-orchestrator.yml",
        model_repository / "schemas" / "publication.schema.json",
        model_repository / "tools" / "model_registry" / "__init__.py",
        model_repository / "tools" / "model_registry" / "errors.py",
        model_repository / "tools" / "model_registry" / "fetch.py",
        model_repository / "tools" / "model_registry" / "model.py",
        model_repository / "tools" / "model_registry" / "registry.py",
        model_repository / "tools" / "model_registry" / "source.py",
        model_repository / "tools" / "runtime" / "execute.py",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# delivery-only change\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


def test_build_fingerprint_ignores_model_documentation_and_board_stage_scripts(model_repository: Path) -> None:
    first = _fingerprint(model_repository, "alpha", "2.1.3")
    model_dir = model_repository / "models" / "alpha"
    (model_dir / "README.md").write_text("documentation only\n", encoding="utf-8")
    (model_dir / "hero.png").write_bytes(b"documentation image only")
    (model_dir / "scripts" / "performance.py").write_text("report = model.benchmark()\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


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


def test_matrix_records_fingerprints_without_catalog_carry_forward(model_repository: Path) -> None:
    models = load_models(model_repository)
    matrix = build_matrix(
        models,
        ["alpha", "bravo"],
        repository=model_repository,
        compiler_version="2.1.3",
    )

    assert {entry["model_id"] for entry in matrix["include"]} == {"alpha", "bravo"}
    assert all(len(entry["build_fingerprint"]) == 64 for entry in matrix["include"])

@pytest.mark.parametrize("grouped", [False, True])
def test_artifact_suffix_updates_catalog_without_rebuilding(model_repository: Path, grouped: bool) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    model = yaml.safe_load(manifest.read_text())
    if grouped:
        _declare_grouped_invocation(model)
        manifest.write_text(yaml.safe_dump(model))
    base = "1" * 40
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
                "compile_fingerprint": entry["build_fingerprint"], "artifact_sha256": "a" * 64,
                "run_id": f"run-{model_id}-{component_id}",
            }
            for entry in matrix["include"] if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in models
    ]}
    previous = build_catalog_snapshot(repository=model_repository, branch="develop", git_sha=base, previous=prior)
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
                load_models(model_repository),
                sorted(models),
                repository=model_repository,
                compiler_version=COMPILER_VERSION,
            )["include"]
        ] == [entry["build_fingerprint"] for entry in matrix["include"]]
        current = build_catalog_snapshot(
            repository=model_repository, branch="develop", git_sha=head, previous=previous,
        )
        alpha = next(item for item in current["models"] if item["id"] == "alpha")
        build = next(item for item in alpha["builds"] if item["component_id"] == "model")
        assert build.get("artifact_suffix") == suffix
        if suffix is None:
            assert "artifact_suffix" not in build
        normalized = [copy.deepcopy(previous["models"]), copy.deepcopy(current["models"])]
        for entries in normalized:
            for item in entries:
                for entry in item["builds"]:
                    entry.pop("artifact_suffix", None)
        # All identities, hashes and stage fingerprints are retained, including siblings.
        assert normalized[0] == normalized[1]
        previous, base = current, head

    binding["artifact_suffix"] = "a" * 256
    manifest.write_text(yaml.safe_dump(model))
    with pytest.raises(ValidationError, match="filename limit"):
        build_catalog_snapshot(repository=model_repository, branch="develop", git_sha=head, previous=previous)


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
                "compile_fingerprint": entry["build_fingerprint"], "artifact_sha256": "a" * 64,
                "run_id": f"run-{model_id}-{component_id}",
            }
            for entry in matrix["include"] if entry["model_id"] == model_id
            for component_id in entry["component_ids"]
        ]} for model_id in models
    ]}
    if source == "stale":
        prior["models"][0]["builds"][0]["compile_fingerprint"] = "0" * 64
    manifest = model_repository / "models" / "alpha" / "model.yaml"
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
            repository=model_repository, api_url="https://writer.example.invalid",
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
            expected_pending = {"alpha"} if source == "stale" else set(models)
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
def test_compile_key_ignores_board_stage_and_conversion_files(model_repository: Path, relative: str) -> None:
    path = model_repository / "models" / "alpha" / relative
    path.write_text("# initial\n", encoding="utf-8")
    first = _fingerprint(model_repository, "alpha", "2.1.3")

    path.write_text("# revised\n", encoding="utf-8")

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first


def test_compile_key_ignores_benchmark_and_accuracy_declarations(model_repository: Path) -> None:
    manifest = model_repository / "models" / "alpha" / "model.yaml"
    first = _fingerprint(model_repository, "alpha", "2.1.3")

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

    assert _fingerprint(model_repository, "alpha", "2.1.3") == first
