from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from model_registry.compile_matrix import build_matrix
from model_registry.dependencies import resolve_model_compiler
from model_registry.model import load_models
from model_registry.registry import RegistryPublicationError, build_catalog_snapshot
from model_registry.stage_keys import (
    ACCURACY_TOOLING,
    BOARD_STAGE_TOOLING,
    COMPILE_TOOLING,
    PERFORMANCE_TOOLING,
    compile_key,
)
from model_registry.stage_plan import applicable_stages, plan_stages, resolve_base_branch

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]



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


def test_resolve_base_ignores_feature_branches(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(repository)], check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=repository, check=True)

    def commit(branch: str, name: str, start: str | None = None) -> str:
        subprocess.run(["git", "switch", "-qC", branch, *([start] if start else [])], cwd=repository, check=True)
        (repository / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-qm", name], cwd=repository, check=True)
        subprocess.run(["git", "push", "-q", "origin", branch], cwd=repository, check=True)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()

    commit("develop", "base")
    shared = commit("feat/a", "shared")
    commit("feat/a", "a")
    feature = commit("feat/a", "a-fix")
    commit("feat/b", "b", shared)
    commit("develop", "develop-moved", "origin/develop")
    commit("release-1.0", "release", "origin/develop")
    release_feature = commit("feat/r", "r")

    def base(branch: str, head: str, preferred: str | None = None) -> str:
        return resolve_base_branch(repository, current_branch=branch, head=head, preferred_branch=preferred)[
            "base_branch"
        ]

    # feat/b is 3 commits from feat/a, develop is 4: only base branches qualify.
    assert base("feat/a", feature) == "develop"
    assert base("feat/r", release_feature) == "release-1.0"
    assert base("feat/r", release_feature, preferred="feat/a") == "feat/a"


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
    models = load_models(repository)
    matrix = build_matrix(models, sorted(models), repository=repository, compiler_version=COMPILER_VERSION)
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
    ]} for model_id in models]}
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
    entries = build_matrix(load_models(stage_repository), [PLAN_MODEL])["include"]
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
            compile_key(stage_repository, PLAN_MODEL, profile, "model", COMPILER_VERSION)
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


@pytest.mark.parametrize(
    "relative", sorted(set(ACCURACY_TOOLING) - set(BOARD_STAGE_TOOLING) - {"scripts/run_model_accuracy.sh"})
)
def test_shared_accuracy_tooling_change_measures_accuracy_only(stage_repository: Path, relative: str) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, f"../../{relative}")

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
