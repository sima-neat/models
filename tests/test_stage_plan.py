from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from model_registry.compile_matrix import build_matrix
from model_registry.dependencies import resolve_model_compiler
from model_registry.errors import ValidationError
from model_registry.model import load_models
from model_registry.registry import RegistryPublicationError, build_catalog_snapshot
from model_registry.stage_keys import (
    ACCURACY_TOOLING,
    BOARD_STAGE_TOOLING,
    COMPILE_TOOLING,
    PERFORMANCE_TOOLING,
    compile_key,
)
from model_registry.stage_plan import (
    applicable_stages,
    is_long_lived,
    merge_catalog_sources,
    plan_stages,
    resolve_base_branches,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fixture_model"
SCHEMA = ROOT / "schemas" / "model.schema.json"
TAXONOMY = ROOT / "taxonomies" / "model-taxonomy.yaml"
COMPILER_VERSION = resolve_model_compiler(ROOT)["identity"]



@pytest.fixture
def branch_repository(tmp_path: Path):
    """A clone whose origin has develop, a release branch, and feature branches cut from each."""
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

    heads = {"develop": commit("develop", "base")}
    shared = commit("feat/a", "shared")
    commit("feat/a", "a")
    heads["feat/a"] = commit("feat/a", "a-fix")
    heads["feat/b"] = commit("feat/b", "b", shared)
    heads["develop"] = commit("develop", "develop-moved", "origin/develop")
    heads["release-1.0"] = commit("release-1.0", "release", "origin/develop")
    heads["feat/r"] = commit("feat/r", "r")
    return repository, heads


def test_resolve_base_branches_orders_base_branches_by_distance(branch_repository) -> None:
    repository, heads = branch_repository

    def bases(branch: str) -> list[str]:
        return resolve_base_branches(repository, current_branch=branch, head=heads[branch])

    # feat/b shares a commit with feat/a but is a sibling feature branch, never a base.
    assert bases("feat/a") == ["develop", "release-1.0"]
    assert bases("feat/r") == ["release-1.0", "develop"]
    # A base branch is not its own base; its own catalog is read separately.
    assert bases("develop") == ["release-1.0"]
    assert bases("release-1.0") == ["develop"]


def test_resolve_base_branches_without_other_base_branches_is_empty(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    subprocess.run(["git", "clone", "-q", str(remote), str(repository)], check=True)
    subprocess.run(["git", "config", "user.email", "models-test@example.invalid"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Models Test"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=repository, check=True)
    subprocess.run(["git", "branch", "-M", "develop"], cwd=repository, check=True)
    subprocess.run(["git", "push", "-q", "origin", "develop"], cwd=repository, check=True)

    assert resolve_base_branches(repository, current_branch="develop", head="HEAD") == []


def test_resolve_base_branches_uses_an_explicit_base(branch_repository) -> None:
    repository, heads = branch_repository

    def bases(branch: str, preferred: str) -> list[str]:
        return resolve_base_branches(repository, current_branch=branch, head=heads[branch], preferred_branch=preferred)

    assert bases("feat/r", "feat/a") == ["feat/a"]
    assert bases("develop", "develop") == []
    with pytest.raises(ValidationError, match="'feat/missing' was not fetched"):
        bases("feat/a", "feat/missing")


@pytest.mark.parametrize(
    ("branch", "long_lived"),
    [
        ("develop", True),
        ("main", True),
        ("release-1.0", True),
        ("integration/vision", True),
        ("feat/a", False),
        ("fix/develop", False),
    ],
)
def test_long_lived_branches(branch: str, long_lived: bool) -> None:
    assert is_long_lived(branch) is long_lived


def _source(*profiles: tuple[str, str, str], model_id: str = "x") -> dict:
    """A catalog document with one build and one package per (profile, compile key, origin)."""
    return {"models": [{
        "id": model_id,
        "builds": [
            {"build_profile_id": profile, "component_id": "model", "compile_fingerprint": key, "run_id": origin}
            for profile, key, origin in profiles
        ],
        "packages": [
            {"build_profile_id": profile, "package_id": "default", "run_id": origin}
            for profile, _, origin in profiles
        ],
    }]}


MERGE_MATRIX = [
    {"model_id": "x", "build_profile_id": profile, "component_ids": ["model"], "build_fingerprint": key}
    for profile, key in (("p1", "k1"), ("p2", "k2"))
]


def _origins(document: dict) -> dict[tuple[str, str], str]:
    return {
        (kind, item["build_profile_id"]): item["run_id"]
        for model in document["models"]
        for kind in ("builds", "packages")
        for item in model[kind]
    }


def test_merge_takes_each_profile_from_the_first_source_carrying_its_keys() -> None:
    # The branch compiled p1, then merged a develop that recompiled p2.
    own = _source(("p1", "k1", "own"), ("p2", "old", "own"))
    develop = _source(("p1", "old", "develop"), ("p2", "k2", "develop"))

    merged = merge_catalog_sources([own, develop], MERGE_MATRIX)

    assert _origins(merged) == {
        ("builds", "p1"): "own",
        ("packages", "p1"): "own",
        ("builds", "p2"): "develop",
        ("packages", "p2"): "develop",
    }


def test_merge_prefers_earlier_sources_when_several_match() -> None:
    merged = merge_catalog_sources(
        [_source(("p1", "k1", "own")), _source(("p1", "k1", "develop"), ("p2", "k2", "develop"))], MERGE_MATRIX
    )

    assert _origins(merged)[("builds", "p1")] == "own"
    assert _origins(merged)[("builds", "p2")] == "develop"


def test_merge_keeps_the_first_unmatched_profile_and_drops_models_outside_head() -> None:
    own = _source(("p1", "stale", "own"))
    develop = _source(("p1", "older", "develop"))
    removed = _source(("p1", "k1", "removed"), model_id="removed")

    merged = merge_catalog_sources([own, develop, removed], MERGE_MATRIX)

    # The planner still sees p1's prior build, at a key that no longer matches, and compiles it.
    assert [model["id"] for model in merged["models"]] == ["x"]
    assert _origins(merged) == {("builds", "p1"): "own", ("packages", "p1"): "own"}
    assert merge_catalog_sources([], MERGE_MATRIX) is None


def test_merge_requires_every_component_key_of_a_profile() -> None:
    matrix = [{"model_id": "x", "build_profile_id": "p1", "component_ids": ["a", "b"], "build_fingerprint": "k1"}]
    partial = {"models": [{"id": "x", "builds": [
        {"build_profile_id": "p1", "component_id": "a", "compile_fingerprint": "k1", "run_id": "own"},
    ]}]}
    complete = {"models": [{"id": "x", "builds": [
        {"build_profile_id": "p1", "component_id": component, "compile_fingerprint": "k1", "run_id": "develop"}
        for component in ("a", "b")
    ]}]}

    merged = merge_catalog_sources([partial, complete], matrix)

    assert [build["run_id"] for build in merged["models"][0]["builds"]] == ["develop", "develop"]


# Stage planning runs against a copy of resnet_50: w16a16 declares only performance,
# w8a8 declares performance and DevKit accuracy over the default package.
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


def _plan(repository: Path, registry: FakeRegistry | None, branch: str = BRANCH, **kwargs) -> dict:
    return plan_stages(
        repository,
        compiler_version=COMPILER_VERSION,
        read=registry.read if registry is not None else None,
        read_api_url=READ_API,
        branch=branch,
        **kwargs,
    )


def _catalog_reads(registry: FakeRegistry) -> list[str]:
    return [
        unquote(url.rsplit("git_ref=", 1)[1]).removeprefix("refs/heads/")
        for url in registry.urls
        if "/catalogs/current?" in url
    ]


def _stale(document: dict, profile_id: str) -> dict:
    """The catalog as published before the profile's compile key changed."""
    stale = copy.deepcopy(document)
    for build in stale["models"][0]["builds"]:
        if build["build_profile_id"] == profile_id:
            build["compile_fingerprint"] = "0" * 64
    return stale


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
        ("w16a16", ["performance"]),
        ("w8a8", ["performance", "accuracy"]),
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
            for profile in ("w16a16", "w8a8")
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
    assert builds["w16a16"]["validation_fingerprint"] is None
    assert all(
        any(f"fingerprint={build['benchmark_fingerprint']}&" in url for url in registry.urls)
        for build in builds.values()
    )
    assert any(f"fingerprint={builds['w8a8']['validation_fingerprint']}&" in url for url in registry.urls)


def test_compile_key_change_compiles_with_every_applicable_stage(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "scripts/compile.py")

    plan = _plan(stage_repository, registry)

    assert _selected(plan, "compile") == [
        ("w16a16", ["performance"], ["model"]),
        ("w8a8", ["performance", "accuracy"], ["model"]),
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
        ("w16a16", ["performance"], ["model"]),
        ("w8a8", ["performance"], ["model"]),
    ]


def test_accuracy_script_change_measures_accuracy_only(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "scripts/accuracy.py")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [("w8a8", ["accuracy"], [])]


@pytest.mark.parametrize(
    "relative", sorted(set(ACCURACY_TOOLING) - set(BOARD_STAGE_TOOLING) - {"scripts/run_model_accuracy.sh"})
)
def test_shared_accuracy_tooling_change_measures_accuracy_only(stage_repository: Path, relative: str) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, f"../../{relative}")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [("w8a8", ["accuracy"], [])]


def test_shared_compile_runner_change_compiles_every_build(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, "../../scripts/run_model_compilation.sh")

    plan = _plan(stage_repository, registry)

    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["w16a16", "w8a8"]


@pytest.mark.parametrize("relative", BOARD_STAGE_TOOLING)
def test_board_stage_tooling_change_measures_both_stages(stage_repository: Path, relative: str) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))
    _edit(stage_repository, f"../../{relative}")

    plan = _plan(stage_repository, registry)

    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [
        ("w16a16", ["performance"], ["model"]),
        ("w8a8", ["performance", "accuracy"], ["model"]),
    ]


def test_pending_or_missing_build_compiles_its_invocation(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    runs = _published_runs(document)
    model = document["models"][0]
    int8 = next(build for build in model["builds"] if build["build_profile_id"] == "w8a8")
    # A cancelled run leaves its invocation in pending_builds rather than builds.
    model["builds"].remove(int8)
    document["pending_builds"] = [{
        "model_id": PLAN_MODEL,
        "build_profile_id": "w8a8",
        "compile_invocation_id": "model",
        "component_ids": ["model"],
        "compile_fingerprint": int8["compile_fingerprint"],
    }]
    registry = FakeRegistry({BRANCH: document}, runs)

    plan = _plan(stage_repository, registry)

    assert _selected(plan, "compile") == [("w8a8", ["performance", "accuracy"], ["model"])]
    assert plan["measure"]["include"] == []


def test_force_rebuild_compiles_published_invocations(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({BRANCH: document}, _published_runs(document))

    plan = _plan(stage_repository, registry, force_rebuild_all=True)

    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["w16a16", "w8a8"]
    assert plan["measure"]["include"] == []


def test_branch_without_catalog_plans_against_base_branch_catalog(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry({"develop": document}, _published_runs(document))
    _edit(stage_repository, "scripts/accuracy.py")

    plan = _plan(stage_repository, registry, base_branches=["develop"])

    assert plan["catalog_found"] is True
    assert plan["compile"]["include"] == []
    assert _selected(plan, "measure") == [("w8a8", ["accuracy"], [])]
    assert _catalog_reads(registry) == [BRANCH, "develop"]
    assert all(url.startswith(f"{READ_API}/v1/repositories/sima-neat/models/") for url in registry.urls)


def test_feature_branch_reuses_its_own_and_base_builds(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    # The branch compiled bf16, then merged a develop that recompiled int8.
    registry = FakeRegistry(
        {BRANCH: _stale(document, "w8a8"), "develop": _stale(document, "w16a16")},
        _published_runs(document),
    )

    plan = _plan(stage_repository, registry, base_branches=["develop", "release-1.0"])

    assert plan["compile"]["include"] == []
    assert plan["measure"]["include"] == []
    assert _catalog_reads(registry) == [BRANCH, "develop", "release-1.0"]


def test_long_lived_branch_reuses_only_its_own_catalog(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry(
        {"develop": _stale(document, "w8a8"), "release-1.0": document}, _published_runs(document)
    )

    plan = _plan(stage_repository, registry, branch="develop", base_branches=["release-1.0"])

    assert _selected(plan, "compile") == [("w8a8", ["performance", "accuracy"], ["model"])]
    assert _catalog_reads(registry) == ["develop"]


def test_long_lived_branch_first_push_inherits_from_the_closest_base(stage_repository: Path) -> None:
    document = _published_catalog(stage_repository)
    registry = FakeRegistry(
        {"develop": _stale(document, "w8a8"), "integration/vision": document}, _published_runs(document)
    )

    plan = _plan(stage_repository, registry, branch="release-1.0", base_branches=["develop", "integration/vision"])

    assert _selected(plan, "compile") == [("w8a8", ["performance", "accuracy"], ["model"])]
    assert _catalog_reads(registry) == ["release-1.0", "develop"]


@pytest.mark.parametrize("base_branches", [None, [], ["develop"]])
def test_no_catalog_compiles_everything(stage_repository: Path, base_branches: list[str] | None) -> None:
    registry = FakeRegistry({}, set())

    plan = _plan(stage_repository, registry, base_branches=base_branches)

    assert plan["catalog_found"] is False
    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["w16a16", "w8a8"]
    assert plan["measure"]["include"] == []


def test_offline_plan_compiles_everything(stage_repository: Path) -> None:
    plan = _plan(stage_repository, None)

    assert plan["catalog_found"] is False
    assert _selected(plan, "compile") == [
        ("w16a16", ["performance"], ["model"]),
        ("w8a8", ["performance", "accuracy"], ["model"]),
    ]


def _mark_on_demand(repository: Path) -> None:
    definition = repository / "models" / PLAN_MODEL / "model.yaml"
    definition.write_text(definition.read_text(encoding="utf-8") + "\nbuild-on-demand: true\n", encoding="utf-8")


@pytest.mark.parametrize("force_rebuild_all", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_on_demand_models_never_compile_or_measure_automatically(
    stage_repository: Path, force_rebuild_all: bool, published: bool,
) -> None:
    document = _published_catalog(stage_repository)
    _mark_on_demand(stage_repository)
    _edit(stage_repository, "scripts/compile.py")
    registry = FakeRegistry({BRANCH: document} if published else {}, set())

    plan = _plan(stage_repository, registry, force_rebuild_all=force_rebuild_all)

    assert plan["compile"] == {"include": []}
    assert plan["measure"] == {"include": []}
    assert registry.resolved() == []


def test_manual_model_only_selection_keeps_ordinary_builds(stage_repository: Path) -> None:
    plan = _plan(stage_repository, None, model_ids=[PLAN_MODEL], include_on_demand=True)

    assert [entry["build_profile_id"] for entry in plan["compile"]["include"]] == ["w16a16", "w8a8"]
    assert all("build_on_demand" not in entry for entry in plan["compile"]["include"])


@pytest.mark.parametrize("scope", [{}, {"model_ids": [PLAN_MODEL]}, {"profile_ids": ["w16a16"]}])
def test_manual_on_demand_build_requires_explicit_model_and_profile(stage_repository: Path, scope: dict) -> None:
    _mark_on_demand(stage_repository)

    with pytest.raises(ValidationError, match="explicit --models and --profiles"):
        _plan(stage_repository, None, include_on_demand=True, **scope)


def test_manual_on_demand_build_selects_only_requested_model_and_profile(stage_repository: Path) -> None:
    _mark_on_demand(stage_repository)
    shutil.copytree(FIXTURE, stage_repository / "models" / "fixture_model")

    plan = _plan(
        stage_repository, None, model_ids=[PLAN_MODEL], profile_ids=["w16a16"], include_on_demand=True,
    )

    assert len(plan["compile"]["include"]) == 1
    entry = plan["compile"]["include"][0]
    assert (entry["model_id"], entry["build_profile_id"], entry["build_on_demand"]) == (
        PLAN_MODEL, "w16a16", True,
    )


@pytest.mark.parametrize("incremental", [False, True])
def test_catalog_lists_unbuilt_on_demand_models_without_pending_builds(
    stage_repository: Path, incremental: bool,
) -> None:
    _mark_on_demand(stage_repository)

    document = build_catalog_snapshot(
        repository=stage_repository, branch=BRANCH, git_sha="b" * 40, incremental=incremental,
    )

    assert document["models"][0]["build_on_demand"] is True
    assert document["models"][0]["builds"] == []
    assert document["models"][0]["packages"] == []
    assert document.get("pending_builds", []) == []
    requirements = document["models"][0]["build_requirements"]
    matrix = build_matrix(load_models(stage_repository), [PLAN_MODEL], repository=stage_repository,
                          compiler_version=COMPILER_VERSION)["include"]
    for entry in matrix:
        profile = requirements[entry["build_profile_id"]]
        invocation = next(item for item in profile["invocations"] if item["id"] == entry["compile_invocation_id"])
        assert invocation["compile_fingerprint"] == entry["build_fingerprint"]
        assert invocation["component_ids"] == sorted(entry["component_ids"])
        assert profile["compiler_version"] == resolve_model_compiler(stage_repository)["version"]


@pytest.mark.parametrize("candidate_receipts", [False, True])
def test_catalog_keeps_approved_on_demand_builds_and_packages_after_input_changes(
    stage_repository: Path, tmp_path: Path, candidate_receipts: bool,
) -> None:
    previous = _published_catalog(stage_repository)
    approved = previous["models"][0]
    approved["builds"][0]["compiler_identity"] = "approved-compiler"
    approved["packages"] = [{
        "package_id": "default", "build_profile_id": "w16a16", "run_id": "approved-package",
        "artifact_sha256": "c" * 64, "size_bytes": 20, "package_fingerprint": "d" * 64,
        "component_run_ids": {"model": approved["builds"][0]["run_id"]},
        "layout": "deployment_files", "storage_backend": "huggingface",
        "deployment_size_bytes": 200, "file_count": 5,
    }]
    _mark_on_demand(stage_repository)
    _edit(stage_repository, "scripts/compile.py")
    invocation = build_matrix(
        load_models(stage_repository), [PLAN_MODEL], ["w16a16"],
        repository=stage_repository, compiler_version=COMPILER_VERSION,
    )["include"][0]
    assert approved["builds"][0]["compile_fingerprint"] != invocation["build_fingerprint"]
    selected_matrix = tmp_path / "selected.json"
    selected_matrix.write_text(json.dumps({"include": [invocation]}), encoding="utf-8")
    compiled_dir, package_dir = tmp_path / "compiled", tmp_path / "packages"
    if candidate_receipts:
        component = compiled_dir / PLAN_MODEL / "w16a16" / "model"
        component.mkdir(parents=True)
        (component / "registry-publication.json").write_text(json.dumps({"runs": [{
            "model_id": PLAN_MODEL, "build_profile_id": "w16a16", "component_id": "model",
            "run_id": "candidate-compile", "artifact_sha256": "e" * 64,
            "compile_fingerprint": invocation["build_fingerprint"],
        }]}), encoding="utf-8")
        package = package_dir / PLAN_MODEL / "w16a16" / "packages" / "default"
        package.mkdir(parents=True)
        (package / "registry-publication.json").write_text(json.dumps({"packages": [{
            **approved["packages"][0], "model_id": PLAN_MODEL, "run_id": "candidate-package",
            "manifest_sha256": "d" * 64, "component_run_ids": {"model": "candidate-compile"},
        }]}), encoding="utf-8")

    document = build_catalog_snapshot(
        repository=stage_repository, branch=BRANCH, git_sha="b" * 40, previous=previous,
        selected_matrix=selected_matrix, compiled_dir=compiled_dir, package_dir=package_dir, incremental=True,
    )

    assert document["models"][0]["builds"] == approved["builds"]
    assert document["models"][0]["packages"] == approved["packages"]
    assert document["pending_builds"] == []


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
        _plan(stage_repository, registry, base_branches=["develop"])
