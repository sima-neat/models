from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .artifact import record_compile_outputs, validate_compile_outputs, verify_artifact, verify_mpk
from .benchmark_declarations import BENCHMARK_STAGES, benchmarked_profiles, reference_metrics_by_model
from .compile_matrix import build_matrix
from .dependencies import resolve_benchmark_runtime, resolve_model_compiler
from .documentation import load_model_documentation
from .errors import ModelRegistryError
from .fetch import Fetcher, resolve_dataset
from .mla_stage_simulation import write_mla_stage_simulation
from .model import component_for, component_pipelines, load_and_validate_model, load_models, profile_component_for
from .model_tests import TEST_OUTCOMES
from .runner_approval import authorize_runner_matrix
from .stage_plan import plan_stages, resolve_base_branches

AUTHOR_COMMANDS = frozenset(
    {
        "fetch",
        "fetch-calibration",
        "fetch-fidelity-dataset",
        "matrix",
        "record-compile-outputs",
        "authorize-runner-matrix",
        "assemble-package",
        "resolve-compiler",
        "simulate-mla-stages",
        "validate",
        "validate-models",
        "validate-compile-outputs",
        "verify-mpk",
        "verify-artifact",
    }
)


def tool_version() -> str:
    try:
        return version("sima-neat-model-registry")
    except PackageNotFoundError:
        return "source"


def _parser(*, author_safe: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-registry", description="SiMa Neat Model Registry lifecycle tooling")
    parser.add_argument("--version", action="version", version=f"%(prog)s {tool_version()}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate a model directory against the model schema")
    validate.add_argument("model_dir", type=Path)
    validate.add_argument("--schema", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")

    validate_models = subcommands.add_parser(
        "validate-models", help="validate every model manifest and its taxonomy categorization"
    )
    validate_models.add_argument("--repository", type=Path, default=Path.cwd())
    validate_models.add_argument("--json", action="store_true", dest="json_output")

    conversion = subcommands.add_parser("publish-conversion", help="publish a validated conversion output")
    conversion.add_argument("model_id")
    conversion.add_argument("output_dir", type=Path)
    conversion.add_argument("--repository", type=Path, default=Path.cwd())
    conversion.add_argument("--api-url", required=True)
    conversion.add_argument("--bucket", required=True)
    conversion.add_argument("--kms-key-id")
    conversion.add_argument("--aws-region", default="us-west-2")
    conversion.add_argument("--environment", required=True)
    conversion.add_argument("--validation", required=True, choices=["passed"])

    entrypoint = subcommands.add_parser(
        "conversion-entrypoint", help="print the conversion output compile.py receives as --model"
    )
    entrypoint.add_argument("model_dir", type=Path)

    fetch = subcommands.add_parser("fetch", help="download and verify a model's immutable source")
    fetch.add_argument("model_dir", type=Path)
    fetch.add_argument("--schema", type=Path)
    fetch.add_argument("--cache-dir", type=Path)
    fetch.add_argument("--component", help="component ID; optional when the model contains one component")
    fetch.add_argument("--json", action="store_true", dest="json_output")

    calibration = subcommands.add_parser(
        "fetch-calibration", help="download and verify a model's immutable calibration snapshot"
    )
    calibration.add_argument("model_dir", type=Path)
    calibration.add_argument("--schema", type=Path)
    calibration.add_argument("--cache-dir", type=Path)
    calibration.add_argument("--build-profile", required=True, help="build profile that declares the calibration")
    calibration.add_argument("--component", required=True, help="component within the selected build profile")
    calibration.add_argument("--json", action="store_true", dest="json_output")

    fidelity_dataset = subcommands.add_parser(
        "fetch-fidelity-dataset", help="download and verify the dataset numerical fidelity compares on"
    )
    fidelity_dataset.add_argument("model_dir", type=Path)
    fidelity_dataset.add_argument("--schema", type=Path)
    fidelity_dataset.add_argument("--cache-dir", type=Path)
    fidelity_dataset.add_argument("--build-profile", required=True)
    fidelity_dataset.add_argument("--component", required=True, help="component within the selected build profile")
    fidelity_dataset.add_argument("--json", action="store_true", dest="json_output")

    matrix = subcommands.add_parser("matrix", help="emit the deterministic compile invocation matrix")
    matrix.add_argument("--repository", type=Path, default=Path.cwd())
    matrix.add_argument("--models", help="comma-separated model IDs; every model when omitted")
    matrix.add_argument("--profiles", help="comma-separated build profile IDs within the selected models")
    matrix.add_argument("--components", help="comma-separated component IDs within the selected models")
    matrix.add_argument("--compiler-version", help="also emit each invocation's compile fingerprint")

    pipelines = subcommands.add_parser(
        "component-pipelines", help="write the order each multi-component model runs in, for the model browser"
    )
    pipelines.add_argument("--repository", type=Path, default=Path.cwd())
    pipelines.add_argument("--output", type=Path, required=True)

    reference = subcommands.add_parser(
        "reference-metrics", help="write each model's published and FP32 baseline scores, for the model browser"
    )
    reference.add_argument("--repository", type=Path, default=Path.cwd())
    reference.add_argument("--output", type=Path, required=True)

    stages = subcommands.add_parser(
        "plan-stages", help="select changed compile invocations and the benchmarks reused builds still need"
    )
    stages.add_argument("--repository", type=Path, default=Path.cwd())
    stages.add_argument("--compiler-version", required=True)
    stages.add_argument("--branch", required=True)
    stages.add_argument("--base-branches", help="comma-separated branches whose catalogs builds may come from")
    stages.add_argument("--read-api-url", help="public artifact-registry API; omit with --offline")
    stages.add_argument("--api-url", default=os.environ.get("VULCAN_ARTIFACT_REGISTRY_API_URL"))
    stages.add_argument("--aws-region", default="us-west-2")
    stages.add_argument("--offline", action="store_true", help="read no registry state; select every invocation")
    stages.add_argument("--models", help="comma-separated model IDs; every model when omitted")
    stages.add_argument("--profiles", help="comma-separated build profile IDs within the selected models")
    stages.add_argument("--components", help="comma-separated component IDs within the selected models")
    stages.add_argument("--force-rebuild-all", action="store_true")
    stages.add_argument("--include-on-demand", action="store_true", help="build explicitly selected on-demand models")

    authorize = subcommands.add_parser(
        "authorize-runner-matrix",
        help="authorize compile runner requests against a Terraform-published Vulcan specification",
    )
    authorize.add_argument("--matrix", type=Path, required=True)
    authorize.add_argument("--vulcan-spec", type=Path, required=True)
    authorize.add_argument("--repository-name", required=True)

    base = subcommands.add_parser("resolve-base", help="list the base branches whose catalogs builds may come from")
    base.add_argument("--repository", type=Path, default=Path.cwd())
    base.add_argument("--current-branch", required=True)
    base.add_argument("--head", required=True)
    base.add_argument("--preferred-branch")

    compiler = subcommands.add_parser("resolve-compiler", help="resolve the repository-wide Model Compiler dependency")
    compiler.add_argument("--repository", type=Path, default=Path.cwd())

    benchmark_runtime = subcommands.add_parser(
        "resolve-benchmark-runtime", help="resolve pinned DevKit and benchmark runtime dependencies"
    )
    benchmark_runtime.add_argument("--repository", type=Path, default=Path.cwd())

    devkit_accuracy = subcommands.add_parser(
        "resolve-devkit-accuracy", help="resolve each DevKit accuracy dataset of a build profile into the cache"
    )
    devkit_accuracy.add_argument("model_dir", type=Path)
    devkit_accuracy.add_argument("--schema", type=Path)
    devkit_accuracy.add_argument("--cache-dir", type=Path)
    devkit_accuracy.add_argument("--build-profile", required=True)

    inputs = subcommands.add_parser(
        "model-component-mpks", help="print component=path for each component MPK a build profile's benchmarks run"
    )
    inputs.add_argument("compiled_dir", type=Path, help="root holding <model>/<profile>/compile-outputs.json")
    inputs.add_argument("--repository", type=Path, default=Path.cwd())
    inputs.add_argument("--model", required=True)
    inputs.add_argument("--build-profile", required=True)

    check = subcommands.add_parser(
        "check-report", help="check a model report against model.yaml and record the DevKit it ran on"
    )
    check.add_argument("output_dir", type=Path, help="the stage output holding model-report.json")
    check.add_argument("--repository", type=Path, default=Path.cwd())
    check.add_argument("--stage", choices=BENCHMARK_STAGES, required=True)
    check.add_argument("--model", required=True)
    check.add_argument("--build-profile", required=True)

    mpk = subcommands.add_parser(
        "verify-mpk", aliases=["verify-artifact"], help="validate and checksum a compiled artifact"
    )
    mpk.add_argument("output_dir", type=Path)
    mpk.add_argument("--write-manifest", action="store_true")
    mpk.add_argument("--json", action="store_true", dest="json_output")
    mpk.add_argument("--model-id")
    mpk.add_argument("--build-profile-id")
    mpk.add_argument("--component-id")
    mpk.add_argument("--target")
    mpk.add_argument("--compiler-version")
    mpk.add_argument("--compiler-label")
    mpk.add_argument("--source-sha256")
    mpk.add_argument("--calibration-sha256")
    mpk.add_argument("--git-sha")
    mpk.add_argument("--environment")
    mpk.add_argument("--compile-fingerprint")
    mpk.add_argument("--requested-runner-class")
    mpk.add_argument("--requested-runner-market")
    mpk.add_argument("--runner-market")
    mpk.add_argument("--runner-name")
    mpk.add_argument("--runner-instance-id")
    mpk.add_argument("--runner-instance-type")
    mpk.add_argument("--runner-actual-market")
    mpk.add_argument("--runner-class")
    mpk.add_argument("--runner-approval-id")
    mpk.add_argument("--runner-approval-spec-revision")

    compile_outputs = subcommands.add_parser(
        "validate-compile-outputs", help="validate every MPK declared by one compile invocation"
    )
    compile_outputs.add_argument("manifest", type=Path)
    compile_outputs.add_argument("--output-root", type=Path, required=True)
    compile_outputs.add_argument("--model-id", required=True)
    compile_outputs.add_argument("--build-profile-id", required=True)
    compile_outputs.add_argument("--compile-invocation-id", required=True)
    compile_outputs.add_argument("--components", required=True, help="comma-separated component IDs")
    compile_outputs.add_argument("--json", action="store_true", dest="json_output")

    record_outputs = subcommands.add_parser(
        "record-compile-outputs", help="write compile-outputs.json for a recipe that wrote only component MPKs"
    )
    record_outputs.add_argument("output_root", type=Path)
    record_outputs.add_argument("--model-id", required=True)
    record_outputs.add_argument("--build-profile-id", required=True)
    record_outputs.add_argument("--compile-invocation-id", required=True)
    record_outputs.add_argument("--components", required=True, help="comma-separated component IDs")

    simulate = subcommands.add_parser(
        "simulate-mla-stages", help="write mla-stage-simulation.json: mla-msim's estimate of each compiled MLA stage"
    )
    simulate.add_argument("output_dir", type=Path, help="the build profile's compile output, with compile-outputs.json")

    package = subcommands.add_parser(
        "assemble-package", help="assemble deterministic package bundles from verified component MPKs"
    )
    package.add_argument("--repository", type=Path, default=Path.cwd())
    package.add_argument("--compiled-dir", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--model")
    package.add_argument("--build-profile")
    package.add_argument("--package")
    package.add_argument("--all", action="store_true", dest="all_packages")
    package.add_argument("--json", action="store_true", dest="json_output")

    publish = subcommands.add_parser("publish", help="publish completed model compilations to the Vulcan registry")
    publish.add_argument("--repository", type=Path, default=Path.cwd())
    publish.add_argument("--dist-dir", type=Path, required=True)
    publish.add_argument("--api-url", required=True)
    publish.add_argument("--aws-region", default="us-west-2")
    publish.add_argument("--bucket", required=True)
    publish.add_argument("--kms-key-id")
    publish.add_argument("--environment", choices=("staging", "production"), required=True)
    publish.add_argument("--branch", required=True)
    publish.add_argument("--git-sha", required=True)
    publish.add_argument("--github-run-id", required=True)
    publish.add_argument("--github-run-attempt", required=True)
    publish.add_argument("--github-run-url", required=True)
    publish.add_argument("--json", action="store_true", dest="json_output")

    candidate = subcommands.add_parser(
        "publish-candidate", help="publish and retain a complete on-demand model/profile"
    )
    candidate.add_argument("--repository", type=Path, default=Path.cwd())
    candidate.add_argument("--dist-dir", type=Path, required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)
    candidate.add_argument("--model", required=True)
    candidate.add_argument("--build-profile", required=True)
    candidate.add_argument("--api-url", required=True)
    candidate.add_argument("--aws-region", default="us-west-2")
    candidate.add_argument("--bucket", required=True)
    candidate.add_argument("--kms-key-id")
    candidate.add_argument("--environment", choices=("staging", "production"), required=True)
    candidate.add_argument("--branch", required=True)
    candidate.add_argument("--git-sha", required=True)
    candidate.add_argument("--github-run-id", required=True)
    candidate.add_argument("--github-run-attempt", required=True)
    candidate.add_argument("--github-run-url", required=True)
    candidate.add_argument("--summary", type=Path)
    candidate.add_argument("--json", action="store_true", dest="json_output")

    branch_deployment = subcommands.add_parser(
        "publish-branch-deployment", help="select a finalized candidate in its source feature-branch catalog"
    )
    branch_deployment.add_argument("--candidate", type=Path, required=True)
    branch_deployment.add_argument("--api-url", required=True)
    branch_deployment.add_argument("--read-api-url", required=True)
    branch_deployment.add_argument("--aws-region", default="us-west-2")

    publish_package = subcommands.add_parser(
        "publish-package", help="publish assembled full-model packages to the Vulcan registry"
    )
    publish_package.add_argument("--repository", type=Path, default=Path.cwd())
    publish_package.add_argument("--package-dir", type=Path, required=True)
    publish_package.add_argument("--api-url", required=True)
    publish_package.add_argument("--aws-region", default="us-west-2")
    publish_package.add_argument("--bucket", required=True)
    publish_package.add_argument("--kms-key-id")
    publish_package.add_argument("--environment", choices=("staging", "production"), required=True)
    publish_package.add_argument("--branch", required=True)
    publish_package.add_argument("--git-sha", required=True)
    publish_package.add_argument("--github-run-id", required=True)
    publish_package.add_argument("--github-run-attempt", required=True)
    publish_package.add_argument("--github-run-url", required=True)
    publish_package.add_argument("--json", action="store_true", dest="json_output")

    publish_report = subcommands.add_parser(
        "publish-report", help="publish a checked model report as a whole-model benchmark run"
    )
    publish_report.add_argument("output_dir", type=Path, help="the stage output check-report accepted")
    publish_report.add_argument("--stage", choices=BENCHMARK_STAGES, required=True)
    publish_report.add_argument(
        "--repository", type=Path, required=True, help="models checkout at the measured revision"
    )
    publish_report.add_argument(
        "--compiled-dir", type=Path, required=True, help="the registered build and package the report measured"
    )
    publish_report.add_argument("--model", required=True)
    publish_report.add_argument("--build-profile", required=True)
    publish_report.add_argument("--api-url", required=True)
    publish_report.add_argument("--aws-region", default="us-west-2")
    publish_report.add_argument("--bucket", required=True)
    publish_report.add_argument("--kms-key-id")
    publish_report.add_argument("--environment", choices=("staging", "production"), required=True)
    publish_report.add_argument("--branch", required=True)
    publish_report.add_argument("--git-sha", required=True)
    publish_report.add_argument("--github-run-url")
    publish_report.add_argument(
        "--test-status", choices=TEST_OUTCOMES, required=True, help="outcome of the model's tests on the report"
    )
    publish_report.add_argument("--json", action="store_true", dest="json_output")

    publish_catalog_parser = subcommands.add_parser(
        "publish-catalog", help="publish and atomically promote a complete model catalog snapshot"
    )
    publish_catalog_parser.add_argument("--repository", type=Path, default=Path.cwd())
    publish_catalog_parser.add_argument("--compiled-dir", type=Path)
    publish_catalog_parser.add_argument("--package-dir", type=Path)
    publish_catalog_parser.add_argument(
        "--selected-matrix",
        type=Path,
        help="omit prior packages for build profiles selected for replacement",
    )
    publish_catalog_parser.add_argument(
        "--force-rebuild-all", action="store_true",
        help="reject forced replacement of packages already published in the current revision",
    )
    publish_catalog_parser.add_argument("--api-url", required=True)
    publish_catalog_parser.add_argument("--read-api-url", required=True)
    publish_catalog_parser.add_argument("--aws-region", default="us-west-2")
    publish_catalog_parser.add_argument("--bucket", required=True)
    publish_catalog_parser.add_argument("--kms-key-id")
    publish_catalog_parser.add_argument("--environment", required=True)
    publish_catalog_parser.add_argument("--branch", required=True)
    publish_catalog_parser.add_argument(
        "--base-branches", help="comma-separated branches whose catalogs builds may come from"
    )
    publish_catalog_parser.add_argument("--git-sha", required=True)
    publish_catalog_parser.add_argument("--github-run-url")
    publish_catalog_parser.add_argument("--json", action="store_true", dest="json_output")

    stage_registry = subcommands.add_parser(
        "stage-registry-build", help="restore a current catalog build from durable registry MPKs"
    )
    stage_registry.add_argument("--repository", type=Path, default=Path.cwd())
    stage_registry.add_argument("--output-dir", type=Path, required=True)
    stage_registry.add_argument("--api-url", required=True)
    stage_registry.add_argument("--branch", required=True)
    stage_registry.add_argument("--catalog-revision", required=True)
    stage_registry.add_argument("--model-id", required=True)
    stage_registry.add_argument("--build-profile-id", required=True)
    stage_registry.add_argument("--compile-invocation-id", required=True)
    stage_registry.add_argument("--json", action="store_true", dest="json_output")

    if not author_safe:
        from .promotions import add_parser

        add_parser(subcommands)

    if author_safe:
        for command in tuple(subcommands.choices):
            if command not in AUTHOR_COMMANDS:
                del subcommands.choices[command]
        subcommands._choices_actions = [
            choice for choice in subcommands._choices_actions if choice.dest in AUTHOR_COMMANDS
        ]
    return parser


def _validate(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    result = {"model_id": model["id"], "model_dir": str(args.model_dir.resolve()), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid model: {model['id']}")
    return 0


def _validate_models(args: argparse.Namespace) -> int:
    repository = args.repository.expanduser().resolve()
    models = load_models(repository)
    for model_id, model in sorted(models.items()):
        load_model_documentation(
            repository / "models" / model_id,
            model["display_name"],
            repository=repository,
        )
    result = {"model_count": len(models), "model_ids": sorted(models), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid models: {len(models)}")
    return 0


def _validate_compile_outputs(args: argparse.Namespace) -> int:
    component_ids = [item.strip() for item in args.components.split(",") if item.strip()]
    if not component_ids:
        raise ModelRegistryError("--components must contain at least one component ID")
    result = validate_compile_outputs(
        args.manifest,
        args.output_root,
        model_id=args.model_id,
        build_profile_id=args.build_profile_id,
        compile_invocation_id=args.compile_invocation_id,
        component_ids=component_ids,
    )
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Valid compile outputs: {len(result['outputs'])} component(s)")
    return 0


def _selected_component(model: dict[str, object], component_id: str | None) -> dict[str, object]:
    components = model["components"]
    if component_id is None:
        if len(components) != 1:
            raise ModelRegistryError("--component is required when the model contains multiple components")
        return components[0]
    return component_for(model, component_id)


def _fetch(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    component = _selected_component(model, args.component)
    from .conversion import apply_selection, fetch_conversion

    apply_selection(model, root=args.model_dir.resolve().parents[1])
    if component["source"]["type"] == "conversion":
        path, cache_hit, source_sha256, size_bytes = fetch_conversion(component["source"], args.cache_dir)
    else:
        path, cache_hit, source_sha256, size_bytes = Fetcher(args.cache_dir).fetch_source(component["source"])
    source_root = path.parent
    if component["source"]["type"] == "file_set":
        source_root = path
        for _ in Path(component["source"]["entrypoint"]).parts:
            source_root = source_root.parent
    result = {
        "cache_hit": cache_hit,
        "component_id": component["id"],
        "model_id": model["id"],
        "path": str(path),
        "root": str(source_root),
        "sha256": source_sha256,
        "size_bytes": size_bytes,
    }
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        state = "verified cache hit" if cache_hit else "downloaded and verified"
        print(f"{state}: {path}")
    return 0


def _fetch_calibration(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    _, binding = profile_component_for(model, args.build_profile, args.component)
    calibration_id = binding.get("calibration")
    calibration = next((item for item in model["calibrations"] if item["id"] == calibration_id), None)
    if calibration is None:
        result = {
            "kind": "none",
            "model_id": model["id"],
            "component_id": args.component,
            "build_profile_id": args.build_profile,
            "path": "",
            "sha256": "",
            "size_bytes": 0,
        }
    else:
        dataset = calibration["dataset"]
        path = resolve_dataset(dataset, args.cache_dir)
        result = {
            "kind": "dataset",
            "model_id": model["id"],
            "component_id": args.component,
            "build_profile_id": args.build_profile,
            "path": str(path),
            "sha256": dataset["sha256"],
            "size_bytes": dataset["size_bytes"],
        }
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    elif result["path"]:
        print(f"Verified calibration: {result['path']}")
    else:
        print(f"No calibration required: {result['model_id']}")
    return 0


def _fetch_fidelity_dataset(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    profile_component_for(model, args.build_profile, args.component)
    selected = model.get("numerical_fidelity")
    result = {
        "kind": "none",
        "model_id": model["id"],
        "component_id": args.component,
        "build_profile_id": args.build_profile,
        "path": "",
        "sha256": "",
        "size_bytes": 0,
    }
    if selected is not None:
        dataset = selected["dataset"]
        result.update(
            kind="dataset",
            path=str(resolve_dataset(dataset, args.cache_dir)),
            sha256=dataset["sha256"],
            size_bytes=dataset["size_bytes"],
        )
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    elif result["path"]:
        print(f"Verified fidelity dataset: {result['path']}")
    else:
        print(f"No numerical fidelity declared: {result['model_id']}")
    return 0


def _id_list(value: str | None, option: str) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ModelRegistryError(f"{option} must contain at least one ID")
    return items


def _branch_list(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _matrix(args: argparse.Namespace) -> int:
    models = load_models(args.repository)
    result = build_matrix(
        models,
        _id_list(args.models, "--models") or sorted(models),
        _id_list(args.profiles, "--profiles"),
        _id_list(args.components, "--components"),
        repository=args.repository if args.compiler_version is not None else None,
        compiler_version=args.compiler_version,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _component_pipelines(args: argparse.Namespace) -> int:
    pipelines = component_pipelines(load_models(args.repository))
    args.output.write_text(json.dumps(pipelines, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


def _reference_metrics(args: argparse.Namespace) -> int:
    reference = reference_metrics_by_model(load_models(args.repository))
    args.output.write_text(json.dumps(reference, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


def _plan_stages(args: argparse.Namespace) -> int:
    from .registry import SigV4RegistryClient, _public_registry_json

    if args.offline == bool(args.read_api_url):
        raise ModelRegistryError("pass exactly one of --read-api-url or --offline")
    result = plan_stages(
        args.repository,
        compiler_version=args.compiler_version,
        read=None if args.offline else _public_registry_json,
        read_api_url=args.read_api_url,
        branch=args.branch,
        base_branches=_branch_list(args.base_branches),
        model_ids=_id_list(args.models, "--models"),
        profile_ids=_id_list(args.profiles, "--profiles"),
        component_ids=_id_list(args.components, "--components"),
        force_rebuild_all=args.force_rebuild_all,
        include_on_demand=args.include_on_demand,
        conversion_client=SigV4RegistryClient(args.api_url, args.aws_region) if args.api_url else None,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _authorize_runner_matrix(args: argparse.Namespace) -> int:
    try:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        vulcan_spec = json.loads(args.vulcan_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"could not load runner authorization input: {exc}") from exc
    if not isinstance(matrix, dict) or not isinstance(vulcan_spec, dict):
        raise ModelRegistryError("runner authorization inputs must be JSON objects")
    result = authorize_runner_matrix(matrix, vulcan_spec, repository=args.repository_name)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_base(args: argparse.Namespace) -> int:
    base_branches = resolve_base_branches(
        args.repository,
        current_branch=args.current_branch,
        head=args.head,
        preferred_branch=args.preferred_branch,
    )
    print(json.dumps({"base_branches": base_branches}, sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_compiler(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_model_compiler(args.repository), sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_benchmark_runtime(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_benchmark_runtime(args.repository), sort_keys=True, separators=(",", ":")))
    return 0


def _verify_mpk(args: argparse.Namespace) -> int:
    fields = (
        "model_id",
        "build_profile_id",
        "component_id",
        "target",
        "compiler_version",
        "compiler_label",
        "source_sha256",
        "calibration_sha256",
        "git_sha",
        "environment",
        "compile_fingerprint",
        "requested_runner_class",
        "requested_runner_market",
        "runner_market",
        "runner_name",
        "runner_instance_id",
        "runner_instance_type",
        "runner_actual_market",
        "runner_class",
        "runner_approval_id",
        "runner_approval_spec_revision",
    )
    provenance = {field: getattr(args, field) for field in fields if getattr(args, field) is not None}
    if (args.output_dir / "deployment").is_dir():
        try:
            provenance["llima_version"] = version("sima-lmm")
        except PackageNotFoundError:
            pass  # Deployed artifacts can also be verified without installing the compiler.
    verify = verify_artifact if args.command == "verify-artifact" else verify_mpk
    result = verify(args.output_dir, write_manifest=args.write_manifest, provenance=provenance)
    print(json.dumps(result, sort_keys=True) if args.json_output
          else f"Valid artifact: {result.get('manifest', result.get('archive'))}")
    return 0


def _assemble_package(args: argparse.Namespace) -> int:
    from .package import assemble_available_packages, assemble_package

    explicit = (args.model, args.build_profile, args.package)
    if args.all_packages == any(value is not None for value in explicit):
        raise ModelRegistryError(
            "select --all or provide --model, --build-profile, and --package"
        )
    if args.all_packages:
        results = assemble_available_packages(
            repository=args.repository,
            compiled_dir=args.compiled_dir,
            output_dir=args.output_dir,
        )
    else:
        if any(value is None for value in explicit):
            raise ModelRegistryError("--model, --build-profile, and --package must be supplied together")
        results = [
            assemble_package(
                repository=args.repository,
                compiled_dir=args.compiled_dir,
                output_dir=args.output_dir,
                model_id=args.model,
                build_profile_id=args.build_profile,
                package_id=args.package,
            )
        ]
    payload = {"count": len(results), "packages": results}
    print(json.dumps(payload, sort_keys=True) if args.json_output else f"Assembled {len(results)} package(s)")
    return 0


def _publish(args: argparse.Namespace) -> int:
    from .registry import publish_compilations

    result = publish_compilations(
        repository=args.repository,
        dist_dir=args.dist_dir,
        api_url=args.api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        git_sha=args.git_sha,
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
        github_run_url=args.github_run_url,
    )
    runs = {(run["model_id"], run["build_profile_id"], run["component_id"]): run for run in result["runs"]}
    if len(runs) != result["count"]:
        raise ModelRegistryError("published compilation result contains duplicate components")
    for (model_id, profile_id, component_id), run in runs.items():
        receipt = {**result, "count": 1, "runs": [run]}
        path = args.dist_dir / model_id / profile_id / component_id / "registry-publication.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Published {result['count']} compilation(s)"
    print(message)
    return 0


def _publish_package(args: argparse.Namespace) -> int:
    from .registry import publish_packages

    result = publish_packages(
        repository=args.repository,
        package_dir=args.package_dir,
        api_url=args.api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        git_sha=args.git_sha,
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
        github_run_url=args.github_run_url,
    )
    for package in result["packages"]:
        path = (
            args.package_dir / package["model_id"] / package["build_profile_id"]
            / "packages" / package["package_id"] / "registry-publication.json"
        )
        path.write_text(json.dumps({"packages": [package]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Published {result['count']} package(s)"
    print(message)
    return 0


def _publish_candidate(args: argparse.Namespace) -> int:
    from .candidates import publish_candidate

    result = publish_candidate(
        repository=args.repository, dist_dir=args.dist_dir, output_dir=args.output_dir,
        model_id=args.model, build_profile_id=args.build_profile,
        api_url=args.api_url, aws_region=args.aws_region, bucket=args.bucket, kms_key_id=args.kms_key_id,
        environment=args.environment, branch=args.branch, git_sha=args.git_sha,
        github_run_id=args.github_run_id, github_run_attempt=args.github_run_attempt,
        github_run_url=args.github_run_url,
    )
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as stream:
            stream.write(
                f"Candidate ready: **{args.model}/{args.build_profile}** ({args.environment})\n\n"
                f"- Deployment: `{result['deployment_id']}`\n"
                f"- Manifest: `{result['manifest_digest']}`\n"
                f"- Source: `{result['git_ref']}` at `{result['commit_sha']}`\n"
                f"- Protection: `{json.dumps(result['protection'], sort_keys=True)}`\n"
                f"- [Build and compile validation]({args.github_run_url})\n\n"
                "Candidate finalized; approval, DevKit qualification, and catalog promotion are separate.\n"
            )
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Candidate ready: {result['deployment_id']}")
    return 0


def _resolve_devkit_accuracy(args: argparse.Namespace) -> int:
    """Print one JSON line per dataset the profile's accuracy evaluates, in model.yaml order; none without one."""
    model = load_and_validate_model(args.model_dir, args.schema)
    if args.build_profile not in benchmarked_profiles(model, "accuracy"):
        return 0
    for name, dataset in model["accuracy"]["datasets"].items():
        print(json.dumps({
            "name": name,
            "path": str(resolve_dataset(dataset, args.cache_dir)),
            "sha256": dataset["sha256"],
            "size_bytes": dataset["size_bytes"],
        }, sort_keys=True))
    return 0


def _check_report(args: argparse.Namespace) -> int:
    from model_report import REPORT_FILE

    from .report_publication import ENVIRONMENT_FILE, check_report, runner_environment

    report = check_report(args.output_dir / REPORT_FILE, args.repository, args.model, args.build_profile, args.stage)
    (args.output_dir / ENVIRONMENT_FILE).write_text(
        json.dumps(runner_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.stage == "accuracy":
        for name, result in report["datasets"].items():
            print(f"Checked the accuracy report on {name}: {', '.join(sorted(result['metrics']))}")
    else:
        print(f"Checked the {args.stage} report: {', '.join(sorted(report['metrics']))}")
    return 0


def _publish_report(args: argparse.Namespace) -> int:
    from .report_publication import publish_report

    result = publish_report(
        output_dir=args.output_dir,
        stage=args.stage,
        repository=args.repository,
        compiled_dir=args.compiled_dir,
        model_id=args.model,
        build_profile_id=args.build_profile,
        api_url=args.api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        git_sha=args.git_sha,
        github_run_url=args.github_run_url,
        test_status=args.test_status,
    )
    message = (
        json.dumps(result, sort_keys=True)
        if args.json_output
        else f"Published {args.stage} {result['benchmark_fingerprint']} "
        f"for {result['model_id']}/{result['build_profile_id']}"
    )
    print(message)
    return 0


def _publish_catalog(args: argparse.Namespace) -> int:
    from .registry import publish_catalog

    result = publish_catalog(
        repository=args.repository,
        compiled_dir=args.compiled_dir,
        package_dir=args.package_dir,
        selected_matrix=args.selected_matrix,
        force_rebuild_all=args.force_rebuild_all,
        api_url=args.api_url,
        read_api_url=args.read_api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        base_branches=_branch_list(args.base_branches),
        git_sha=args.git_sha,
        github_run_url=args.github_run_url,
    )
    message = (
        json.dumps(result, sort_keys=True)
        if args.json_output
        else f"Published catalog {result['catalog']['catalog_revision']}"
    )
    print(message)
    return 0


def _stage_registry_build(args: argparse.Namespace) -> int:
    from .registry import stage_catalog_build

    result = stage_catalog_build(
        repository=args.repository,
        output_dir=args.output_dir,
        api_url=args.api_url,
        branch=args.branch,
        catalog_revision=args.catalog_revision,
        model_id=args.model_id,
        build_profile_id=args.build_profile_id,
        compile_invocation_id=args.compile_invocation_id,
    )
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Staged {result['count']} registry MPK(s)")
    return 0


def _main(argv: Sequence[str] | None, *, author_safe: bool) -> int:
    parser = _parser(author_safe=author_safe)
    args = parser.parse_args(argv)
    try:
        if args.command == "promotion":
            from .promotions import run

            return run(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "validate-models":
            return _validate_models(args)
        if args.command == "validate-compile-outputs":
            return _validate_compile_outputs(args)
        if args.command == "record-compile-outputs":
            print(record_compile_outputs(
                args.output_root,
                model_id=args.model_id,
                build_profile_id=args.build_profile_id,
                compile_invocation_id=args.compile_invocation_id,
                component_ids=_id_list(args.components, "--components"),
            ))
            return 0
        if args.command == "conversion-entrypoint":
            from .conversion import conversion_outputs

            print(conversion_outputs(args.model_dir)[0])
            return 0
        if args.command == "publish-conversion":
            from .conversion import publish_conversion
            from .registry import S3ArtifactStore, SigV4RegistryClient

            result = publish_conversion(
                args.repository.resolve(), args.model_id, args.output_dir.resolve(),
                client=SigV4RegistryClient(args.api_url, args.aws_region),
                store=S3ArtifactStore(args.bucket, kms_key_id=args.kms_key_id), bucket=args.bucket,
                environment=args.environment, branch=os.environ["GITHUB_REF_NAME"],
                git_sha=os.environ["GITHUB_SHA"], workflow=os.environ["GITHUB_RUN_ID"], validation=args.validation,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "simulate-mla-stages":
            print(write_mla_stage_simulation(args.output_dir))
            return 0
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "fetch-calibration":
            return _fetch_calibration(args)
        if args.command == "fetch-fidelity-dataset":
            return _fetch_fidelity_dataset(args)
        if args.command == "matrix":
            return _matrix(args)
        if args.command == "component-pipelines":
            return _component_pipelines(args)
        if args.command == "reference-metrics":
            return _reference_metrics(args)
        if args.command == "plan-stages":
            return _plan_stages(args)
        if args.command == "authorize-runner-matrix":
            return _authorize_runner_matrix(args)
        if args.command == "resolve-base":
            return _resolve_base(args)
        if args.command == "resolve-compiler":
            return _resolve_compiler(args)
        if args.command == "resolve-benchmark-runtime":
            return _resolve_benchmark_runtime(args)
        if args.command == "resolve-devkit-accuracy":
            return _resolve_devkit_accuracy(args)
        if args.command == "model-component-mpks":
            from .report_publication import model_component_mpks

            print("\n".join(model_component_mpks(args.compiled_dir, args.repository, args.model, args.build_profile)))
            return 0
        if args.command == "check-report":
            return _check_report(args)
        if args.command in {"verify-mpk", "verify-artifact"}:
            return _verify_mpk(args)
        if args.command == "assemble-package":
            return _assemble_package(args)
        if args.command == "publish":
            return _publish(args)
        if args.command == "publish-candidate":
            return _publish_candidate(args)
        if args.command == "publish-branch-deployment":
            from .candidates import publish_branch_deployment

            result = publish_branch_deployment(
                args.candidate, api_url=args.api_url, read_api_url=args.read_api_url, aws_region=args.aws_region,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "publish-package":
            return _publish_package(args)
        if args.command == "publish-report":
            return _publish_report(args)
        if args.command == "publish-catalog":
            return _publish_catalog(args)
        if args.command == "stage-registry-build":
            return _stage_registry_build(args)
        parser.error(f"unsupported command: {args.command}")
    except ModelRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv, author_safe=False)


def author_main(argv: Sequence[str] | None = None) -> int:
    return _main(argv, author_safe=True)


if __name__ == "__main__":
    raise SystemExit(main())
