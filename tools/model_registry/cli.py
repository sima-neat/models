from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .accuracy import evaluate_accuracy, resolve_dataset
from .accuracy_config import find_accuracy_config, find_devkit_accuracy_config, validation_components
from .artifact import validate_compile_outputs, verify_mpk
from .artifact_metadata import validate_artifact_metadata
from .catalog import (
    build_matrix,
    discover_catalog,
    resolve_base_branch,
    select_model_ids,
    semantic_change_plan,
)
from .dependencies import resolve_benchmark_runtime, resolve_model_compiler
from .documentation import load_model_documentation
from .errors import ModelRegistryError
from .fetch import Fetcher
from .fidelity_config import find_recipe_fidelity_config
from .model import component_for, load_and_validate_model, profile_component_for
from .runner_approval import authorize_runner_matrix

AUTHOR_COMMANDS = frozenset(
    {
        "fetch",
        "fetch-calibration",
        "fetch-accuracy-dataset",
        "evaluate",
        "matrix",
        "authorize-runner-matrix",
        "assemble-package",
        "resolve-compiler",
        "validate",
        "validate-catalog",
        "validate-compile-outputs",
        "verify-mpk",
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

    validate = subcommands.add_parser("validate", help="validate a model directory against its schema version")
    validate.add_argument("model_dir", type=Path)
    validate.add_argument("--schema", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")

    validate_catalog = subcommands.add_parser(
        "validate-catalog", help="validate every model manifest and its taxonomy categorization"
    )
    validate_catalog.add_argument("--repository", type=Path, default=Path.cwd())
    validate_catalog.add_argument("--json", action="store_true", dest="json_output")

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

    accuracy_dataset = subcommands.add_parser(
        "fetch-accuracy-dataset", help="download and verify a model's immutable correctness dataset"
    )
    accuracy_dataset.add_argument("model_dir", type=Path)
    accuracy_dataset.add_argument("--schema", type=Path)
    accuracy_dataset.add_argument("--cache-dir", type=Path)
    accuracy_dataset.add_argument(
        "--build-profile", required=True, help="resolve accuracy only when its config applies"
    )
    accuracy_dataset.add_argument(
        "--component", required=True, help="component evaluated by the selected build profile"
    )
    accuracy_dataset.add_argument("--json", action="store_true", dest="json_output")

    evaluate = subcommands.add_parser("evaluate", help="run a declarative model accuracy evaluation")
    evaluate.add_argument("model_dir", type=Path)
    evaluate.add_argument("--build-profile", required=True)
    evaluate.add_argument("--component", required=True)
    evaluate.add_argument("--candidate-model", type=Path, required=True, help="saved candidate .sima graph")
    evaluate.add_argument("--reference-model", type=Path, help="verified ONNX source; fetched when omitted")
    evaluate.add_argument("--dataset", type=Path, help="verified dataset archive; resolved when omitted")
    evaluate.add_argument("--dataset-cache-dir", type=Path)
    evaluate.add_argument("--model-cache-dir", type=Path)
    evaluate.add_argument("--test-id")
    evaluate.add_argument("--profile", default="regular")
    evaluate.add_argument("--samples", type=int)
    evaluate.add_argument("--workers", type=int)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--json", action="store_true", dest="json_output")

    matrix = subcommands.add_parser("matrix", help="emit a deterministic changed-model compilation matrix")
    matrix.add_argument("--repository", type=Path, default=Path.cwd())
    matrix.add_argument("--all", action="store_true", dest="all_models")
    matrix.add_argument("--models", help="comma-separated model IDs")
    matrix.add_argument("--profiles", help="comma-separated build profile IDs within the selected models")
    matrix.add_argument("--components", help="comma-separated component IDs within the selected models")
    matrix.add_argument("--changed-path", action="append", default=[])
    matrix.add_argument("--base")
    matrix.add_argument("--head")
    matrix.add_argument("--compiler-version")
    matrix.add_argument("--force-rebuild-all", action="store_true")

    authorize = subcommands.add_parser(
        "authorize-runner-matrix",
        help="authorize compile runner requests against a Terraform-published Vulcan specification",
    )
    authorize.add_argument("--matrix", type=Path, required=True)
    authorize.add_argument("--vulcan-spec", type=Path, required=True)
    authorize.add_argument("--repository-name", required=True)
    authorize.add_argument("--environment", required=True)

    base = subcommands.add_parser("resolve-base", help="resolve the closest base branch and merge base")
    base.add_argument("--repository", type=Path, default=Path.cwd())
    base.add_argument("--current-branch", required=True)
    base.add_argument("--head", required=True)
    base.add_argument("--preferred-branch")
    base.add_argument("--default-branch", default="develop")

    compiler = subcommands.add_parser("resolve-compiler", help="resolve the repository-wide Model Compiler dependency")
    compiler.add_argument("--repository", type=Path, default=Path.cwd())

    benchmark_runtime = subcommands.add_parser(
        "resolve-benchmark-runtime", help="resolve pinned DevKit and benchmark runtime dependencies"
    )
    benchmark_runtime.add_argument("--repository", type=Path, default=Path.cwd())

    core_report = subcommands.add_parser(
        "import-core-benchmark", help="attach registry provenance to a Core benchmark report"
    )
    core_report.add_argument("--manifest", type=Path, required=True)
    core_report.add_argument("--report", type=Path, required=True)
    core_report.add_argument("--output", type=Path, required=True)
    core_report.add_argument("--schema", type=Path, required=True)

    devkit_accuracy = subcommands.add_parser(
        "resolve-devkit-accuracy", help="resolve a build profile's DevKit accuracy entry and its cached dataset"
    )
    devkit_accuracy.add_argument("model_dir", type=Path)
    devkit_accuracy.add_argument("--schema", type=Path)
    devkit_accuracy.add_argument("--cache-dir", type=Path)
    devkit_accuracy.add_argument("--build-profile", required=True)

    accuracy_import = subcommands.add_parser(
        "import-accuracy", help="attach model, dataset and MPK identity to scripts/accuracy.py output"
    )
    accuracy_import.add_argument("--repository", type=Path, default=Path.cwd())
    accuracy_import.add_argument("--model-id", required=True)
    accuracy_import.add_argument("--build-profile-id", required=True)
    accuracy_import.add_argument("--script-output", type=Path, required=True)
    accuracy_import.add_argument(
        "--result-root", type=Path, required=True, help="benchmark result root with each component's manifest"
    )
    accuracy_import.add_argument("--output", type=Path, required=True)

    benchmark_manifest = subcommands.add_parser(
        "make-benchmark-manifest", help="generate a direct-DevKit benchmark manifest for one compiled model"
    )
    benchmark_manifest.add_argument("--repository", type=Path, default=Path.cwd())
    benchmark_manifest.add_argument("--compiled-dir", type=Path, required=True)
    benchmark_manifest.add_argument("--output", type=Path, required=True)
    benchmark_manifest.add_argument("--branch", required=True)
    benchmark_manifest.add_argument("--git-sha", required=True)
    benchmark_manifest.add_argument("--model", required=True)
    benchmark_manifest.add_argument("--build-profile", required=True)
    benchmark_manifest.add_argument("--component", required=True)
    benchmark_manifest.add_argument("--json", action="store_true", dest="json_output")

    mpk = subcommands.add_parser("verify-mpk", help="validate and checksum one compiler MPK output")
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
    mpk.add_argument("--build-fingerprint")
    mpk.add_argument("--requested-runner-class")
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
    package.add_argument("--resource-cache-dir", type=Path)
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

    publish_benchmark = subcommands.add_parser(
        "publish-benchmark", help="publish a benchmark run linked to its Vulcan model compilation run"
    )
    publish_benchmark.add_argument("--result", type=Path, required=True)
    publish_benchmark.add_argument("--log", type=Path)
    publish_benchmark.add_argument("--schema", type=Path)
    publish_benchmark.add_argument("--api-url", required=True)
    publish_benchmark.add_argument("--aws-region", default="us-west-2")
    publish_benchmark.add_argument("--bucket", required=True)
    publish_benchmark.add_argument("--kms-key-id")
    publish_benchmark.add_argument("--environment", choices=("staging", "production"), required=True)
    publish_benchmark.add_argument("--branch", required=True)
    publish_benchmark.add_argument("--git-sha", required=True)
    publish_benchmark.add_argument("--github-run-url")
    publish_benchmark.add_argument("--compile-run-id")
    publish_benchmark.add_argument("--accuracy-result", type=Path, help="DevKit accuracy result this component carries")
    publish_benchmark.add_argument("--accuracy-schema", type=Path)
    publish_benchmark.add_argument("--json", action="store_true", dest="json_output")

    benchmark_set = subcommands.add_parser(
        "validate-benchmark-set", help="validate a complete benchmark result set before publication"
    )
    benchmark_set.add_argument("--repository", type=Path, default=Path.cwd())
    benchmark_set.add_argument("--result-root", type=Path, required=True)
    benchmark_set.add_argument("--model-id", required=True)
    benchmark_set.add_argument("--build-profile-id", required=True)
    benchmark_set.add_argument("--components", required=True)
    benchmark_set.add_argument("--git-sha", required=True)
    benchmark_set.add_argument("--accuracy-result", type=Path, help="DevKit accuracy result, when one was measured")
    benchmark_set.add_argument("--json", action="store_true", dest="json_output")

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
    publish_catalog_parser.add_argument("--api-url", required=True)
    publish_catalog_parser.add_argument("--read-api-url", required=True)
    publish_catalog_parser.add_argument("--aws-region", default="us-west-2")
    publish_catalog_parser.add_argument("--bucket", required=True)
    publish_catalog_parser.add_argument("--kms-key-id")
    publish_catalog_parser.add_argument("--environment", required=True)
    publish_catalog_parser.add_argument("--branch", required=True)
    publish_catalog_parser.add_argument("--base-branch", help="reuse this catalog when the target branch has none")
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
    validate_artifact_metadata(model)
    result = {"model_id": model["id"], "model_dir": str(args.model_dir.resolve()), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid model: {model['id']}")
    return 0


def _validate_catalog(args: argparse.Namespace) -> int:
    repository = args.repository.expanduser().resolve()
    catalog = discover_catalog(repository)
    for model_id, model in sorted(catalog.items()):
        load_model_documentation(
            repository / "models" / model_id,
            model["display_name"],
            repository=repository,
        )
    result = {"model_count": len(catalog), "model_ids": sorted(catalog), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid model catalog: {len(catalog)} models")
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
    if calibration is None or calibration["status"] != "ready":
        result = {
            "kind": "none",
            "model_id": model["id"],
            "component_id": args.component,
            "build_profile_id": args.build_profile,
            "path": "",
            "sha256": "",
            "size_bytes": 0,
        }
    elif "dataset" in calibration:
        dataset = calibration["dataset"]
        path = resolve_dataset(dataset, None, args.cache_dir)
        result = {
            "kind": "dataset",
            "model_id": model["id"],
            "component_id": args.component,
            "build_profile_id": args.build_profile,
            "path": str(path),
            "sha256": dataset["sha256"],
            "size_bytes": dataset["size_bytes"],
        }
    else:
        path, cache_hit, source_sha256, size_bytes = Fetcher(args.cache_dir).fetch_source(calibration["source"])
        result = {
            "kind": "source",
            "cache_hit": cache_hit,
            "model_id": model["id"],
            "component_id": args.component,
            "build_profile_id": args.build_profile,
            "path": str(path),
            "sha256": source_sha256,
            "size_bytes": size_bytes,
        }
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    elif result["path"]:
        print(f"Verified calibration: {result['path']}")
    else:
        print(f"No calibration required: {result['model_id']}")
    return 0


def _fetch_accuracy_dataset(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    profile_component_for(model, args.build_profile, args.component)
    selected = find_accuracy_config(
        args.model_dir,
        model,
        build_profile_id=args.build_profile,
        component_id=args.component,
    )
    if selected is None:
        selected = find_recipe_fidelity_config(
            args.model_dir,
            model,
            build_profile_id=args.build_profile,
            component_id=args.component,
        )
    if selected is None:
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
        dataset = selected["dataset"]
        path = resolve_dataset(dataset, None, args.cache_dir)
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
        print(f"Verified accuracy dataset: {result['path']}")
    else:
        print(f"No applicable executable accuracy dataset configured: {result['model_id']}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate_accuracy(
        args.model_dir,
        args.build_profile,
        args.component,
        args.candidate_model,
        reference_model=args.reference_model,
        dataset_path=args.dataset,
        dataset_cache_dir=args.dataset_cache_dir,
        model_cache_dir=args.model_cache_dir,
        test_id=args.test_id,
        profile_name=args.profile,
        sample_count=args.samples,
        workers=args.workers,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json_output:
        print(json.dumps(report, sort_keys=True))
    else:
        samples = report["dataset"]
        candidate = report["metrics"]["candidate"]
        print(
            f"Accuracy {report['model']['id']}/{report['model']['build_profile_id']}/"
            f"{report['model']['component_id']}: "
            f"samples={samples['evaluated_samples']}/{samples['available_samples']} "
            f"top1={candidate['top1_accuracy']:.4f} output={output}"
        )
    return 0


def _matrix(args: argparse.Namespace) -> int:
    catalog = discover_catalog(args.repository)
    selection_modes = (
        int(args.all_models) + int(args.models is not None) + int(bool(args.changed_path or args.base or args.head))
    )
    if selection_modes != 1:
        raise ModelRegistryError("select exactly one of --all, --models, or changed paths/--base and --head")
    if args.all_models:
        model_ids = sorted(catalog)
    elif args.models is not None:
        model_ids = [item.strip() for item in args.models.split(",") if item.strip()]
        if not model_ids:
            raise ModelRegistryError("--models must contain at least one model ID")
    elif args.base or args.head:
        if not args.base or not args.head:
            raise ModelRegistryError("--base and --head must be supplied together")
        if args.profiles is not None or args.components is not None:
            raise ModelRegistryError("--profiles and --components are not supported for semantic revision comparison")
        if args.compiler_version is None:
            raise ModelRegistryError("--compiler-version is required for semantic revision comparison")
        result = semantic_change_plan(
            args.repository,
            args.base,
            args.head,
            compiler_version=args.compiler_version,
            force_rebuild_all=args.force_rebuild_all,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    else:
        paths = list(args.changed_path)
        model_ids = select_model_ids(catalog, paths)
    profile_ids = None
    if args.profiles is not None:
        profile_ids = [item.strip() for item in args.profiles.split(",") if item.strip()]
        if not profile_ids:
            raise ModelRegistryError("--profiles must contain at least one build profile ID")
    component_ids = None
    if args.components is not None:
        component_ids = [item.strip() for item in args.components.split(",") if item.strip()]
        if not component_ids:
            raise ModelRegistryError("--components must contain at least one component ID")
    print(
        json.dumps(
            build_matrix(
                catalog,
                model_ids,
                profile_ids,
                component_ids,
                repository=args.repository if args.compiler_version is not None else None,
                compiler_version=args.compiler_version,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _authorize_runner_matrix(args: argparse.Namespace) -> int:
    try:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        vulcan_spec = json.loads(args.vulcan_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"could not load runner authorization input: {exc}") from exc
    if not isinstance(matrix, dict) or not isinstance(vulcan_spec, dict):
        raise ModelRegistryError("runner authorization inputs must be JSON objects")
    result = authorize_runner_matrix(
        matrix,
        vulcan_spec,
        repository=args.repository_name,
        environment=args.environment,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_base(args: argparse.Namespace) -> int:
    result = resolve_base_branch(
        args.repository,
        current_branch=args.current_branch,
        head=args.head,
        preferred_branch=args.preferred_branch,
        default_branch=args.default_branch,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_compiler(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_model_compiler(args.repository), sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_benchmark_runtime(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_benchmark_runtime(args.repository), sort_keys=True, separators=(",", ":")))
    return 0


def _make_benchmark_manifest(args: argparse.Namespace) -> int:
    from .benchmark_manifest import make_benchmark_manifest

    result = make_benchmark_manifest(
        repository=args.repository,
        compiled_dir=args.compiled_dir,
        output_path=args.output,
        branch=args.branch,
        git_sha=args.git_sha,
        model_id=args.model,
        build_profile_id=args.build_profile,
        component_id=args.component,
    )
    print(json.dumps(result, sort_keys=True) if args.json_output else result["manifest"])
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
        "build_fingerprint",
        "requested_runner_class",
        "runner_class",
        "runner_approval_id",
        "runner_approval_spec_revision",
    )
    provenance = {field: getattr(args, field) for field in fields if getattr(args, field) is not None}
    result = verify_mpk(args.output_dir, write_manifest=args.write_manifest, provenance=provenance)
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid MPK: {result['archive']}")
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
            resource_cache_dir=args.resource_cache_dir,
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
                resource_cache_dir=args.resource_cache_dir,
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


def _publish_benchmark(args: argparse.Namespace) -> int:
    from .benchmark_publication import publish_benchmark_result

    result = publish_benchmark_result(
        result_path=args.result,
        log_path=args.log,
        schema_path=args.schema,
        api_url=args.api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        git_sha=args.git_sha,
        github_run_url=args.github_run_url,
        compile_run_id=args.compile_run_id,
        accuracy_result_path=args.accuracy_result,
        accuracy_schema_path=args.accuracy_schema,
    )
    message = (
        json.dumps(result, sort_keys=True)
        if args.json_output
        else f"Published benchmark {result['result_fingerprint']} for "
        f"{result['model_id']}/{result['build_profile_id']}/{result['component_id']}"
    )
    print(message)
    return 0


def _validate_benchmark_set(args: argparse.Namespace) -> int:
    from .benchmark_publication import validate_benchmark_set

    component_ids = [item.strip() for item in args.components.split(",") if item.strip()]
    result = validate_benchmark_set(
        repository=args.repository,
        result_root=args.result_root,
        model_id=args.model_id,
        build_profile_id=args.build_profile_id,
        component_ids=component_ids,
        git_sha=args.git_sha,
        accuracy_result=args.accuracy_result,
    )
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid benchmark set: {result['count']}")
    return 0


def _resolve_devkit_accuracy(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    entry = find_devkit_accuracy_config(model, args.build_profile)
    if entry is None:
        result = {"kind": "none", "model_id": model["id"], "build_profile_id": args.build_profile}
    else:
        result = {
            "kind": "dataset",
            "model_id": model["id"],
            "build_profile_id": args.build_profile,
            "validation_id": entry["id"],
            "result_component": entry["result_component"],
            "components": validation_components(model, entry),
            "path": str(resolve_dataset(entry["dataset"], None, args.cache_dir)),
            "sha256": entry["dataset"]["sha256"],
            "size_bytes": entry["dataset"]["size_bytes"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


def _import_accuracy(args: argparse.Namespace) -> int:
    from .accuracy_result import ACCURACY_RESULT_SCHEMA, build_accuracy_result, load_accuracy_result

    result = build_accuracy_result(
        repository=args.repository,
        model_id=args.model_id,
        build_profile_id=args.build_profile_id,
        script_output=args.script_output,
        result_root=args.result_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_accuracy_result(args.output, Path(args.repository).expanduser().resolve() / ACCURACY_RESULT_SCHEMA)
    print(json.dumps({"output": str(args.output), "metrics": sorted(result["metrics"])}, sort_keys=True))
    return 0


def _publish_catalog(args: argparse.Namespace) -> int:
    from .registry import publish_catalog

    result = publish_catalog(
        repository=args.repository,
        compiled_dir=args.compiled_dir,
        package_dir=args.package_dir,
        selected_matrix=args.selected_matrix,
        api_url=args.api_url,
        read_api_url=args.read_api_url,
        aws_region=args.aws_region,
        bucket=args.bucket,
        kms_key_id=args.kms_key_id,
        environment=args.environment,
        branch=args.branch,
        base_branch=args.base_branch,
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
        if args.command == "validate":
            return _validate(args)
        if args.command == "validate-catalog":
            return _validate_catalog(args)
        if args.command == "validate-compile-outputs":
            return _validate_compile_outputs(args)
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "fetch-calibration":
            return _fetch_calibration(args)
        if args.command == "fetch-accuracy-dataset":
            return _fetch_accuracy_dataset(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "matrix":
            return _matrix(args)
        if args.command == "authorize-runner-matrix":
            return _authorize_runner_matrix(args)
        if args.command == "resolve-base":
            return _resolve_base(args)
        if args.command == "resolve-compiler":
            return _resolve_compiler(args)
        if args.command == "resolve-benchmark-runtime":
            return _resolve_benchmark_runtime(args)
        if args.command == "import-core-benchmark":
            from .core_benchmark import import_core_benchmark

            result = import_core_benchmark(args.manifest, args.report, args.output, schema_path=args.schema)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "resolve-devkit-accuracy":
            return _resolve_devkit_accuracy(args)
        if args.command == "import-accuracy":
            return _import_accuracy(args)
        if args.command == "make-benchmark-manifest":
            return _make_benchmark_manifest(args)
        if args.command == "verify-mpk":
            return _verify_mpk(args)
        if args.command == "assemble-package":
            return _assemble_package(args)
        if args.command == "publish":
            return _publish(args)
        if args.command == "publish-package":
            return _publish_package(args)
        if args.command == "publish-benchmark":
            return _publish_benchmark(args)
        if args.command == "validate-benchmark-set":
            return _validate_benchmark_set(args)
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
