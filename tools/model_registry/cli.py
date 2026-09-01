from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .accuracy import evaluate_accuracy, resolve_dataset
from .artifact import verify_mpk
from .catalog import build_matrix, discover_catalog, git_changed_paths, resolve_base_branch, select_model_ids
from .dependencies import resolve_benchmark_runtime, resolve_model_compiler
from .errors import ModelRegistryError
from .fetch import Fetcher
from .model import load_and_validate_model, variant_requires_calibration
from .published import validate_latest_index
from .source import SourceSpec

AUTHOR_COMMANDS = frozenset(
    {
        "fetch",
        "fetch-calibration",
        "evaluate",
        "matrix",
        "resolve-compiler",
        "validate",
        "validate-catalog",
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
    fetch.add_argument("--json", action="store_true", dest="json_output")

    calibration = subcommands.add_parser(
        "fetch-calibration", help="download and verify a model's immutable calibration snapshot"
    )
    calibration.add_argument("model_dir", type=Path)
    calibration.add_argument("--schema", type=Path)
    calibration.add_argument("--cache-dir", type=Path)
    calibration.add_argument("--variant", help="resolve calibration only when this compile variant requires it")
    calibration.add_argument("--json", action="store_true", dest="json_output")

    evaluate = subcommands.add_parser("evaluate", help="run a declarative model accuracy evaluation")
    evaluate.add_argument("model_dir", type=Path)
    evaluate.add_argument("--variant", required=True)
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
    matrix.add_argument("--variants", help="comma-separated variant IDs within the selected models")
    matrix.add_argument("--changed-path", action="append", default=[])
    matrix.add_argument("--base")
    matrix.add_argument("--head")
    matrix.add_argument("--compiler-version")
    matrix.add_argument("--previous-index", type=Path)

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

    benchmark_manifest = subcommands.add_parser(
        "make-benchmark-manifest", help="generate a direct-DevKit benchmark manifest for one compiled model"
    )
    benchmark_manifest.add_argument("--repository", type=Path, default=Path.cwd())
    benchmark_manifest.add_argument("--compiled-dir", type=Path, required=True)
    benchmark_manifest.add_argument("--output", type=Path, required=True)
    benchmark_manifest.add_argument("--branch", required=True)
    benchmark_manifest.add_argument("--git-sha", required=True)
    benchmark_manifest.add_argument("--model", required=True)
    benchmark_manifest.add_argument("--variant", required=True)
    benchmark_manifest.add_argument("--json", action="store_true", dest="json_output")

    mpk = subcommands.add_parser("verify-mpk", help="validate and checksum one compiler MPK output")
    mpk.add_argument("output_dir", type=Path)
    mpk.add_argument("--write-manifest", action="store_true")
    mpk.add_argument("--json", action="store_true", dest="json_output")
    mpk.add_argument("--model-id")
    mpk.add_argument("--variant-id")
    mpk.add_argument("--target")
    mpk.add_argument("--compiler-version")
    mpk.add_argument("--compiler-label")
    mpk.add_argument("--source-sha256")
    mpk.add_argument("--calibration-sha256")
    mpk.add_argument("--git-sha")
    mpk.add_argument("--environment")
    mpk.add_argument("--build-fingerprint")

    package = subcommands.add_parser("package", help="assemble compiled MPKs into sima-cli package metadata")
    package.add_argument("--repository", type=Path, default=Path.cwd())
    package.add_argument("--compiled-dir", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--git-sha", required=True)
    package.add_argument("--environment", choices=("staging", "production"), required=True)
    package.add_argument("--branch", required=True)
    package.add_argument("--previous-index", type=Path)
    package.add_argument(
        "--allow-partial-catalog",
        action="store_true",
        help="omit unbuilt models with no previous index entry (intended for feature branches)",
    )
    package.add_argument("--json", action="store_true", dest="json_output")

    previous = subcommands.add_parser("fetch-previous", help="fetch the previous model-aware latest index")
    previous.add_argument("--environment", choices=("staging", "production"), required=True)
    previous.add_argument("--branch", required=True)
    previous.add_argument("--fallback-branch")
    previous.add_argument("--output-file", type=Path, required=True)
    previous.add_argument("--json", action="store_true", dest="json_output")

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
    publish_benchmark.add_argument("--json", action="store_true", dest="json_output")

    publish_taxonomy_parser = subcommands.add_parser(
        "publish-taxonomy", help="publish the canonical model taxonomy to the Vulcan registry"
    )
    publish_taxonomy_parser.add_argument("--repository", type=Path, default=Path.cwd())
    publish_taxonomy_parser.add_argument("--api-url", required=True)
    publish_taxonomy_parser.add_argument("--aws-region", default="us-west-2")
    publish_taxonomy_parser.add_argument("--json", action="store_true", dest="json_output")

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


def _validate_catalog(args: argparse.Namespace) -> int:
    catalog = discover_catalog(args.repository)
    result = {"model_count": len(catalog), "model_ids": sorted(catalog), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid model catalog: {len(catalog)} models")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    source = SourceSpec.from_mapping(model["source"])
    path, cache_hit = Fetcher(args.cache_dir).fetch(source)
    result = {
        "cache_hit": cache_hit,
        "model_id": model["id"],
        "path": str(path),
        "sha256": source.sha256,
        "size_bytes": source.size_bytes,
    }
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    else:
        state = "verified cache hit" if cache_hit else "downloaded and verified"
        print(f"{state}: {path}")
    return 0


def _fetch_calibration(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    calibration = model["calibration"]
    requires_calibration = True
    if args.variant:
        variants = [item for item in model["compile_variants"] if item["id"] == args.variant]
        if len(variants) != 1:
            raise ModelRegistryError(f"unknown compile variant: {args.variant}")
        requires_calibration = variant_requires_calibration(variants[0])
    if calibration["status"] != "ready" or not requires_calibration:
        result = {"kind": "none", "model_id": model["id"], "path": "", "sha256": "", "size_bytes": 0}
    elif "dataset" in calibration:
        dataset = calibration["dataset"]
        path = resolve_dataset(dataset, None, args.cache_dir)
        result = {
            "kind": "dataset",
            "model_id": model["id"],
            "path": str(path),
            "sha256": dataset["sha256"],
            "size_bytes": dataset["size_bytes"],
        }
    else:
        source = SourceSpec.from_mapping(calibration["source"])
        path, cache_hit = Fetcher(args.cache_dir).fetch(source)
        result = {
            "kind": "source",
            "cache_hit": cache_hit,
            "model_id": model["id"],
            "path": str(path),
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
        }
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    elif result["path"]:
        print(f"Verified calibration: {result['path']}")
    else:
        print(f"No calibration required: {result['model_id']}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate_accuracy(
        args.model_dir,
        args.variant,
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
            f"Accuracy {report['model']['id']}/{report['model']['variant_id']}: "
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
    else:
        paths = list(args.changed_path)
        if args.base or args.head:
            if not args.base or not args.head:
                raise ModelRegistryError("--base and --head must be supplied together")
            paths.extend(git_changed_paths(args.repository, args.base, args.head))
        model_ids = select_model_ids(catalog, paths)
    variant_ids = None
    if args.variants is not None:
        variant_ids = [item.strip() for item in args.variants.split(",") if item.strip()]
        if not variant_ids:
            raise ModelRegistryError("--variants must contain at least one variant ID")
    previous_models = None
    if args.previous_index is not None:
        if args.compiler_version is None:
            raise ModelRegistryError("--compiler-version is required with --previous-index")
        try:
            value = json.loads(args.previous_index.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRegistryError(f"could not read previous index {args.previous_index}: {exc}") from exc
        previous_models = validate_latest_index(value)["models"]
    print(
        json.dumps(
            build_matrix(
                catalog,
                model_ids,
                variant_ids,
                repository=args.repository if args.compiler_version is not None else None,
                compiler_version=args.compiler_version,
                previous_models=previous_models,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
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
        variant_id=args.variant,
    )
    print(json.dumps(result, sort_keys=True) if args.json_output else result["manifest"])
    return 0


def _verify_mpk(args: argparse.Namespace) -> int:
    fields = (
        "model_id",
        "variant_id",
        "target",
        "compiler_version",
        "compiler_label",
        "source_sha256",
        "calibration_sha256",
        "git_sha",
        "environment",
        "build_fingerprint",
    )
    provenance = {field: getattr(args, field) for field in fields if getattr(args, field) is not None}
    result = verify_mpk(args.output_dir, write_manifest=args.write_manifest, provenance=provenance)
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid MPK: {result['archive']}")
    return 0


def _package(args: argparse.Namespace) -> int:
    from .package import package_compiled_models

    result = package_compiled_models(
        args.repository,
        args.compiled_dir,
        args.output_dir,
        args.git_sha,
        environment=args.environment,
        branch=args.branch,
        previous_index=args.previous_index,
        allow_partial_catalog=args.allow_partial_catalog,
    )
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Packaged {len(result['models'])} variant(s)"
    print(message)
    return 0


def _fetch_previous(args: argparse.Namespace) -> int:
    from .published import fetch_previous_index

    index = fetch_previous_index(args.environment, args.branch, args.output_file)
    source_branch = args.branch if index is not None else None
    if index is None and args.fallback_branch and args.fallback_branch != args.branch:
        index = fetch_previous_index(args.environment, args.fallback_branch, args.output_file)
        source_branch = args.fallback_branch if index is not None else None
    result = {
        "found": index is not None,
        "catalog_tag": index.get("catalog_tag") if index else None,
        "source_branch": source_branch,
    }
    print(json.dumps(result, sort_keys=True) if args.json_output else (result["catalog_tag"] or "No previous index"))
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
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Published {result['count']} compilation(s)"
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
    )
    message = (
        json.dumps(result, sort_keys=True)
        if args.json_output
        else f"Published benchmark {result['result_fingerprint']} for {result['model_id']}/{result['variant_id']}"
    )
    print(message)
    return 0


def _publish_taxonomy(args: argparse.Namespace) -> int:
    from .registry import publish_taxonomy

    result = publish_taxonomy(
        repository=args.repository,
        api_url=args.api_url,
        aws_region=args.aws_region,
    )
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Published taxonomy v{result['version']}"
    print(message)
    return 0


def _main(argv: Sequence[str] | None, *, author_safe: bool) -> int:
    parser = _parser(author_safe=author_safe)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "validate-catalog":
            return _validate_catalog(args)
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "fetch-calibration":
            return _fetch_calibration(args)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "matrix":
            return _matrix(args)
        if args.command == "resolve-base":
            return _resolve_base(args)
        if args.command == "resolve-compiler":
            return _resolve_compiler(args)
        if args.command == "resolve-benchmark-runtime":
            return _resolve_benchmark_runtime(args)
        if args.command == "make-benchmark-manifest":
            return _make_benchmark_manifest(args)
        if args.command == "verify-mpk":
            return _verify_mpk(args)
        if args.command == "package":
            return _package(args)
        if args.command == "fetch-previous":
            return _fetch_previous(args)
        if args.command == "publish":
            return _publish(args)
        if args.command == "publish-benchmark":
            return _publish_benchmark(args)
        if args.command == "publish-taxonomy":
            return _publish_taxonomy(args)
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
