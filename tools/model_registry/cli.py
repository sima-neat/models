from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .artifact import verify_mpk
from .catalog import build_matrix, discover_catalog, git_changed_paths, select_model_ids
from .dependencies import resolve_model_compiler
from .errors import ModelRegistryError
from .fetch import Fetcher
from .model import load_and_validate_model
from .package import package_compiled_models
from .published import fetch_previous_index
from .source import SourceSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-registry", description="SiMa Neat Model Registry lifecycle tooling")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate a model directory against schema v1")
    validate.add_argument("model_dir", type=Path)
    validate.add_argument("--schema", type=Path)
    validate.add_argument("--json", action="store_true", dest="json_output")

    fetch = subcommands.add_parser("fetch", help="download and verify a model's immutable source")
    fetch.add_argument("model_dir", type=Path)
    fetch.add_argument("--schema", type=Path)
    fetch.add_argument("--cache-dir", type=Path)
    fetch.add_argument("--json", action="store_true", dest="json_output")

    matrix = subcommands.add_parser("matrix", help="emit a deterministic changed-model compilation matrix")
    matrix.add_argument("--repository", type=Path, default=Path.cwd())
    matrix.add_argument("--all", action="store_true", dest="all_models")
    matrix.add_argument("--models", help="comma-separated model IDs")
    matrix.add_argument("--changed-path", action="append", default=[])
    matrix.add_argument("--base")
    matrix.add_argument("--head")

    compiler = subcommands.add_parser(
        "resolve-compiler", help="resolve the repository-wide Model Compiler dependency"
    )
    compiler.add_argument("--repository", type=Path, default=Path.cwd())

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
    mpk.add_argument("--git-sha")
    mpk.add_argument("--environment")

    package = subcommands.add_parser("package", help="assemble compiled MPKs into sima-cli package metadata")
    package.add_argument("--repository", type=Path, default=Path.cwd())
    package.add_argument("--compiled-dir", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    package.add_argument("--git-sha", required=True)
    package.add_argument("--environment", choices=("staging", "production"), required=True)
    package.add_argument("--branch", required=True)
    package.add_argument("--previous-index", type=Path)
    package.add_argument("--json", action="store_true", dest="json_output")

    previous = subcommands.add_parser("fetch-previous", help="fetch the previous model-aware latest index")
    previous.add_argument("--environment", choices=("staging", "production"), required=True)
    previous.add_argument("--branch", required=True)
    previous.add_argument("--fallback-branch")
    previous.add_argument("--output-file", type=Path, required=True)
    previous.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _validate(args: argparse.Namespace) -> int:
    model = load_and_validate_model(args.model_dir, args.schema)
    result = {"model_id": model["id"], "model_dir": str(args.model_dir.resolve()), "valid": True}
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid model: {model['id']}")
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
    print(json.dumps(build_matrix(catalog, model_ids), sort_keys=True, separators=(",", ":")))
    return 0


def _resolve_compiler(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_model_compiler(args.repository), sort_keys=True, separators=(",", ":")))
    return 0


def _verify_mpk(args: argparse.Namespace) -> int:
    fields = (
        "model_id",
        "variant_id",
        "target",
        "compiler_version",
        "compiler_label",
        "source_sha256",
        "git_sha",
        "environment",
    )
    provenance = {field: getattr(args, field) for field in fields if getattr(args, field) is not None}
    result = verify_mpk(args.output_dir, write_manifest=args.write_manifest, provenance=provenance)
    print(json.dumps(result, sort_keys=True) if args.json_output else f"Valid MPK: {result['archive']}")
    return 0


def _package(args: argparse.Namespace) -> int:
    result = package_compiled_models(
        args.repository,
        args.compiled_dir,
        args.output_dir,
        args.git_sha,
        environment=args.environment,
        branch=args.branch,
        previous_index=args.previous_index,
    )
    message = json.dumps(result, sort_keys=True) if args.json_output else f"Packaged {len(result['models'])} variant(s)"
    print(message)
    return 0


def _fetch_previous(args: argparse.Namespace) -> int:
    index = fetch_previous_index(args.environment, args.branch, args.output_file)
    source_branch = args.branch if index is not None else None
    if index is None and args.fallback_branch and args.fallback_branch != args.branch:
        index = fetch_previous_index(args.environment, args.fallback_branch, args.output_file)
        source_branch = args.fallback_branch if index is not None else None
    result = {"found": index is not None, "catalog_tag": index.get("catalog_tag") if index else None,
              "source_branch": source_branch}
    print(json.dumps(result, sort_keys=True) if args.json_output else (result["catalog_tag"] or "No previous index"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "fetch":
            return _fetch(args)
        if args.command == "matrix":
            return _matrix(args)
        if args.command == "resolve-compiler":
            return _resolve_compiler(args)
        if args.command == "verify-mpk":
            return _verify_mpk(args)
        if args.command == "package":
            return _package(args)
        if args.command == "fetch-previous":
            return _fetch_previous(args)
        parser.error(f"unsupported command: {args.command}")
    except ModelRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
