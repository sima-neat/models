from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

from .errors import ValidationError
from .fetch import Fetcher, sha256_file
from .model import load_and_validate_model
from .source import source_identity


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    archive.addfile(_tar_info(name, len(payload)), io.BytesIO(payload))


def _add_file(archive: tarfile.TarFile, name: str, path: Path) -> None:
    with path.open("rb") as stream:
        archive.addfile(_tar_info(name, path.stat().st_size), stream)


def assemble_package(
    *,
    repository: str | Path,
    compiled_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    build_profile_id: str,
    package_id: str,
    resource_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    compiled = Path(compiled_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    model_dir = root / "models" / model_id
    model = load_and_validate_model(model_dir, root / "schemas" / "model.schema.json")
    profiles = {item["id"]: item for item in model["build_profiles"]}
    packages = {item["id"]: item for item in model["packages"]}
    resources = {item["id"]: item for item in model["resources"]}
    components = {item["id"]: item for item in model["components"]}
    compositions = {item["id"]: item for item in model["compositions"]}
    if build_profile_id not in profiles:
        raise ValidationError(f"unknown build profile: {model_id}/{build_profile_id}")
    if package_id not in packages:
        raise ValidationError(f"unknown package: {model_id}/{package_id}")
    package = packages[package_id]
    if build_profile_id not in package["build_profiles"]:
        raise ValidationError(f"package {model_id}/{package_id} does not support build profile {build_profile_id}")

    component_entries: list[dict[str, Any]] = []
    component_files: list[tuple[str, Path]] = []
    provenance_identity: dict[str, str] | None = None
    for component_id in sorted(package["components"]):
        result_dir = compiled / model_id / build_profile_id / component_id
        result = _load_json(result_dir / "compile-result.json", "compile result")
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            raise ValidationError(f"{result_dir}: compile result provenance must be an object")
        expected = {
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "component_id": component_id,
            "target": profiles[build_profile_id]["target"],
            "source_sha256": source_identity(components[component_id]["source"]),
        }
        for field, value in expected.items():
            if provenance.get(field) != value:
                raise ValidationError(f"{result_dir}: provenance {field} must be {value!r}")
        current_identity = {
            field: str(provenance.get(field) or "")
            for field in ("git_sha", "environment", "compiler_version", "compiler_label", "target")
        }
        if provenance_identity is None:
            provenance_identity = current_identity
        elif current_identity != provenance_identity:
            raise ValidationError(f"{result_dir}: component provenance is incompatible with its package siblings")

        archive_name = result.get("archive")
        if not isinstance(archive_name, str) or not archive_name:
            raise ValidationError(f"{result_dir}: compile result archive must be a non-empty string")
        component_archive = result_dir / archive_name
        if component_archive.parent != result_dir or not component_archive.is_file():
            raise ValidationError(f"{result_dir}: component archive is missing")
        digest, size = sha256_file(component_archive)
        if digest != result.get("sha256") or size != result.get("size_bytes"):
            raise ValidationError(f"{result_dir}: component archive does not match compile-result.json")
        package_path = f"components/{component_id}/{archive_name}"
        component_files.append((package_path, component_archive))
        publication_path = result_dir / "registry-publication.json"
        compile_run_id = None
        if publication_path.is_file():
            publication = _load_json(publication_path, "registry publication")
            runs = publication.get("runs", [])
            if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict):
                compile_run_id = runs[0].get("run_id")
        component_entries.append(
            {
                "id": component_id,
                "role": components[component_id]["role"],
                "path": package_path,
                "sha256": digest,
                "size_bytes": size,
                "source_sha256": expected["source_sha256"],
                "compile_run_id": compile_run_id,
            }
        )

    fetcher = Fetcher(resource_cache_dir)
    resource_entries: list[dict[str, Any]] = []
    resource_files: list[tuple[str, Path]] = []
    occupied_paths = {name for name, _ in component_files} | {"manifest.json"}
    for resource_id in sorted(package["resources"]):
        resource = resources[resource_id]
        source = resource["source"]
        path, _, digest, size = fetcher.fetch_source(source)
        install_path = PurePosixPath(resource["install_path"]).as_posix()
        paths: list[str] = []
        if source["type"] == "file":
            paths.append(install_path)
            files = [(install_path, path)]
        else:
            source_root = path
            for _ in PurePosixPath(source["entrypoint"]).parts:
                source_root = source_root.parent
            files = []
            for member in source["files"]:
                package_path = (PurePosixPath(install_path) / member["path"]).as_posix()
                paths.append(package_path)
                files.append((package_path, source_root / member["path"]))
        for package_path, resource_path in files:
            if package_path in occupied_paths:
                raise ValidationError(f"package path is declared more than once: {package_path}")
            occupied_paths.add(package_path)
            resource_files.append((package_path, resource_path))
        resource_entries.append(
            {
                "id": resource_id,
                "paths": paths,
                "source_sha256": digest,
                "size_bytes": size,
            }
        )

    composition = compositions[package["composition"]]
    composition_files: list[tuple[str, Path]] = []
    if composition["type"] == "single":
        packaged_composition = composition
    else:
        composition_path = f"composition/{Path(composition['file']).name}"
        composition_files.append((composition_path, model_dir / composition["file"]))
        packaged_composition = {
            "id": composition["id"],
            "type": composition["type"],
            "schema_version": composition["schema_version"],
            "path": composition_path,
        }

    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "display_name": model["display_name"],
        "package_id": package_id,
        "build_profile_id": build_profile_id,
        "composition": packaged_composition,
        "target": profiles[build_profile_id]["target"],
        "provenance": provenance_identity or {},
        "components": component_entries,
        "resources": resource_entries,
        "license": model["license"],
    }
    package_schema = _load_json(root / "schemas" / "package-manifest-v1.schema.json", "package schema")
    errors = sorted(
        Draft202012Validator(package_schema).iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "manifest"
        raise ValidationError(f"package manifest {location}: {first.message}")
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    package_dir = output_root / model_id / build_profile_id / "packages" / package_id
    package_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{model_id}_{build_profile_id}_{package_id}_bundle.tar"
    bundle_path = package_dir / bundle_name
    temporary = package_dir / f".{bundle_name}.partial"
    prefix = f"{model_id}/"
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
            _add_bytes(archive, f"{prefix}manifest.json", manifest_bytes)
            for name, path in sorted((*component_files, *resource_files, *composition_files)):
                _add_file(archive, f"{prefix}{name}", path)
        temporary.replace(bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest, size = sha256_file(bundle_path)
    result = {
        "schema_version": 1,
        "status": "success",
        "model_id": model_id,
        "package_id": package_id,
        "build_profile_id": build_profile_id,
        "archive": bundle_name,
        "sha256": digest,
        "size_bytes": size,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest,
    }
    (package_dir / "package-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def assemble_available_packages(
    *,
    repository: str | Path,
    compiled_dir: str | Path,
    output_dir: str | Path,
    resource_cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(repository).expanduser().resolve()
    results: list[dict[str, Any]] = []
    for model_file in sorted((root / "models").glob("*/model.yaml")):
        model = load_and_validate_model(model_file.parent, root / "schemas" / "model.schema.json")
        for package in sorted(model["packages"], key=lambda item: item["id"]):
            for profile_id in sorted(package["build_profiles"]):
                required = [
                    Path(compiled_dir).expanduser().resolve()
                    / model["id"]
                    / profile_id
                    / component_id
                    / "compile-result.json"
                    for component_id in package["components"]
                ]
                if required and all(path.is_file() for path in required):
                    results.append(
                        assemble_package(
                            repository=root,
                            compiled_dir=compiled_dir,
                            output_dir=output_dir,
                            model_id=model["id"],
                            build_profile_id=profile_id,
                            package_id=package["id"],
                            resource_cache_dir=resource_cache_dir,
                        )
                    )
    return results
