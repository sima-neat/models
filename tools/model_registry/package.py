from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

from .artifact import deployment_files, verify_artifact
from .errors import SourcePolicyError, ValidationError
from .fetch import Fetcher, sha256_file
from .mla_stage_simulation import SIMULATION_FILE
from .model import _expand_source, load_and_validate_model, mpk_filename, source_members
from .runner_approval import runner_provenance
from .source import source_identity

# scripts/compile.py writes the profile's numerical fidelity here; the package run publishes it.
FIDELITY_FILE = "fidelity.json"
# What the compile writes about the whole profile beside its components; each package carries them to publication.
PROFILE_RESULT_FILES = (FIDELITY_FILE, SIMULATION_FILE)

def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object: {path}")
    return value


def _zip_info(name: str) -> zipfile.ZipInfo:
    """Return a platform-independent, non-executable ZIP member descriptor."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def _add_file(archive: zipfile.ZipFile, name: str, path: Path) -> None:
    with path.open("rb") as stream:
        archive.writestr(_zip_info(name), stream.read())


def _link_or_copy(source: str, destination: str) -> str:
    """Reuse local payloads without copying multi-gigabyte weights when possible."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copyfile(source, destination)


def package_resources(model: dict[str, Any]) -> list[dict[str, Any]]:
    """model.yaml's resources, each with its expanded source; every package of the model installs all of them."""
    return [
        {"id": resource_id, "source": _expand_source(resource["source"]), "install_path": resource["install_path"]}
        for resource_id, resource in sorted(model.get("resources", {}).items())
    ]


def _reject_overlapping_paths(paths: list[str]) -> None:
    """A package path may appear once and may not sit below another packaged file."""
    occupied: set[str] = set()
    for path in paths:
        if path in occupied:
            raise ValidationError(f"package path is declared more than once: {path}")
        occupied.add(path)
    for path in sorted(occupied):
        if any(parent.as_posix() in occupied for parent in PurePosixPath(path).parents):
            raise ValidationError(f"package paths overlap: {path}")


def _resource_paths(resource: dict[str, Any]) -> list[str]:
    """Where a resource's files land in the package, from its declaration alone."""
    install_path = PurePosixPath(resource["install_path"])
    if resource["source"]["type"] == "file":
        return [install_path.as_posix()]
    return [(install_path / member["path"]).as_posix() for member in resource["source"]["files"]]


def validate_package_resources(model: dict[str, Any]) -> None:
    """Hold each resource's source to the compile source's rules, and their package paths to the package's,
    so a bad one fails before the compile."""
    resources = package_resources(model)
    for resource in resources:
        source = resource["source"]
        try:
            members = dict(source_members(source))
        except (KeyError, TypeError, ValueError, SourcePolicyError) as exc:
            raise ValidationError(f"{model['id']}: resource {resource['id']}: source policy: {exc}") from exc
        if source["type"] == "file_set" and source["entrypoint"] not in members:
            raise ValidationError(
                f"{model['id']}: resource {resource['id']}: main_file {source['entrypoint']!r} is not one of its files"
            )
    if not resources:
        return
    resource_paths = [path for resource in resources for path in _resource_paths(resource)]
    # Every package of the model installs all resources beside its components' archives.
    archives_by_profile = [
        [mpk_filename(model, profile["id"], item["id"]) for item in profile["components"]]
        for profile in model.get("build_profiles", [])
    ]
    for archives in archives_by_profile or [[]]:
        try:
            _reject_overlapping_paths([*archives, *resource_paths])
        except ValidationError as exc:
            raise ValidationError(f"{model['id']}: resources: {exc}") from exc


def fetch_package_resources(model: dict[str, Any], output_dir: Path, cache_dir: Path | None = None) -> Path | None:
    """Lay out the model's resources as its package installs them; None for a model without resources."""
    resources = package_resources(model)
    if not resources:
        return None
    fetcher = Fetcher(cache_dir)
    for resource in resources:
        for name, path in _resource_files(fetcher, resource)[1]:
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    return output_dir


def _runtime_files(profile_dir: Path) -> list[tuple[str, Path]]:
    """Files the compile recipe declared beside its MPKs in compile-outputs.json, such as runtime tables."""
    manifest_file = profile_dir / "compile-outputs.json"
    if not manifest_file.is_file():
        return []
    files = []
    for item in _load_json(manifest_file, "compile outputs").get("runtime_files", []):
        path = profile_dir / item["path"]
        if not path.is_file() or sha256_file(path)[0] != item["sha256"]:
            raise ValidationError(f"{manifest_file}: runtime file {item['path']} is missing or changed")
        files.append((item["path"], path))
    return files


def _resource_files(fetcher: Fetcher, resource: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    """Fetch one pinned resource and place its files under install_path."""
    source = resource["source"]
    path, _, digest, size = fetcher.fetch_source(source)
    if source["type"] == "file":
        local = [path]
    else:
        source_root = path
        for _ in PurePosixPath(source["entrypoint"]).parts:
            source_root = source_root.parent
        local = [source_root / member["path"] for member in source["files"]]
    files = list(zip(_resource_paths(resource), local, strict=True))
    entry = {"id": resource["id"], "paths": [name for name, _ in files], "source_sha256": digest, "size_bytes": size}
    return entry, files


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
    from .conversion import apply_selection

    apply_selection(model, root=root)
    profiles = {item["id"]: item for item in model["build_profiles"]}
    packages = {item["id"]: item for item in model["packages"]}
    components = {item["id"]: item for item in model["components"]}
    compositions = {item["id"]: item for item in model["compositions"]}
    if build_profile_id not in profiles:
        raise ValidationError(f"unknown build profile: {model_id}/{build_profile_id}")
    if package_id not in packages:
        raise ValidationError(f"unknown package: {model_id}/{package_id}")
    package = packages[package_id]
    if build_profile_id not in package["build_profiles"]:
        raise ValidationError(f"package {model_id}/{package_id} does not support build profile {build_profile_id}")

    native_deployment = None
    native_files: list[dict[str, Any]] = []
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
        current_identity.update(runner_provenance(provenance))
        if provenance.get("llima_version"):
            current_identity["llima_version"] = str(provenance["llima_version"])
        if provenance_identity is None:
            provenance_identity = current_identity
        elif current_identity != provenance_identity:
            raise ValidationError(f"{result_dir}: component provenance is incompatible with its package siblings")

        native = components[component_id]["interface"]["format"] == "huggingface"
        archive_name = result.get("manifest" if native else "archive")
        if not isinstance(archive_name, str) or not archive_name:
            raise ValidationError(f"{result_dir}: compile result archive must be a non-empty string")
        component_archive = result_dir / archive_name
        if component_archive.parent != result_dir or not component_archive.is_file():
            raise ValidationError(f"{result_dir}: component archive is missing")
        digest, size = sha256_file(component_archive)
        if native:
            verified = verify_artifact(result_dir)
            size = verified["size_bytes"]
            native_files = deployment_files(_load_json(component_archive, "deployment manifest"))
        if digest != result.get("sha256") or size != result.get("size_bytes"):
            raise ValidationError(f"{result_dir}: component archive does not match compile-result.json")
        if native:
            if package["composition"] not in compositions or compositions[package["composition"]]["type"] != "single":
                raise ValidationError("A LLiMa package uses a single deployment composition")
            if len(package["components"]) != 1 or model.get("resources"):
                raise ValidationError("A LLiMa package contains one complete deployment and no extra resources")
            if result.get("format") != "llima":
                raise ValidationError("Hugging Face compile result must declare format llima")
            native_deployment = result_dir / "deployment"
        package_path = "." if native else archive_name
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
                **({"format": "llima"} if native else {}),
                "path": package_path,
                "sha256": digest,
                "size_bytes": size,
                "source_sha256": expected["source_sha256"],
                "compile_run_id": compile_run_id,
            }
        )

    composition = compositions[package["composition"]]
    if composition["type"] == "single":
        packaged_composition = composition
    else:
        packaged_composition = {"id": composition["id"], "type": composition["type"]}

    extra_files = _runtime_files(compiled / model_id / build_profile_id)
    resource_entries: list[dict[str, Any]] = []
    resources = package_resources(model)
    if resources:
        fetcher = Fetcher(resource_cache_dir)
        for resource in resources:
            entry, files = _resource_files(fetcher, resource)
            resource_entries.append(entry)
            extra_files.extend(files)
    _reject_overlapping_paths([name for name, _ in (*component_files, *extra_files)])
    packaged_files = sorted((*component_files, *extra_files))
    installed_files = [
        {"path": name, "sha256": sha256_file(path)[0], "size_bytes": path.stat().st_size}
        for name, path in packaged_files
    ]
    if native_deployment is not None:
        if extra_files:
            raise ValidationError("A LLiMa package ships its deployment folder and no runtime files")
        installed_files = native_files
    manifest = {
        "model_id": model_id,
        "display_name": model["display_name"],
        "package_id": package_id,
        "build_profile_id": build_profile_id,
        "composition": packaged_composition,
        "target": profiles[build_profile_id]["target"],
        "provenance": provenance_identity or {},
        "components": component_entries,
        "resources": resource_entries,
        "files": installed_files,
        "license": model["license"],
    }
    package_schema = _load_json(root / "schemas" / "package-manifest.schema.json", "package schema")
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
    for name in PROFILE_RESULT_FILES:
        result_file = compiled / model_id / build_profile_id / name
        if result_file.is_file():
            shutil.copyfile(result_file, package_dir / name)
    if native_deployment is not None:
        deployment = package_dir / "deployment"
        if deployment.resolve().is_relative_to(native_deployment.resolve()):
            raise ValidationError("package output must be outside the compiled deployment")
        if deployment.exists():
            shutil.rmtree(deployment)
        shutil.copytree(native_deployment, deployment, copy_function=_link_or_copy)
        (deployment / "manifest.json").write_bytes(manifest_bytes)
        result = {
            "status": "success", "model_id": model_id, "package_id": package_id,
            "build_profile_id": build_profile_id, "layout": "directory",
            "manifest_path": "deployment/manifest.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": sum(item["size_bytes"] for item in installed_files),
            "built_at": datetime.now(timezone.utc).isoformat(), "manifest": manifest,
        }
        (package_dir / "package-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    # A single-component model's MPK is the package itself; anything more ships as a flat ZIP of the installed files.
    if len(packaged_files) == 1:
        bundle_name = packaged_files[0][0]
    else:
        bundle_name = f"{model_id}_{build_profile_id}_{package_id}.zip"
    bundle_path = package_dir / bundle_name
    temporary = package_dir / f".{bundle_name}.partial"
    try:
        if len(packaged_files) == 1:
            shutil.copyfile(packaged_files[0][1], temporary)
        else:
            with zipfile.ZipFile(temporary, "w", allowZip64=True, compression=zipfile.ZIP_STORED) as archive:
                for name, path in packaged_files:
                    _add_file(archive, name, path)
        temporary.replace(bundle_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest, size = sha256_file(bundle_path)
    result = {
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
