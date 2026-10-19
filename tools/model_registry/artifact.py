from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ValidationError
from .fetch import sha256_file


def validate_compile_outputs(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    model_id: str,
    build_profile_id: str,
    compile_invocation_id: str,
    component_ids: list[str],
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{manifest_file}: could not load compile outputs: {exc}") from exc

    # A grouped invocation compiles several components from one derived graph set, so it must also
    # record each component's derived graph and the invocation's correctness result.
    grouped = len(component_ids) > 1
    fields = {"model_id", "build_profile_id", "compile_invocation_id", "outputs"}
    if grouped:
        fields.update({"derived_graphs", "correctness"})
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValidationError(f"{manifest_file}: compile outputs must contain {sorted(fields)}")
    identity = {
        "model_id": model_id,
        "build_profile_id": build_profile_id,
        "compile_invocation_id": compile_invocation_id,
    }
    for field, expected in identity.items():
        if manifest.get(field) != expected:
            raise ValidationError(f"{manifest_file}: {field} must be {expected!r}, found {manifest.get(field)!r}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValidationError(f"{manifest_file}: outputs must be a non-empty list")
    if len(component_ids) != len(set(component_ids)):
        raise ValidationError(f"{manifest_file}: expected component IDs must be unique")
    expected_components = set(component_ids)
    resolved: list[dict[str, str]] = []
    seen_components: set[str] = set()
    for index, output in enumerate(outputs):
        path_field, sha_field = (
            compile_output_fields(output) if isinstance(output, dict) else ("mpk_path", "mpk_sha256")
        )
        output_fields = {"component_id", path_field, sha_field}
        if not isinstance(output, dict) or set(output) != output_fields:
            raise ValidationError(f"{manifest_file}: outputs[{index}] must contain {sorted(output_fields)}")
        component_id = output.get("component_id")
        if not isinstance(component_id, str) or component_id not in expected_components:
            raise ValidationError(f"{manifest_file}: outputs[{index}] has unknown component {component_id!r}")
        if component_id in seen_components:
            raise ValidationError(f"{manifest_file}: component {component_id!r} appears more than once")
        relative = PurePosixPath(str(output.get(path_field, "")))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValidationError(f"{manifest_file}: unsafe MPK path {relative}")
        suffix = "deployment-manifest.json" if path_field == "artifact_path" else "_mpk.tar.gz"
        if relative.parts[0] != component_id or not relative.name.endswith(suffix):
            raise ValidationError(f"{manifest_file}: {component_id!r} MPK must be under its component directory")
        archive = root.joinpath(*relative.parts)
        try:
            archive.resolve().relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"{manifest_file}: unsafe MPK path {relative}") from exc
        if not archive.is_file():
            raise ValidationError(f"{manifest_file}: MPK does not exist: {relative}")
        digest, _ = sha256_file(archive)
        if output.get(sha_field) != digest:
            raise ValidationError(f"{manifest_file}: {component_id!r} MPK SHA-256 does not match")
        archives = compiled_artifacts(archive.parent)
        if archives != [archive]:
            raise ValidationError(
                f"{manifest_file}: expected exactly one MPK for {component_id}, found {len(archives)}"
            )
        seen_components.add(component_id)
        if path_field == "artifact_path":
            verify_artifact(archive.parent)
        resolved.append({"component_id": component_id, path_field: str(archive), sha_field: digest})

    missing = sorted(expected_components - seen_components)
    if missing:
        raise ValidationError(f"{manifest_file}: missing component outputs: {', '.join(missing)}")
    declared_archives = {Path(output[compile_output_fields(output)[0]]).resolve() for output in resolved}
    actual_archives = {
        path.resolve() for pattern in ("*_mpk.tar.gz", "deployment-manifest.json") for path in root.rglob(pattern)
    }
    if actual_archives != declared_archives:
        raise ValidationError(f"{manifest_file}: output tree contains undeclared MPK archives")

    result = {**identity, "outputs": resolved}
    if grouped:
        graphs = manifest["derived_graphs"]
        if not isinstance(graphs, list) or len(graphs) != len(component_ids):
            raise ValidationError(f"{manifest_file}: derived_graphs must cover every component")
        graph_ids: set[str] = set()
        for index, graph in enumerate(graphs):
            if not isinstance(graph, dict) or set(graph) != {"component_id", "sha256"}:
                raise ValidationError(
                    f"{manifest_file}: derived_graphs[{index}] must contain component_id and sha256"
                )
            component_id = graph["component_id"]
            digest = graph["sha256"]
            if component_id not in expected_components or component_id in graph_ids:
                raise ValidationError(f"{manifest_file}: invalid derived graph component {component_id!r}")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValidationError(f"{manifest_file}: invalid derived graph SHA-256")
            graph_ids.add(component_id)
        if graph_ids != expected_components:
            raise ValidationError(f"{manifest_file}: derived_graphs must cover every component")

        correctness = manifest["correctness"]
        if not isinstance(correctness, dict) or set(correctness) != {"component_id", "path", "sha256"}:
            raise ValidationError(f"{manifest_file}: correctness must contain component_id, path, and sha256")
        component_id = correctness["component_id"]
        if component_id not in expected_components:
            raise ValidationError(f"{manifest_file}: correctness references unknown component {component_id!r}")
        relative = PurePosixPath(str(correctness["path"]))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValidationError(f"{manifest_file}: unsafe correctness path {relative}")
        if relative.parts[0] != component_id or relative.name != "fidelity.json":
            raise ValidationError(f"{manifest_file}: correctness must be the owning component's fidelity.json")
        correctness_path = root.joinpath(*relative.parts)
        if not correctness_path.is_file():
            raise ValidationError(f"{manifest_file}: correctness result does not exist: {relative}")
        digest, _ = sha256_file(correctness_path)
        if correctness["sha256"] != digest:
            raise ValidationError(f"{manifest_file}: correctness SHA-256 does not match")
        result.update(
            {
                "derived_graphs": graphs,
                "correctness": {
                    "component_id": component_id,
                    "path": str(correctness_path),
                    "sha256": digest,
                },
            }
        )
    return result


def record_compile_outputs(
    output_root: str | Path,
    *,
    model_id: str,
    build_profile_id: str,
    compile_invocation_id: str,
    component_ids: list[str],
) -> Path:
    """Write compile-outputs.json for a recipe that wrote one MPK per component directory and no manifest."""
    root = Path(output_root).expanduser().resolve()
    outputs = []
    for component_id in component_ids:
        archives = compiled_artifacts(root / component_id)
        if len(archives) != 1:
            raise ValidationError(f"expected one MPK in {root / component_id}, found {len(archives)}")
        native = archives[0].name == "deployment-manifest.json"
        outputs.append({
            "component_id": component_id,
            "artifact_path" if native else "mpk_path": f"{component_id}/{archives[0].name}",
            "artifact_sha256" if native else "mpk_sha256": sha256_file(archives[0])[0],
        })
    manifest = root / "compile-outputs.json"
    manifest.write_text(
        json.dumps({
            "model_id": model_id,
            "build_profile_id": build_profile_id,
            "compile_invocation_id": compile_invocation_id,
            "outputs": outputs,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_mpk(
    output_dir: str | Path,
    *,
    write_manifest: bool = False,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    directory = Path(output_dir).expanduser().resolve()
    archives = sorted(directory.glob("*_mpk.tar.gz")) if directory.is_dir() else []
    if len(archives) != 1:
        raise ValidationError(f"{directory}: expected exactly one *_mpk.tar.gz archive, found {len(archives)}")

    archive = archives[0]
    try:
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ValidationError(f"{archive}: invalid MPK archive: {exc}") from exc

    names: list[str] = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValidationError(f"{archive}: unsafe archive member: {member.name}")
        names.append(member.name)
    if not any(name.endswith(".elf") for name in names):
        raise ValidationError(f"{archive}: MPK does not contain an ELF payload")
    if not any(name.endswith("_mpk.json") for name in names):
        raise ValidationError(f"{archive}: MPK does not contain _mpk.json metadata")

    digest, size = sha256_file(archive)
    result = {
        "archive": archive.name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
        "sha256": digest,
        "size_bytes": size,
        "status": "success",
    }
    if write_manifest:
        (directory / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        (directory / "compile-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def compile_output_fields(output: dict[str, Any]) -> tuple[str, str]:
    """Read native artifact fields while retaining existing MPK manifests."""
    return ("artifact_path", "artifact_sha256") if "artifact_path" in output else ("mpk_path", "mpk_sha256")


def compiled_artifacts(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*_mpk.tar.gz"), *directory.glob("deployment-manifest.json")])


def deployment_artifact_name(path: str) -> str:
    """A stable registry name for a deployment-relative path, independent of storage."""
    return "file-" + hashlib.sha256(path.encode()).hexdigest()


def deployment_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate storage-independent file references before publishing or downloading."""
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("deployment manifest requires a non-empty files list")
    names: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "artifact", "sha256", "size_bytes"}:
            raise ValidationError("deployment files require path, artifact, sha256 and size_bytes")
        name = item["path"]
        if not isinstance(name, str):
            raise ValidationError("deployment file path must be a relative POSIX path")
        path = PurePosixPath(name)
        if (not name or not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or any(":" in part or "\\" in part or "\x00" in part or part.endswith((".", " "))
                       for part in path.parts)
                or path.parts[0].casefold() == "manifest.json" or name.casefold() in names):
            raise ValidationError(f"unsafe or duplicate deployment path: {name}")
        names.add(name.casefold())
        digest, size = item["sha256"], item["size_bytes"]
        if (item["artifact"] != deployment_artifact_name(name)
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or type(size) is not int or size < 0):
            raise ValidationError(f"invalid deployment file metadata: {name}")
    if any(parent.as_posix().casefold() in names for name in names for parent in PurePosixPath(name).parents):
        raise ValidationError("deployment file paths overlap")
    return files


def verify_llima(deployment: Path, *, allow_manifest: bool = False) -> list[dict[str, Any]]:
    """Validate a deployed LLiMa folder and checksum every regular file."""
    if deployment.is_symlink() or not deployment.is_dir():
        raise ValidationError(f"LLiMa deployment not found or is a symbolic link: {deployment}")
    files = []
    names: set[str] = set()
    for path in sorted(deployment.rglob("*")):
        relative = path.relative_to(deployment).as_posix()
        if allow_manifest and relative == "manifest.json" and path.is_file() and not path.is_symlink():
            continue
        if (path.is_symlink() or not (path.is_file() or path.is_dir())
                or any(":" in part or "\\" in part or part.endswith((".", " "))
                       for part in path.relative_to(deployment).parts)
                or path.relative_to(deployment).parts[0].casefold() == "manifest.json"
                or relative.casefold() in names):
            raise ValidationError(f"{deployment}: unsafe or duplicate deployment path: {relative}")
        names.add(relative.casefold())
        if path.is_file():
            digest, size = sha256_file(path)
            files.append({"path": relative, "artifact": deployment_artifact_name(relative),
                          "sha256": digest, "size_bytes": size})
    if not any("elf_files" in PurePosixPath(item["path"]).parts[:-1] and item["path"].endswith(".elf")
               for item in files):
        raise ValidationError(f"{deployment}: LLiMa deployment contains no ELF files")
    return deployment_files({"files": files})


def llima_compiled_parameters(deployment: Path) -> dict[str, Any]:
    """Read target settings from a single deployment or a speculative pair."""
    paths = [deployment / "devkit/vlm_config.json", *sorted(deployment.glob("*/devkit/vlm_config.json"))]
    targets = []
    for path in paths:
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        if not ((config.get("lm_cfg") or {}).get("speculative_decoding_cfg") or {}).get("is_draft"):
            targets.append(config)
    if not targets:
        return {}
    if len(targets) != 1:
        raise ValidationError(f"{deployment}: expected one target configuration")
    config = targets[0]
    pipeline = config["pipeline_cfg"]
    parameters = {
        name: pipeline[field]
        for name, field in (("max_num_tokens", "max_num_tokens"),
                            ("language_group_size", "input_token_group_size"))
        if field in pipeline
    }
    image_size = (config.get("vm_cfg") or {}).get("image_size")
    if type(image_size) is int and image_size > 0:
        image_size = [image_size, image_size]
    if (isinstance(image_size, list) and len(image_size) == 2
            and all(type(value) is int and value > 0 for value in image_size)):
        parameters.update(input_height=image_size[0], input_width=image_size[1])
    speculative = (config.get("lm_cfg") or {}).get("speculative_decoding_cfg") or {}
    if speculative.get("method"):
        parameters["speculative_method"] = speculative["method"]
    return parameters


def verify_artifact(output_dir: str | Path, *, write_manifest: bool = False,
                    provenance: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify one MPK or complete LLiMa deployment and record its provenance."""
    directory = Path(output_dir).expanduser().resolve()
    deployment = directory / "deployment"
    if not deployment.exists():
        return verify_mpk(directory, write_manifest=write_manifest, provenance=provenance)
    if list(directory.glob("*_mpk.tar.gz")):
        raise ValidationError(f"{directory}: expected one compiled artifact")
    files = verify_llima(deployment)
    payload = json.dumps({"format": "llima", "files": files}, indent=2, sort_keys=True).encode() + b"\n"
    manifest = directory / "deployment-manifest.json"
    if write_manifest:
        manifest.write_bytes(payload)
    elif not manifest.is_file() or manifest.read_bytes() != payload:
        raise ValidationError(f"{manifest}: deployment files do not match the manifest")
    digest = hashlib.sha256(payload).hexdigest()
    result = {"manifest": manifest.name, "format": "llima", "layout": "directory",
              "built_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance or {},
              "sha256": digest, "size_bytes": sum(item["size_bytes"] for item in files), "status": "success",
              "compiled_parameters": llima_compiled_parameters(deployment)}
    if write_manifest:
        (directory / "compile-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result
