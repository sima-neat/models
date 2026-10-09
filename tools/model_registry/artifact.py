from __future__ import annotations

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
        output_fields = {"component_id", "mpk_path", "mpk_sha256"}
        if not isinstance(output, dict) or set(output) != output_fields:
            raise ValidationError(f"{manifest_file}: outputs[{index}] must contain {sorted(output_fields)}")
        component_id = output.get("component_id")
        if not isinstance(component_id, str) or component_id not in expected_components:
            raise ValidationError(f"{manifest_file}: outputs[{index}] has unknown component {component_id!r}")
        if component_id in seen_components:
            raise ValidationError(f"{manifest_file}: component {component_id!r} appears more than once")
        relative = PurePosixPath(str(output.get("mpk_path", "")))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValidationError(f"{manifest_file}: unsafe MPK path {relative}")
        if relative.parts[0] != component_id or not relative.name.endswith("_mpk.tar.gz"):
            raise ValidationError(f"{manifest_file}: {component_id!r} MPK must be under its component directory")
        archive = root.joinpath(*relative.parts)
        try:
            archive.resolve().relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"{manifest_file}: unsafe MPK path {relative}") from exc
        if not archive.is_file():
            raise ValidationError(f"{manifest_file}: MPK does not exist: {relative}")
        digest, _ = sha256_file(archive)
        if output.get("mpk_sha256") != digest:
            raise ValidationError(f"{manifest_file}: {component_id!r} MPK SHA-256 does not match")
        archives = sorted(archive.parent.glob("*_mpk.tar.gz"))
        if archives != [archive]:
            raise ValidationError(
                f"{manifest_file}: expected exactly one MPK for {component_id}, found {len(archives)}"
            )
        seen_components.add(component_id)
        resolved.append({"component_id": component_id, "mpk_path": str(archive), "mpk_sha256": digest})

    missing = sorted(expected_components - seen_components)
    if missing:
        raise ValidationError(f"{manifest_file}: missing component outputs: {', '.join(missing)}")
    declared_archives = {Path(output["mpk_path"]).resolve() for output in resolved}
    actual_archives = {path.resolve() for path in root.rglob("*_mpk.tar.gz")}
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
        archives = sorted((root / component_id).glob("*_mpk.tar.gz"))
        if len(archives) != 1:
            raise ValidationError(f"expected one MPK in {root / component_id}, found {len(archives)}")
        outputs.append({
            "component_id": component_id,
            "mpk_path": f"{component_id}/{archives[0].name}",
            "mpk_sha256": sha256_file(archives[0])[0],
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
