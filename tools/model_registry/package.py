from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact import verify_mpk
from .catalog import discover_catalog, model_build_fingerprint
from .errors import ValidationError
from .fetch import sha256_file
from .model import variant_requires_calibration
from .published import artifact_url, validate_latest_index
from .taxonomy import (
    TAXONOMY_RELATIVE_PATH,
    canonical_taxonomy_document,
    load_taxonomy,
    load_taxonomy_for_model,
    validate_model_categorization,
)


def _size_text(size_bytes: int) -> str:
    for suffix, divisor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if size_bytes >= divisor:
            return f"{size_bytes / divisor:.1f}".rstrip("0").rstrip(".") + suffix
    return "1KB" if size_bytes else "0KB"


def _metadata(
    *,
    name: str,
    version: str,
    description: str,
    resources: list[str],
    checksums: dict[str, str],
    selectables: list[dict[str, str]],
    size_bytes: int,
) -> dict[str, Any]:
    if not resources:
        raise ValidationError(f"{name}: sima-cli metadata requires at least one non-selectable resource")
    size = _size_text(size_bytes)
    return {
        "name": name,
        "version": version,
        "release": "",
        "description": description,
        "platforms": [],
        "resources": resources,
        "resources-checksum": checksums,
        "selectable-resources": selectables,
        "size": {"download": size, "install": size},
        "installation": {
            "script": 'echo "Compiled model MPK artifacts downloaded"',
            "post-message": "[bold]Compiled model artifacts are ready.[/bold]\n",
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_compiled_models(
    repository: str | Path,
    compiled_dir: str | Path,
    output_dir: str | Path,
    git_sha: str,
    *,
    environment: str = "staging",
    branch: str = "main",
    previous_index: str | Path | None = None,
    allow_partial_catalog: bool = False,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    compiled = Path(compiled_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValidationError(f"package output directory already exists: {output}")
    if not compiled.is_dir():
        raise ValidationError(f"compiled model directory not found: {compiled}")
    if len(git_sha) != 40 or any(c not in "0123456789abcdef" for c in git_sha.lower()):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip():
        raise ValidationError("branch must not be empty")

    catalog = discover_catalog(root)
    taxonomy = load_taxonomy(root / TAXONOMY_RELATIVE_PATH)
    categorizations = {
        model_id: validate_model_categorization(model, load_taxonomy_for_model(root, model))
        for model_id, model in catalog.items()
    }
    version = git_sha[:12]
    previous_models: dict[str, Any] = {}
    if previous_index is not None:
        path = Path(previous_index).expanduser().resolve()
        if path.is_file():
            previous_models = validate_latest_index(json.loads(path.read_text(encoding="utf-8")))["models"]

    current_results = sorted(compiled.glob("*/*/compile-result.json"))
    current_by_model: dict[str, list[dict[str, Any]]] = {}
    for result_path in current_results:
        model_id, variant_id = result_path.relative_to(compiled).parts[:2]
        if model_id not in catalog:
            raise ValidationError(f"compiled output references unknown model: {model_id}")
        variants = {item["id"]: item for item in catalog[model_id]["compile_variants"]}
        if variant_id not in variants:
            raise ValidationError(f"compiled output references unknown variant: {model_id}/{variant_id}")
        if variants[variant_id].get("status", "supported") == "unsupported":
            raise ValidationError(
                f"compiled output references unsupported variant: {model_id}/{variant_id}: "
                f"{variants[variant_id]['reason']}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = result.get("provenance", {})
        expected_provenance = {
            "model_id": model_id,
            "variant_id": variant_id,
            "git_sha": git_sha,
            "target": variants[variant_id]["target"],
            "environment": environment,
        }
        compiler_version = provenance.get("compiler_version")
        if isinstance(compiler_version, str) and compiler_version.strip():
            expected_provenance["build_fingerprint"] = model_build_fingerprint(root, model_id, compiler_version)
        for key, value in expected_provenance.items():
            if provenance.get(key) != value:
                raise ValidationError(
                    f"{result_path}: provenance {key} must be {value!r}, found {provenance.get(key)!r}"
                )
        for key in ("compiler_version", "compiler_label"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise ValidationError(f"{result_path}: provenance {key} must be a non-empty string")
        source_sha256 = provenance.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256.lower())
        ):
            raise ValidationError(f"{result_path}: provenance source_sha256 must be a SHA-256 digest")
        declared_calibration = catalog[model_id]["calibration"]
        calibration_sha256 = provenance.get("calibration_sha256")
        requires_calibration = variant_requires_calibration(variants[variant_id])
        if declared_calibration["status"] == "ready" and requires_calibration:
            calibration_input = declared_calibration.get("source") or declared_calibration.get("dataset")
            expected_calibration = calibration_input["sha256"]
            if calibration_sha256 != expected_calibration:
                raise ValidationError(
                    f"{result_path}: provenance calibration_sha256 must be {expected_calibration!r}, "
                    f"found {calibration_sha256!r}"
                )
        elif calibration_sha256 is not None:
            raise ValidationError(f"{result_path}: unexpected calibration provenance for {model_id}/{variant_id}")
        verified = verify_mpk(result_path.parent)
        if result.get("sha256") != verified["sha256"] or result.get("archive") != verified["archive"]:
            raise ValidationError(f"{result_path}: compile result does not match the MPK archive")
        destination = output / model_id / variant_id
        destination.mkdir(parents=True, exist_ok=True)
        for source in (result_path.parent / verified["archive"], result_path.parent / "SHA256SUMS", result_path):
            if not source.is_file():
                raise ValidationError(f"{result_path.parent}: incomplete compiled model output")
            shutil.copy2(source, destination / source.name)
        current_by_model.setdefault(model_id, []).append(
            {
                "target": variants[variant_id]["target"],
                "variant_id": variant_id,
                "mpk": f"{model_id}/{variant_id}/{verified['archive']}",
                "sha256": verified["sha256"],
                "size_bytes": verified["size_bytes"],
                "built_at": result.get("built_at"),
                "provenance": provenance,
            }
        )

    index_models: dict[str, Any] = {}
    for model_id, model in sorted(catalog.items()):
        expected_variants = {
            item["id"] for item in model["compile_variants"] if item.get("status", "supported") == "supported"
        }
        if model_id in current_by_model:
            built = current_by_model[model_id]
            found_variants = {item["variant_id"] for item in built}
            if found_variants != expected_variants:
                raise ValidationError(
                    f"{model_id}: compiled variants {sorted(found_variants)} "
                    f"do not match catalog {sorted(expected_variants)}"
                )
            built_at_values = [str(item["built_at"] or "") for item in built]
            if any(not value for value in built_at_values):
                raise ValidationError(f"{model_id}: compile result is missing built_at")
            index_models[model_id] = {
                "display_name": model["display_name"],
                "description": model["description"],
                **({"categorization": categorizations[model_id]} if categorizations[model_id] is not None else {}),
                "latest_tag": version,
                "artifact_branch": branch,
                "built_at": max(built_at_values),
                "build_fingerprint": built[0]["provenance"]["build_fingerprint"],
                "variants": {
                    item["variant_id"]: {
                        "target": item["target"],
                        "mpk": item["mpk"],
                        "sha256": item["sha256"],
                        "size_bytes": item["size_bytes"],
                        "compiler_version": item["provenance"].get("compiler_version", ""),
                        "compiler_label": item["provenance"].get("compiler_label", ""),
                        "source_sha256": item["provenance"].get("source_sha256", ""),
                        **(
                            {"calibration_sha256": item["provenance"]["calibration_sha256"]}
                            if item["provenance"].get("calibration_sha256")
                            else {}
                        ),
                    }
                    for item in built
                },
            }
        else:
            entry = previous_models.get(model_id)
            if not isinstance(entry, dict):
                if allow_partial_catalog:
                    continue
                raise ValidationError(f"{model_id}: no compiled output and no previous latest.json entry")
            if set(entry.get("variants", {})) != expected_variants:
                raise ValidationError(
                    f"{model_id}: previous variants do not match the current catalog; rebuild the model"
                )
            entry = dict(entry)
            entry["display_name"] = model["display_name"]
            entry["description"] = model["description"]
            if categorizations[model_id] is None:
                entry.pop("categorization", None)
            else:
                entry["categorization"] = categorizations[model_id]
            index_models[model_id] = entry

    packaged_models: list[dict[str, Any]] = []
    root_selectables: list[dict[str, str]] = []
    for model_id, entry in sorted(index_models.items()):
        for variant_id, variant in sorted(entry["variants"].items()):
            url = artifact_url(environment, entry["artifact_branch"], entry["latest_tag"], variant["mpk"])
            packaged_models.append(
                {
                    "id": model_id,
                    "display_name": entry["display_name"],
                    "variant_id": variant_id,
                    **({"categorization": entry["categorization"]} if entry.get("categorization") else {}),
                    "target": variant["target"],
                    "mpk": variant["mpk"],
                    "artifact_url": url,
                    "sha256": variant["sha256"],
                    "size_bytes": variant["size_bytes"],
                    "compiled_git_sha": entry["latest_tag"],
                    "artifact_branch": entry["artifact_branch"],
                    "built_at": entry["built_at"],
                }
            )
            root_selectables.append(
                {
                    "name": f"{entry['display_name']} ({variant_id})",
                    "url": "",
                    "resource": url,
                    "checksum": variant["sha256"],
                }
            )

    for model_id, built in current_by_model.items():
        model_dir = output / model_id
        resources = [Path(item["mpk"]).relative_to(model_id).as_posix() for item in built]
        checksums = {resource: sha256_file(model_dir / resource)[0] for resource in resources}
        size_bytes = sum((model_dir / resource).stat().st_size for resource in resources)
        _write_json(
            model_dir / "metadata.json",
            _metadata(
                name=f"gh:sima-neat/models/{model_id}",
                version=version,
                description=catalog[model_id]["description"],
                resources=resources,
                checksums=checksums,
                selectables=[],
                size_bytes=size_bytes,
            ),
        )

    taxonomy_path = output / "taxonomy.json"
    _write_json(taxonomy_path, canonical_taxonomy_document(taxonomy))
    taxonomy_digest, taxonomy_size = sha256_file(taxonomy_path)
    taxonomy_reference = {"version": taxonomy["version"], "path": "taxonomy.json", "sha256": taxonomy_digest}

    latest_index = {
        "schema_version": 2,
        "repository": "sima-neat/models",
        "branch": branch,
        "catalog_tag": version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": taxonomy_reference,
        "models": index_models,
    }
    _write_json(output / "latest.json", latest_index)
    catalog_document = {
        "schema_version": 2,
        "repository": "sima-neat/models",
        "commit": git_sha,
        "commit_folder": version,
        "taxonomy": taxonomy_reference,
        "models": packaged_models,
    }
    _write_json(output / "catalog.json", catalog_document)
    catalog_digest, catalog_size = sha256_file(output / "catalog.json")
    _write_json(
        output / "metadata.json",
        _metadata(
            name="gh:sima-neat/models",
            version=version,
            description="SiMa Neat compiled model MPK catalog",
            resources=["catalog.json", "taxonomy.json"],
            checksums={"catalog.json": catalog_digest, "taxonomy.json": taxonomy_digest},
            selectables=root_selectables,
            size_bytes=taxonomy_size + catalog_size + sum(item["size_bytes"] for item in packaged_models),
        ),
    )
    return catalog_document
