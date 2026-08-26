from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from model_registry.artifact import verify_mpk
from model_registry.catalog import discover_catalog, model_build_fingerprint
from model_registry.errors import ValidationError
from model_registry.package import package_compiled_models

ROOT = Path(__file__).resolve().parents[2]
GIT_SHA = "a" * 40


def _compiled_fixture(
    root: Path,
    *,
    git_sha: str = GIT_SHA,
    models: tuple[str, ...] | None = None,
) -> None:
    for model_id, model in discover_catalog(ROOT).items():
        if models is not None and model_id not in models:
            continue
        for variant in model["compile_variants"]:
            if variant.get("status", "supported") == "unsupported":
                continue
            variant_id = variant["id"]
            output = root / model_id / variant_id
            output.mkdir(parents=True)
            archive = output / f"{model_id}_{variant_id}_mpk.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for name in (f"{model_id}/model.elf", f"{model_id}/{model_id}_mpk.json"):
                    content = b"fixture"
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    stream.addfile(info, io.BytesIO(content))
            provenance = {
                "model_id": model_id,
                "variant_id": variant_id,
                "git_sha": git_sha,
                "target": variant["target"],
                "environment": "staging",
                "compiler_version": "2.1.3",
                "compiler_label": "model-compiler-2.1.3",
                "source_sha256": "c" * 64,
                "build_fingerprint": model_build_fingerprint(ROOT, model_id, "2.1.3"),
            }
            calibration = model["calibration"]
            if calibration["status"] == "ready":
                provenance["calibration_sha256"] = calibration["source"]["sha256"]
            verify_mpk(
                output,
                write_manifest=True,
                provenance=provenance,
            )


def test_package_builds_root_and_per_model_sima_cli_metadata(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled"
    _compiled_fixture(compiled)
    destination = tmp_path / "dist"

    result = package_compiled_models(ROOT, compiled, destination, GIT_SHA, branch="develop")

    assert result["commit_folder"] == GIT_SHA[:12]
    root_metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert root_metadata["name"] == "gh:sima-neat/models"
    assert root_metadata["resources"] == ["catalog.json", "taxonomy.json"]
    assert root_metadata["selectable-resources"][0]["resource"].endswith("_mpk.tar.gz")
    assert len(root_metadata["selectable-resources"][0]["checksum"]) == 64
    assert root_metadata["selectable-resources"][0]["resource"].startswith(
        "https://artifacts.stg.neat.sima.ai/models/develop/aaaaaaaaaaaa/"
    )

    latest = json.loads((destination / "latest.json").read_text(encoding="utf-8"))
    latest_schema = json.loads((ROOT / "schemas" / "latest-index-v2.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(latest_schema, format_checker=FormatChecker()).validate(latest)
    assert latest["schema_version"] == 2
    assert latest["taxonomy"]["path"] == "taxonomy.json"
    assert latest["models"]["resnet_50"]["categorization"] == {
        "taxonomy_version": 2,
        "model_type": "vision",
        "tasks": ["image_classification"],
        "modalities": {"inputs": ["image"], "outputs": ["class_scores"]},
        "architecture": {"id": "resnet_50", "paradigm": "cnn", "variant": "imagenet_1k"},
        "tags": [],
    }
    assert latest["catalog_tag"] == GIT_SHA[:12]
    assert latest["models"]["resnet_50"]["latest_tag"] == GIT_SHA[:12]
    assert latest["models"]["resnet_50"]["artifact_branch"] == "develop"
    assert latest["models"]["resnet_50"]["built_at"]
    assert len(latest["models"]["resnet_50"]["build_fingerprint"]) == 64
    taxonomy = json.loads((destination / "taxonomy.json").read_text(encoding="utf-8"))
    assert taxonomy["tasks"]["image_classification"]["display_name"] == "Image Classification"
    assert latest["taxonomy"]["sha256"] == hashlib.sha256((destination / "taxonomy.json").read_bytes()).hexdigest()

    catalog_model = next(item for item in result["models"] if item["id"] == "resnet_50")
    assert catalog_model["categorization"]["architecture"]["paradigm"] == "cnn"

    model_metadata = json.loads((destination / "resnet_50" / "metadata.json").read_text(encoding="utf-8"))
    assert model_metadata["name"] == "gh:sima-neat/models/resnet_50"
    assert model_metadata["resources"] == [
        "modalix_bf16/resnet_50_modalix_bf16_mpk.tar.gz",
        "modalix_int8/resnet_50_modalix_int8_mpk.tar.gz",
    ]
    assert set(model_metadata["resources"]) == set(model_metadata["resources-checksum"])
    assert (destination / "resnet_50" / "modalix_bf16" / "compile-result.json").is_file()
    assert (destination / "resnet_50" / "modalix_bf16" / "SHA256SUMS").is_file()
    assert (destination / "resnet_50" / "modalix_int8" / "compile-result.json").is_file()
    assert (destination / "resnet_50" / "modalix_int8" / "SHA256SUMS").is_file()


def test_package_rejects_mismatched_compile_provenance(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled"
    _compiled_fixture(compiled, git_sha="b" * 40)
    with pytest.raises(ValidationError, match="provenance git_sha"):
        package_compiled_models(ROOT, compiled, tmp_path / "dist", GIT_SHA)


def test_package_rejects_existing_output_directory(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled"
    _compiled_fixture(compiled)
    destination = tmp_path / "dist"
    destination.mkdir()
    with pytest.raises(ValidationError, match="already exists"):
        package_compiled_models(ROOT, compiled, destination, GIT_SHA)


def test_package_rejects_unsupported_compiled_variant_with_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = discover_catalog(ROOT)
    variant = next(
        item for item in catalog["ssd_mobilenet_v3"]["compile_variants"] if item["id"] == "modalix_int8_tess_mla"
    )
    variant["status"] = "unsupported"
    variant["reason"] = "fixture compiler evidence"
    monkeypatch.setattr("model_registry.package.discover_catalog", lambda _root: catalog)
    result = tmp_path / "compiled" / "ssd_mobilenet_v3" / "modalix_int8_tess_mla" / "compile-result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ValidationError,
        match="compiled output references unsupported variant.*fixture compiler evidence",
    ):
        package_compiled_models(ROOT, tmp_path / "compiled", tmp_path / "dist", GIT_SHA)


def test_package_carries_forward_pointers_without_copying_unchanged_mpk(tmp_path: Path) -> None:
    first_compiled = tmp_path / "first-compiled"
    _compiled_fixture(first_compiled)
    previous = tmp_path / "previous"
    package_compiled_models(ROOT, first_compiled, previous, GIT_SHA, branch="main")

    no_changes = tmp_path / "no-changes"
    no_changes.mkdir()
    next_sha = "b" * 40
    destination = tmp_path / "next"
    result = package_compiled_models(
        ROOT,
        no_changes,
        destination,
        next_sha,
        branch="feature/catalog",
        previous_index=previous / "latest.json",
    )

    assert result["commit"] == next_sha
    assert result["models"][0]["compiled_git_sha"] == GIT_SHA[:12]
    assert result["models"][0]["artifact_branch"] == "main"
    assert not (destination / result["models"][0]["mpk"]).exists()
    assert not (destination / "resnet_50" / "metadata.json").exists()
    latest = json.loads((destination / "latest.json").read_text(encoding="utf-8"))
    assert latest["catalog_tag"] == next_sha[:12]
    assert latest["models"]["resnet_50"]["latest_tag"] == GIT_SHA[:12]
    assert latest["models"]["resnet_50"]["artifact_branch"] == "main"
    assert latest["models"]["resnet_50"]["categorization"]["tasks"] == ["image_classification"]
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert "/models/main/aaaaaaaaaaaa/resnet_50/" in metadata["selectable-resources"][0]["resource"]


def test_first_package_requires_every_catalog_model(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    with pytest.raises(ValidationError, match="no compiled output and no previous"):
        package_compiled_models(ROOT, compiled, tmp_path / "dist", GIT_SHA)


def test_feature_package_can_start_with_only_its_compiled_model(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled"
    _compiled_fixture(compiled, models=("yolo26_tiny_drone",))
    destination = tmp_path / "dist"

    result = package_compiled_models(
        ROOT,
        compiled,
        destination,
        GIT_SHA,
        branch="feature/yolo26",
        allow_partial_catalog=True,
    )

    assert {model["id"] for model in result["models"]} == {"yolo26_tiny_drone"}
    latest = json.loads((destination / "latest.json").read_text(encoding="utf-8"))
    assert set(latest["models"]) == {"yolo26_tiny_drone"}
    assert (destination / "yolo26_tiny_drone" / "metadata.json").is_file()
