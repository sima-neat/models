from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from model_registry.artifact import validate_compile_outputs, verify_mpk
from model_registry.errors import ValidationError
from model_registry.fetch import sha256_file


def _archive(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name in names:
            content = b"fixture"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            stream.addfile(info, io.BytesIO(content))


def _output(component: str, archive: Path, path: str | None = None) -> dict[str, str]:
    digest, _ = sha256_file(archive)
    return {
        "component_id": component,
        "mpk_path": path or f"{component}/{archive.name}",
        "mpk_sha256": digest,
    }


def _grouped_manifest(tmp_path: Path, outputs: list[dict[str, str]]) -> dict[str, object]:
    fidelity = tmp_path / "transformer" / "fidelity.json"
    fidelity.parent.mkdir(exist_ok=True)
    fidelity.write_text("{}\n", encoding="utf-8")
    digest, _ = sha256_file(fidelity)
    return {
        "schema_version": 2,
        "model_id": "rf_detr_small",
        "build_profile_id": "modalix_int8_bf16",
        "compile_invocation_id": "rf_detr",
        "outputs": outputs,
        "derived_graphs": [
            {"component_id": component, "sha256": "a" * 64}
            for component in ("backbone", "transformer")
        ],
        "correctness": {
            "component_id": "transformer",
            "path": "transformer/fidelity.json",
            "sha256": digest,
        },
    }


def test_verify_mpk_writes_checksums_and_result(tmp_path: Path) -> None:
    archive = tmp_path / "resnet_50_mpk.tar.gz"
    _archive(archive, ["resnet_50/model.elf", "resnet_50/resnet_50_mpk.json"])
    result = verify_mpk(tmp_path, write_manifest=True, provenance={"model_id": "resnet_50"})
    assert result["archive"] == archive.name
    assert len(result["sha256"]) == 64
    assert archive.name in (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    written = json.loads((tmp_path / "compile-result.json").read_text(encoding="utf-8"))
    assert written["archive"] == archive.name
    assert written["built_at"].endswith("+00:00")
    assert written["provenance"]["model_id"] == "resnet_50"
    assert written["status"] == "success"


@pytest.mark.parametrize(
    "names, message",
    [
        (["model/model_mpk.json"], "ELF payload"),
        (["model/model.elf"], "_mpk.json"),
        (["../model.elf", "model_mpk.json"], "unsafe archive member"),
    ],
)
def test_verify_mpk_rejects_incomplete_or_unsafe_archives(tmp_path: Path, names: list[str], message: str) -> None:
    _archive(tmp_path / "broken_mpk.tar.gz", names)
    with pytest.raises(ValidationError, match=message):
        verify_mpk(tmp_path)


def test_verify_mpk_requires_exactly_one_archive(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        verify_mpk(tmp_path)


def test_validate_compile_outputs_accepts_exact_component_set(tmp_path: Path) -> None:
    for component in ("backbone", "transformer"):
        directory = tmp_path / component
        directory.mkdir()
        _archive(directory / f"{component}_mpk.tar.gz", [f"{component}/model.elf", f"{component}/model_mpk.json"])
    manifest = tmp_path / "compile-outputs.json"
    manifest.write_text(
        json.dumps(
            _grouped_manifest(
                tmp_path,
                [
                    _output("backbone", tmp_path / "backbone" / "backbone_mpk.tar.gz"),
                    _output("transformer", tmp_path / "transformer" / "transformer_mpk.tar.gz"),
                ],
            )
        ),
        encoding="utf-8",
    )

    result = validate_compile_outputs(
        manifest,
        tmp_path,
        model_id="rf_detr_small",
        build_profile_id="modalix_int8_bf16",
        compile_invocation_id="rf_detr",
        component_ids=["backbone", "transformer"],
    )

    assert [item["component_id"] for item in result["outputs"]] == ["backbone", "transformer"]
    assert all(Path(item["mpk_path"]).is_file() for item in result["outputs"])


@pytest.mark.parametrize(
    "path, message",
    [
        ("../backbone_mpk.tar.gz", "unsafe MPK path"),
        ("backbone/backbone_mpk.tar.gz", "missing component"),
    ],
)
def test_validate_compile_outputs_rejects_incomplete_or_unsafe_manifest(
    tmp_path: Path, path: str, message: str
) -> None:
    directory = tmp_path / "backbone"
    directory.mkdir()
    _archive(directory / "backbone_mpk.tar.gz", ["backbone/model.elf", "backbone/model_mpk.json"])
    manifest = tmp_path / "compile-outputs.json"
    manifest.write_text(
        json.dumps(
            _grouped_manifest(
                tmp_path,
                [_output("backbone", directory / "backbone_mpk.tar.gz", path)],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=message):
        validate_compile_outputs(
            manifest,
            tmp_path,
            model_id="rf_detr_small",
            build_profile_id="modalix_int8_bf16",
            compile_invocation_id="rf_detr",
            component_ids=["backbone", "transformer"],
        )


def test_validate_compile_outputs_rejects_mpk_hash_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "backbone"
    directory.mkdir()
    archive = directory / "backbone_mpk.tar.gz"
    _archive(archive, ["backbone/model.elf", "backbone/model_mpk.json"])
    manifest = tmp_path / "compile-outputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "rf_detr_small",
                "build_profile_id": "modalix_int8_bf16",
                "compile_invocation_id": "rf_detr",
                "outputs": [
                    {
                        "component_id": "backbone",
                        "mpk_path": "backbone/backbone_mpk.tar.gz",
                        "mpk_sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="SHA-256 does not match"):
        validate_compile_outputs(
            manifest,
            tmp_path,
            model_id="rf_detr_small",
            build_profile_id="modalix_int8_bf16",
            compile_invocation_id="rf_detr",
            component_ids=["backbone"],
        )


def test_validate_compile_outputs_rejects_undeclared_mpk(tmp_path: Path) -> None:
    directory = tmp_path / "backbone"
    directory.mkdir()
    archive = directory / "backbone_mpk.tar.gz"
    _archive(archive, ["backbone/model.elf", "backbone/model_mpk.json"])
    extra = tmp_path / "compiler-state" / "extra_mpk.tar.gz"
    extra.parent.mkdir()
    _archive(extra, ["extra/model.elf", "extra/model_mpk.json"])
    manifest = tmp_path / "compile-outputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "rf_detr_small",
                "build_profile_id": "modalix_int8_bf16",
                "compile_invocation_id": "backbone",
                "outputs": [_output("backbone", archive)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="undeclared MPK"):
        validate_compile_outputs(
            manifest,
            tmp_path,
            model_id="rf_detr_small",
            build_profile_id="modalix_int8_bf16",
            compile_invocation_id="backbone",
            component_ids=["backbone"],
        )
