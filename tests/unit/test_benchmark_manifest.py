from __future__ import annotations

import hashlib
import json
from pathlib import Path

import model_registry.benchmark_manifest as benchmark_manifest
import pytest
from model_registry.benchmark_manifest import make_benchmark_manifest, validate_benchmark_components
from model_registry.errors import ValidationError
from model_registry.model import load_and_validate_model

ROOT = Path(__file__).resolve().parents[2]


def test_dispatch_validates_complete_compile_but_benchmarks_only_declared_subset(tmp_path: Path, monkeypatch) -> None:
    model = load_and_validate_model(ROOT / "models" / "rf_detr_small", ROOT / "schemas" / "model.schema.json")
    model["tests"]["performance"] = model["tests"]["performance"][:1]
    monkeypatch.setattr(benchmark_manifest, "load_and_validate_model", lambda *args: model)
    output_root = tmp_path / "rf_detr_small" / "modalix_int8_bf16"
    outputs = []
    for component in ("backbone", "transformer"):
        directory = output_root / component
        directory.mkdir(parents=True)
        archive = directory / f"{component}_mpk.tar.gz"
        archive.write_bytes(component.encode())
        outputs.append(
            {
                "component_id": component,
                "mpk_path": f"{component}/{archive.name}",
                "mpk_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    fidelity = output_root / "transformer" / "fidelity.json"
    fidelity.write_text("{}")
    manifest = {
        "schema_version": 2,
        "model_id": "rf_detr_small",
        "build_profile_id": "modalix_int8_bf16",
        "compile_invocation_id": "rf_detr",
        "outputs": outputs,
        "derived_graphs": [
            {"component_id": component, "sha256": "a" * 64} for component in ("backbone", "transformer")
        ],
        "correctness": {
            "component_id": "transformer",
            "path": "transformer/fidelity.json",
            "sha256": hashlib.sha256(fidelity.read_bytes()).hexdigest(),
        },
    }
    (output_root / "compile-outputs.json").write_text(json.dumps(manifest))
    args = (ROOT, tmp_path, "rf_detr_small", "modalix_int8_bf16", "rf_detr")

    validate_benchmark_components(*args, ["backbone"])
    for components in (["transformer"], ["backbone", "transformer"], ["backbone", "backbone"], []):
        with pytest.raises(ValidationError, match="declared performance tests"):
            validate_benchmark_components(*args, components)
    (output_root / "transformer" / "transformer_mpk.tar.gz").unlink()
    with pytest.raises(ValidationError, match="MPK does not exist"):
        validate_benchmark_components(*args, ["backbone"])


@pytest.mark.parametrize(
    ("model_id", "build_profile_id", "component_id"),
    [
        ("resnet_50", "modalix_bf16", "model"),
        ("ssd_mobilenet_v1", "modalix_int8", "model"),
        ("yolo26_tiny_drone", "modalix_int8_tessellation_mla", "model"),
        ("rf_detr_small", "modalix_int8_bf16", "backbone"),
        ("rf_detr_small", "modalix_int8_bf16", "transformer"),
    ],
)
def test_generates_direct_devkit_manifest_for_compiled_model(
    tmp_path: Path, model_id: str, build_profile_id: str, component_id: str
) -> None:
    component = tmp_path / "compiled" / model_id / build_profile_id / component_id
    component.mkdir(parents=True)
    archive = component / f"{model_id}_mpk.tar.gz"
    archive.write_bytes(b"compiled model")
    (component / "compile-result.json").write_text(
        json.dumps(
            {
                "archive": f"{model_id}_mpk.tar.gz",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "provenance": {"compiler_version": "2.1.3"},
            }
        ),
        encoding="utf-8",
    )
    (component / "registry-publication.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model_id": model_id,
                        "build_profile_id": build_profile_id,
                        "component_id": component_id,
                        "run_id": "00000000-0000-4000-8000-000000000001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = make_benchmark_manifest(
        repository=ROOT,
        compiled_dir=tmp_path / "compiled",
        output_path=tmp_path / "performance-test.json",
        branch="feature/benchmark",
        git_sha="b" * 40,
        model_id=model_id,
        build_profile_id=build_profile_id,
        component_id=component_id,
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

    dependencies = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compatibility"] == {
        "platform_version": dependencies["platform-version"],
    }
    assert manifest["model"]["id"] == model_id
    assert manifest["model"]["build_profile_id"] == build_profile_id
    assert manifest["model"]["component_id"] == component_id
    assert manifest["model"]["path"].endswith(f"/{model_id}/{build_profile_id}/{component_id}/{model_id}_mpk.tar.gz")
    assert manifest["model"]["uri"] == "vulcan://runs/00000000-0000-4000-8000-000000000001/artifacts/mpk"
    assert manifest["provenance"]["execution_provider"] == "github-actions-devkit-runner"
    assert manifest["script"] == str(ROOT / "models" / model_id / "tests" / "benchmark.py")
    assert "build" not in manifest["runtime"]["devkit"]
    assert "benchmark" not in manifest
    assert "model_options" not in manifest
