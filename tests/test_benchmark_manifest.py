from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from model_registry.benchmark_manifest import make_benchmark_manifest, validate_benchmark_components
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def compile_component(tmp_path: Path, model_id: str, build_profile_id: str, component_id: str, run_id: str) -> None:
    """Leave a component's compiled MPK and its registered compile run where the benchmark stage reads them."""
    component = tmp_path / "compiled" / model_id / build_profile_id / component_id
    component.mkdir(parents=True)
    archive = component / f"{model_id}_mpk.tar.gz"
    archive.write_bytes(f"compiled {component_id}".encode())
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
                        "run_id": run_id,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_dispatch_validates_complete_compile_but_benchmarks_only_declared_subset(tmp_path: Path) -> None:
    output_root = tmp_path / "rf_detr_small" / "w8a8_w16a16"
    outputs = []
    for component in ("backbone", "decoder"):
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
    fidelity = output_root / "decoder" / "fidelity.json"
    fidelity.write_text("{}")
    manifest = {
        "model_id": "rf_detr_small",
        "build_profile_id": "w8a8_w16a16",
        "compile_invocation_id": "rf_detr",
        "outputs": outputs,
        "derived_graphs": [
            {"component_id": component, "sha256": "a" * 64} for component in ("backbone", "decoder")
        ],
        "correctness": {
            "component_id": "decoder",
            "path": "decoder/fidelity.json",
            "sha256": hashlib.sha256(fidelity.read_bytes()).hexdigest(),
        },
    }
    (output_root / "compile-outputs.json").write_text(json.dumps(manifest))
    args = (ROOT, tmp_path, "rf_detr_small", "w8a8_w16a16", "rf_detr")

    # The package's benchmark runs under its last component.
    validate_benchmark_components(*args, ["decoder"])
    # An accuracy-only dispatch measures no component's performance.
    validate_benchmark_components(*args, [])
    for components in (["backbone"], ["backbone", "decoder"], ["decoder", "decoder"]):
        with pytest.raises(ValidationError, match="declared benchmark entries"):
            validate_benchmark_components(*args, components)
    (output_root / "backbone" / "backbone_mpk.tar.gz").unlink()
    with pytest.raises(ValidationError, match="MPK does not exist"):
        validate_benchmark_components(*args, ["decoder"])


@pytest.mark.parametrize(
    ("model_id", "build_profile_id", "component_id"),
    [
        ("resnet_50", "w16a16", "model"),
        ("ssd_mobilenet_v1", "w8a8", "model"),
        ("yolo26_tiny_drone", "w8a8", "model"),
    ],
)
def test_generates_direct_devkit_manifest_for_compiled_model(
    tmp_path: Path, model_id: str, build_profile_id: str, component_id: str
) -> None:
    compile_component(tmp_path, model_id, build_profile_id, component_id, "00000000-0000-4000-8000-000000000001")

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
    assert manifest["script"] == str(ROOT / "models" / model_id / "scripts/performance.py")
    assert "benchmark_options" not in manifest
    assert "build" not in manifest["runtime"]["devkit"]
    assert "benchmark" not in manifest
    assert "model_options" not in manifest


def test_a_package_manifest_lists_each_build_its_last_component_runs(tmp_path: Path) -> None:
    run_ids = {"backbone": "00000000-0000-4000-8000-000000000011", "decoder": "00000000-0000-4000-8000-000000000012"}
    for component_id, run_id in run_ids.items():
        compile_component(tmp_path, "rf_detr_small", "w8a8_w16a16", component_id, run_id)

    result = make_benchmark_manifest(
        repository=ROOT,
        compiled_dir=tmp_path / "compiled",
        output_path=tmp_path / "performance-test.json",
        branch="feature/benchmark",
        git_sha="b" * 40,
        model_id="rf_detr_small",
        build_profile_id="w8a8_w16a16",
        component_id="decoder",
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

    assert manifest["model"] == {
        "id": "rf_detr_small",
        "build_profile_id": "w8a8_w16a16",
        "component_id": None,
        "package_id": "default",
        "path": None,
        "sha256": None,
        "uri": None,
    }
    assert [(item["component_id"], item["compile_run_id"]) for item in manifest["components"]] == list(run_ids.items())
    assert all(Path(item["path"]).is_file() for item in manifest["components"])
    assert manifest["provenance"]["compile_run_id"] == run_ids["decoder"]
    assert manifest["script"] == str(ROOT / "models/rf_detr_small/scripts/performance.py")
    assert "benchmark_options" not in manifest
