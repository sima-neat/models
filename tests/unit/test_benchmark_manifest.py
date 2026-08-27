from __future__ import annotations

import json
from pathlib import Path

import pytest
from model_registry.benchmark_manifest import make_benchmark_manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("model_id", "variant_id"),
    [
        ("resnet_50", "modalix_bf16"),
        ("ssd_mobilenet_v1", "modalix_int8_tess_mla"),
    ],
)
def test_generates_direct_devkit_manifest_for_compiled_model(
    tmp_path: Path, model_id: str, variant_id: str
) -> None:
    variant = tmp_path / "compiled" / model_id / variant_id
    variant.mkdir(parents=True)
    (variant / "compile-result.json").write_text(
        json.dumps(
            {"archive": f"{model_id}_mpk.tar.gz", "sha256": "a" * 64, "provenance": {"compiler_version": "2.1.3"}}
        ),
        encoding="utf-8",
    )
    (variant / "registry-publication.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model_id": model_id,
                        "variant_id": variant_id,
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
        variant_id=variant_id,
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

    assert manifest["compatibility"] == {"platform_version": "2.1.3", "neat_version": "0.4.0"}
    assert manifest["model"]["id"] == model_id
    assert manifest["model"]["variant_id"] == variant_id
    assert manifest["model"]["path"].endswith(f"/{model_id}/{variant_id}/{model_id}_mpk.tar.gz")
    assert manifest["model"]["uri"] == "vulcan://runs/00000000-0000-4000-8000-000000000001/artifacts/mpk"
    assert manifest["provenance"]["execution_provider"] == "github-actions-devkit-runner"
