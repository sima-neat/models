from __future__ import annotations

import json
from pathlib import Path

from model_registry.benchmark_manifest import make_resnet50_benchmark_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_generates_direct_devkit_resnet50_manifest(tmp_path: Path) -> None:
    variant = tmp_path / "compiled" / "resnet_50" / "modalix_bf16"
    variant.mkdir(parents=True)
    (variant / "compile-result.json").write_text(
        json.dumps(
            {"archive": "resnet_50_mpk.tar.gz", "sha256": "a" * 64, "provenance": {"compiler_version": "2.1.3"}}
        ),
        encoding="utf-8",
    )
    (variant / "registry-publication.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model_id": "resnet_50",
                        "variant_id": "modalix_bf16",
                        "run_id": "00000000-0000-4000-8000-000000000001",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = make_resnet50_benchmark_manifest(
        repository=ROOT,
        compiled_dir=tmp_path / "compiled",
        output_path=tmp_path / "performance-test.json",
        branch="feature/benchmark",
        git_sha="b" * 40,
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))

    assert manifest["compatibility"] == {"platform_version": "2.1.3", "neat_version": "0.4.0"}
    assert manifest["model"]["path"].endswith("/resnet_50/modalix_bf16/resnet_50_mpk.tar.gz")
    assert manifest["model"]["uri"] == "vulcan://runs/00000000-0000-4000-8000-000000000001/artifacts/mpk"
    assert manifest["provenance"]["execution_provider"] == "github-actions-devkit-runner"
