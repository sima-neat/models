from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from model_registry.accuracy_result import (
    ACCURACY_RESULT_SCHEMA,
    accuracy_measurements,
    compiled_artifacts,
    load_accuracy_result,
)
from model_registry.cli import main
from model_registry.errors import ValidationError

ROOT = Path(__file__).resolve().parents[1]
GIT_SHA = "a" * 40


def compiled_build(tmp_path: Path, components: tuple[str, ...] = ("backbone", "transformer")) -> tuple[Path, dict]:
    """Write an RF-DETR Small compilation with one MPK per component and its compile-outputs.json."""
    compiled = tmp_path / "compiled"
    build = compiled / "rf_detr_small" / "modalix_int8_bf16"
    outputs = []
    digests = {}
    for component_id in components:
        archive = build / component_id / f"rf_detr_small_{component_id}_mpk.tar.gz"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(f"{component_id} mpk".encode())
        digests[component_id] = hashlib.sha256(archive.read_bytes()).hexdigest()
        outputs.append(
            {
                "component_id": component_id,
                "mpk_path": f"{component_id}/{archive.name}",
                "mpk_sha256": digests[component_id],
            }
        )
    (build / "compile-outputs.json").write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    return compiled, digests


def script_output(tmp_path: Path, metrics: dict[str, float]) -> Path:
    path = tmp_path / "script-output.json"
    output = {
        "core_version": "3.0.0",
        "dataset": {"split": "val2017", "evaluated_samples": 5000, "available_samples": 5000},
        "metrics": {name: {"value": value, "unit": "ratio", "direction": "higher"} for name, value in metrics.items()},
        "details": {"inference_seconds": 187.7},
    }
    path.write_text(json.dumps(output), encoding="utf-8")
    return path


def import_accuracy(tmp_path: Path, metrics: dict[str, float], git_sha: str = GIT_SHA) -> int:
    compiled, _ = compiled_build(tmp_path)
    return main(
        [
            "import-accuracy",
            "--repository", str(ROOT),
            "--model-id", "rf_detr_small",
            "--build-profile-id", "modalix_int8_bf16",
            "--script-output", str(script_output(tmp_path, metrics)),
            "--compiled-dir", str(compiled),
            "--git-sha", git_sha,
            "--output", str(tmp_path / "accuracy-result.json"),
        ]
    )  # fmt: skip


def test_import_accuracy_records_the_entry_dataset_and_compiled_mpks(tmp_path: Path) -> None:
    assert import_accuracy(tmp_path, {"coco_map": 0.4694, "coco_ap50": 0.6537}) == 0

    report = load_accuracy_result(tmp_path / "accuracy-result.json", ROOT / ACCURACY_RESULT_SCHEMA)
    _, digests = compiled_build(tmp_path / "expected")
    assert report["model"]["component_id"] == "transformer"
    assert report["model"]["package_id"] == "default"
    assert report["model"]["artifacts"] == digests
    assert report["validation"] == {"id": "coco_accuracy", "engine": "devkit_script", "status": "ready"}
    assert report["dataset"]["sha256"] == "a40de1caf42b47c5a8976eb97efe3c83f3a41337c4d16d318eb15d65c293f2da"
    assert report["execution"]["models_commit"] == GIT_SHA
    script = ROOT / "models" / "rf_detr_small" / "scripts" / "accuracy.py"
    assert report["execution"]["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    names = [item["name"] for item in accuracy_measurements(report, "f" * 64)]
    assert names == ["model.accuracy.coco_ap50", "model.accuracy.coco_map"]


def test_import_accuracy_attaches_the_published_score_as_each_metrics_reference(tmp_path: Path) -> None:
    assert import_accuracy(tmp_path, {"coco_map": 0.4694, "coco_ap50": 0.6537, "coco_ap75": 0.5017}) == 0

    metrics = load_accuracy_result(tmp_path / "accuracy-result.json", ROOT / ACCURACY_RESULT_SCHEMA)["metrics"]
    assert metrics["coco_map"] == {"value": 0.4694, "unit": "ratio", "direction": "higher", "reference": 0.530}
    assert metrics["coco_ap50"]["reference"] == 0.721
    assert "reference" not in metrics["coco_ap75"]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"coco_map": 0.4, "top1": 0.9}, "undeclared metrics: top1"),
        ({"coco_ap50": 0.6}, "did not measure required metrics: coco_map"),
    ],
)
def test_import_accuracy_rejects_metrics_outside_the_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], metrics: dict[str, float], expected: str
) -> None:
    assert import_accuracy(tmp_path, metrics) == 2
    assert expected in capsys.readouterr().err


def test_import_accuracy_rejects_a_git_sha_that_is_not_a_full_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert import_accuracy(tmp_path, {"coco_map": 0.4694}, git_sha="abc123") == 2
    assert "40 lowercase hexadecimal" in capsys.readouterr().err


def test_compiled_artifacts_returns_each_scoped_component_digest(tmp_path: Path) -> None:
    compiled, digests = compiled_build(tmp_path)

    artifacts = compiled_artifacts(compiled, "rf_detr_small", "modalix_int8_bf16", ["backbone", "transformer"])

    assert artifacts == digests


def test_compiled_artifacts_rejects_a_missing_component(tmp_path: Path) -> None:
    compiled, _ = compiled_build(tmp_path, components=("backbone",))

    with pytest.raises(ValidationError, match="exactly one compiled transformer MPK"):
        compiled_artifacts(compiled, "rf_detr_small", "modalix_int8_bf16", ["backbone", "transformer"])


def test_compiled_artifacts_rejects_an_mpk_that_does_not_match_its_recorded_sha256(tmp_path: Path) -> None:
    compiled, _ = compiled_build(tmp_path)
    archive = compiled / "rf_detr_small" / "modalix_int8_bf16" / "transformer" / "rf_detr_small_transformer_mpk.tar.gz"
    archive.write_bytes(b"rebuilt after the manifest was written")

    with pytest.raises(ValidationError, match="transformer MPK SHA-256 does not match"):
        compiled_artifacts(compiled, "rf_detr_small", "modalix_int8_bf16", ["backbone", "transformer"])


def test_compiled_artifacts_rejects_a_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="could not read compile outputs"):
        compiled_artifacts(tmp_path, "rf_detr_small", "modalix_int8_bf16", ["backbone"])


def test_resolve_devkit_accuracy_reports_models_without_it(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["resolve-devkit-accuracy", str(ROOT / "models" / "resnet_50"), "--build-profile", "modalix_bf16"]) == 0

    assert json.loads(capsys.readouterr().out)["kind"] == "none"
