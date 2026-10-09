from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from model_registry.accuracy_result import ACCURACY_RESULT_SCHEMA, accuracy_measurements, load_accuracy_result
from model_registry.cli import main

ROOT = Path(__file__).resolve().parents[2]
MPK_SHA256 = {"backbone": "b" * 64, "transformer": "c" * 64}


def benchmark_root(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark"
    for component_id, sha256 in MPK_SHA256.items():
        (root / component_id).mkdir(parents=True)
        manifest = {"model": {"sha256": sha256}, "provenance": {"models_commit": "a" * 40}}
        (root / component_id / "performance-test.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


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


def import_accuracy(tmp_path: Path, metrics: dict[str, float]) -> int:
    return main(
        [
            "import-accuracy",
            "--repository", str(ROOT),
            "--model-id", "rf_detr_small",
            "--build-profile-id", "modalix_int8_bf16",
            "--script-output", str(script_output(tmp_path, metrics)),
            "--result-root", str(benchmark_root(tmp_path)),
            "--output", str(tmp_path / "accuracy-result-v2.json"),
        ]
    )  # fmt: skip


def test_import_accuracy_records_the_entry_dataset_and_benchmarked_mpks(tmp_path: Path) -> None:
    assert import_accuracy(tmp_path, {"coco_map": 0.4694, "coco_ap50": 0.6537}) == 0

    report = load_accuracy_result(tmp_path / "accuracy-result-v2.json", ROOT / ACCURACY_RESULT_SCHEMA)
    assert report["model"]["component_id"] == "transformer"
    assert report["model"]["package_id"] == "default"
    assert report["model"]["artifacts"] == MPK_SHA256
    assert report["validation"] == {"id": "coco_accuracy", "engine": "devkit_script", "status": "ready"}
    assert report["dataset"]["sha256"] == "abb616d0eaf3d55d98d44ed092e05e2df6f87264d7f8426550d62402ef222e78"
    assert report["execution"]["models_commit"] == "a" * 40
    script = ROOT / "models" / "rf_detr_small" / "scripts" / "accuracy.py"
    assert report["execution"]["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    names = [item["name"] for item in accuracy_measurements(report, "f" * 64)]
    assert names == ["model.accuracy.coco_ap50", "model.accuracy.coco_map"]


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


def test_resolve_devkit_accuracy_reports_models_without_it(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["resolve-devkit-accuracy", str(ROOT / "models" / "resnet_50"), "--build-profile", "modalix_bf16"]) == 0

    assert json.loads(capsys.readouterr().out)["kind"] == "none"
