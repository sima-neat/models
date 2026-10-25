from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from model_registry.errors import ValidationError
from model_registry.fingerprints import benchmark_fingerprint
from model_registry.model import load_and_validate_model
from model_registry.registry import S3Object
from model_registry.report_publication import (
    _registered_builds,
    check_report,
    compiled_mpks,
    model_component_mpks,
    publish_report,
)
from model_report import METRICS, REPORT_FILE

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RUN = "00000000-0000-4000-8000-00000000000a"
BENCHMARK_RUN = "00000000-0000-4000-8000-00000000000b"
PERFORMANCE_METRICS = {
    "latency_avg_ms": 2.5, "throughput_fps": 400.0, "avg_power_watts": 8.0, "energy_per_inference_joules": 0.02,
}


class FakeStore:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def ensure_native_sha256(self, path: Path, key: str, sha256: str, *, media_type: str = "") -> S3Object:
        self.keys.append(key)
        return S3Object(key=key, version_id="version-1", sha256=sha256, size_bytes=path.stat().st_size)


def model(model_id: str, repository: Path = ROOT) -> dict:
    return load_and_validate_model(repository / "models" / model_id, repository / "schemas" / "model.schema.json")


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def registered_build(root: Path, model_id: str, profile_id: str, components: list[str]) -> dict[str, dict]:
    """A compiled profile with each component's compile receipt and the profile's package receipt."""
    profile_dir = root / model_id / profile_id
    builds, outputs = {}, []
    for index, component in enumerate(components):
        mpk = profile_dir / component / f"{component}_mpk.tar.gz"
        mpk.parent.mkdir(parents=True)
        mpk.write_bytes(component.encode())
        digest = hashlib.sha256(component.encode()).hexdigest()
        run_id = f"00000000-0000-4000-8000-{index + 1:012d}"
        outputs.append({"component_id": component, "mpk_path": f"{component}/{mpk.name}", "mpk_sha256": digest})
        write_json(profile_dir / component / "registry-publication.json",
                   {"runs": [{"component_id": component, "run_id": run_id}]})
        builds[component] = {"run_id": run_id, "artifact_sha256": digest}
    write_json(profile_dir / "compile-outputs.json", {"outputs": outputs})
    write_json(profile_dir / "packages" / "default" / "registry-publication.json", {"packages": [{
        "model_id": model_id, "build_profile_id": profile_id, "package_id": "default", "run_id": PACKAGE_RUN,
        "component_run_ids": {component: build["run_id"] for component, build in builds.items()},
    }]})
    return builds


def performance_report(components: list[str], **changes: object) -> dict:
    return {
        "schema": "sima.model-report", "schema_version": 1, "stage": "performance",
        "runtime": {"pyneat": "3.0.0", "build": {}}, "workload": "MPK default", "metrics": dict(PERFORMANCE_METRICS),
        "telemetry": {},
        "components": {component: {
            "execution_plan": {},
            "mla_layer_stats": {"status": "not requested", "layers": []},
            "mla_stage_stats": {"status": "not requested", "stages": []},
        } for component in components},
        **changes,
    }  # fmt: skip


def dataset_result(**changes: object) -> dict:
    return {
        "split": "val", "evaluated_samples": 500, "available_samples": 50000,
        "metrics": {"top1_accuracy": 0.77, "top5_accuracy": 0.93}, "details": {}, **changes,
    }  # fmt: skip


IMAGENET = "imagenet-2012-mini"


def accuracy_report(datasets: dict | None = None) -> dict:
    """ResNet-50's accuracy report, on its one dataset unless ``datasets`` names others."""
    return {
        "schema": "sima.model-report", "schema_version": 1, "stage": "accuracy",
        "runtime": {"pyneat": "3.0.0", "build": {}}, "datasets": datasets or {IMAGENET: dataset_result()},
    }  # fmt: skip


def test_model_component_mpks_name_each_components_verified_mpk_in_model_order(tmp_path: Path) -> None:
    builds = registered_build(tmp_path, "rf_detr_nano", "w8a8_w16a16", ["decoder", "backbone"])
    profile_dir = tmp_path / "rf_detr_nano" / "w8a8_w16a16"

    mpks = compiled_mpks(tmp_path, model("rf_detr_nano"), "w8a8_w16a16")

    assert list(mpks) == ["backbone", "decoder"]
    assert {component: mpk["sha256"] for component, mpk in mpks.items()} == {
        component: build["artifact_sha256"] for component, build in builds.items()
    }
    assert model_component_mpks(tmp_path, ROOT, "rf_detr_nano", "w8a8_w16a16") == [
        f"backbone={profile_dir / 'backbone' / 'backbone_mpk.tar.gz'}",
        f"decoder={profile_dir / 'decoder' / 'decoder_mpk.tar.gz'}",
    ]


def test_compiled_mpks_reject_an_mpk_that_does_not_match_its_recorded_digest(tmp_path: Path) -> None:
    registered_build(tmp_path, "rf_detr_nano", "w8a8_w16a16", ["backbone", "decoder"])
    (tmp_path / "rf_detr_nano" / "w8a8_w16a16" / "decoder" / "decoder_mpk.tar.gz").write_bytes(b"rebuilt")

    with pytest.raises(ValidationError, match="MPK SHA-256 does not match compile-outputs.json"):
        compiled_mpks(tmp_path, model("rf_detr_nano"), "w8a8_w16a16")


def test_registered_builds_require_the_package_to_link_the_benchmarked_builds(tmp_path: Path) -> None:
    builds = registered_build(tmp_path, "rf_detr_nano", "w8a8_w16a16", ["backbone", "decoder"])
    assert _registered_builds(tmp_path, model("rf_detr_nano"), "w8a8_w16a16") == (builds, PACKAGE_RUN)

    receipt = tmp_path / "rf_detr_nano" / "w8a8_w16a16" / "packages" / "default" / "registry-publication.json"
    document = json.loads(receipt.read_text(encoding="utf-8"))
    # A package published from an earlier backbone build does not cover this benchmark.
    document["packages"][0]["component_run_ids"]["backbone"] = "00000000-0000-4000-8000-000000000099"
    write_json(receipt, document)

    with pytest.raises(ValidationError, match="the package does not link the benchmarked component builds"):
        _registered_builds(tmp_path, model("rf_detr_nano"), "w8a8_w16a16")


@pytest.mark.parametrize(
    ("model_id", "profile_id", "stage", "report", "message"),
    [
        ("resnet_50", "w8a8", "accuracy", performance_report(["model"]), "is a performance report, not accuracy"),
        ("resnet_50", "w8a8", "performance", performance_report(["model"], metrics={**PERFORMANCE_METRICS,
         "top1_accuracy": 0.7}), "unknown performance metrics: top1_accuracy"),
        ("resnet_50", "w8a8", "performance", performance_report(["model"], metrics={"latency_avg_ms": 2.5}),
         "required performance metrics were not measured: throughput_fps, avg_power_watts"),
        ("resnet_50", "w8a8", "accuracy", accuracy_report({IMAGENET: dataset_result(metrics={"top5_accuracy": 0.93})}),
         f"{IMAGENET}: required accuracy metrics were not measured: top1_accuracy"),
        ("resnet_50", "w8a8", "accuracy", accuracy_report({"other": dataset_result()}),
         rf"must give exactly the datasets \['{IMAGENET}'\]; missing \['{IMAGENET}'\], undeclared \['other'\]"),
        ("resnet_50", "w8a8", "accuracy", accuracy_report({IMAGENET: dataset_result(), "other": dataset_result()}),
         r"missing \[\], undeclared \['other'\]"),
        ("resnet_50", "w8a8", "accuracy", {**accuracy_report(), "metrics": {"top1_accuracy": 0.77}},
         "is not valid under any of the given schemas"),
        ("rf_detr_nano", "w8a8_w16a16", "performance", performance_report(["decoder", "backbone"]),
         r"must describe the components \['backbone', 'decoder'\], in order"),
        # The BF16 build declares no accuracy qualification.
        ("resnet_50", "w16a16", "accuracy", accuracy_report(), "does not measure accuracy of w16a16"),
    ],
)  # fmt: skip
def test_check_report_holds_the_report_to_model_yaml(
    tmp_path: Path, model_id: str, profile_id: str, stage: str, report: dict, message: str
) -> None:
    write_json(tmp_path / REPORT_FILE, report)

    with pytest.raises(ValidationError, match=message):
        check_report(tmp_path / REPORT_FILE, ROOT, model_id, profile_id, stage)


def publish(
    tmp_path: Path, model_id: str, profile_id: str, stage: str, report: dict, repository: Path = ROOT
) -> tuple[list, dict]:
    output = tmp_path / f"_{stage}"
    write_json(output / REPORT_FILE, report)
    write_json(output / "environment.json", {
        "board_model": "Modalix", "kernel_release": "6.1", "machine": "aarch64", "cpu_count": 8,
        "cpu_governor": "userspace", "mla_frequency_hz": 1000000000, "cpu_frequency_khz": 1400000,
    })  # fmt: skip
    (output / f"{stage}.log").write_text("measured\n", encoding="utf-8")
    calls = []

    def request(method: str, path: str, body: dict | None, key: str) -> dict:
        calls.append((method, path, body))
        if path == "/v1/runs":
            return {"run_id": BENCHMARK_RUN}
        return {"artifact_id": f"artifact-{len(calls)}"} if "/artifacts/" in path else {}

    result = publish_report(
        output_dir=output, stage=stage, repository=repository, compiled_dir=tmp_path / "_compiled", model_id=model_id,
        build_profile_id=profile_id, api_url="https://writer.example.invalid", aws_region="us-west-2",
        bucket="model-artifacts", environment="staging", branch="feat/reports", git_sha="a" * 40,
        test_status="failed", request=request, artifact_store=FakeStore(),
    )  # fmt: skip
    return calls, result


def test_publish_report_publishes_one_whole_model_run_under_the_package_run(tmp_path: Path) -> None:
    builds = registered_build(tmp_path / "_compiled", "rf_detr_nano", "w8a8_w16a16", ["backbone", "decoder"])
    report = performance_report(["backbone", "decoder"])

    calls, result = publish(tmp_path, "rf_detr_nano", "w8a8_w16a16", "performance", report)

    fingerprint = benchmark_fingerprint(ROOT, model("rf_detr_nano"), "w8a8_w16a16", builds, "performance")
    paths = [path for _, path, _ in calls]
    create = calls[0][2]
    assert create["run_type"] == "model_performance"
    assert create["parent_run_id"] == PACKAGE_RUN
    assert create["run_fingerprint"] == fingerprint
    assert result["benchmark_fingerprint"] == fingerprint and result["package_run_id"] == PACKAGE_RUN
    # Each component's compile run is an input of the one whole-model run.
    inputs = {path.rsplit("/", 1)[1]: body for _, path, body in calls if "/inputs/" in path}
    assert {name: body["source_reference"] for name, body in inputs.items()} == {
        f"compiled-model-{component}": f"vulcan://runs/{build['run_id']}/artifacts/mpk"
        for component, build in builds.items()
    }
    assert {body["sha256"] for body in inputs.values()} == {build["artifact_sha256"] for build in builds.values()}
    test_path = f"/v1/runs/{BENCHMARK_RUN}/test-results/performance/model"
    test_result = next(body for _, path, body in calls if path == test_path)
    assert test_result["status"] == "failed"
    assert test_result["summary"] == report
    assert test_result["report_artifact_id"] and test_result["log_artifact_id"]
    measurements = next(body for _, path, body in calls if path.endswith("/measurements/batch"))["measurements"]
    assert {item["name"]: item["value"] for item in measurements} == {
        METRICS[name]["registry_name"]: value for name, value in PERFORMANCE_METRICS.items()
    }
    assert all(
        item["measurement_key"].startswith(f"{fingerprint}-") and item["sample_count"] == 1 for item in measurements
    )
    assert paths[-1] == f"/v1/runs/{BENCHMARK_RUN}/finalize"


def test_publish_report_counts_each_evaluated_accuracy_sample(tmp_path: Path) -> None:
    registered_build(tmp_path / "_compiled", "resnet_50", "w8a8", ["model"])

    calls, result = publish(tmp_path, "resnet_50", "w8a8", "accuracy", accuracy_report())

    assert calls[0][2]["run_type"] == result["run_type"] == "model_accuracy"
    measurements = next(body for _, path, body in calls if path.endswith("/measurements/batch"))["measurements"]
    assert sorted(item["name"] for item in measurements) == [
        "model.accuracy.top1_accuracy", "model.accuracy.top5_accuracy",
    ]
    assert {item["sample_count"] for item in measurements} == {500}


def test_publish_report_measures_each_dataset_apart(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for relative in ("schemas", "taxonomies", "datasets", "deps", "scripts", "tools", ".github/workflows",
                     "models/resnet_50"):
        shutil.copytree(ROOT / relative, repository / relative, ignore=shutil.ignore_patterns("__pycache__"))
    model_dir = repository / "models" / "resnet_50"
    authored = yaml.safe_load((model_dir / "model.yaml").read_text(encoding="utf-8"))
    authored["accuracy"]["datasets"].append({**authored["accuracy"]["datasets"][0], "variant": "small"})
    (model_dir / "model.yaml").write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    tests = yaml.safe_load((model_dir / "tests" / "tests.yaml").read_text(encoding="utf-8"))
    for expected in tests["accuracy"]["expected"].values():
        expected["imagenet-2012-small"] = expected[IMAGENET]
    (model_dir / "tests" / "tests.yaml").write_text(yaml.safe_dump(tests), encoding="utf-8")
    builds = registered_build(tmp_path / "_compiled", "resnet_50", "w8a8", ["model"])
    report = accuracy_report({
        IMAGENET: dataset_result(),
        "imagenet-2012-small": dataset_result(evaluated_samples=100, metrics={"top1_accuracy": 0.7}),
    })

    calls, result = publish(tmp_path, "resnet_50", "w8a8", "accuracy", report, repository)

    fingerprint = benchmark_fingerprint(repository, model("resnet_50", repository), "w8a8", builds, "accuracy")
    assert [body["run_type"] for _, path, body in calls if path == "/v1/runs"] == ["model_accuracy"]
    assert result["benchmark_fingerprint"] == fingerprint
    measurements = next(body for _, path, body in calls if path.endswith("/measurements/batch"))["measurements"]
    subject = {"model_id": "resnet_50", "build_profile_id": "w8a8"}
    assert sorted(
        (item["measurement_key"], item["name"], item["sample_count"], tuple(sorted(item["dimensions"].items())))
        for item in measurements
    ) == sorted([
        (f"{fingerprint}-{IMAGENET}-top1_accuracy", "model.accuracy.top1_accuracy", 500,
         tuple(sorted({**subject, "dataset": IMAGENET}.items()))),
        (f"{fingerprint}-{IMAGENET}-top5_accuracy", "model.accuracy.top5_accuracy", 500,
         tuple(sorted({**subject, "dataset": IMAGENET}.items()))),
        (f"{fingerprint}-imagenet-2012-small-top1_accuracy", "model.accuracy.top1_accuracy", 100,
         tuple(sorted({**subject, "dataset": "imagenet-2012-small"}.items()))),
    ])
