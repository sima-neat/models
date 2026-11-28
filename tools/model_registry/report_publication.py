"""Check a build profile's model report on the DevKit and publish it as one whole-model benchmark run.

scripts/performance.py and scripts/accuracy.py write model-report.json with tools/model_report. On the board,
check_report() holds the report to model.yaml and records the board it ran on. publish_report() then publishes
it unchanged under the profile's package run: the report, Core's evidence and the log as artifacts, the model's
test outcome, and one measurement per metric, per dataset for accuracy. The run fingerprint is the stage's
benchmark fingerprint, so a registry run with the same fingerprint means the stage need not run again.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator
from model_report import CORE_REPORT_FILE, METRICS, REPORT_FILE, REQUIRED_PERFORMANCE_METRICS, accuracy_value_error

from .benchmark_declarations import benchmark_runner, benchmarked_profiles
from .dependencies import catalog_inputs, resolve_benchmark_runtime
from .errors import ValidationError
from .fetch import sha256_file
from .fingerprints import BENCHMARK_RUN_TYPES, benchmark_fingerprint
from .model import build_profile_for, load_and_validate_model
from .model_tests import require_test_outcome
from .registry import S3ArtifactStore, SigV4RegistryClient, canonical_sha256

JsonObject = dict[str, Any]
Request = Callable[[str, str, JsonObject | None, str], JsonObject]

ENVIRONMENT_FILE = "environment.json"
# The publish-package receipt the compile workflow, or a registry restore, leaves beside the profile's components.
PACKAGE_RECEIPT = "packages/default/registry-publication.json"
# run_model_performance.sh sets the MLA and CPU clocks before measuring; the environment records the clocks it ran at.
MLA_FREQUENCY = Path("/sys/class/devfreq/5000000.mla/cur_freq")
CPU_FREQUENCY = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")
CPU_GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def _model(root: Path, model_id: str) -> JsonObject:
    return load_and_validate_model(root / "models" / model_id, root / "schemas" / "model.schema.json")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {label} {path}: {exc}") from exc


def compiled_outputs(compiled_dir: str | Path, model: JsonObject, build_profile_id: str) -> dict[str, JsonObject]:
    """Each component's archive or native manifest path and SHA-256, as compile-outputs.json records them."""
    profile_dir = Path(compiled_dir).expanduser().resolve() / model["id"] / build_profile_id
    outputs = _read_json(profile_dir / "compile-outputs.json", "compile outputs").get("outputs", [])
    recorded = {}
    for component in model["components"]:
        matches = [item for item in outputs if isinstance(item, dict) and item.get("component_id") == component["id"]]
        native = component["interface"]["format"] == "huggingface"
        path_field, sha_field = ("artifact_path", "artifact_sha256") if native else ("mpk_path", "mpk_sha256")
        if len(matches) != 1 or path_field not in matches[0]:
            raise ValidationError(f"{profile_dir}: the benchmark needs exactly one compiled {component['id']} artifact")
        digest = matches[0].get(sha_field)
        # The benchmark fingerprint and the published run name the build by this digest.
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValidationError(
                f"{profile_dir}/compile-outputs.json: {sha_field} of {component['id']} must be a lowercase SHA-256 "
                f"digest, not {digest!r}"
            )
        recorded[component["id"]] = {"path": profile_dir / matches[0][path_field], "sha256": digest}
    return recorded


def compiled_components(compiled_dir: str | Path, model: JsonObject, build_profile_id: str) -> dict[str, JsonObject]:
    """Each compiled component archive or native manifest, checked against compile-outputs.json."""
    components = compiled_outputs(compiled_dir, model, build_profile_id)
    native = model["components"][0]["interface"]["format"] == "huggingface"
    for output in components.values():
        if sha256_file(output["path"])[0] != output["sha256"]:
            label = "artifact" if native else "MPK"
            raise ValidationError(f"{output['path']}: {label} SHA-256 does not match compile-outputs.json")
    return components


def compiled_mpks(compiled_dir: str | Path, model: JsonObject, build_profile_id: str) -> dict[str, JsonObject]:
    """Each verified MPK for a Core benchmark."""
    if any(component["interface"]["format"] != "onnx" for component in model["components"]):
        raise ValidationError("model-component-mpks requires ONNX components")
    return compiled_components(compiled_dir, model, build_profile_id)


def model_component_mpks(
    compiled_dir: str | Path, repository: str | Path, model_id: str, build_profile_id: str
) -> list[str]:
    """``component=path`` for each MPK the model's script runs, in model.yaml order."""
    mpks = compiled_mpks(compiled_dir, _model(Path(repository).expanduser().resolve(), model_id), build_profile_id)
    return [f"{component}={mpk['path']}" for component, mpk in mpks.items()]


def model_component_artifacts(
    compiled_dir: str | Path, repository: str | Path, model_id: str, build_profile_id: str
) -> list[str]:
    """``component=path`` for each verified MPK or native deployment manifest."""
    components = compiled_components(
        compiled_dir, _model(Path(repository).expanduser().resolve(), model_id), build_profile_id
    )
    return [f"{component}={artifact['path']}" for component, artifact in components.items()]


def _registered_builds(
    compiled_dir: Path, model: JsonObject, build_profile_id: str
) -> tuple[dict[str, JsonObject], str]:
    """Each component's compile run and MPK digest, and the profile's package run, from the staged receipts and
    compile-outputs.json alone: publication never holds the build's files."""
    profile_dir = compiled_dir / model["id"] / build_profile_id
    builds = {}
    for component_id, mpk in compiled_outputs(compiled_dir, model, build_profile_id).items():
        path = profile_dir / component_id / "registry-publication.json"
        runs = _read_json(path, "compile publication").get("runs", [])
        if len(runs) != 1 or runs[0].get("component_id") != component_id:
            raise ValidationError(f"{path}: the publication does not belong to {component_id}")
        builds[component_id] = {"run_id": _run_id(runs[0].get("run_id"), path), "artifact_sha256": mpk["sha256"]}
    path = profile_dir / PACKAGE_RECEIPT
    packages = [
        item
        for item in _read_json(path, "package publication").get("packages", [])
        if (item.get("model_id"), item.get("build_profile_id"), item.get("package_id"))
        == (model["id"], build_profile_id, "default")
    ]
    if len(packages) != 1:
        raise ValidationError(f"{path}: a benchmark needs the profile's published package")
    if packages[0].get("component_run_ids") != {component: build["run_id"] for component, build in builds.items()}:
        raise ValidationError(f"{path}: the package does not link the benchmarked component builds")
    return builds, _run_id(packages[0].get("run_id"), path)


def _run_id(value: Any, path: Path) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError as exc:
        raise ValidationError(f"{path}: run ID must be a UUID") from exc


def runner_environment(*, native: bool = False) -> JsonObject:
    """The DevKit's installation and clocks, without taking performance measurements."""
    # Core installs the Neat runtime as Scheduler packages named simaai-*.
    patterns = ["sima-lmm-*"] if native else ["*neat*", "simaai-*"]
    packages = subprocess.run(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n", *patterns],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    installed = {}
    for line in packages.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0] == "ii " and fields[2].strip():
            installed[fields[1]] = fields[2]
    return {
        "buildinfo": Path("/etc/buildinfo").read_text(encoding="utf-8"),
        "packages": installed,
        "board_model": "Modalix",
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "cpu_governor": CPU_GOVERNOR.read_text(encoding="utf-8").strip() if CPU_GOVERNOR.exists() else None,
        "mla_frequency_hz": int(MLA_FREQUENCY.read_text(encoding="utf-8")) if MLA_FREQUENCY.exists() else None,
        "cpu_frequency_khz": int(CPU_FREQUENCY.read_text(encoding="utf-8")) if CPU_FREQUENCY.exists() else None,
        "runner": os.environ.get("RUNNER_NAME", platform.node()),
    }


def check_report(
    path: str | Path, repository: str | Path, model_id: str, build_profile_id: str, stage: str
) -> JsonObject:
    """Read model-report.json and require what model.yaml declares of the profile's stage."""
    root = Path(repository).expanduser().resolve()
    report_path = Path(path).expanduser().resolve()
    model = _model(root, model_id)
    if build_profile_id not in benchmarked_profiles(model, stage):
        raise ValidationError(f"{model_id} does not measure {stage} of {build_profile_id}")
    report = _read_json(report_path, "model report")
    schema = _read_json(root / "schemas" / "model-report.schema.json", "schema")
    errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.absolute_path))
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "report"
        raise ValidationError(f"{report_path}: {location}: {errors[0].message}")
    if report["stage"] != stage:
        raise ValidationError(f"{report_path}: the report is a {report['stage']} report, not {stage}")
    if stage == "accuracy":
        datasets = model["accuracy"]["datasets"]
        missing = [name for name in datasets if name not in report["datasets"]]
        undeclared = sorted(set(report["datasets"]) - set(datasets))
        if missing or undeclared:
            raise ValidationError(
                f"{report_path}: the report must give exactly the datasets {list(datasets)}; "
                f"missing {missing}, undeclared {undeclared}"
            )
        for name, dataset in datasets.items():
            measured = report["datasets"][name]["metrics"]
            _check_metrics(f"{report_path}: {name}", stage, measured, dataset["required_metrics"])
        return report
    # A speech model is packaged like a language model, and PyNeat measures it, so it reports like a vision model.
    generates_text = benchmark_runner(model) == "mole"
    if generates_text:
        from model_report import telemetry_metrics
        from model_report.generation_benchmark import validate_report

        if "generation" not in report:
            raise ValidationError(f"{report_path}: a text generation performance report needs generation workloads")
        try:
            validate_report(report["generation"])
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        for workload in report["generation"]["workloads"]:
            _check_metrics(str(report_path), stage,
                           {name: metric["value"] for name, metric in workload["metrics"].items()},
                           ["ttft_ms", "prefill_tps", "decode_tps"] if workload["kind"] in ("text", "text_dataset")
                           else ["ttft_ms", "decode_tps"])
            _check_metrics(str(report_path), stage, telemetry_metrics(workload.get("telemetry", {})),
                           ["avg_power_watts"])
            if "baseline" in workload:
                _check_metrics(str(report_path), stage, telemetry_metrics(workload["baseline"].get("telemetry", {})),
                               ["avg_power_watts"])
        if report["generation"]["settings"].get("decoding") == "speculative":
            declared = build_profile_for(model, build_profile_id).get("benchmark_baseline")
            baseline = report["generation"]["baseline"]
            if not declared or (baseline["model_id"], baseline["build_profile_id"]) != (
                declared["model_id"], declared["build_profile"]
            ):
                raise ValidationError("generation report does not use the declared benchmark baseline")
        return report
    if "generation" in report:
        raise ValidationError(f"{report_path}: generation workloads require a model that generates text")
    scope = model["performance"].get("scope", "model")
    if report.get("scope", "model") != scope:
        raise ValidationError(f"{report_path}: performance scope must be {scope}")
    if scope == "components":
        for name, component in report["components"].items():
            _check_metrics(f"{report_path}: {name}", stage, component["metrics"], REQUIRED_PERFORMANCE_METRICS)
    else:
        _check_metrics(str(report_path), stage, report["metrics"], REQUIRED_PERFORMANCE_METRICS)
    components = [component["id"] for component in model["components"]]
    if list(report["components"]) != components:
        raise ValidationError(f"{report_path}: the report must describe the components {components}, in order")
    return report


def _check_metrics(label: str, stage: str, metrics: dict[str, float], required: list[str]) -> None:
    unknown = sorted(name for name in metrics if METRICS.get(name, {}).get("stage") != stage)
    if unknown:
        raise ValidationError(f"{label}: unknown {stage} metrics: {', '.join(unknown)}")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValidationError(f"{label}: required {stage} metrics were not measured: {', '.join(missing)}")
    if stage == "accuracy":
        # The schema cannot bound a score by its metric's unit, so a report written without from_accuracy() stops here.
        errors = [error for name, value in metrics.items() if (error := accuracy_value_error(name, value))]
        if errors:
            raise ValidationError(f"{label}: {'; '.join(errors)}")


def put_run_file(
    client: SigV4RegistryClient,
    store: S3ArtifactStore,
    path: Path,
    *,
    run_id: str,
    name: str,
    artifact_type: str,
    media_type: str,
    bucket: str,
    s3_key: str,
    metadata: JsonObject,
    stable_key: str,
) -> JsonObject:
    """Store the file under its SHA-256 and register it as the run's artifact `name`."""
    stored = store.ensure_native_sha256(path, s3_key, sha256_file(path)[0], media_type=media_type)
    return client.request(
        "PUT",
        f"/v1/runs/{run_id}/artifacts/{name}",
        {
            "artifact_type": artifact_type,
            "s3_bucket": bucket,
            "s3_key": stored.key,
            "s3_version_id": stored.version_id,
            "sha256": stored.sha256,
            "size_bytes": stored.size_bytes,
            "media_type": media_type,
            "retention_class": "branch",
            "metadata": metadata,
        },
        f"{stable_key}-{name}",
    )


def _measurement(key: str, name: str, value: float, sample_count: int, dimensions: JsonObject) -> JsonObject:
    return {
        "measurement_key": key,
        "name": METRICS[name]["registry_name"],
        "unit": METRICS[name]["unit"],
        "value": value,
        "direction": METRICS[name]["direction"],
        "sample_count": sample_count,
        "aggregation": METRICS[name]["aggregation"],
        "dimensions": dict(dimensions),
    }


def publish_report(
    *,
    output_dir: str | Path,
    stage: str,
    repository: str | Path,
    compiled_dir: str | Path,
    model_id: str,
    build_profile_id: str,
    api_url: str,
    aws_region: str,
    bucket: str,
    environment: str,
    branch: str,
    git_sha: str,
    test_status: str,
    kms_key_id: str | None = None,
    github_run_url: str | None = None,
    request: Request | None = None,
    artifact_store: S3ArtifactStore | None = None,
) -> JsonObject:
    """Publish a profile's checked model report as a whole-model benchmark run under its package run.

    ``output_dir`` holds model-report.json, environment.json, any Core evidence, and ``<stage>.log``.
    The test result carries ``test_status``, the outcome of the model's tests on the report; the run still
    completes when they fail, so the stage is not measured again.
    """
    test_status = require_test_outcome(test_status)
    root = Path(repository).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    commit_sha = git_sha.lower()
    if len(commit_sha) != 40 or any(character not in "0123456789abcdef" for character in commit_sha):
        raise ValidationError("git SHA must contain exactly 40 hexadecimal characters")
    if not branch.strip():
        raise ValidationError("branch must not be empty")
    report = check_report(output / REPORT_FILE, root, model_id, build_profile_id, stage)
    model = _model(root, model_id)
    builds, package_run_id = _registered_builds(Path(compiled_dir).expanduser().resolve(), model, build_profile_id)
    baseline_builds = {}
    baseline = report.get("generation", {}).get("baseline")
    if baseline:
        reference = _model(root, baseline["model_id"])
        baseline_builds, _ = _registered_builds(Path(compiled_dir).resolve(), reference, baseline["build_profile_id"])
        if len(baseline_builds) != 1 or next(iter(baseline_builds.values())) != {
            "run_id": baseline["compile_run_id"], "artifact_sha256": baseline["artifact_sha256"]
        }:
            raise ValidationError("baseline report does not match its registered compilation")
    fingerprint = benchmark_fingerprint(root, model, build_profile_id, builds, stage)
    run_type = BENCHMARK_RUN_TYPES[stage]
    native = "generation" in report
    tool = "llima" if native else "pyneat"
    board = {**_read_json(output / ENVIRONMENT_FILE, "runner environment"), tool: report["runtime"][tool]}
    runtime = resolve_benchmark_runtime(root)
    inputs = catalog_inputs(root)
    report_digest = canonical_sha256(report)
    stable_digest = hashlib.sha256(f"{package_run_id}:{fingerprint}:{report_digest}".encode()).hexdigest()
    stable_key = f"models-{stage}-{stable_digest}"
    prefix = f"models/{quote(branch, safe='')}/{commit_sha[:12]}/{model_id}/{build_profile_id}/{stage}/{fingerprint}"
    subject = {"model_id": model_id, "build_profile_id": build_profile_id}
    client = SigV4RegistryClient(api_url, aws_region, request=request)
    store = artifact_store or S3ArtifactStore(bucket, kms_key_id=kms_key_id)

    create_body: JsonObject = {
        "run_type": run_type,
        "parent_run_id": package_run_id,
        "commit_sha": commit_sha,
        "git_ref": f"refs/heads/{branch}",
        "run_fingerprint": fingerprint,
        "environment": {
            "fingerprint": canonical_sha256(board),
            "toolchain_name": "sima-neat",
            "toolchain_version": report["runtime"][tool],
            "target_platform": runtime["devkit"]["platform_version"],
            "target_device": board["board_model"],
            "operating_system": board["kernel_release"],
            "properties": {
                "vulcan_environment": environment,
                **{
                    key: board[key]
                    for key in ("machine", "cpu_count", "cpu_governor", "mla_frequency_hz", "cpu_frequency_khz")
                },
                **({"catalog_inputs": inputs} if inputs else {}),
            },
        },
        "metadata": {**subject, "stage": stage, "package_run_id": package_run_id},
    }
    if github_run_url:
        create_body.update({"ci_provider": "github-actions", "ci_run_url": github_run_url})
    created = client.request("POST", "/v1/runs", create_body, f"{stable_key}-create")
    run_id = _run_id(created.get("run_id"), output / REPORT_FILE)

    for component_id, build in builds.items():
        name = f"compiled-model-{component_id.replace('_', '-')}"
        client.request(
            "PUT",
            f"/v1/runs/{run_id}/inputs/{name}",
            {
                "input_type": "model-pack",
                "source_reference": f"vulcan://runs/{build['run_id']}/artifacts/{'llima' if native else 'mpk'}",
                "sha256": build["artifact_sha256"],
                "media_type": "application/json" if native else "application/gzip",
                "metadata": {**subject, "component_id": component_id, "compile_run_id": build["run_id"]},
            },
            f"{stable_key}-{name}",
        )
    for component_id, build in baseline_builds.items():
        name = f"baseline-model-{component_id.replace('_', '-')}"
        client.request("PUT", f"/v1/runs/{run_id}/inputs/{name}", {
            "input_type": "model-pack",
            "source_reference": f"vulcan://runs/{build['run_id']}/artifacts/llima",
            "sha256": build["artifact_sha256"], "media_type": "application/json",
            "metadata": {"model_id": baseline["model_id"], "build_profile_id": baseline["build_profile_id"],
                         "component_id": component_id, "compile_run_id": build["run_id"]},
        }, f"{stable_key}-{name}")
    # Vulcan completes a run only once its canonical parameters are stored.
    parameters = {stage: model[stage], "builds": builds, "runtime": runtime}
    if baseline:
        parameters["benchmark_baseline"] = baseline
    client.request(
        "PUT",
        f"/v1/runs/{run_id}/parameters",
        {"parameters": parameters, "canonical_sha256": canonical_sha256(parameters)},
        f"{stable_key}-parameters",
    )

    files = {"model-report": (REPORT_FILE, "model-report", "application/json")}
    if (output / CORE_REPORT_FILE).is_file():
        files["llima-benchmark-report" if native else "core-benchmark-report"] = (
            CORE_REPORT_FILE, "benchmark-report", "application/json"
        )
    if report.get("scope") == "components":
        for component in report["components"]:
            filename = f"{component}-{CORE_REPORT_FILE}"
            if (output / filename).is_file():
                files[f"core-benchmark-report-{component.replace('_', '-')}"] = (
                    filename,
                    "benchmark-report",
                    "application/json",
                )
    if (output / f"{stage}.log").is_file():
        files[f"{stage}-log"] = (f"{stage}.log", "log", "text/plain")
    artifacts = {
        name: put_run_file(
            client,
            store,
            output / filename,
            run_id=run_id,
            name=name,
            artifact_type=artifact_type,
            media_type=media_type,
            bucket=bucket,
            s3_key=f"{prefix}/{report_digest}/{filename}",
            metadata={**subject, "stage": stage},
            stable_key=stable_key,
        )
        for name, (filename, artifact_type, media_type) in files.items()
    }
    summary = report
    if native:
        # Full-dataset token timings belong in the artifact, not an unbounded inline API response.
        def compact(workload):
            return {key: compact(value) if key == "baseline" else value
                    for key, value in workload.items() if key != "requests"}

        summary = {**report, "generation": {**report["generation"],
            "workloads": [compact(workload) for workload in report["generation"]["workloads"]]}}
    test_body: JsonObject = {
        "status": test_status,
        # The browser uses aggregates; Raw result retains every request in the registered artifact.
        "summary": summary,
        "report_artifact_id": artifacts["model-report"]["artifact_id"],
    }
    if f"{stage}-log" in artifacts:
        test_body["log_artifact_id"] = artifacts[f"{stage}-log"]["artifact_id"]
    client.request("PUT", f"/v1/runs/{run_id}/test-results/{stage}/model", test_body, f"{stable_key}-test-result")
    measurements = (
        [
            _measurement(
                f"{fingerprint}-{dataset}-{name}",
                name,
                value,
                result["evaluated_samples"],
                {**subject, "dataset": dataset},
            )
            for dataset, result in sorted(report["datasets"].items())
            for name, value in sorted(result["metrics"].items())
        ]
        if stage == "accuracy"
        else [
            _measurement(f"{fingerprint}-{component}-{name}", name, value, 1, {**subject, "component_id": component})
            for component, result in report["components"].items()
            for name, value in sorted(result["metrics"].items())
        ]
        if report.get("scope") == "components"
        else [
            _measurement(f"{fingerprint}-{name}", name, value, 1, subject)
            for name, value in sorted(report["metrics"].items())
        ]
    )
    if native:
        from model_report import telemetry_metrics

        measurements = []
        for workload in report["generation"]["workloads"]:
            scopes = [("target", workload)]
            if "baseline" in workload:
                scopes.append(("baseline", workload["baseline"]))
            for role, reading in scopes:
                dimensions = {**subject, "workload_id": workload["id"], "workload_kind": workload["kind"],
                              "role": role, "decoding": "ordinary" if role == "baseline" else
                              report["generation"]["settings"].get("decoding", "ordinary")}
                if reading["kind"] == "text":
                    dimensions["input_tokens"] = str(reading["input_tokens"])
                if "dataset" in reading:
                    dimensions["dataset"] = reading["dataset"]["name"]
                if role == "baseline":
                    dimensions.update(baseline_model_id=baseline["model_id"],
                                      baseline_build_profile_id=baseline["build_profile_id"])
                metrics = {name: metric["value"] for name, metric in reading["metrics"].items()}
                metrics.update(telemetry_metrics(reading.get("telemetry", {})))
                count = reading["metrics"]["decode_tps"]["sample_count"]
                measurements.extend(
                    _measurement(f"{fingerprint}-{workload['id']}-{role}-{name}", name, value, count, dimensions)
                    for name, value in sorted(metrics.items())
                )
    # check_report() requires the declared metrics, so a report always has at least one.
    client.request(
        "POST", f"/v1/runs/{run_id}/measurements/batch", {"measurements": measurements}, f"{stable_key}-measurements"
    )
    client.request("POST", f"/v1/runs/{run_id}/finalize", {"status": "completed"}, f"{stable_key}-finalize")
    return {
        "run_id": run_id,
        "run_type": run_type,
        "package_run_id": package_run_id,
        **subject,
        "benchmark_fingerprint": fingerprint,
        "test_status": test_status,
    }
