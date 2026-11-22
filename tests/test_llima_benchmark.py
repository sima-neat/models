"""GenAI measurement boundaries and registered compilation reuse."""

import json
from pathlib import Path

import pytest
from model_registry.errors import ValidationError
from model_registry.report_publication import check_report
from model_report import from_generation
from model_report import generation_benchmark as bench
from model_report.board_telemetry import summarize

from tests.test_report_publication import publish, registered_build

ROOT = Path(__file__).resolve().parents[1]


def text_workload(length):
    requests = [
        {
            "phase": "warmup" if i == 0 else "measure",
            "input_tokens": length,
            "token_durations_s": [0.1, 0.02, 0.04],
            "generated_tokens": 3,
            "decode_tokens": 2,
            "ttft_s": 0.1,
            "decode_time_s": 0.06,
        }
        for i in range(6)
    ]
    workload = {"id": f"text_n{length}", "kind": "text", "input_tokens": length, "requests": requests}
    workload.update(telemetry=summarize({}), throughput_duration_ms=0.0)
    workload["metrics"] = bench.workload_metrics(workload)
    return workload


def report():
    workloads = [text_workload(length) for length in bench.input_buckets(4096)]
    image = {
        "id": "image",
        "kind": "image_text",
        "image_size": [448, 448],
        "telemetry": summarize({}),
        "throughput_duration_ms": 0.0,
        "requests": [
            {
                "phase": "warmup" if i == 0 else "measure",
                "ttft_s": 0.13,
                "input_tokens": 512,
                "token_durations_s": [0.13, 0.01, 0.02],
                "generated_tokens": 3,
                "decode_tokens": 2,
                "decode_time_s": 0.03,
            }
            for i in range(6)
        ],
    }
    image["metrics"] = bench.workload_metrics(image)
    return {
        "status": "passed",
        "settings": {"max_context": 4096, "image_size": [448, 448]},
        "runtime": {"mole_version": "0.6.0+test"},
        "timing": {},
        "workloads": [*workloads, image],
    }


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (1152, [128]),
        (4096, [128, 256, 512, 1024, 2048, 3072]),
        (8192, [128, 256, 512, 1024, 2048, 4096, 7168]),
    ],
)
def test_buckets_cover_headroom_boundary(context, expected):
    assert bench.input_buckets(context) == expected


def test_text_rate_uses_actual_decode_time_and_excludes_ttft():
    workload = text_workload(128)
    metrics = bench.workload_metrics(workload)
    assert metrics["ttft_ms"]["value"] == pytest.approx(100)
    assert metrics["prefill_tps"]["value"] == pytest.approx(1280)
    assert metrics["decode_tps"]["value"] == pytest.approx(2 / 0.06)
    assert metrics["decode_tps"]["sample_count"] == 5


@pytest.mark.parametrize("damage", ["missing_bucket", "missing_run", "nan", "counts", "summary", "size"])
def test_failed_or_inconsistent_measurements_are_not_publishable(damage):
    data = report()
    if damage == "missing_bucket":
        data["workloads"].pop(1)
    if damage == "missing_run":
        data["workloads"][0]["requests"].pop()
    if damage == "nan":
        data["workloads"][0]["requests"][1]["ttft_s"] = float("nan")
    if damage == "counts":
        data["workloads"][0]["requests"][1]["generated_tokens"] += 1
    if damage == "summary":
        data["workloads"][0]["metrics"]["decode_tps"]["value"] = 999
    if damage == "size":
        data["workloads"][-1]["image_size"] = [1024, 1024]
    with pytest.raises(ValueError):
        bench.validate_report(data)


def test_exact_prompt_lengths_preserve_native_assistant_suffix():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            assert kwargs["enable_thinking"] is False
            return [1000, *messages[0]["content"].encode(), 1001, 1002]

    for length in (128, 256, 3072):
        tokens = bench._prompt(Tokenizer(), length)
        assert len(tokens) == length and tokens[0] == 1000 and tokens[-2:] == [1001, 1002]


def test_text_client_uses_the_resolved_target_tokenizer(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    target = tmp_path / "target"
    (target / "devkit").mkdir(parents=True)
    (target / "devkit/vlm_config.json").write_text('{"target": true}')
    calls = []

    def resolve(deployment):
        assert deployment == tmp_path
        return target, tmp_path / "draft"

    def client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(zmq_socket=SimpleNamespace(close=lambda **_: None),
                               zmq_context=SimpleNamespace(term=lambda: None))

    monkeypatch.setitem(sys.modules, "sima_lmm.devkit.model_manager", SimpleNamespace(
        resolve_target_and_draft_paths=resolve))
    monkeypatch.setitem(sys.modules, "sima_lmm.mole.modalixlm", SimpleNamespace(ModalixLM=client))
    assert bench._config(tmp_path) == {"target": True}
    with bench._client("board", tmp_path, 4096):
        pass
    assert calls == [{"board": "board", "pretrained": str(target / "devkit"), "device": "cpu", "max_length": 4096}]


def generation_report():
    data = report()
    data["settings"].update(repetitions=5, warmup=1)
    for workload in data["workloads"]:
        workload["telemetry"] = summarize({
            "metrics": [{"key": "power_current_watts", "unit": "W"}],
            "samples": [{"values": {"power_current_watts": 14.0}}],
        })
    return from_generation(data, {"llima": "0.6.0", "mole": "0.6.0", "build": {}}).document


def test_generation_uses_the_unified_report_and_checks_every_workload(tmp_path):
    data = generation_report()
    path = tmp_path / "model-report.json"
    path.write_text(json.dumps(data))
    assert check_report(path, ROOT, "qwen3_5_0_8b", "w4a16", "performance") == data
    assert data["schema_version"] == 1
    data["generation"]["workloads"][-1]["telemetry"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="required performance metrics were not measured"):
        check_report(path, ROOT, "qwen3_5_0_8b", "w4a16", "performance")


def test_generation_publication_uses_native_inputs_and_keeps_workload_dimensions(tmp_path):
    import hashlib

    compiled = tmp_path / "_compiled"
    registered_build(compiled, "qwen3_5_0_8b", "w4a16", ["model"])
    profile = compiled / "qwen3_5_0_8b/w4a16"
    manifest = profile / "model/deployment-manifest.json"
    manifest.write_text("{}")
    (profile / "compile-outputs.json").write_text(json.dumps({"outputs": [{
        "component_id": "model", "artifact_path": "model/deployment-manifest.json",
        "artifact_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }]}))
    calls, result = publish(tmp_path, "qwen3_5_0_8b", "w4a16", "performance", generation_report())
    assert result["run_type"] == "model_performance"
    inputs = [body for _, path, body in calls if "/inputs/" in path]
    assert inputs[0]["source_reference"].endswith("/artifacts/llima")
    measurements = next(body["measurements"] for _, path, body in calls if path.endswith("/measurements/batch"))
    assert {row["dimensions"]["workload_id"] for row in measurements} == {w["id"] for w in report()["workloads"]}
    image = [row for row in measurements if row["dimensions"]["workload_id"] == "image"]
    assert all("input_tokens" not in row["dimensions"] for row in image)
    assert {row["name"] for row in image} == {
        "model.generation.ttft", "model.generation.decode_rate", "board.power.avg", "board.power.peak",
    }


def dataset_workload(name="gsm8k", *, image=False, ordinary=False):
    requests = []
    for index in (0, 0, 1, 2):
        durations = [0.1, 0.04, 0.08] if ordinary else [0.1, 0.02, 0.04]
        requests.append({
            "phase": "warmup" if not requests else "measure", "status": "ok",
            "sample_id": str(index), "input_tokens": (index + 1) * 32,
            "prompt_sha256": f"prompt-{index}", "output_budget": 256,
            "token_durations_s": durations, "ttft_s": durations[0],
            "generated_tokens": 3, "decode_tokens": 2, "decode_time_s": sum(durations[1:]),
            "accepted_draft_tokens": None if ordinary else 2,
        })
    workload = {
        "id": f"dataset_{name}", "kind": "image_dataset" if image else "text_dataset",
        "dataset": {"name": name, "available_samples": 3, "selection": "all",
                    "sources": [{"repository": "example/dataset", "revision": "a" * 40}]},
        "requests": requests,
        "telemetry": generation_report()["generation"]["workloads"][0]["telemetry"],
    }
    workload["metrics"] = bench.workload_metrics(workload)
    return workload


def test_dataset_comparison_weights_actual_tokens_and_keeps_zero_draft_contribution():
    workload = dataset_workload()
    workload["baseline"] = dataset_workload(ordinary=True)
    # Ordinary MoLE warmup includes the first token in addition to its decode budget.
    workload["baseline"]["requests"][0]["output_budget"] = 257
    metrics = bench.workload_metrics(workload)
    assert metrics["decode_speedup"]["value"] == pytest.approx(2)
    assert metrics["draft_contribution_pct"]["value"] == pytest.approx(200 / 3)
    assert metrics["decode_tps"]["sample_count"] == 3
    assert metrics["prefill_tps"]["value"] == pytest.approx((32 + 64 + 96) / 0.3)
    for row in workload["requests"]:
        row["accepted_draft_tokens"] = 0
    assert bench.workload_metrics(workload)["draft_contribution_pct"]["value"] == 0


@pytest.mark.parametrize("damage", ["prompt", "budget", "source", "counter", "missing_sample"])
def test_comparison_rejects_unmatched_or_speculative_baselines(damage):
    workload = dataset_workload()
    baseline = workload["baseline"] = dataset_workload(ordinary=True)
    if damage == "prompt":
        baseline["requests"][1]["prompt_sha256"] = "different"
    if damage == "budget":
        baseline["requests"][1]["output_budget"] = 128
    if damage == "source":
        baseline["dataset"]["sources"][0]["revision"] = "b" * 40
    if damage == "counter":
        baseline["requests"][1]["accepted_draft_tokens"] = 0
    if damage == "missing_sample":
        baseline["requests"].pop()
    baseline["metrics"] = bench.workload_metrics(baseline) if damage != "missing_sample" else baseline["metrics"]
    with pytest.raises(ValueError):
        bench.workload_metrics(workload)


def test_image_dataset_uses_native_decode_timings_without_a_text_prefill_rate():
    workload = dataset_workload("chartqa", image=True)
    metrics = bench.workload_metrics(workload)
    assert metrics["decode_tps"]["value"] == pytest.approx(2 / 0.06)
    assert "prefill_tps" not in metrics
    assert "draft_contribution_pct" in metrics


@pytest.mark.parametrize("target_tools,baseline_tools", [(True, True), (False, True), (True, False), (False, False)])
def test_bfcl_requires_native_tool_support_in_both_deployments(monkeypatch, tmp_path, target_tools, baseline_tools):
    import sys
    from types import SimpleNamespace

    support = {"target": target_tools, "baseline": baseline_tools}

    def tokenizer(path, *, local_files_only):
        assert local_files_only
        template = "{{ tools }}" if support[Path(path).parent.name] else "{{ messages }}"
        return SimpleNamespace(chat_template=template)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)))
    monkeypatch.setattr(bench, "_target", lambda path: path)
    names, skipped = bench._speculative_datasets(tmp_path / "target", tmp_path / "baseline", vision=True)
    unsupported = not (target_tools and baseline_tools)
    assert ("bfcl-v3" not in names) == unsupported
    assert set(names) == set((*bench.TEXT_DATASETS, *bench.IMAGE_DATASETS)) - set(skipped)
    assert set(skipped) == ({"bfcl-v3"} if unsupported else set())
    if skipped:
        assert skipped["bfcl-v3"] == f"{'baseline' if target_tools else 'target'} tokenizer has no native tool template"


def test_dataset_runner_reuses_mole_with_no_sample_limit_and_excludes_warmup(monkeypatch, tmp_path):
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    import numpy as np

    phases, limits = [], []

    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([1, 2, 3], dtype=np.int64)

    class Client:
        def _zmq_infer(self, *args):
            return None

    class Telemetry:
        def __init__(self, name):
            pass

        @contextmanager
        def measure(self, enabled=True):
            phases.append(enabled)
            yield

        def report(self):
            return {}

    @contextmanager
    def client(*args, **kwargs):
        yield Client()

    def run(lm, config):
        limits.append(config.limit)
        workload = dataset_workload(config.performance.dataset)
        for _ in workload["requests"]:
            lm._zmq_infer(Tensor(), "generate_for_perf")
        return {"requests": workload["requests"], "dataset": workload["dataset"], "evaluator": {}}

    monkeypatch.setitem(sys.modules, "sima_lmm.mole.perf_runner", SimpleNamespace(
        PerformanceOptions=lambda **kwargs: SimpleNamespace(**kwargs), run_performance=run))
    monkeypatch.setattr(bench, "_client", client)
    monkeypatch.setattr(bench, "BoardTelemetry", Telemetry)
    workloads = bench._datasets(None, tmp_path, tmp_path, 4096, bench.TEXT_DATASETS)
    assert limits == [None, None, None]
    assert phases == [False, True, True, True] * 3
    assert all(len(row["prompt_sha256"]) == 64 for workload in workloads for row in workload["requests"])


def test_declared_baseline_stages_its_deployment_and_records_the_compilation(monkeypatch, tmp_path):
    from model_registry import registry

    calls = []

    def stage(**kwargs):
        calls.append(kwargs)
        return {"count": 1, "outputs": [{"component_id": "model", "run_id": "ordinary-run",
                                         "artifact_sha256": "a" * 64}]}

    monkeypatch.setattr(registry, "stage_catalog_build", stage)
    registry.stage_benchmark_baseline(repository=ROOT, output_dir=tmp_path,
        model_id="gemma_4_e2b_it_textonly_mtp", build_profile_id="w4a16", api_url="https://example.invalid")
    assert calls[0]["model_id"] == "gemma_4_e2b_it_textonly"
    assert calls[0]["compile_invocation_id"] == "w4a16"
    receipt = json.loads((tmp_path / "benchmark-baseline.json").read_text())
    assert receipt["compile_run_id"] == "ordinary-run"
    assert receipt["deployment"] == "gemma_4_e2b_it_textonly/w4a16/model/deployment"


def test_speculative_target_and_its_baseline_download_by_compile_run(monkeypatch, tmp_path):
    import hashlib

    from model_registry import registry
    from model_registry.fetch import BuildFile, StagedBuildFiles, download_build

    api = "https://registry.example.invalid"
    reads, objects = {}, {}

    def staged_deployment(model_id, run_id, paths):
        """Stage a LLiMa deployment as stage-registry-build does: its file list, no file."""
        files = []
        for index, path in enumerate(paths):
            payload = f"{model_id}/{path}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            artifact = f"file-{index}"
            files.append(BuildFile(f"model/deployment/{path}", run_id, artifact, digest, len(payload)))
            reads[f"{api}/v1/runs/{run_id}/artifacts/{artifact}"] = {"artifact": {
                "sha256": digest, "size_bytes": len(payload), "download_url": f"https://objects.invalid/{digest}",
            }}
            objects[f"https://objects.invalid/{digest}"] = payload
        profile = tmp_path / model_id / "w4a16"
        profile.mkdir(parents=True)
        StagedBuildFiles(api, tuple(files)).write(profile)

    def stage(**kwargs):
        staged_deployment(kwargs["model_id"], "ordinary-run", ["devkit/vlm_config.json", "elf_files/language.elf"])
        return {"count": 1, "outputs": [{"component_id": "model", "run_id": "ordinary-run",
                                         "artifact_sha256": "a" * 64}]}

    # The target's single component holds the draft beside the target, so one file list covers both.
    staged_deployment("gemma_4_e2b_it_textonly_mtp", "target-run",
                      ["devkit/vlm_config.json", "elf_files/language.elf", "draft/elf_files/draft.elf"])
    monkeypatch.setattr(registry, "stage_catalog_build", stage)
    registry.stage_benchmark_baseline(repository=ROOT, output_dir=tmp_path,
        model_id="gemma_4_e2b_it_textonly_mtp", build_profile_id="w4a16", api_url=api)

    counts = download_build(tmp_path, "gemma_4_e2b_it_textonly_mtp", "w4a16", read=reads.__getitem__,
                            download=objects.__getitem__)

    assert counts == {"gemma_4_e2b_it_textonly_mtp/w4a16": 3, "gemma_4_e2b_it_textonly/w4a16": 2}
    target = tmp_path / "gemma_4_e2b_it_textonly_mtp/w4a16/model/deployment"
    draft = b"gemma_4_e2b_it_textonly_mtp/draft/elf_files/draft.elf"
    assert (target / "draft/elf_files/draft.elf").read_bytes() == draft
    receipt = json.loads((tmp_path / "benchmark-baseline.json").read_text())
    baseline = tmp_path / receipt["deployment"]
    assert (baseline / "elf_files/language.elf").read_bytes() == b"gemma_4_e2b_it_textonly/elf_files/language.elf"


@pytest.mark.parametrize("skip_bfcl", [False, True])
def test_speculative_publication_registers_both_models_and_per_dataset_counts(tmp_path, skip_bfcl):
    import hashlib

    baseline_model = "gemma_4_e2b_it_textonly"
    target_model = "gemma_4_e2b_it_textonly_mtp"
    digest = hashlib.sha256(b"{}").hexdigest()
    for name in (target_model, baseline_model):
        registered_build(tmp_path / "_compiled", name, "w4a16", ["model"])
        profile = tmp_path / "_compiled" / name / "w4a16"
        (profile / "model/deployment-manifest.json").write_text("{}")
        (profile / "compile-outputs.json").write_text(json.dumps({"outputs": [{
            "component_id": "model", "artifact_path": "model/deployment-manifest.json", "artifact_sha256": digest,
        }]}))
    skipped = {"bfcl-v3": "baseline tokenizer has no native tool template"} if skip_bfcl else {}
    workloads = []
    for name in bench.TEXT_DATASETS:
        if name in skipped:
            continue
        workload = dataset_workload(name)
        workload["baseline"] = dataset_workload(name, ordinary=True)
        workload["metrics"] = bench.workload_metrics(workload)
        workloads.append(workload)
    generation = {"status": "passed", "workloads": workloads, "runtime": {}, "timing": {},
        "settings": {"max_context": 4096, "image_size": None, "decoding": "speculative", "repetitions": 1},
        "baseline": {"model_id": baseline_model, "build_profile_id": "w4a16",
                     "compile_run_id": "00000000-0000-4000-8000-000000000001", "artifact_sha256": digest}}
    if skipped:
        generation["skipped_workloads"] = skipped
    document = from_generation(generation, {"llima": "0.6.0", "mole": "0.6.0", "build": {}}).document
    calls, _ = publish(tmp_path, target_model, "w4a16", "performance", document)
    inputs = [path for _, path, _ in calls if "/inputs/" in path]
    assert any(path.endswith("/baseline-model-model") for path in inputs)
    measurements = next(body["measurements"] for _, path, body in calls if path.endswith("/measurements/batch"))
    assert {row["dimensions"]["dataset"] for row in measurements} == set(bench.TEXT_DATASETS) - set(skipped)
    assert {row["dimensions"]["role"] for row in measurements} == {"target", "baseline"}
    assert all(row["sample_count"] == 3 for row in measurements)
    assert any(row["name"] == "model.generation.decode_speedup" for row in measurements)
    summary = next(body["summary"] for _, path, body in calls if "/test-results/" in path)
    assert summary["generation"].get("skipped_workloads", {}) == skipped
    assert "requests" not in summary["generation"]["workloads"][0]
    assert "requests" not in summary["generation"]["workloads"][0]["baseline"]
    assert summary["generation"]["workloads"][0]["metrics"] == workloads[0]["metrics"]
    assert document["generation"]["workloads"][0]["requests"]  # The artifact remains complete.
    if skipped:
        generation.pop("skipped_workloads")
        with pytest.raises(ValueError, match="do not cover"):
            bench.validate_report(generation)
