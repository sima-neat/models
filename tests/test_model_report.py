import json
import runpy
import subprocess
import sys
import types
from pathlib import Path

import model_report
import pytest
import yaml
from jsonschema import Draft202012Validator

REPOSITORY = Path(__file__).resolve().parents[1]
CORE_REPORT = json.loads((REPOSITORY / "tests/fixtures/core-benchmark-report.json").read_text(encoding="utf-8"))
SCHEMA = Draft202012Validator(json.loads((REPOSITORY / "schemas/model-report.schema.json").read_text()))


class Saved:
    """A Core result whose save() writes the JSON Core would."""

    def __init__(self, document):
        self.document = document

    def save(self, path):
        Path(path).write_text(json.dumps(self.document), encoding="utf-8")
        return path


class Model:
    def __init__(self, name, batch_size=1):
        self.name, self.batch_size = name, batch_size

    def compiled_batch_size(self):
        return self.batch_size

    def execution_plan(self):
        plan = {"schema": "sima.neat.execution_plan", "model": self.name}
        return types.SimpleNamespace(to_json=lambda: json.dumps(plan))


@pytest.fixture(autouse=True)
def pyneat(monkeypatch):
    module = types.SimpleNamespace(__version__="3.0.0", build_info=lambda: {"version": "3.0.0"})
    monkeypatch.setitem(sys.modules, "pyneat", module)


def saved_report(path):
    report = json.loads((path / model_report.REPORT_FILE).read_text(encoding="utf-8"))
    SCHEMA.validate(report)
    return report


def test_performance_report_reads_core_report_and_saves_it_as_evidence(tmp_path):
    model_report.performance(Saved(CORE_REPORT), {"model": Model("resnet")}, workload="MPK default").save(tmp_path)

    report = saved_report(tmp_path)
    assert report["metrics"]["latency_avg_ms"] == CORE_REPORT["latency_ms"]
    assert report["metrics"]["throughput_fps"] == CORE_REPORT["fps"]
    assert report["metrics"]["avg_power_watts"] == CORE_REPORT["telemetry"]["power_som_watts"]["mean"]
    assert report["components"]["model"]["layer_stats"] == CORE_REPORT["layer_stats"]
    assert report["telemetry"] == CORE_REPORT["telemetry"]
    assert report["runtime"] == {"pyneat": "3.0.0", "build": {"version": "3.0.0"}}
    assert json.loads((tmp_path / model_report.CORE_REPORT_FILE).read_text()) == CORE_REPORT


def test_energy_per_inference_matches_cores_energy_for_its_sample_count(tmp_path):
    # The recorded report measured 1000 samples of batch 1; Core's energy covers those samples.
    metrics = model_report.performance(Saved(CORE_REPORT), {"model": Model("resnet")}, workload="w").document["metrics"]

    assert metrics["energy_per_inference_joules"] == pytest.approx(CORE_REPORT["energy_joules"] / 1000)


def test_energy_per_inference_does_not_depend_on_the_compiled_batch(tmp_path):
    # Core's throughput already counts each inference of a batch, so equal power and rate mean equal energy.
    single = model_report.performance(Saved(CORE_REPORT), {"model": Model("m")}, workload="w").document["metrics"]
    batched = model_report.performance(Saved(CORE_REPORT), {"model": Model("m", 4)}, workload="w").document["metrics"]

    assert batched["energy_per_inference_joules"] == pytest.approx(single["energy_per_inference_joules"])


def test_a_value_core_did_not_measure_is_absent(tmp_path):
    unmeasured = {**CORE_REPORT, "latency_p99_ms": 0.0, "telemetry": {**CORE_REPORT["telemetry"], "power_som_watts": {
        **CORE_REPORT["telemetry"]["power_som_watts"], "samples": 0}}}

    metrics = model_report.performance(Saved(unmeasured), {"model": Model("m")}, workload="w").document["metrics"]

    assert "latency_p99_ms" not in metrics
    assert "avg_power_watts" not in metrics and "energy_per_inference_joules" not in metrics


def test_a_script_timed_model_passes_its_telemetry_and_each_components_layer_stats(tmp_path):
    measured = {**CORE_REPORT, "telemetry": {}, "layer_stats": []}
    timings = {"backbone": [{"name": "conv", "stage_index": 0, "total_execution_time_us": 5}], "decoder": []}

    model_report.performance(
        Saved(measured),
        {"backbone": Model("backbone"), "decoder": Model("decoder")},
        workload="Backbone, proposals, decoder",
        telemetry=Saved(CORE_REPORT["telemetry"]),
        layer_stats=timings,
    ).save(tmp_path)

    report = saved_report(tmp_path)
    assert list(report["components"]) == ["backbone", "decoder"]
    assert report["components"]["backbone"]["layer_stats"] == timings["backbone"]
    assert report["metrics"]["avg_power_watts"] == CORE_REPORT["telemetry"]["power_som_watts"]["mean"]


def test_several_components_need_their_layer_stats():
    with pytest.raises(ValueError, match="layer_stats"):
        model_report.performance(Saved(CORE_REPORT), {"a": Model("a"), "b": Model("b")}, workload="w")


def test_telemetry_in_an_unexpected_unit_is_rejected():
    power = {**CORE_REPORT["telemetry"]["power_som_watts"], "unit": "mW"}
    telemetry = {**CORE_REPORT["telemetry"], "power_som_watts": power}

    with pytest.raises(ValueError, match="power_som_watts"):
        model_report.performance(Saved({**CORE_REPORT, "telemetry": telemetry}), {"m": Model("m")}, workload="w")


def test_accuracy_report_validates_against_the_schema(tmp_path):
    coco = {"split": "val2017", "evaluated_samples": 5000, "available_samples": 5000}
    model_report.accuracy({
        "validation": {**coco, "metrics": {"coco_map": 0.4, "coco_ap50": 0.55}, "details": {"detections": 12}},
        "sampled": {**coco, "evaluated_samples": 500, "metrics": {"coco_map": 0.39}},
    }).save(tmp_path)

    report = saved_report(tmp_path)
    assert report["stage"] == "accuracy"
    assert list(report["datasets"]) == ["validation", "sampled"]
    assert report["datasets"]["validation"]["metrics"] == {"coco_map": 0.4, "coco_ap50": 0.55}
    assert report["datasets"]["sampled"] == {**coco, "evaluated_samples": 500, "metrics": {"coco_map": 0.39},
                                             "details": {}}


SAMPLES = {"split": "val", "evaluated_samples": 1, "available_samples": 1}


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ({}, "at least one dataset"),
        ({"validation": {**SAMPLES, "metrics": {"made_up_metric": 0.1}}}, "validation: unknown"),
        ({"validation": {**SAMPLES, "metrics": {"latency_avg_ms": 0.1}}}, "validation: unknown"),
        ({"validation": {**SAMPLES, "metrics": {"coco_map": 1.5}}}, "ratio"),
        ({"validation": {**SAMPLES, "evaluated_samples": 2, "metrics": {"coco_map": 0.5}}}, "evaluated_samples"),
        ({"validation": {"metrics": {"coco_map": 0.5}}}, "validation must give metrics, split"),
        ({"validation": {**SAMPLES, "metrics": {"coco_map": 0.5}, "notes": "x"}}, "validation must give"),
    ],
)
def test_accuracy_rejects_what_the_registry_cannot_publish(datasets, message):
    with pytest.raises(ValueError, match=message):
        model_report.accuracy(datasets)


def test_every_metric_names_its_stage_unit_direction_and_measurement():
    for name, metric in model_report.METRICS.items():
        assert metric["stage"] in {"performance", "accuracy"}, name
        assert metric["direction"] in {"higher", "lower", "neutral"}, name
        assert metric["unit"] and metric["label"] and metric["registry_name"], name
        assert not metric.get("required") or metric["stage"] == "performance", name
    assert model_report.REQUIRED_PERFORMANCE_METRICS == [
        "latency_avg_ms", "throughput_fps", "avg_power_watts", "energy_per_inference_joules"
    ]


def test_the_package_needs_only_the_standard_library():
    probe = (
        "import sys, model_report; "
        "third_party = {name.partition('.')[0] for name in sys.modules} - set(sys.stdlib_module_names); "
        "print(sorted(third_party - {'model_report', '__main__'}))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(REPOSITORY / 'tools')!r}); {probe}"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]"


def single_component_performance_scripts():
    """Single-component models' scripts, which run Model.benchmark; multi-component models test their pipeline."""
    scripts = []
    for path in sorted(REPOSITORY.glob("models/*/model.yaml")):
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "performance" in model and "components" not in model["compilation"]:
            scripts.append(path.parent / "scripts/performance.py")
    assert scripts
    return scripts


@pytest.mark.parametrize("script", single_component_performance_scripts(), ids=lambda path: path.parents[1].name)
def test_every_single_component_script_saves_a_report_from_its_benchmark(script, monkeypatch, tmp_path):
    loaded, requests = [], []

    class ScriptModel(Model):
        def __init__(self, *args):
            super().__init__("model")
            loaded.append(args)

        def benchmark(self, options):
            requests.append(options)
            return Saved(CORE_REPORT)

        def info(self):
            return types.SimpleNamespace(selection=types.SimpleNamespace(selected_post_kind="boxdecode"))

    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        __version__="3.0.0", build_info=lambda: {"version": "3.0.0"}, Model=ScriptModel,
        ModelOptions=types.SimpleNamespace, BenchmarkOptions=types.SimpleNamespace,
        BoxDecodeType=types.SimpleNamespace(
            YoloV8="YoloV8", YoloV26="YoloV26", YoloV26Pose="YoloV26Pose", YoloV26Seg="YoloV26Seg"
        ),
    ))
    monkeypatch.setattr(sys, "argv", [str(script), "--mpk", "model.tar.gz", "--output", str(tmp_path)])

    runpy.run_path(str(script), run_name="__main__")

    assert [args[0] for args in loaded] == ["model.tar.gz"]
    assert [(r.num_samples, r.min_duration_ms, r.include_layer_stats) for r in requests] == [(1000, 60000, True)]
    report = saved_report(tmp_path)
    assert list(report["components"]) == ["model"]
    assert report["components"]["model"]["layer_stats"] == CORE_REPORT["layer_stats"]
    assert report["workload"]
