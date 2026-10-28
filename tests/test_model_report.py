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
    model_report.from_performance(Saved(CORE_REPORT), {"model": Model("resnet")}, workload="MPK default").save(tmp_path)

    report = saved_report(tmp_path)
    assert report["metrics"]["latency_avg_ms"] == CORE_REPORT["latency_ms"]
    assert report["metrics"]["throughput_fps"] == CORE_REPORT["fps"]
    assert report["metrics"]["avg_power_watts"] == CORE_REPORT["telemetry"]["power_som_watts"]["mean"]
    assert report["components"]["model"]["mla_layer_stats"] == CORE_REPORT["mla_layer_stats"]
    assert report["components"]["model"]["mla_stage_stats"] == CORE_REPORT["mla_stage_stats"]
    assert report["telemetry"] == CORE_REPORT["telemetry"]
    assert report["runtime"] == {"pyneat": "3.0.0", "build": {"version": "3.0.0"}}
    assert json.loads((tmp_path / model_report.CORE_REPORT_FILE).read_text()) == CORE_REPORT


def test_energy_per_inference_matches_cores_energy_for_its_sample_count(tmp_path):
    # The recorded report measured 1000 samples of batch 1; Core's energy covers those samples.
    report = model_report.from_performance(Saved(CORE_REPORT), {"model": Model("resnet")}, workload="w")
    metrics = report.document["metrics"]

    assert metrics["energy_per_inference_joules"] == pytest.approx(CORE_REPORT["energy_joules"] / 1000)


def test_energy_per_inference_does_not_depend_on_the_compiled_batch(tmp_path):
    # Core's throughput already counts each inference of a batch, so equal power and rate mean equal energy.
    single = model_report.from_performance(Saved(CORE_REPORT), {"model": Model("m")}, workload="w")
    batched = model_report.from_performance(Saved(CORE_REPORT), {"model": Model("m", 4)}, workload="w")

    assert batched.document["metrics"]["energy_per_inference_joules"] == pytest.approx(
        single.document["metrics"]["energy_per_inference_joules"])


def test_a_value_core_did_not_measure_is_absent(tmp_path):
    unmeasured = {**CORE_REPORT, "latency_p99_ms": 0.0, "telemetry": {**CORE_REPORT["telemetry"], "power_som_watts": {
        **CORE_REPORT["telemetry"]["power_som_watts"], "samples": 0}}}

    metrics = model_report.from_performance(Saved(unmeasured), {"model": Model("m")}, workload="w").document["metrics"]

    assert "latency_p99_ms" not in metrics
    assert "avg_power_watts" not in metrics and "energy_per_inference_joules" not in metrics


def test_memory_is_mla_and_ev74_memory_and_never_the_cma_pool():
    telemetry = CORE_REPORT["telemetry"]
    metrics = model_report.from_performance(Saved(CORE_REPORT), {"model": Model("m")}, workload="w").document["metrics"]

    assert metrics["mla_memory_peak_mib"] == telemetry["mla_memory_mib"]["maximum"]
    assert metrics["ev74_memory_avg_mib"] == telemetry["ev74_memory_mib"]["mean"]
    assert metrics["ev74_memory_peak_mib"] == telemetry["ev74_memory_mib"]["maximum"]

    # A Core from before EV74 memory reports the whole CMA pool in use instead, which is not the model's.
    earlier = {key: value for key, value in telemetry.items() if key != "ev74_memory_mib"}
    earlier["cma_memory_mib"] = {**telemetry["mla_memory_mib"], "key": "ev74_cma_used_mb", "mean": 863.8}
    metrics = model_report.from_performance(
        Saved({**CORE_REPORT, "telemetry": earlier}), {"model": Model("m")}, workload="w").document["metrics"]

    assert not any("ev74" in name or "cma" in name for name in metrics)


def test_a_script_timed_model_passes_its_telemetry_and_each_components_mla_stats(tmp_path):
    # A hand-filled BenchmarkReport holds no MLA stats; each component's come from its own Core benchmark.
    measured = {**CORE_REPORT, "telemetry": {}, "mla_layer_stats": {"status": "not requested", "layers": []},
                "mla_stage_stats": {"status": "not requested", "stages": []}}
    layer_stats = {"backbone": CORE_REPORT["mla_layer_stats"], "decoder": {"status": "mla-rt not found", "layers": []}}
    stage_stats = {"backbone": CORE_REPORT["mla_stage_stats"], "decoder": {"status": "mla-rt not found", "stages": []}}

    model_report.from_performance(
        Saved(measured),
        {"backbone": Model("backbone"), "decoder": Model("decoder")},
        workload="Backbone, proposals, decoder",
        telemetry=Saved(CORE_REPORT["telemetry"]),
        mla_layer_stats=layer_stats,
        mla_stage_stats=stage_stats,
    ).save(tmp_path)

    report = saved_report(tmp_path)
    assert list(report["components"]) == ["backbone", "decoder"]
    for component in ("backbone", "decoder"):
        assert report["components"][component]["mla_layer_stats"] == layer_stats[component]
        assert report["components"][component]["mla_stage_stats"] == stage_stats[component]
    assert report["metrics"]["avg_power_watts"] == CORE_REPORT["telemetry"]["power_som_watts"]["mean"]


def test_several_components_need_their_mla_stats():
    with pytest.raises(ValueError, match="mla_layer_stats"):
        model_report.from_performance(Saved(CORE_REPORT), {"a": Model("a"), "b": Model("b")}, workload="w")


def test_telemetry_in_an_unexpected_unit_is_rejected():
    power = {**CORE_REPORT["telemetry"]["power_som_watts"], "unit": "mW"}
    telemetry = {**CORE_REPORT["telemetry"], "power_som_watts": power}

    with pytest.raises(ValueError, match="power_som_watts"):
        model_report.from_performance(Saved({**CORE_REPORT, "telemetry": telemetry}), {"m": Model("m")}, workload="w")


def test_accuracy_report_validates_against_the_schema(tmp_path):
    coco = {"split": "val2017", "evaluated_samples": 5000, "available_samples": 5000}
    model_report.from_accuracy({
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
        model_report.from_accuracy(datasets)


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


class BenchmarkOptions:
    """PyNeat's BenchmarkOptions, which rejects a field Core does not have."""

    __slots__ = (
        "num_samples", "min_duration_ms", "include_mla_layer_stats", "include_mla_stage_stats", "resize_mode", "inputs"
    )


class ModelOptions:
    """PyNeat's ModelOptions, with the preprocessing fields the scripts set."""

    def __init__(self):
        self.preprocess = types.SimpleNamespace(
            resize=types.SimpleNamespace(), color_convert=types.SimpleNamespace(), normalize=types.SimpleNamespace()
        )


class Tensor:
    """A PyNeat tensor made from a NumPy array, keeping the array and the options it was made with."""

    def __init__(self, array, **options):
        self.array, self.options = array, options

    @classmethod
    def from_numpy(cls, array, copy=False, **options):
        return cls(array.copy() if copy else array, **options)


# The PyNeat names the performance scripts read, each standing for itself.
PYNEAT_ENUMS = {
    "InputKind": ("Image", "Tensor"),
    "AutoFlag": ("On", "Off"),
    "ResizeMode": ("Letterbox", "Stretch"),
    "NormalizePreset": ("ImageNet",),
    "PixelFormat": ("RGB",),
    "TensorMemory": ("EV74",),
    "BoxDecodeType": ("YoloV26", "YoloV26Pose", "YoloV26Seg"),
}
SCRIPT_PYNEAT = {
    enum: types.SimpleNamespace(**{value: value for value in values}) for enum, values in PYNEAT_ENUMS.items()
}
# A 4:3 COCO image letterboxes to 960x720 in the 1280x720 frame, with 160 grey columns on each side.
SOURCE_SIZE, IMAGE_COLUMNS, FILL = (480, 640), slice(160, 1120), 114


def write_image(path, value):
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((*SOURCE_SIZE, 3), value, np.uint8))


def benchmark_dataset(root):
    """A dataset with the files each script samples from: COCO val2017 and ImageNet's val split. Only the second
    COCO image labels a person, so a pose script must pick it."""
    write_image(root / "val2017/000000000001.jpg", 200)
    write_image(root / "val2017/000000000002.jpg", 200)
    write_image(root / "val/n01440764/ILSVRC2012_val_00000001.webp", 200)
    keypoints = {
        "images": [{"id": 1, "file_name": "000000000001.jpg"}, {"id": 2, "file_name": "000000000002.jpg"}],
        "annotations": [{"image_id": 1, "num_keypoints": 0}, {"image_id": 2, "num_keypoints": 17}],
    }
    (root / "annotations").mkdir()
    (root / "annotations/person_keypoints_val2017.json").write_text(json.dumps(keypoints), encoding="utf-8")
    return root


def assert_letterboxed_frame(tensor):
    frame = tensor.array
    assert (frame.shape, str(frame.dtype)) == ((720, 1280, 3), "uint8")
    assert tensor.options["image_format"] == "RGB"
    assert (frame[:, : IMAGE_COLUMNS.start] == FILL).all() and (frame[:, IMAGE_COLUMNS.stop :] == FILL).all()
    assert (frame[:, IMAGE_COLUMNS] == 200).all()


# Models whose inputs are not images benchmark a fixed tensor input on the tensor route.
TENSOR_INPUT_MODELS = {"superglue_outdoor"}


def performance_scripts(*, components: bool) -> list[Path]:
    """Benchmarked models' scripts: single-component ones run Model.benchmark, multi-component ones time their
    pipeline and take each component's MLA stats from its own Model.benchmark."""
    scripts = []
    for path in sorted(REPOSITORY.glob("models/*/model.yaml")):
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "performance" in model and ("components" in model["compilation"]) == components:
            scripts.append(path.parent / "scripts/performance.py")
    assert scripts
    return scripts


@pytest.mark.parametrize(
    "script",
    [path for path in performance_scripts(components=False) if path.parents[1].name not in TENSOR_INPUT_MODELS],
    ids=lambda path: path.parents[1].name,
)
def test_every_single_component_script_benchmarks_a_dataset_image(script, monkeypatch, tmp_path):
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
        ModelOptions=ModelOptions, BenchmarkOptions=BenchmarkOptions, Tensor=Tensor, **SCRIPT_PYNEAT,
    ))
    dataset = benchmark_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        str(script), "--mpk", "model.tar.gz", "--dataset", f"source={dataset}", "--output", str(output),
    ])

    runpy.run_path(str(script), run_name="__main__")

    assert [args[0] for args in loaded] == ["model.tar.gz"]
    assert [(r.num_samples, r.min_duration_ms, r.include_mla_layer_stats, r.include_mla_stage_stats)
            for r in requests] == [(1000, 60000, True, True)]
    # Neat preprocesses the frame inside the measured pipeline.
    options = loaded[0][1]
    assert options.preprocess.kind == "Image"
    assert (options.preprocess.enable, options.preprocess.resize.enable, options.preprocess.normalize.enable) == (
        "On", "On", "On"
    )
    (frame,) = requests[0].inputs
    assert_letterboxed_frame(frame)
    report = saved_report(output)
    if getattr(options, "decode_type", "").startswith("YoloV26"):
        assert (options.preprocess.resize.mode, options.top_k, options.score_threshold) == ("Letterbox", 10, 0.25)
        # Pose accuracy scores only images with a labelled person.
        expected = "000000000002.jpg" if options.decode_type == "YoloV26Pose" else ".jpg"
        assert expected in report["workload"]
    else:
        assert (options.preprocess.resize.mode, options.preprocess.preset) == ("Stretch", "ImageNet")
        assert "ILSVRC2012_val_00000001.webp" in report["workload"]
    assert list(report["components"]) == ["model"]
    assert report["components"]["model"]["mla_layer_stats"] == CORE_REPORT["mla_layer_stats"]
    assert report["components"]["model"]["mla_stage_stats"] == CORE_REPORT["mla_stage_stats"]


def superpoint_dataset(root):
    """A yfcc100m-keypoints val directory of three images, two with 512 keypoints at the 640 x 480 input's centre."""
    import numpy as np

    rows = []
    for index, count in enumerate((512, 100, 512)):
        stem = root / "val" / "a" / str(index)
        stem.mkdir(parents=True)
        np.save(stem / "keypoints.npy", np.tile(np.array([320, 240], np.float32), (count, 1)))
        np.save(stem / "scores.npy", np.full(count, 0.5, np.float32))
        np.save(stem / "descriptors.npy", np.full((count, 256), 1 / 16, np.float32))
        rows.append(f"a/{index}.jpg 1280 960 0 {count}")
    (root / "annotations").mkdir()
    (root / "annotations" / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("with_dataset", [True, False], ids=["dataset", "fixed"])
@pytest.mark.parametrize("model_name", sorted(TENSOR_INPUT_MODELS))
def test_every_tensor_input_script_times_its_cpu_matching(model_name, with_dataset, monkeypatch, tmp_path):
    import numpy as np

    requests = []

    def rows(stats, field):
        return types.SimpleNamespace(status=stats["status"], **{field: [
            types.SimpleNamespace(**row) for row in stats[field]
        ]})

    class ScriptModel:
        def __init__(self, mpk, options=None):
            self.mpk, self.options = mpk, options

        def benchmark(self, options):
            requests.append(options)
            return types.SimpleNamespace(
                mla_layer_stats=rows(CORE_REPORT["mla_layer_stats"], "layers"),
                mla_stage_stats=rows(CORE_REPORT["mla_stage_stats"], "stages"),
            )

    class Runner:
        """Returns couplings in Neat's channels-last [1, 1, i, j] shape, each keypoint scoring itself highest."""

        def run(self, tensors, timeout_ms):
            couplings = np.full((513, 513), -5.0, np.float32)
            np.fill_diagonal(couplings, 5.0)
            return [types.SimpleNamespace(to_numpy=lambda copy: couplings.reshape(1, 1, 513, 513))]

    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        Model=ScriptModel, ModelOptions=ModelOptions, BenchmarkOptions=BenchmarkOptions, Tensor=Tensor,
        **SCRIPT_PYNEAT,
    ))
    script = REPOSITORY / "models" / model_name / "scripts/performance.py"
    monkeypatch.syspath_prepend(str(script.parent))
    functions = runpy.run_path(str(script))

    options = functions["tensor_options"]()
    assert options.preprocess.kind == "Tensor"
    assert (options.preprocess.resize.enable, options.preprocess.normalize.enable) == ("Off", "Off")
    if with_dataset:
        workload, images = functions["dataset_images"](superpoint_dataset(tmp_path / "dataset"))
        # The image with 100 keypoints is never picked.
        assert workload == "yfcc100m-keypoints-val a/0.jpg and a/2.jpg" or workload == (
            "yfcc100m-keypoints-val a/2.jpg and a/0.jpg"
        )
    else:
        workload, images = functions["fixed_images"]()
    tensors = functions["pair_tensors"](images)
    assert [tensor.shape for tensor in tensors] == [(1, 1, 512, 2), (1, 1, 512, 1), (1, 1, 512, 256)] * 2
    if with_dataset:
        # The dataset's keypoints sit at the image centre, which SuperGlue's frame maps to the origin.
        assert all((tensor == 0).all() for tensor in tensors[0::3])
    # Every timed pair runs the CPU Sinkhorn iterations and mutual-match extraction after the MPK.
    matches = functions["match_pair"](Runner(), tensors)
    assert matches.tolist() == list(range(512))
    layer_stats, stage_stats = functions["mla_stats"]("model.tar.gz")
    assert [(r.include_mla_layer_stats, r.include_mla_stage_stats) for r in requests] == [(True, True)]
    assert (layer_stats, stage_stats) == (CORE_REPORT["mla_layer_stats"], CORE_REPORT["mla_stage_stats"])


@pytest.mark.parametrize("script", performance_scripts(components=True), ids=lambda path: path.parents[1].name)
def test_every_multi_component_script_keeps_cores_mla_stats(script, monkeypatch):
    requests = []

    def rows(stats, field):
        return types.SimpleNamespace(status=stats["status"], **{field: [
            types.SimpleNamespace(**row) for row in stats[field]
        ]})

    class ScriptModel:
        def __init__(self, mpk):
            self.mpk = mpk

        def benchmark(self, options):
            requests.append(options)
            return types.SimpleNamespace(
                mla_layer_stats=rows(CORE_REPORT["mla_layer_stats"], "layers"),
                mla_stage_stats=rows(CORE_REPORT["mla_stage_stats"], "stages"),
            )

    pyneat = types.SimpleNamespace(Model=ScriptModel, BenchmarkOptions=BenchmarkOptions)
    monkeypatch.setitem(sys.modules, "pyneat", pyneat)

    layer_stats, stage_stats = runpy.run_path(str(script))["mla_stats"](Path("backbone.tar.gz"))

    assert [(r.include_mla_layer_stats, r.include_mla_stage_stats) for r in requests] == [(True, True)]
    assert (layer_stats, stage_stats) == (CORE_REPORT["mla_layer_stats"], CORE_REPORT["mla_stage_stats"])


@pytest.mark.parametrize("script", performance_scripts(components=True), ids=lambda path: path.parents[1].name)
def test_every_multi_component_script_feeds_its_backbone_a_dataset_frame(script, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        ModelOptions=ModelOptions, Tensor=Tensor, **SCRIPT_PYNEAT,
    ))
    functions = runpy.run_path(str(script))

    image = functions["sample_image"](benchmark_dataset(tmp_path))
    frame = functions["frame_tensor"](image)

    assert image.parent.name == "val2017"
    assert_letterboxed_frame(frame)
    assert frame.options["memory"] == "EV74"
    options = functions["image_options"]()
    assert (options.preprocess.kind, options.preprocess.resize.mode, options.preprocess.preset) == (
        "Image", "Stretch", "ImageNet"
    )


def test_model_code_uses_cores_mla_stats_names():
    """Core renamed include_layer_stats and total_execution_time_us; PyNeat rejects the old option."""
    old_names = ("include_layer_stats", "total_execution_time_us")
    uses = [
        f"{path.relative_to(REPOSITORY)}: {name}"
        for path in sorted(REPOSITORY.glob("models/**/*.py"))
        for name in old_names
        if name in path.read_text(encoding="utf-8")
    ]
    assert uses == []
