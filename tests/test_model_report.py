import json
import runpy
import shutil
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


def test_accuracy_report_keeps_disparity_errors_in_their_units(tmp_path):
    # OpenStereo reports EPE in pixels and D1-all as a percentage, both lower is better; neither is a 0-1 ratio.
    scene_flow = {"split": "test", "evaluated_samples": 4370, "available_samples": 4370}
    result = {**scene_flow, "metrics": {"epe": 1.25, "d1_all": 2.82}}
    model_report.from_accuracy({"scene-flow-val": result}).save(tmp_path)

    assert saved_report(tmp_path)["datasets"]["scene-flow-val"]["metrics"] == {"epe": 1.25, "d1_all": 2.82}


SAMPLES = {"split": "val", "evaluated_samples": 1, "available_samples": 1}


def test_depth_error_is_unbounded_but_threshold_accuracy_is_a_fraction(tmp_path):
    result = {**SAMPLES, "metrics": {"abs_rel": 1.25, "delta1": 0.89}}
    model_report.from_accuracy({"nyu-depth-v2-default": result}).save(tmp_path)
    assert saved_report(tmp_path)["datasets"]["nyu-depth-v2-default"]["metrics"] == result["metrics"]


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        ({}, "at least one dataset"),
        ({"validation": {**SAMPLES, "metrics": {"made_up_metric": 0.1}}}, "validation: unknown"),
        ({"validation": {**SAMPLES, "metrics": {"latency_avg_ms": 0.1}}}, "validation: unknown"),
        ({"validation": {**SAMPLES, "metrics": {"coco_map": 1.5}}}, "ratio"),
        ({"validation": {**SAMPLES, "metrics": {"d1_all": 101.0}}}, "d1_all must be a percentage from 0 to 100"),
        ({"validation": {**SAMPLES, "metrics": {"epe": -0.1}}}, "epe must be at least 0 px"),
        ({"validation": {**SAMPLES, "metrics": {"epe": float("inf")}}}, "epe must be at least 0 px"),
        ({"validation": {**SAMPLES, "metrics": {"abs_rel": -0.1}}}, "abs_rel must be at least 0 ratio"),
        ({"validation": {**SAMPLES, "metrics": {"abs_rel": float("inf")}}}, "abs_rel must be at least 0 ratio"),
        ({"validation": {**SAMPLES, "metrics": {"abs_rel": float("nan")}}}, "abs_rel must be at least 0 ratio"),
        ({"validation": {**SAMPLES, "metrics": {"delta1": 1.01}}}, "delta1 must be a ratio from 0 to 1"),
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
        "import sys, model_report, model_report.generation_benchmark, model_report.board_telemetry; "
        "third_party = {name.partition('.')[0] for name in sys.modules} - set(sys.stdlib_module_names); "
        "print(sorted(third_party - {'model_report', '__main__'}))"
    )
    result = subprocess.run(
        # Exclude site packages and system sitecustomize; the package promises a standard-library-only import.
        [sys.executable, "-I", "-S", "-c", f"import sys; sys.path.insert(0, {str(REPOSITORY / 'tools')!r}); {probe}"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]"


def test_native_performance_script_runs_without_registry_package(tmp_path):
    shutil.copytree(REPOSITORY / "tools/model_report", tmp_path / "model_report")
    script = REPOSITORY / "models/qwen3_0_6b/scripts/performance.py"
    probe = (
        f"import sys, runpy; sys.path.insert(0, {str(tmp_path)!r}); "
        f"sys.argv = [{str(script)!r}, '--help']; runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-c", probe], capture_output=True, text=True, check=True)
    assert "--deployment" in result.stdout and "--output" in result.stdout


class BenchmarkOptions:
    """PyNeat's BenchmarkOptions, which rejects a field Core does not have."""

    __slots__ = (
        "num_samples", "min_duration_ms", "include_mla_layer_stats", "include_mla_stage_stats", "resize_mode", "inputs"
    )


class ModelOptions:
    """PyNeat's ModelOptions, with the preprocessing and execution fields the scripts set."""

    def __init__(self):
        self.preprocess = types.SimpleNamespace(
            resize=types.SimpleNamespace(), normalize=types.SimpleNamespace(), color_convert=types.SimpleNamespace()
        )
        self.advanced_execution = types.SimpleNamespace()
        self.processcvu = types.SimpleNamespace()
        self.superpoint = types.SimpleNamespace()
        self.semantic_segmentation = types.SimpleNamespace()


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
    "BoxDecodeType": (
        "YoloV8", "YoloV26", "YoloV26Pose", "YoloV26Seg", "YoloV8Obb", "YoloV26Obb", "SuperPoint", "YoloX", "Detr"
    ),
    "BoxDecodeTypeOption": ("Split3Interleaved",),
    "SuperPointProfile": ("MagicLeapDemoV1", "LightGlueV1"),
}
SCRIPT_PYNEAT = {
    enum: types.SimpleNamespace(**{value: value for value in values}) for enum, values in PYNEAT_ENUMS.items()
}
# Color conversion takes its own enum, not PixelFormat; distinct values catch a script passing the wrong one.
SCRIPT_PYNEAT["PreprocessColorFormat"] = types.SimpleNamespace(
    **{value: f"PreprocessColorFormat.{value}" for value in ("RGB", "BGR", "GRAY8")}
)
# A 4:3 COCO image letterboxes to 960x720 in the 1280x720 frame, with 160 grey columns on each side.
SOURCE_SIZE, IMAGE_COLUMNS, FILL = (480, 640), slice(160, 1120), 114
# The ImageNet image a classifier benchmarks, by the variant of its accuracy dataset, as its workload names it.
IMAGENET_SAMPLES = {"mini": "n01440764/ILSVRC2012_val_00000001.webp", "val": "val2012/ILSVRC2012_val_00000001.JPEG"}
# A Scene Flow view is 960x540; a stereo script joins the pair side by side, left then right, unletterboxed.
STEREO_VIEW, STEREO_LEFT, STEREO_RIGHT = (540, 960), 60, 90
STEREO_PAIR = "FlyingThings3D/frames_finalpass/TEST/A/0000/{view}/0006.png"


def write_image(path, value, size=SOURCE_SIZE):
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((*size, 3), value, np.uint8))


def benchmark_dataset(root):
    """A dataset with the files each script samples from: COCO val2017, ImageNet's val2012 and ImageNet mini's val
    split, a DAVIS val sequence, an HPatches val sequence, DOTA's val images, Pittsburgh 30k, ADE20K's validation
    images, and the Scene Flow test list. Only the second COCO image labels a person, so a pose script must pick it."""
    write_image(root / STEREO_PAIR.format(view="left"), STEREO_LEFT, STEREO_VIEW)
    write_image(root / STEREO_PAIR.format(view="right"), STEREO_RIGHT, STEREO_VIEW)
    disparity = STEREO_PAIR.replace("frames_finalpass", "disparity").format(view="left").replace(".png", ".pfm")
    pair = f"{STEREO_PAIR.format(view='left')} {STEREO_PAIR.format(view='right')} {disparity}\n"
    (root / "sceneflow_finalpass_test.txt").write_text(pair, encoding="utf-8")
    write_image(root / "val/bear/00000.jpg", 200)
    (root / "val.txt").write_text("bear\n", encoding="utf-8")
    write_image(root / "val2017/000000000001.jpg", 200)
    write_image(root / "val2017/000000000002.jpg", 200)
    write_image(root / "val2012/ILSVRC2012_val_00000001.JPEG", 200)
    write_image(root / "val/n01440764/ILSVRC2012_val_00000001.webp", 200)
    write_image(root / "val/images/P0003.jpg", 200)
    write_image(root / "val/v_scene/1.ppm", 200)
    write_image(root / "val/rgb_0001.jpg", 200)
    write_image(root / "test/rgb_0001.jpg", 200)
    write_image(root / "images/000/000000_pitch1_yaw1.jpg", 200)
    # Under the archive's top folder, so that no other script's images/ glob finds it.
    write_image(root / "ade20k-val/images/validation/ADE_val_00000001.jpg", 200)
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


def assert_side_by_side_frame(tensor):
    """One unletterboxed RGB frame holding the stereo pair: the left view, then the right view."""
    frame = tensor.array
    assert (frame.shape, str(frame.dtype)) == ((STEREO_VIEW[0], 2 * STEREO_VIEW[1], 3), "uint8")
    assert tensor.options["image_format"] == "RGB"
    assert (frame[:, : STEREO_VIEW[1]] == STEREO_LEFT).all() and (frame[:, STEREO_VIEW[1] :] == STEREO_RIGHT).all()


def first_accuracy_dataset(script: Path) -> str | None:
    model = yaml.safe_load((script.parents[1] / "model.yaml").read_text(encoding="utf-8"))
    datasets = model.get("accuracy", {}).get("datasets", [])
    return datasets[0]["id"] if datasets else None


# Video models with no accuracy dataset get no --dataset and benchmark a fixed clip their script builds.
FIXED_CLIP_MODELS = {"vjepa2_1_vitb_16x256"}


def performance_scripts(*, components: bool, accuracy: bool | None = None) -> list[Path]:
    """Benchmarked models' scripts: single-component ones run Model.benchmark, multi-component ones time their
    pipeline and take each component's MLA stats from its own Model.benchmark. accuracy selects the models that
    do or do not declare accuracy datasets; only those benchmark a dataset image."""
    scripts = []
    for path in sorted(REPOSITORY.glob("models/*/model.yaml")):
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        if accuracy is not None and ("accuracy" in model) != accuracy:
            continue
        if ("performance" in model and model["compilation"]["format"] == "onnx"
                and ("components" in model["compilation"]) == components):
            scripts.append(path.parent / "scripts/performance.py")
    assert scripts
    return scripts


def takes_images(script: Path) -> bool:
    """Whether the model a performance script measures declares an image input."""
    model = yaml.safe_load((script.parents[1] / "model.yaml").read_text(encoding="utf-8"))
    return "image" in model["modalities"]["inputs"]


# A model whose inputs are not images, such as a keypoint matcher, benchmarks tensors its own tests cover.
@pytest.mark.parametrize(
    "script",
    [path for path in performance_scripts(components=False) if takes_images(path) and first_accuracy_dataset(path)],
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
    declaration = yaml.safe_load((script.parents[1] / "model.yaml").read_text())["accuracy"]["datasets"][0]
    source_id = f"{declaration['id']}-{declaration['variant']}"
    monkeypatch.setattr(sys, "argv", [
        str(script), "--mpk", "model.tar.gz", "--dataset", f"{source_id}={dataset}", "--output", str(output),
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
    # Neat sizes its input buffers, which the execution plan shows as its input, for the benchmark frame.
    height, width = frame.array.shape[:2]
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (width, height)
    report = saved_report(output)
    if first_accuracy_dataset(script) == "scene-flow":
        # A stereo model benchmarks a Scene Flow test pair as its side-by-side frame, stretched to the model input.
        assert_side_by_side_frame(frame)
        assert (options.preprocess.resize.mode, options.preprocess.preset) == ("Stretch", "ImageNet")
        assert STEREO_PAIR.format(view="left") in report["workload"]
    elif first_accuracy_dataset(script) == "nyu-depth-v2":
        assert_letterboxed_frame(frame)
        test_split = declaration["variant"] == "val"
        assert (options.preprocess.resize.mode, options.preprocess.preset) == (
            "Stretch" if test_split else "Letterbox", "ImageNet"
        )
        assert "rgb_0001.jpg" in report["workload"]
    elif getattr(options, "decode_type", "").startswith(("YoloV26", "YoloV8", "YoloX")):
        # Box decoders take a probability threshold, in [0, 1).
        assert_letterboxed_frame(frame)
        assert (options.preprocess.resize.mode, options.top_k, options.score_threshold) == ("Letterbox", 10, 0.25)
        if options.decode_type in {"YoloV8", "YoloV8Obb"}:
            assert options.nms_iou_threshold == 0.7
            assert "NMS IoU 0.7" in report["workload"]
        elif options.decode_type == "YoloX":
            color, normalize = options.preprocess.color_convert, options.preprocess.normalize
            assert (color.enable, color.input_format, color.output_format) == (
                "On", "PreprocessColorFormat.RGB", "PreprocessColorFormat.BGR"
            )
            assert options.decode_type_option == "Split3Interleaved"
            assert options.preprocess.resize.pad_value == 114
            assert normalize.mean == [0.0] * 3 and normalize.stddev == [1.0 / 255] * 3
            assert normalize.has_explicit_stats
            assert options.nms_iou_threshold == 0.65
            assert "NMS IoU 0.65" in report["workload"]
        # Pose accuracy scores only images with a labelled person; oriented boxes measure DOTA's aerial images.
        expected = {
            "YoloV26Pose": "000000000002.jpg", "YoloV8Obb": "P0003.jpg", "YoloV26Obb": "P0003.jpg",
        }.get(options.decode_type, ".jpg")
        assert expected in report["workload"]
    elif getattr(options, "decode_type", "") == "Detr":
        # DETR stretches the frame to its input with the ImageNet preset and decodes it with Neat's DETR decoder,
        # which keeps the top 10 above 0.25 and runs no NMS.
        assert_letterboxed_frame(frame)
        assert (options.preprocess.resize.mode, options.preprocess.preset) == ("Stretch", "ImageNet")
        assert (options.top_k, options.score_threshold) == (10, 0.25)
        assert not hasattr(options, "nms_iou_threshold")
        assert "DETR box decoder, top 10 above 0.25" in report["workload"]
        assert any(f"{name} in a 1280x720" in report["workload"] for name in ("000000000001.jpg", "000000000002.jpg"))
    elif getattr(options, "decode_type", "") == "SuperPoint":
        # SuperPoint takes the frame stretched to its input, converted to GRAY8, scaled to [0, 1] without a preset,
        # and decodes it with Magic Leap's demo settings, as the accuracy stage does.
        assert_letterboxed_frame(frame)
        color = options.preprocess.color_convert
        assert (options.preprocess.resize.mode, color.enable, color.input_format, color.output_format) == (
            "Stretch", "On", "PreprocessColorFormat.RGB", "PreprocessColorFormat.GRAY8"
        )
        assert not hasattr(options.preprocess, "preset")
        assert (options.superpoint.profile, options.top_k) == ("MagicLeapDemoV1", 1000)
        assert "v_scene/1.ppm" in report["workload"]
    elif first_accuracy_dataset(script) == "ade20k":
        # A semantic segmenter benchmarks an ADE20K validation image, stretched to its input with the ImageNet preset.
        assert_letterboxed_frame(frame)
        assert (options.preprocess.resize.mode, options.preprocess.preset) == ("Stretch", "ImageNet")
        assert "ADE_val_00000001.jpg" in report["workload"]
        # Neat decodes the class scores into a class map of the whole frame, one class index per pixel.
        segmentation = options.semantic_segmentation
        assert (segmentation.enable, segmentation.output_width, segmentation.output_height) == (True, width, height)
        assert f"{width}x{height} class map" in report["workload"]
    elif first_accuracy_dataset(script) == "pitts30k":
        # A place-recognition model benchmarks a Pittsburgh 30k image, stretched to its input with the ImageNet preset.
        assert_letterboxed_frame(frame)
        assert (options.preprocess.resize.mode, options.preprocess.preset) == ("Stretch", "ImageNet")
        assert "_pitch1_yaw1.jpg" in report["workload"]
    else:
        assert_letterboxed_frame(frame)
        assert options.preprocess.resize.mode == "Stretch"
        # A classifier normalizes with the ImageNet preset, or with the mean and std of its own preprocessing.
        normalize = options.preprocess.normalize
        if getattr(normalize, "has_explicit_stats", False):
            calibration = runpy.run_path(str(script.parent / "calibration.py"))
            assert list(normalize.mean) == pytest.approx(calibration["MEAN"].tolist())
            assert list(normalize.stddev) == pytest.approx(calibration["STD"].tolist())
        else:
            assert options.preprocess.preset == "ImageNet"
        metadata = yaml.safe_load((script.parents[1] / "model.yaml").read_text(encoding="utf-8"))
        assert IMAGENET_SAMPLES[metadata["accuracy"]["datasets"][0]["variant"]] in report["workload"]
    assert list(report["components"]) == ["model"]
    assert report["components"]["model"]["mla_layer_stats"] == CORE_REPORT["mla_layer_stats"]
    assert report["components"]["model"]["mla_stage_stats"] == CORE_REPORT["mla_stage_stats"]


# A model without an accuracy dataset takes no --dataset and builds its own frame.
@pytest.mark.parametrize(
    "script",
    [path for path in performance_scripts(components=False) if takes_images(path) and not first_accuracy_dataset(path)],
    ids=lambda path: path.parents[1].name,
)
def test_every_single_component_script_without_a_dataset_benchmarks_a_fixed_frame(script, monkeypatch, tmp_path):
    import numpy as np

    loaded, requests = [], []

    class ScriptModel(Model):
        def __init__(self, *args):
            super().__init__("model")
            loaded.append(args)

        def benchmark(self, options):
            requests.append(options)
            return Saved(CORE_REPORT)

    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        __version__="3.0.0", build_info=lambda: {"version": "3.0.0"}, Model=ScriptModel,
        ModelOptions=ModelOptions, BenchmarkOptions=BenchmarkOptions, Tensor=Tensor, **SCRIPT_PYNEAT,
    ))
    for run in ("first", "second"):
        monkeypatch.setattr(sys, "argv", [str(script), "--mpk", "model.tar.gz", "--output", str(tmp_path / run)])
        runpy.run_path(str(script), run_name="__main__")

    assert [args[0] for args in loaded] == ["model.tar.gz"] * 2
    assert [(r.num_samples, r.min_duration_ms, r.include_mla_layer_stats, r.include_mla_stage_stats)
            for r in requests] == [(1000, 60000, True, True)] * 2
    # Neat preprocesses the frame inside the measured pipeline.
    options = loaded[0][1]
    assert options.preprocess.kind == "Image"
    assert (options.preprocess.enable, options.preprocess.resize.enable, options.preprocess.normalize.enable) == (
        "On", "On", "On"
    )
    # Every run measures the same 720p frame, and not a blank one.
    (first,), (second,) = (request.inputs for request in requests)
    assert (first.array.shape, str(first.array.dtype), first.options["image_format"]) == (
        (720, 1280, 3), "uint8", "RGB"
    )
    assert np.array_equal(first.array, second.array) and first.array.std() > 10
    # Neat sizes its input buffers, which the execution plan shows as its input, for the benchmark frame.
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (1280, 720)
    report = saved_report(tmp_path / "first")
    assert report["workload"]
    assert list(report["components"]) == ["model"]
    assert report["components"]["model"]["mla_layer_stats"] == CORE_REPORT["mla_layer_stats"]
    assert report["components"]["model"]["mla_stage_stats"] == CORE_REPORT["mla_stage_stats"]


@pytest.mark.parametrize(
    "script", [path for path in performance_scripts(components=True)
               if yaml.safe_load((path.parents[1] / "model.yaml").read_text())["performance"].get("scope")
               != "components"],
    ids=lambda path: path.parents[1].name,
)
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
    # A script may import a helper beside it, as it does when run from its directory.
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace())

    layer_stats, stage_stats = runpy.run_path(str(script))["mla_stats"](Path("backbone.tar.gz"))

    assert [(r.include_mla_layer_stats, r.include_mla_stage_stats) for r in requests] == [(True, True)]
    assert (layer_stats, stage_stats) == (CORE_REPORT["mla_layer_stats"], CORE_REPORT["mla_stage_stats"])


@pytest.mark.parametrize(
    "script",
    [path for path in performance_scripts(components=True, accuracy=True)
     if path.parents[1].name not in FIXED_CLIP_MODELS
     and yaml.safe_load((path.parents[1] / "model.yaml").read_text())["performance"].get("scope") != "components"],
    ids=lambda path: path.parents[1].name,
)
def test_every_multi_component_script_feeds_its_backbone_a_dataset_frame(script, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        ModelOptions=ModelOptions, Tensor=Tensor, **SCRIPT_PYNEAT,
    ))
    functions = runpy.run_path(str(script))

    image = functions["sample_image"](benchmark_dataset(tmp_path))
    frame = functions["frame_tensor"](image)

    metadata = yaml.safe_load((script.parents[1] / "model.yaml").read_text(encoding="utf-8"))
    splits = {"coco-2017": "val2017", "imagenet-2012": "val2012", "davis-2017": "bear"}
    assert image.parent.name == splits[metadata["accuracy"]["datasets"][0]["id"]]
    assert_letterboxed_frame(frame)
    assert frame.options["memory"] == "EV74"
    options = functions["image_options"]()
    assert (options.preprocess.kind, options.preprocess.resize.mode) == ("Image", "Stretch")
    # The first component normalizes with the ImageNet preset, or with the mean and std of its own preprocessing.
    normalize = options.preprocess.normalize
    if getattr(normalize, "has_explicit_stats", False):
        calibration = runpy.run_path(str(script.parent / "calibration.py"))
        assert list(normalize.mean) == pytest.approx(calibration["MEAN"].tolist())
        assert list(normalize.stddev) == pytest.approx(calibration["STD"].tolist())
    else:
        assert options.preprocess.preset == "ImageNet"
    height, width = frame.array.shape[:2]
    assert (options.preprocess.input_max_width, options.preprocess.input_max_height) == (width, height)


@pytest.mark.parametrize("model_name", sorted(FIXED_CLIP_MODELS))
def test_every_fixed_clip_script_builds_the_same_prepared_clip(model_name, monkeypatch):
    script = REPOSITORY / "models" / model_name / "scripts/performance.py"
    monkeypatch.setitem(sys.modules, "pyneat", types.SimpleNamespace(
        ModelOptions=ModelOptions, Tensor=Tensor, **SCRIPT_PYNEAT,
    ))
    monkeypatch.syspath_prepend(str(script.parent))
    functions = runpy.run_path(str(script))

    first, second = functions["synthetic_clip"](), functions["synthetic_clip"]()

    assert first.ndim == 4 and str(first.dtype) == "float32"
    assert (first == second).all()
    assert first.std() > 0.1
    options = functions["tensor_options"]()
    assert options.preprocess.kind == "Tensor"
    assert (options.preprocess.resize.enable, options.preprocess.color_convert.enable,
            options.preprocess.normalize.enable) == ("Off", "Off", "Off")


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


def test_independent_components_keep_their_own_metrics_telemetry_and_evidence(tmp_path):
    image = model_report.from_performance(Saved(CORE_REPORT), {"image_encoder": Model("image")}, workload="image")
    text_core = {**CORE_REPORT, "fps": 600.0, "latency_ms": 2.0,
                 "telemetry": {**CORE_REPORT["telemetry"], "power_som_watts": {
                     **CORE_REPORT["telemetry"]["power_som_watts"], "mean": 4.0}}}
    text = model_report.from_performance(Saved(text_core), {"text_encoder": Model("text")}, workload="text")
    model_report.from_component_performance({"image_encoder": image, "text_encoder": text}).save(tmp_path)
    report = saved_report(tmp_path)
    assert report["scope"] == "components" and "metrics" not in report and "telemetry" not in report
    for name, expected in (("image_encoder", image), ("text_encoder", text)):
        component = report["components"][name]
        for key in ("metrics", "telemetry", "workload"):
            assert component[key] == expected.document[key]
        assert (tmp_path / f"{name}-benchmark-report.json").is_file()
    assert report["components"]["text_encoder"]["metrics"]["energy_per_inference_joules"] == pytest.approx(4 / 600)


def test_component_reports_reject_wrong_components_and_mixed_runtimes():
    image = model_report.from_performance(Saved(CORE_REPORT), {"image": Model("image")}, workload="image")
    with pytest.raises(ValueError, match="exactly that component"):
        model_report.from_component_performance({"text": image})
    image.document["runtime"]["pyneat"] = "another-runtime"
    with pytest.raises(ValueError, match="same runtime"):
        model_report.from_component_performance({"image": image})
    with pytest.raises(ValueError, match="at least one component"):
        model_report.from_component_performance({})
