# Coding style for model scripts

A reviewer, and the next contributor, read a model's scripts to learn what the package does to
the upstream model and how its numbers were measured. Write each script so that reader follows
it in one pass. Two model folders are the reference:

- [`models/yolo26n`](../models/yolo26n) for one component. It has every script and test file
  this page covers, and its `performance.py` selects Neat's `YoloV26` box decoder.
- [`models/rf_detr_nano`](../models/rf_detr_nano) for several components with host work
  between them: the backbone, the top 300 proposals on the CPU, then the decoder.

Read both before writing or reviewing a model script. These rules cover ONNX models. An LLM or
VLM recipe follows [LLiMa language and vision-language
models](../docs/adding-a-model.md#llima-language-and-vision-language-models).
[`rules.md` "Rules no check catches"](rules.md#rules-no-check-catches) owns what `convert.py`
exports and what `surgery.py` rewrites, where board-only code lives, how the benchmark scripts
stand alone, how scripts read conversion output names, and how DevKit tests import. The rules
below assume those.

## How the code reads

A violation is code a reader misreads or cannot follow. A name that only differs from the one
you would pick is not one. Each pair below shows the same code twice, and code that matches a
pair takes the shape of its good side. The good sides come from `models/yolo26n` and
`models/rf_detr_nano`.

### The main function reads as its steps

A reader learns what the build does from `compile_model` alone. Each step is one line that
names what it produces, and its detail lives in the helper of that name.

```python
# Bad: the steps are there, but the reader has to find where each one starts and ends.
def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    model = load_metadata(MODEL_DIR)
    profiles = [profile for profile in model["build_profiles"] if profile["id"] == profile_id]
    if len(profiles) != 1:
        raise ValueError(f"model.yaml declares no build profile {profile_id!r}")
    parameters = next(item["parameters"] for item in profiles[0]["components"] if item["id"] == COMPONENT_ID)
    if parameters["precision"] == "w8a8":
        quantization = QuantizationParams(calibration_method=HistogramMSEMethod(parameters["calibration_bins"]))
        samples = calibration_images(dataset, int(parameters["calibration_samples"]))
    else:
        quantization = QuantizationParams(calibration_method=SkipCalibration())
        samples = None
    graph = prepare_model(source, output_dir / "graphs")
    return _quantize_and_compile(graph, quantization, samples, output_dir)
```

```python
# Good: five lines tell the whole build. Open a helper only for the step you care about.
def compile_model(source: Path, output_dir: Path, profile_id: str, dataset: Path | None) -> Path:
    parameters = _parameters(profile_id)                # the build profile from model.yaml
    quantization = _quantization(parameters)            # formats and calibration method
    samples = _calibration_samples(parameters, dataset) # None when the build skips calibration

    graph = prepare_model(source, output_dir / "graphs")
    return _quantize_and_compile(graph, quantization, samples, output_dir)
```

Both sides do the same work. Extract a step when it has a name the reader would use, such as
"read the build profile". Keep a line inline when a function name would only restate it. This
pair is shortened to show the shape.

### A name says what the value is

Name a value by its meaning, with its unit when it has one, in the terms `model.yaml` and the
README use. A function that does work gets a verb or names what it returns. A constant sits at
the top of its module, in upper case, named for its role.

```python
# Bad: d, n, x, and two bare values tell the reader nothing, and the error hides both counts.
def sample(d: Path, n: int) -> list[str]:
    names = [x for x in _image_names(d) if PurePosixPath(x).parent.name == "cal2017"]
    if len(names) < n:
        raise ValueError("not enough images")
    return random.Random(20260930).sample(names, n)
```

```python
# Good
CALIBRATION_SPLIT = "cal2017"
CALIBRATION_SEED = 20260930


def calibration_names(dataset: Path, count: int) -> list[str]:
    names = [name for name in _image_names(dataset) if PurePosixPath(name).parent.name == CALIBRATION_SPLIT]
    if count < 1 or len(names) < count:
        raise ValueError(f"cannot select {count} calibration images from {len(names)} {CALIBRATION_SPLIT} images")
    return random.Random(CALIBRATION_SEED).sample(names, count)
```

### Each fact has one owner

`model.yaml` owns build parameters, `surgery.py` owns the component interface, and
`calibration.py` owns preprocessing and sample selection. Other scripts import the value.

```python
# Bad: compile.py repeats what surgery.py defines, and the two copies drift.
INPUT = "images"
INPUT_SHAPE = [1, 3, 640, 640]
```

```python
# Good
from surgery import INPUT, INPUT_SHAPE
```

### A condition tests the fact the branch depends on

A component ID, a model name, or a count that stands in for the fact breaks when the next
component or model is added.

```python
# Bad: the branch is about calibration, and the test is about which component this is.
if component_id == "backbone":
    samples = calibration_images(dataset, int(parameters["calibration_samples"]))
```

```python
# Good
if "calibration_samples" in parameters:
    samples = calibration_images(dataset, int(parameters["calibration_samples"]))
```

### Each precision states its own settings

One `QuantizationParams` call per precision sets every field that build needs.

```python
# Bad: both builds share one chain, so no single place shows what w8a16 gets.
config = default_quantization.with_weight_quantization(int8_weights)
if precision == "w8a8":
    config = config.with_calibration(HistogramMSEMethod(calibration_bins))
else:
    config = config.with_activation_quantization(bfloat16_scheme()).with_calibration(SkipCalibration())
return config.with_per_token_quant(precision == "w8a16")
```

```python
# Good
if precision == "w8a8":
    return QuantizationParams(
        calibration_method=HistogramMSEMethod(calibration_bins),
        activation_quantization_scheme=quantization_scheme(asymmetric=True, per_channel=False, bits=8),
        weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        biascorr_type=BiasCorrectionType.REGULAR,
    )
if precision == "w8a16":
    return QuantizationParams(
        calibration_method=SkipCalibration(),
        activation_quantization_scheme=bfloat16_scheme(),
        weight_quantization_scheme=quantization_scheme(asymmetric=False, per_channel=True, bits=8),
        # Each token's activations get their own INT8 scale before an INT8-weight operation.
        per_token_quant=True,
    )
```

### A build without calibration passes no data

```python
# Bad: a zero image satisfies quantize(), and the build reads as calibrated.
if precision == "w8a8":
    samples = calibration_images(dataset, calibration_samples)
else:
    samples = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
```

```python
# Good
calibration_data = None
if precision == "w8a8":
    samples = calibration_images(dataset, calibration_samples)
    calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
```

### A comment says what the code cannot

Write a comment where a reader would otherwise stop and ask why: a chosen value, a step that
looks wrong but is right, or a fact from upstream or the compiler. Keep it to one or two lines,
name the reason or its source, and describe the code as it is now. Leave code that reads on its
own without one.

```python
# Bad: the comments repeat the code, and the one value that needs a reason has none.
# Number of detections to keep
TOP_K = 300
# The score threshold
SCORE_THRESHOLD = 0.001
LOGIT_FLOOR = -7.0
```

```python
# Good
# Ultralytics' end-to-end selection and validation settings: the best 300 anchor/class pairs scoring above 0.001.
TOP_K = 300
SCORE_THRESHOLD = 0.001
# sigmoid(-7) is about 0.00091, below SCORE_THRESHOLD, so a logit at or below LOGIT_FLOOR is never kept and the
# decoder skips its sigmoid.
LOGIT_FLOOR = -7.0
```

### Every branch, option, and constant is reached

Each one runs from this model's entry point. Remove a branch for a precision `model.yaml` does
not declare, and a helper a copy from another model left unused. A test alone does not reach it.

## Every script

- The module docstring opens with one sentence that says what the script does to which model:
  `"""Cut YOLO26n at its raw one-to-one Detect heads, before the in-graph box decode and TopK."""`.
  A script with a command line passes it to `argparse.ArgumentParser(description=__doc__)`.
- A value that is a choice, such as a seed, a sample count, a tolerance, or a compiler option
  away from its default, carries a comment with its reason or the measurement behind it:
  `# Calibration draws only from cal2017, so no scored val2017 image calibrates the model.` The
  reason for a build parameter sits beside it in `model.yaml`:
  `# 1,024 backbone bins scored 0.7 COCO val2017 mAP above 2,048.`
- An error names the value it rejects and what was expected:
  `model.yaml declares no build profile 'w32a32'`. A bad argument fails through
  `parser.error`. A board script exits with `SystemExit(<message>)` when an input is missing.

## convert.py

- `main` reads the `--conversion` entry with `load_and_validate_model`, fetches its source with
  `Fetcher().fetch_source`, and calls `convert(source, output_dir)`. `convert` creates the
  output directory with `exist_ok=False`.
- `validate_source` rejects a declared source other than the checkpoint the exporter expects.
  `requirements-convert.txt` pins the exporter
  ([pin every input](../docs/adding-a-model.md#3-pin-every-input)).
- `validate_export` checks the written ONNX: the opset, the input and output names in order,
  their float dtypes and static shapes, and that the file holds every tensor itself.
- The script runs the upstream framework and ONNX Runtime on seeded random inputs (`SEEDS`),
  compares the tensors the package keeps with `np.testing.assert_allclose`, and prints one line
  that names the passed check and the output path.

## surgery.py

- The entry point is `prepare_model(source, output_dir)`. It returns the path of each graph it
  writes, one per component in `model.yaml` order.
- It checks the source's inputs and outputs before it rewrites anything, with
  `_require(condition, message)` raising `ValueError("Unsupported <model> graph: ...")`. An
  upstream change then fails at the check instead of inside the compiler.
- Each rewrite finds its node by tensor name and checks the node's operator type before it
  changes the node.
- Module constants name the component interface: `INPUT`, `INPUT_SHAPE`, `OUTPUT_SHAPES`, and
  `OUTPUTS`, plus the boundary tensors of a split, such as RF-DETR's `BOUNDARY_SHAPES`.
- Each graph it writes passes `onnx.checker.check_model` before `prepare_model` returns.

## compile.py

- The command line is the [ONNX recipe
  interface](../docs/adding-a-model.md#6-define-build-profiles). `main` rejects a
  `--compile-invocation` other than `--build-profile`, a `--model` that is not an existing
  `.onnx` file, and a missing dataset path. It then calls `compile_model`, logs each MPK path,
  and returns 0 to `raise SystemExit(main())`.
- It imports the component interface from `surgery.py` and the preprocessing from
  `calibration.py`, and restates neither.
- One function, `_parameters` or `_component_parameters`, reads the build profile from
  `model.yaml` with `load_model` and raises for an undeclared profile.
- One function turns a component's parameters into `QuantizationParams`, with one branch per
  precision. Only a component whose parameters set `calibration_samples` receives calibration
  data. The others use `SkipCalibration()`.
- The compiler works in `tempfile.TemporaryDirectory(prefix=f"{MODEL_ID}-",
  dir=os.environ.get("RUNNER_TEMP"))`. Only the files the interface lists reach
  `--output-dir`.
- A multi-component recipe runs each component's phase in its own process started with
  `multiprocessing.get_context("spawn")`, so the compiler releases its memory before the next
  component. It calibrates a later component on the earlier quantized component's outputs.

## calibration.py

- It holds the preprocessing the README's Inputs section describes (`preprocess_image`), the
  calibration sample selection, and each host step between components, such as RF-DETR's
  `select_top_proposals`.
- The selection reads a sorted list of names from a directory or a tar archive, draws with
  `random.Random(<seed>)`, and fails with both counts when the dataset holds too few samples.
- `accuracy.py` imports the preprocessing and the host steps from `calibration.py`. A copy in
  `accuracy.py` or `performance.py` names, in its docstring, the `calibration.py` function whose
  values it matches: `"""calibration.select_top_proposals, without the Pillow import this stage
  cannot load."""`.

## performance.py

- It follows the [canonical script](../docs/benchmarking.md#the-performance-script) and the
  [benchmark input](../docs/benchmarking.md#benchmark-input), or the [multi-component
  script](../docs/benchmarking.md#multi-component-models).
- `workload` is one line that names the dataset, the image, the frame, the preprocessing, and
  the decoder settings the script chose.

## accuracy.py

- `main(argv=None) -> int` parses the [accuracy
  arguments](../docs/local-accuracy-development.md#the-accuracy-script) and returns 0 to
  `raise SystemExit(main())`.
- `parse_mpks` accepts exactly the component IDs `model.yaml` declares and exits with them:
  `--mpk must name exactly backbone=<path> and decoder=<path>`.
- `--limit <n>` scores `random.Random(0).sample(available, n)`.
- `decode` turns one sample's MPK outputs into the scorer's rows. It first checks the output
  shapes and that every value is finite.
- Progress and scorer messages go to stderr through `log()`. The last line on stdout is the
  rounded metrics as one JSON object.
- Each dataset's report entry holds `metrics`, `split`, `evaluated_samples`,
  `available_samples`, and `details` with `inference_seconds`.

## Tests

- A test imports a model script with the `load_script` fixture
  ([the tests step](../docs/adding-a-model.md#10-write-the-models-tests)).
- A test's name states the behavior it checks: `test_surgery_requires_a_folded_source`.
- A test body sets up its inputs, leaves a blank line, calls the code under test, leaves a blank
  line, and asserts.
- A check function, such as `validate_export` or a comparison, gets one parametrized test. Its
  `valid` case passes, and each other case breaks one property and expects the error.
- Build-stage tests make their own inputs, such as a small graph from `onnx.helper`, a manifest
  written into a tar, or placeholder images in a temporary archive. They download nothing.
- `tests/build/test_conversion.py` covers `validate_source`, `validate_export`, and the
  framework comparison.
- `tests/build/test_compile.py` checks that the recipe reads each declared build profile and
  rejects an undeclared one, and runs each surgery rewrite on a fixture graph.
- `tests/benchmark/test_performance.py` holds the checks the [tests
  step](../docs/adding-a-model.md#set-up-teststestsyaml) shows and the [MLA stats
  check](../docs/benchmarking.md#mla-profiling). For an image model it also checks that the
  first execution-plan edge carries the `[1, 720, 1280, 3]` benchmark frame.
- A host step between components has a test with exact expected values, such as RF-DETR's
  `test_top_proposals_keep_equal_scores_in_order`.
