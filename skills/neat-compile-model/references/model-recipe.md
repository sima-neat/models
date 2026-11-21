# Model recipe

The Models repository keeps one folder per model with every script that converts,
compiles, and measures it. CI runs the same scripts to produce the numbers on the
[Model Browser](https://developer.sima.ai/models/), so your rebuild of a published
model can match its results. A recipe laid out the same way pins every input and
keeps every rewrite in code, so another person can rebuild it.

The public pages hold the full rules:

| Page | It covers |
| --- | --- |
| [Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md) | reproducing one published model, step by step |
| [Define the model contract](https://github.com/sima-neat/models/blob/main/docs/adding-a-model.md) | the folder, input pins, build profiles, and the `compile.py` interface |
| [Converted ONNX artifacts](https://github.com/sima-neat/models/blob/main/docs/conversion-artifacts.md) | what `convert.py` exports and what belongs in `surgery.py` |
| [Qualify locally](https://github.com/sima-neat/models/blob/main/docs/local-development.md) | validating, fetching, compiling, and checking MPKs |
| [`model.yaml` reference](https://github.com/sima-neat/models/blob/main/docs/metadata-schema.md) | every `model.yaml` field |

## The path every model takes

```text
model.yaml pins the official model
  -> convert.py exports it to ONNX
  -> compile.py runs surgery.py, quantizes with calibration data, writes one MPK per component
  -> performance.py and accuracy.py measure the MPKs on a DevKit
```

A model that pins an ONNX file under `compilation.source` has no `convert.py`
and starts at `compile.py`.

## The folder

The folder name is the model ID, in lowercase snake case. `model.yaml` does not
repeat it.

```text
models/<model-id>/
  model.yaml                      # the model's only configuration file
  scripts/
    convert.py                    # only when model.yaml declares a conversion
    requirements-convert.txt      # required with convert.py
    surgery.py                    # only when the recipe rewrites the graph
    calibration.py                # a helper: model-specific calibration or shared code
    compile.py                    # compile stage: writes one MPK per component
    performance.py                # performance stage, when model.yaml declares performance
    accuracy.py                   # accuracy stage, when model.yaml declares accuracy
  tests/                          # the model's tests
```

Put every model-specific setting in `model.yaml`, not in separate configuration
files. Keep board-only code inside `performance.py` or `accuracy.py`. For ONNX,
every `.py` or `.sh` file under `scripts/` except `convert.py`, `performance.py`,
`accuracy.py`, and `fp32_baseline_accuracy.py` is a compile input. A change to a
helper therefore recompiles the model, even when only `accuracy.py` imports it.

## `model.yaml`

`model.yaml` declares everything the scripts use:

| Field | What it holds |
| --- | --- |
| `upstream` | the official model's `repository` and `revision` |
| `license` | the source model's license: `spdx`, `redistribution`, and `attribution` |
| `conversion.source` | the checkpoint `convert.py` downloads, with its SHA-256 and size |
| `conversion.outputs` | the files `convert.py` writes, `[model.onnx]` by default |
| `compilation` | the source's `format` (`onnx` or `huggingface`), its pinned `source` (or `conversion`), and the `components` in run order |
| `calibrations` | the dataset archives build profiles calibrate on |
| `build_profiles` | the variants you can build, with each component's compiler `parameters` |
| `numerical_fidelity` | optional compiled-versus-source comparison during compilation |
| `accuracy`, `performance` | the DevKit stages the model declares |
| `reference_metrics` | the scores the official model publishes |

### Pin every input

Every source file carries its SHA-256 and byte count, so everyone starts from
the same bytes. Prefer a Hugging Face checkpoint pinned to a full commit. Use
another approved form only when Hugging Face does not have the checkpoint, and
say why in a comment in `model.yaml`:

| Host | Form |
| --- | --- |
| Hugging Face | `hf://models/<owner>/<repository>@<commit>/<file>`, with the full 40-character commit |
| GitHub | `https://github.com/<owner>/<repository>/releases/download/<tag>/<file>`, a release file of a public repository |
| GitHub raw files | `https://raw.githubusercontent.com/<owner>/<repository>/<commit>/<file>`, with the full 40-character commit |
| OpenMMLab | `https://download.openmmlab.com/<project>/<version>/<config>/<name>-<hash>.pth`, a released checkpoint whose name ends with the first 8 hex digits of its SHA-256 |

The owner of a repository can replace a GitHub release file, so only
`conversion.source` pins one. Mutable branches, `releases/latest`, credentials,
query parameters, and fragments are rejected. Record each hash from the exact
bytes:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

A source is one file, or a file set with the `main_file` the recipe opens:

```yaml
compilation:
  format: onnx
  source:
    uri: hf://models/owner/repo@0123456789abcdef0123456789abcdef01234567/model.onnx
    sha256: <SHA-256 of model.onnx>
    size_bytes: 123456
```

### Build profiles

A build profile is one variant users can select. It gives every component's
compiler parameters, keyed by component ID. A single-component model has one
component, `model`:

```yaml
build_profiles:
  w8a8:
    calibration_dataset: coco-2017-cal
    components:
      model:
        parameters: {precision: w8a8}
```

Name a profile by the precision token of each component, in component order,
with a repeated token written once: `w8a8`, `w16a16`, or `w8a8_w16a16` for an
INT8 backbone followed by a BF16 decoder. The recipe maps
`parameters.precision` to compiler settings, and the target is `modalix`. A
profile commented out in `model.yaml` is not supported. `model.yaml` names no
compiler version.

### Datasets

Each stage pins the dataset archive it uses with its `id`, `variant`, `sha256`,
and `size_bytes`. An archive goes by its source ID, `<id>-<variant>`, such as
`coco-2017-val`. A build profile names its calibration with
`calibration_dataset`.

## `convert.py`

`convert.py` exports the official model as it is: its forward pass at the input
shape the package ships, from the model's input to the final outputs the package
keeps. It may simplify the ONNX with `onnxsim.simplify`, including constant
folding, when that keeps the same inputs, outputs, and computation.

- It receives `--conversion <entry>` and `--output-dir <directory>`, and writes
  `model.onnx` there, or the files `conversion.outputs` names.
- YOLO26n's `convert.py` downloads its checkpoint, checks it against the SHA-256
  in `model.yaml`, and then exports it.
- `requirements-convert.txt` pins its packages.
  `requirements-convert-nodeps.txt` optionally pins packages to install without
  their dependencies.
- Scripts read the output names from `model.yaml` with `conversion_outputs()`
  and never hard-code them, so a rename takes one edit.
- The output directory must not exist. The script creates it.

`scripts/run_model_conversion.sh <model-dir> default <output-dir>` runs it in a
fresh Python 3.12 environment. `default` names the model's one conversion.

## `surgery.py`

Every other change made for the compiler belongs in `surgery.py`, even one that
leaves the results exactly the same. That includes a cut: an export that ends at
an inner tensor, such as the raw heads before the official decode step, or one
that starts after an official step. This split keeps the converted ONNX
comparable with the official model.

- `compile.py` runs `surgery.py` first. YOLO26n's surgery cuts the graph at its
  raw detection heads, before the box decode. A model without `surgery.py`
  compiles the ONNX as it is.
- For graph surgery in PyTorch, publish the checkpoint with
  `outputs: [weights.pth]`. `surgery.py` edits the model and exports the ONNX to
  compile, so `compilation.format` stays `onnx`.
- When surgery derives files the application runs on the CPU, such as a
  tokenizer or lookup tables, `compile.py` writes them under the profile's output
  directory and lists each `path` and `sha256` as `runtime_files` in
  `compile-outputs.json`.

This skill's repair and optimization steps decide which rewrites `surgery.py`
makes.

## `calibration.py`

`calibration.py` is a helper beside `compile.py` for model-specific calibration
code, such as reading the calibration images and picking the samples with a
fixed seed. ResNet-50's
[`calibration.py`](https://github.com/sima-neat/models/blob/main/models/resnet_50/scripts/calibration.py)
is one example.

- Calibration feeds quantization: the compiler runs sample images to choose the
  INT8 value ranges. It is not accuracy evidence.
- Only a profile with a `calibration_dataset` needs it. In YOLO26n, `w8a8`
  calibrates on 50 images from a folder named `cal2017`, and `w16a16` and
  `w8a16` take no calibration dataset.
- The recipe calibrates the first component on the profile's calibration
  dataset and later components on its outputs.
- Results match the published ones only when calibration reads the same
  samples. Other images give a different INT8 model.
- When `accuracy.py` imports it, as ResNet-50's does, copy it to the DevKit
  beside `accuracy.py`. It stays a compile input.

## `compile.py`

An ONNX recipe takes this interface:

```text
scripts/compile.py --model <verified-source> --output-dir <new-directory> \
  --build-profile <profile-id> --compile-invocation <profile-id> \
  [--dataset <verified-calibration>] [--fidelity-dataset <verified-fidelity-dataset>]
```

`--compile-invocation` repeats the profile ID, because a profile compiles all
its components in one invocation named after it. The recipe does this:

- It builds only a declared build profile, and stops with an error that names
  any other.
- It reads its build profile from `model.yaml`. In the Models repository it
  uses `model_registry.model.load_model`.
- It rewrites the graph with `surgery.py` and quantizes it. It then writes
  exactly one `<model-id>_<profile-id>_mpk.tar.gz` under
  `<output-dir>/<component-id>/` for each component, by passing
  `<model-id>_<profile-id>` to the compiler as the model name. A profile with
  several components adds `_<component-id>` to both.
- It writes `fidelity.json` in `<output-dir>` when `model.yaml` declares
  `numerical_fidelity`, and stops before compiling when `--fidelity-dataset` is
  missing.
- For several components, it also writes `<output-dir>/compile-outputs.json`
  with each component's MPK path and SHA-256 and its derived graph's SHA-256.
- It never downloads undeclared data. Download, checksum, MPK validation, and
  publication stay in shared tooling, out of model code.

### Quantization in `compile.py`

One `QuantizationParams` call per precision sets every field that build needs, so a reader sees
a build's settings in one place.

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
        per_token_quant=True,
    )
raise ValueError(f"unsupported precision: {precision}")
```

A build that skips calibration passes no data. A placeholder makes the build read as calibrated.

```python
# Bad: a zero image stands in for calibration data.
samples = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
```

```python
# Good
calibration_data = None
if "calibration_samples" in parameters:
    samples = calibration_images(dataset, int(parameters["calibration_samples"]))
    calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT): samples}))
```

## Reproduce a published model

[Getting started](https://github.com/sima-neat/models/blob/main/docs/getting-started.md)
walks through YOLO26n `w8a8`. Replace `yolo26n` and `w8a8` for another model or
variant. Run each command from the root of a clone of the Models repository.

| You need | Check it |
| --- | --- |
| Python 3.12 and internet access, for the conversion | `python3.12 --version` |
| The Model Compiler at the version the repository's [`deps/manifest.json`](https://github.com/sima-neat/models/blob/main/deps/manifest.json) pins under `model-compiler`, active in the compile shell | `python -c 'import afe; print("Model Compiler Python APIs available")'` |
| A Modalix DevKit with PyNeat, for the measurements | `~/pyneat/bin/python -c 'import pyneat; print(pyneat.__version__)'` |

1. Pick a build profile declared in `models/yolo26n/model.yaml`.
2. Get the calibration images. Without SiMa access, create a folder named
   `cal2017` with at least 50 COCO images from outside `val2017`, and set
   `calibration` to the directory that holds it. On SiMa's private network or
   VPN, with Dataset Registry access as
   [Datasets](https://github.com/sima-neat/models/blob/main/docs/datasets.md#set-up-artifactory-access-first)
   describes, resolve the registered archive instead:

   ```bash
   calibration="$(model-registry fetch-calibration models/yolo26n \
     --build-profile w8a8 --component model --json | jq -r .path)"
   ```
3. Convert the checkpoint:

   ```bash
   scripts/run_model_conversion.sh models/yolo26n default build/yolo26n/onnx
   ```

4. Compile, with the Model Compiler environment active:

   ```bash
   PYTHONPATH=tools python models/yolo26n/scripts/compile.py \
     --model build/yolo26n/onnx/model.onnx \
     --output-dir build/yolo26n/w8a8 \
     --build-profile w8a8 \
     --compile-invocation w8a8 \
     --dataset "$calibration"
   ```

   `PYTHONPATH=tools` lets `compile.py` import the `model_registry` package that
   reads `model.yaml`. The compiler's summary ends with a line that starts with
   `Compiled model:`.
5. Check the archive:

   ```bash
   PYTHONPATH=tools python -m model_registry verify-mpk build/yolo26n/w8a8/model
   ```

   It prints `Valid artifact: yolo26n_w8a8_mpk.tar.gz`.
6. Measure the MPK on a DevKit with the `neat-benchmark-model` skill.

`scripts/run_model_compilation.sh` is the runner CI uses. For a model whose
`compilation.source` is `conversion`, it fetches the ONNX that CI published,
which needs CI's registry credentials. Call `compile.py` with your own ONNX
instead.

| What you see | Fix |
| --- | --- |
| `No module named 'afe'` | Activate the Model Compiler environment. In the Neat SDK, run `activate-model-compiler`. |
| `No module named 'model_registry'` | Run from the repository root with `PYTHONPATH=tools`. |
| `FileExistsError` from `convert.py` or `compile.py` | Delete the output directory, or name a new one. |
| `model.yaml declares no build profile` or `expected exactly one build profile` | Pass a name from `build_profiles`. |
| `the INT8 build requires --dataset from dataset-registry` | Pass the calibration archive or directory. |
| `cannot select 50 calibration images from 0 cal2017 images` | Put at least 50 images in a folder named `cal2017`. |
| `model.yaml declares numerical_fidelity; pass --fidelity-dataset` | Pass the dataset that `numerical_fidelity.dataset` names. |
| `yolo26n: resolve conversion with plan-stages before compilation` | Pass your own ONNX to `compile.py` with `--model`. |

A model with `compilation.format: huggingface` is a language, vision-language,
or speech model. Its `scripts/compile.sh` compiles the pinned checkpoint through
`scripts/run_llima_compilation.sh` into a deployment folder, not an MPK.
