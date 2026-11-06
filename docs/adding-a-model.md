# Define the model contract

A Model Registry contribution describes a product that developers can find,
build, test, and download reproducibly. For a simple model, that product is one
ONNX input and one compiled MPK. A larger product compiles its source into
several components, which one download package holds. `model.yaml` declares
the product, and the model's `scripts/` directory holds the code that compiles
and measures it.

This guide ends when the contract validates. Then follow [Qualify
locally](local-development.md) to compile and measure every build before you
open a pull request.

## 1. Decide what the product contains

Before creating files, identify:

- the product and its responsible owner;
- the source the recipe compiles, and whether it is one file or a set of
  related files;
- the components the recipe compiles that source into, in run order;
- the public entrypoint, concrete compiler inputs, preprocessing and decoding;
- the build profiles users choose between and each component's compiler
  parameters;
- the calibration, fidelity, and accuracy data each build needs.

Do not add a product whose source, ownership, or license is ambiguous. Record
ownership in `CODEOWNERS`; do not add owner or maturity boilerplate to YAML.

## 2. Create the directory

The directory name supplies the lowercase snake-case model ID; `model.yaml`
does not repeat it. A model directory may contain only these entries:

```text
models/<model-id>/
  README.md
  hero.png                        # 1200x480 catalog card image
  model.yaml                      # the model's only configuration file
  scripts/
    compile.py                    # compile stage: writes one MPK per component
    performance.py                # performance stage, when model.yaml declares performance
    requirements-performance.txt  # optional wheels for performance.py
    accuracy.py                   # accuracy stage, when model.yaml declares accuracy
    requirements-accuracy.txt     # optional wheels for accuracy.py
    convert.py                    # only when model.yaml declares a conversion
    requirements-convert.txt      # required with convert.py
    <helper>.py                   # model-specific surgery, calibration, or shared code
  tests/                          # the model's tests; see step 10
    tests.yaml                    # which stage tests run, and the expected DevKit values
    build/test_conversion.py      # only with convert.py
    build/test_compile.py
    benchmark/test_accuracy.py    # only when model.yaml declares accuracy
    benchmark/test_performance.py # only when model.yaml declares performance
    fixtures/                     # optional test data
```

Put every model-specific setting in `model.yaml`, not in separate
configuration files. [Step 11](#11-write-the-readme) describes the README.

Keep board-only code inside `performance.py` or `accuracy.py`. For ONNX, any other
`.py` or `.sh` file under `scripts/` is a compile input, so a helper that
`accuracy.py` imports recompiles the model when it changes. See [step
9](#9-know-what-a-change-reruns).

## 3. Pin every input

`compilation.source` pins the one source the recipe compiles: one file, or a
file set when the recipe needs related files in a fixed relative layout.
Every file retains its SHA-256 and byte count. Hugging Face file sets can share
one immutable repository/revision declaration; see [Compilation and
sources](metadata-schema.md#compilation-and-sources). Exceptional members can
keep explicit URIs.

Hugging Face sources use a full 40-character repository commit. Approved HTTPS
sources include pinned Hugging Face resources and the ONNX files the registry
converts. GitHub release assets are not sources; datasets come only from the
Dataset Registry.
Mutable branches, credentials, query parameters, fragments, and unapproved
hosts are rejected.

Record hashes from the exact bytes the registry will fetch:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

Do not commit downloaded inputs. Once they are declared, check them:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id> --component <component-id>
```

`--component` may be omitted when the model has one component.

When upstream publishes a checkpoint instead of ONNX, declare it under
`conversion` and provide `scripts/convert.py` with
`scripts/requirements-convert.txt`. `convert.py` receives `--conversion
<entry>` and `--output-dir <directory>` and writes `model.onnx` there.
`scripts/run_model_conversion.sh` runs it in a fresh Python 3.12 environment:

```bash
scripts/run_model_conversion.sh models/<model-id> default /tmp/<model-id>-export
```

The **Vulcan Model Convert** workflow runs the same script and publishes
a validated ONNX bundle. Declare `compilation.source: conversion` after matching
output is registered. Editing the exporter requires a new conversion; compilation
reuses a previous build when the resolved output bytes are unchanged. See
[conversion artifacts](conversion-artifacts.md).

## 4. Declare the compilation

`compilation` names the source's `format`, `onnx` or `huggingface`, and its
pinned `source`. Conventional recipes are `scripts/compile.py` and
`scripts/compile.sh`; do not repeat their paths. A single-component model
compiles one component, `model`. A recipe that splits the source into several
MPKs lists them in run order under `components`; see [Compilation and
sources](metadata-schema.md#compilation-and-sources).

The public entrypoint defaults to `default`. Qwen/Gemma set `entrypoint: generate`
and Whisper sets `entrypoint: transcribe`, independently of the source's `main_file`.
Keep preprocessing, decoding and surgery in executable code and describe their
application-facing behavior in the README. Put concrete compiler shape overrides
in build parameters. Core execution plans supply tensor details after compilation.

## 5. Declare datasets and calibration

Each stage pins the Dataset Registry archive it uses: a `calibrations` entry,
`numerical_fidelity.dataset`, and each `accuracy.datasets` entry. A pin gives
the dataset ID, variant, SHA-256 and size. An archive goes by its Registry
source ID, `<id>-<variant>` such as `coco-2017-val`. A build profile names its
calibration with `calibration_dataset`. Keep sample selections in their
consuming stage.

Calibration is compiler input for quantization. It is not accuracy evidence.
Calibration, fidelity, and accuracy datasets resolve through Dataset Registry
and need working Artifactory access. See [Datasets for local model
development](datasets.md).

## 6. Define build profiles

A build profile is one variant users can select, such as `w8a8`. It gives
every component's compiler parameters, keyed by component ID:

```yaml
build_profiles:
  w8a8:
    calibration_dataset: imagenet-2012-mini
    components:
      model:
        parameters: {precision: w8a8}
```

Name the profile by its variant ID: the precision token of each component from
[`naming-conventions.yaml`](../naming-conventions.yaml), in component order,
with a repeated token written once, such as `w8a8`, `w16a16`, or
`w8a8_w16a16`. Each compiled MPK carries the model and variant ID, such as
`yolo26n_w8a8_mpk.tar.gz`. See [Build profiles](metadata-schema.md#build-profiles)
for the full rules.

A profile compiles all its components in one compile invocation, named after
the profile. The recipe calibrates the first component on the profile's
`calibration_dataset` and later components on its outputs. The build target is
`modalix`; keep runner class and graph-changing parameters explicit.
Profiles whose components differ in precision still declare each component's
precision rather than deriving it from the profile ID.

Speculative profiles can link their normal-decoding model and profile through
`benchmark_baseline`; see [Build profiles](metadata-schema.md#build-profiles).

ONNX recipes use the following interface. For Hugging Face checkpoints, use the
[LLiMa recipe](#llima-language-and-vision-language-models) instead:

```text
scripts/compile.py --model <verified-source> --output-dir <new-directory> \
  --build-profile <profile-id> --compile-invocation <profile-id> \
  [--dataset <verified-calibration>] [--fidelity-dataset <verified-fidelity-dataset>]
```

`--model` is the verified source. The runner passes `--dataset` when the
profile names a calibration and `--fidelity-dataset` when `model.yaml`
declares `numerical_fidelity`. The recipe:

- reads its effective build profile with `model_registry.model.load_model`, so
  dataset names and omitted defaults reach execution consistently;
- writes exactly one `<model-id>_<profile-id>_mpk.tar.gz` under
  `<output-dir>/<component-id>/` for each component, by passing
  `<model-id>_<profile-id>` to the compiler as the model name; a profile with
  several components adds `_<component-id>` to both;
- writes `fidelity.json` in `<output-dir>` when `model.yaml` declares
  `numerical_fidelity`;
- for several components, also writes `<output-dir>/compile-outputs.json`
  with `outputs` (each component's MPK path and SHA-256) and `derived_graphs`
  (each component's derived graph SHA-256); and
- never downloads undeclared data.

For a single-component model, `scripts/run_model_compilation.sh` writes
`compile-outputs.json` when the recipe does not. It then runs
`model-registry simulate-mla-stages`, which writes `mla-stage-simulation.json`:
the Model Compiler simulator's operation counts and latency for each MLA stage. Shared download, checksum, MPK
validation, and publication code belongs in `tools/`, not in model code.

Set `runner_class` and `runner_market` at profile level for the shared compile
runner. They default independently to `default` and `spot`. `expensive` and
`most-expensive` need a sufficient Vulcan size grant; `on-demand` needs a separate
market grant even at default size. An approval permits a request but does not
select it or start a build. Follow [requesting expensive or on-demand
runners](runner-class-requests.md) for manifest examples, evidence, approval,
validation and dispatch. Update the owning README when changing the request.
Do not confuse the market with `build-on-demand`, which controls scheduling.

## 7. Know what the package holds

Every build profile publishes one package, `default`, holding all its
components; there is nothing to declare. For a multi-component product, add
`pipeline` so the model browser shows the order the components run in and the
CPU steps between them; see [Packages and
pipeline](metadata-schema.md#packages-and-pipeline).

## 8. Declare quality and performance

Omit the section of a stage the model does not implement. Each section covers
the whole model: every component of a build profile together. Without
`build_profiles`, `accuracy` and `performance` measure every profile; keep
restrictions such as INT8-only accuracy.

| Section | Measured by | Runs in |
| --- | --- | --- |
| `numerical_fidelity` | `compile.py`, comparing the compiled graph with the source model | compile stage |
| `accuracy` | `accuracy.py`, running the profile's compiled MPKs | accuracy stage |
| `performance` | the model's standalone `performance.py` | performance stage |

Fidelity is diagnostic numerical agreement, not task accuracy. `accuracy`
names the datasets it evaluates, and each dataset's
[requirements file](datasets.md#required-accuracy-metrics) lists the metrics
its result must hold for the model's tasks. Every `performance` report holds
the metrics [`metrics.json`](../tools/model_report/metrics.json) marks
`required`. Decoder choices and measurement settings stay in the scripts. See [Quality and
performance](metadata-schema.md#quality-and-performance).

Record the scores the source model's authors published under
[`reference_metrics`](metadata-schema.md#reference-metrics), with the dataset,
conditions, and pinned page of each. Record only numbers the page prints for
these weights; leave the field out when none are published.

Follow [Accuracy and fidelity](local-accuracy-development.md) for the fidelity
and accuracy scripts, and [DevKit performance](benchmarking.md#the-performance-script)
for the performance script, including [multi-component
models](benchmarking.md#multi-component-models).

## 9. Know what a change reruns

Each stage has a fingerprint, a SHA-256 of every input that decides its result.
CI compiles a build only when neither the branch's catalog nor, on a feature
branch, a base branch's catalog records a build with its current compile
fingerprint. It runs a benchmark (the performance or accuracy stage) only when
no completed registry run carries its current benchmark fingerprint. Models
with `build-on-demand: true` compile only when a manual dispatch selects them,
and their builds are not benchmarked; see [on-demand
builds](#on-demand-builds).

| Stage | Fingerprint inputs |
| --- | --- |
| conversion | what can change the ONNX: the upstream checkpoint, `convert.py` and the files it uses, and the shared export code. See [when to run Convert again](conversion-artifacts.md#when-to-run-convert-again) |
| compile (ONNX) | the model and profile IDs, the target, each component's ID and the source identity and layout, each component's `parameters`, the profile's calibration pin, the `numerical_fidelity` section, compile/surgery code, the Model Compiler version, `scripts/run_model_compilation.sh`, the shared model loader, and the MLA stage simulation (`tools/model_registry/mla_stage_simulation.py`) |
| compile (LLiMa) | checkpoint content hashes, the target, component graph `parameters`, and the pinned Model Compiler identity including its build revision |
| performance | the compile run and MPK digest of every component, the `performance` declaration, the profile's optional `benchmark_baseline`, the model's `performance.py` and `requirements-performance.txt`, the shared benchmark tooling and `scripts/run_model_performance.sh`, and the DevKit runtime pin |
| accuracy | the compile run and MPK digest of every component, each `accuracy` dataset's name, pin, and the metrics its `datasets/<id>.yaml` requires for the model's tasks, `accuracy.py` and `requirements-accuracy.txt`, the shared benchmark tooling, `scripts/run_model_accuracy.sh`, `scripts/cache_accuracy_dataset.sh` and `scripts/install_dataset_registry.sh`, and the DevKit runtime pin |

ONNX compile code includes every `.py` and `.sh` under `scripts/` except
`performance.py`, `accuracy.py`, `convert.py` and `requirements-convert.txt`.
The shared model loader is an input to every ONNX compile fingerprint;
descriptive preprocessing and surgery declarations are not. The shared
benchmark tooling is `BOARD_STAGE_TOOLING`: the benchmarking workflow, `tools/model_report`
(`__init__.py` and `metrics.json`), `schemas/model-report.schema.json`, and
`tools/model_registry/report_publication.py`.

A new compile run changes the benchmark fingerprints too, so a recompiled build
is always benchmarked again.

| Change | Reruns |
| --- | --- |
| README, display metadata, `pipeline`, `reference_metrics`, `runner_class`, `runner_market`, registry tooling outside a stage's tooling | nothing |
| `build-on-demand` | nothing; changes scheduling and catalog metadata only |
| the upstream checkpoint, `convert.py` and the files it uses, or the shared export code | with `source: conversion`, planning fails until you run Vulcan Model Convert again. The model recompiles only if the ONNX changes. |
| a model's tests, or `scripts/run_model_tests.sh` | nothing. The changed test runs the next time its stage runs. |
| a model's `performance.py`, its requirements, or `performance` declaration | that model's performance stages |
| a profile's `benchmark_baseline` | that profile's performance; no compilation |
| `accuracy.py`, its requirements, the `accuracy` datasets, or one of their pins | that model's accuracy stages |
| a task's entry in `datasets/<id>.yaml` | the accuracy stages of models with that task that evaluate the dataset |
| `accuracy.build_profiles`, a new `datasets/` file, or an entry for a task no such model has | nothing; a newly benchmarked profile runs because it has no run yet |
| `scripts/run_model_performance.sh` | every performance stage |
| `scripts/run_model_accuracy.sh`, `scripts/cache_accuracy_dataset.sh`, or `scripts/install_dataset_registry.sh` | every accuracy stage |
| the DevKit runtime pin, the benchmarking workflow, `tools/model_report`, `schemas/model-report.schema.json`, or `report_publication.py` | every performance and accuracy stage |
| a source, a component's `parameters`, the profile's calibration, or `numerical_fidelity` | that model's affected builds, then their benchmarks |
| an ONNX compile script or other ONNX compile input | that model's affected builds, then their benchmarks |
| the Model Compiler pin | every build, then every benchmark |
| `scripts/run_model_compilation.sh` | ONNX builds, then their benchmarks |
| the shared model loader | ONNX builds, then their benchmarks |
| `tools/model_registry/mla_stage_simulation.py` | ONNX builds, then their benchmarks |
| LLiMa scripts, registry helpers, checkpoint repository name, job count, logging or runner class/market | no LLiMa compilation |

The fingerprints are computed by `compile_fingerprint` and `benchmark_fingerprint` in
`tools/model_registry/fingerprints.py`, and the conversion fingerprint by
`conversion_fingerprint` in `tools/model_registry/conversion.py`.

To apply a changed test at once, dispatch **Vulcan Model Orchestrator** with
`models` set to the model and `force_rebuild_all`: it recompiles the model's
builds, which runs the compile tests and remeasures every build. To run a changed
conversion test, dispatch **Vulcan Model Convert**.

## 10. Write the model's tests

Each model carries its own tests in `tests/`, one file per stage, named by
what it checks. A test either exercises the model's scripts, or checks the real
output of its stage by taking that output's fixture from the root
`conftest.py`:

| File | Exercises | Stage-output fixture | That test runs in CI |
| --- | --- | --- | --- |
| `build/test_conversion.py` | `convert.py` | `converted`: the ONNX path | after conversion, before the ONNX upload |
| `build/test_compile.py` | `compile.py`, calibration, surgery | `compiled`: the build's output directory | after compilation, before the build is published |
| `benchmark/test_accuracy.py` | `accuracy.py` | `accuracy`: the build profile's accuracy report | after the DevKit accuracy stage, before publication |
| `benchmark/test_performance.py` | `performance.py` | `performance`: the build profile's performance report | after the DevKit performance stage, before publication |

A test without a stage-output fixture runs in `python -m pytest` on every pull
request. Import a model script with the `load_script` fixture, as in
`load_script("compile")`; it stubs Model Compiler and DevKit modules the host
lacks. `build_profile` names the build profile of the given output.

A failing conversion or compile test stops the ONNX upload or the build's
publication. A failing measurement test publishes its report with test status
`failed`, then fails the run.

### Set up `tests/tests.yaml`

`tests/tests.yaml` has one entry per stage the model has: `conversion` when
the model has `scripts/convert.py`, `compile` always, `accuracy` when
`model.yaml` declares `accuracy`, and `performance` when it declares
`performance`. `model-registry validate` rejects a missing or extra entry.

Each entry sets `run_tests`. `true` runs the stage's test file. `false` skips
it and needs a `skip_reason`; a skipped measurement still publishes its report,
with test status `skipped`. Skip a stage only while its tests cannot run, for
example before the model has a published report to expect values from.

A running `accuracy` or `performance` entry also states what the model must
deliver on a DevKit in `expected`, keyed by every build profile the stage
measures: the latest published values of that profile's whole model. For
accuracy, key the values by each dataset `accuracy` names, and supply finite
values of accuracy metrics from
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json),
covering at least the metrics that dataset requires. Keep optional expectations such as
top-5: a missing expected measurement fails. For performance, supply
`throughput_fps`. Take the values from the model browser's latest `develop`
results, rounded to 4 decimal places for accuracy and 1 for throughput.

A measured value passes when it falls at most 5% below its expected accuracy
or 10% below its expected throughput. For a metric that `metrics.json` marks
lower is better, it may rise by that much instead. Every model uses these limits:
`ACCURACY_ALLOWED_DROP` and `PERFORMANCE_ALLOWED_DROP` in
`tools/model_registry/model_tests.py`.

```yaml
compile:
  run_tests: true
accuracy:
  run_tests: true
  expected:
    w8a8:
      imagenet-2012-mini:
        top1_accuracy: 0.7764
        top5_accuracy: 0.9356
performance:
  run_tests: true
  expected:
    w16a16:
      throughput_fps: 578.6
    w8a8:
      throughput_fps: 1491.4
```

`model-registry validate` requires an `expected` value for every build profile
the stage benchmarks. The benchmark test compares the report's metrics with these
values through one shared function. A performance test also checks that the
report records each component's execution plan, which the model browser draws:

```python
from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(
    performance: dict[str, Any], model_dir: Path, build_profile: str
) -> None:
    assert measured_shortfalls(model_dir, "performance", build_profile, performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    (component,) = performance["components"].values()
    plan = component["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]
```

Update `expected` when a reviewed change moves a published value, such as a
new compiler or a faster script, in the same pull request or right after its
reports publish. No fingerprint reads `tests/`, so editing `tests.yaml` reruns
nothing; the new values apply the next time the stage runs. To apply them
now, dispatch `Vulcan Model Orchestrator` with `models` set to the model and
`force_rebuild_all`, which recompiles and remeasures it.

### Run the tests locally

From a virtual environment with this checkout's development extra installed
(`python -m pip install '.[dev]'`), run the model's tests, and run one stage's
file on an output you produced:

```bash
python -m pytest models/<model-id>/tests
REGISTRY_PYTHON="$(command -v python)" \
  scripts/run_model_tests.sh performance models/<model-id> <output>/model-report.json <profile-id>
```

The accuracy and performance tests need the build profile the report
measured; the conversion and compile tests take only the output.

`run_model_tests.sh` prints `passed`, `failed`, or `skipped`, as CI publishes
it.

## 11. Write the README

The model page shows the README after what it reads from `model.yaml` and the
registry runs: metrics, benchmarks, variants, components, license, and source
links. The README adds what an application developer needs to run the model:
what it finds, how to prepare its input, and how to read its outputs. Write it
from this template:

````markdown
# <display_name>

![<display_name> model hero](hero.png)

## Overview

<50 to 120 words of prose: what the model finds, the input it takes, what it
returns, and the components it runs as.>

## Inputs and outputs

**Input.** <Each preparation step with its exact values: resize, padding,
color order, scaling, and normalization.>

**Between the components.** <Only for a model with several components: what
the application computes on the CPU and which tensors it passes on.>

**Outputs.** <Each actual output tensor by name, and what it holds.>

1. <Each decode step, in order, one line per step.>

<The class-index mapping, and the Neat decoder for these outputs, if one exists.>

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
````

Start from a README of the same kind:
[`yolo26n`](https://github.com/sima-neat/models/blob/develop/models/yolo26n/README.md)
for a one-component detector,
[`rf_detr_nano`](https://github.com/sima-neat/models/blob/develop/models/rf_detr_nano/README.md)
for a CPU step between two components, and
[`resnet_50`](https://github.com/sima-neat/models/blob/develop/models/resnet_50/README.md)
for a classifier. `model-registry validate-models` checks the structure, the
word limits, the Building applications text, and the internal terms a README
leaves out. The writing rules are in
[`AGENTS.md`](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).

Create the hero using the [geometric comic illustration guide](illustration-guide.md),
which includes the fixed palette, Roboto typography, reference heroes and reusable prompts.

## 12. Validate the contract

```bash
model-registry validate models/<model-id>
model-registry validate-models --repository .
```

These checks establish that the contract is coherent, not that it compiles or
runs. Continue with [Qualify locally](local-development.md) and complete the
full loop for every build before submitting.


## On-demand builds

Set top-level `build-on-demand: true` for any model whose expensive builds
should run only on request. The default is `false`; existing automatic ONNX
builds keep their scheduling. Pushes validate on-demand models and publish
their metadata, but never compile or measure them, even when inputs change.
This flag is not part of the compile fingerprint.

To build Qwen3.5-0.8B, dispatch **Vulcan Model Orchestrator** with
`models=qwen3_5_0_8b` and `profiles=w4a16`. Both selections are
required for an on-demand build. Use staging; production candidate publication
is not enabled. Direct `plan-stages` calls must also pass `--include-on-demand`.
The workflow publishes all invocations and packages as one retained candidate.
Its summary shows the candidate ID, immutable digest, source and protection;
on a feature branch it also selects the finalized deployment for that branch's
website and CLI downloads. This does not approve it for Develop or Main.
Retry failed jobs within the seven-day assembly deadline; after expiry start a
new workflow run. See the [architecture contract](https://github.com/sima-neat/models/blob/develop/design-doc/model-registry-architecture.md#on-demand-builds)
for retention and the separate promotion workflow.

## LLiMa language and vision-language models

Declare a complete language model as one component, with `compilation.format:
huggingface`, `compilation.entrypoint: generate`, and a pinned Hugging Face file
set whose `main_file` is `config.json`. Share the repository/revision while retaining every weight,
tokenizer, processor and chat-template file's path, size and SHA-256. The conventional
recipe is `scripts/compile.sh`. The verified downloader assembles the checkpoint
directory; do not list LLiMa's generated layers as components.

A build profile names one compiled configuration. Start with one profile per
model and set `max_num_tokens: 4096` explicitly in its component `parameters`,
alongside `language_group_size`. Add `input_height` and `input_width` only for
VLMs needing a resolution override. Omit both for text-only LLMs and VLMs whose
native resolution is fixed. Published context and image dimensions come from
compiled parameters; no copied image tensor inventory is needed.

The shared runner forwards these parameters as compiler options after the
checkpoint and output arguments; `compile.sh` must pass them to `llima-compile`.
Keep these values in `model.yaml` rather than duplicating them in the recipe.

The recipe takes two positional paths followed by compiler options and only
compiles. Put supporting Python code in adjacent `compile.py` or `config.py`.
The shared runner writes a direct reproduction command with download instructions
and the resolved graph arguments, alongside those Python files. Reproduction
defaults to four processes; CI keeps the recipe's process count. The Qwen recipe
shows the simple command form; Whisper uses `compile.py` and Gemma MTP uses `config.py`.

Run it through the LLiMa runner, with Model Compiler and LLiMa activated. The
runner passes the selected profile to the recipe as `BUILD_PROFILE_ID`:

```bash
scripts/run_llima_compilation.sh models/qwen3_5_0_8b w4a16 w4a16 /path/to/output
```

The runner downloads the pinned files and passes the recipe a temporary symlink
named after `upstream.repository`, preserving the verified cache and giving
generated files the checkpoint's name. It then runs
`llima-deploy` into `<output-root>/<model>/<profile>/<component>/deployment/`.
This folder contains `devkit/`, `elf_files/`, any `npy_files/`, and the standalone
`compile.sh` with the selected context, image dimensions and other flags,
plus any `compile.py` or `config.py` needed to rerun it. Run
`model-registry verify-artifact <component-output> --write-manifest` to check
the deployment and write `deployment-manifest.json` beside it. The verifier
checksums every file and rejects folders without ELFs, unsafe paths, and links.

`compile-outputs.json` uses `artifact_path: model/deployment-manifest.json`
for Qwen's `model` component and `artifact_sha256` for that index file's hash.
ONNX outputs retain their MPK fields. Publication registers the deployment's
files individually, with a JSON index named `llima`. A native full package is
the deployment folder with `manifest.json` at its root; its component uses
`format: llima`, `path: .`. Retrieval preserves the folder and verifies each
file. See the [manifest and metadata contract](metadata-schema.md#llima-checkpoint-and-deployment-formats).
No native stage creates a deployment ZIP.

Whisper sets `entrypoint: transcribe` and uses a model-specific recipe calling
`WhisperModel` directly instead of `llima-compile`. Its profile selects A16W8
and the Whisper mask and log-probe options, without LLM context or image settings.
The runner forwards boolean parameters as `--name` or `--no-name`.

The native compile fingerprint covers checkpoint content, the target, graph parameters
and the pinned compiler identity. Declare graph-changing options in component
`parameters`, including precision, group size, context length and image dimensions.
Script edits, logging, job counts and runner selection do not invalidate it.
README and hero changes remain outside compile identities.

Omit `calibrations`, `numerical_fidelity`, `accuracy`, and `performance` when no
corresponding stage is implemented. A manual Qwen build compiles and publishes only; performance
and accuracy are separate follow-ups [#294](https://github.com/sima-neat/models/issues/294)
and [#295](https://github.com/sima-neat/models/issues/295). The default Qwen profile
requests the expensive runner with twenty compile jobs, 4096 tokens, 448×448 input,
and language group size 128.
