# Define the model contract

A Model Registry contribution describes a product that developers can find,
build, test, and download reproducibly. For a simple model, that product is one
ONNX input and one compiled MPK. A larger product can contain several compiled
components, runtime resources, a composition, and a download package.
`model.yaml` joins those pieces, and the model's `scripts/` directory holds the
code that compiles and measures them.

This guide ends when the contract validates. Then follow [Qualify
locally](local-development.md) to compile and measure every build before you
open a pull request.

## 1. Decide what the product contains

Before creating files, identify:

- the product and its responsible owner;
- each independently compiled ONNX component;
- whether a component source is one file or a set of related files;
- the public entrypoints, tensors, shapes, state, and preprocessing;
- runtime files such as tokenizers, labels, or configuration;
- the target-wide build profiles and component-specific compiler parameters;
- how the components compose and what users download;
- the calibration, fidelity, and accuracy data each build needs.

Do not add a product whose source, ownership, or license is ambiguous.
`experimental` maturity may describe unfinished qualification; it does not
permit mutable provenance.

## 2. Create the directory

The directory name and the `id` in `model.yaml` are the same lowercase
snake-case identifier. A model directory may contain only these entries:

```text
models/<model-id>/
  README.md
  hero.png                      # 1200x480 catalog card image
  model.yaml                    # the model's only configuration file
  scripts/
    compile.py                  # compile stage: writes one MPK per component
    performance.py              # performance stage: Core benchmark of one MPK on a DevKit
    accuracy.py                 # accuracy stage, when model.yaml declares DevKit accuracy
    requirements-accuracy.txt   # optional wheels for accuracy.py
    convert.py                  # only when model.yaml declares a conversion
    requirements-convert.txt    # required with convert.py
    <helper>.py                 # model-specific surgery, calibration, or shared code
  tests/                        # the model's tests; see step 10
    tests.yaml                  # which stage tests run, and the expected DevKit values
    build/test_conversion.py    # only with convert.py
    build/test_compile.py
    measure/test_accuracy.py    # only when model.yaml declares DevKit accuracy
    measure/test_performance.py # only when model.yaml declares a benchmark
    fixtures/                   # optional test data
```

Put every model-specific setting in `model.yaml`, not in separate
configuration files. [Step 11](#11-write-the-readme) describes the README.

Keep board-only code inside `performance.py` or `accuracy.py`. Any other
`.py` or `.sh` file under `scripts/` is a compile input, so a helper that
`accuracy.py` imports recompiles the model when it changes. See [step
9](#9-know-what-a-change-reruns).

## 3. Pin every input

Each compiled component owns its source. Use `type: file` for one artifact and
`type: file_set` when the recipe needs several related files in a fixed
relative layout. Every file gets its own immutable URI, SHA-256, byte count,
and media type.

Hugging Face sources use a full 40-character repository commit. Approved HTTPS
sources include pinned Hugging Face resources and GitHub release assets.
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

The **Vulcan Model Convert** workflow runs the same script, publishes
`model.onnx`, and prints the `source` block to pin in `model.yaml`. Only that
pin reaches the compile stage; editing `convert.py` recompiles nothing.

## 4. Describe components and interfaces

Declare one `components` entry for each independently compiled unit. A
component owns:

- its role and immutable source;
- named ONNX tensor entrypoints or a Hugging Face generation entrypoint;
- tensor, state, streaming, and concrete shape-profile contracts;
- preprocessing and postprocessing pipelines;
- optional model-specific surgery;
- its `recipe`, `scripts/compile.py` or `scripts/compile.sh`;
- optional dependencies on other components.

Names and tensor details must match the pinned inputs. Inspect the ONNX files;
do not copy them from a model card.

## 5. Add resources and calibration

Declare non-compiled files under `resources` and give each its package
`install_path`. Examples include tokenizers, vocabularies, labels, and runtime
configuration. Declare a separate license when a resource has different
distribution terms.

Declare calibration inputs under `calibrations`. A ready calibration references
either an immutable `source` or a registered dataset. A pending entry states
what is missing. A build-profile component binding selects a calibration by ID.

Calibration is compiler input for quantization. It is not accuracy evidence.
Calibration, fidelity, and accuracy datasets resolve through Dataset Registry
and need working Artifactory access. See [Datasets for local model
development](datasets.md).

## 6. Define build profiles and compile invocations

A build profile is one product-wide target configuration, such as
`w8a8`. Bind every participating component and put its compiler
parameters on the binding:

```yaml
build_profiles:
  - id: w8a8
    target: modalix
    components:
      - id: model
        calibration: default
        parameters: {precision: int8}
```

Name the profile by its variant ID: the precision token of each component from
[`naming-conventions.yaml`](../naming-conventions.yaml), in component order,
with a repeated token written once, such as `w8a8`, `w16a16`, or
`w8a8_w16a16`. Each compiled MPK carries the model and variant ID, such as
`yolo26n_w8a8_mpk.tar.gz`. See [Build profiles](metadata-schema.md#build-profiles)
for the full rules.

By default each supported binding is its own compile invocation, named after
its component. Components that share preparation or intermediate data compile
together in one declared invocation:

```yaml
    compile_invocations:
      - id: detector
        components: [backbone, transformer]
```

Grouped components must share one source, one effective runner class, and at
most one calibration, set on the first listed component. A package and a DevKit
accuracy entry must each compile all their components in one invocation.

A binding proven incompatible may stay as `status: unsupported` with a reason.
Do not use it for untested work.

ONNX recipes use the following interface for single and grouped invocations.
For Hugging Face checkpoints, use the [LLiMa recipe](#llima-language-and-vision-language-models) instead:

```text
scripts/compile.py --model <verified-source> --output-dir <new-directory> \
  --build-profile <profile-id> --compile-invocation <invocation-id> \
  [--dataset <verified-calibration>] [--fidelity-dataset <verified-fidelity-dataset>]
```

`--model` is the verified source of the invocation's first component. The
runner passes `--dataset` when the binding selects a ready calibration and
`--fidelity-dataset` when a `fidelity` entry applies. The recipe:

- reads its build profile from `model.yaml`;
- writes exactly one `<model-id>_<profile-id>_mpk.tar.gz` under
  `<output-dir>/<component-id>/` for each component of the invocation, by
  passing `<model-id>_<profile-id>` to the compiler as the model name; a
  profile with several components adds `_<component-id>` to both;
- writes `fidelity.json` beside the MPK of the fidelity entry's
  `result_component` when a `fidelity` entry applies;
- for a grouped invocation, also writes `<output-dir>/compile-outputs.json`
  with `outputs` (each component's MPK path and SHA-256), `derived_graphs`
  (each component's derived graph SHA-256), and `correctness` (the result
  component's `fidelity.json` and its SHA-256); and
- never downloads undeclared data.

For a single-component invocation, `scripts/run_model_compilation.sh` writes
`compile-outputs.json` when the recipe does not. Shared download, checksum, MPK
validation, and publication code belongs in `tools/`, not in model code.

Elevated `runner_class` requests need approval. See [Compile runner class
requests and approvals](runner-class-requests.md).

## 7. Define composition and packages

A one-component product uses a `single` composition naming its component and
entrypoint. A multi-component product uses a `declarative` composition whose
`components`, `connections`, `inputs`, and `outputs` are written inline in
`model.yaml`; see [Compositions and
packages](metadata-schema.md#compositions-and-packages).

Then declare one or more packages. A package selects:

- the components whose MPKs it contains;
- the runtime resources to install;
- the compatible build profiles;
- the composition that explains how to use them.

Exactly one package is `default: true`. CI assembles each package from the
build that compiles its components, so a user downloads the complete product
as one archive.

For a multi-component product, also add `pipeline` so the model browser shows
the order the components run in; see [Compositions and
packages](metadata-schema.md#compositions-and-packages).

## 8. Declare fidelity, accuracy, and benchmark entries

Each entry names its `scope` (a component or a package) and the
`build_profiles` it applies to. IDs are unique within each section.

| Section | Measured by | Runs in |
| --- | --- | --- |
| `fidelity` | `compile.py`, comparing the compiled graph with the source model | compile stage |
| `accuracy` with `engine: devkit_script` | `accuracy.py`, running the compiled MPKs | accuracy stage |
| `accuracy` without `engine` | nothing; the entry records results measured elsewhere | no stage |
| `benchmark` | `performance.py`, once per component | performance stage |

Fidelity is diagnostic numerical agreement, not task accuracy. A build profile
has at most one `devkit_script` accuracy entry. Every component with a
`benchmark` entry in a build profile gets its own performance result.

Record the scores the source model's authors published under
[`published_metrics`](metadata-schema.md#published-metrics), with the dataset,
conditions, and pinned page of each. Record only numbers the page prints for
these weights; leave the field out when none are published.

Follow [Accuracy and fidelity](local-accuracy-development.md) for the fidelity
and accuracy contracts. Start `performance.py` from the [canonical
script](benchmarking.md#the-performance-script): it writes Core's whole
benchmark report, including the board telemetry the model page shows.

## 9. Know what a change reruns

Each stage has a stage key, a SHA-256 of everything its result depends on. CI
compiles a build only when neither the branch's catalog nor, on a feature
branch, a base branch's catalog records a build with its current compile key.
It runs a DevKit stage only when no completed registry run carries that stage's
current key. Models with `build-on-demand: true` require an explicit manual
selection before either stage can run; see [on-demand builds](#on-demand-builds).

| Stage | Key inputs |
| --- | --- |
| compile | the invocation's components and bindings in `model.yaml` (source, interface, surgery, recipe, dependencies, parameters, shape profile, calibration), the build profile's target and compatibility, the `fidelity` entries for the profile, every `.py` and `.sh` file in `scripts/` except `performance.py`, `accuracy.py`, `requirements-accuracy.txt`, `convert.py`, and `requirements-convert.txt`, declared training code, the Model Compiler version, and backend tooling: `scripts/run_model_compilation.sh` for ONNX; `scripts/run_llima_compilation.sh`, `tools/model_registry/artifact.py`, and `tools/model_registry/cli.py` plus the `upstream.repository` basename for LLiMa |
| performance | the compile run and MPK digest, the component's `benchmark` entries, `performance.py`, the shared performance tooling, and the DevKit runtime pin |
| accuracy | the compile runs and MPK digests of its components, the `accuracy` entry, `accuracy.py` and `requirements-accuracy.txt`, the shared accuracy tooling, and the DevKit runtime pin |

A new compile run changes the performance and accuracy keys too, so a
recompiled build is always remeasured.

| Change | Reruns |
| --- | --- |
| README, display metadata, `published_metrics`, `convert.py`, registry tooling outside a stage's tooling | nothing |
| `build-on-demand` | nothing; changes scheduling and catalog metadata only |
| a model's tests under `tests/`, or `scripts/run_model_tests.sh` | nothing; a changed test applies the next time its stage runs |
| `performance.py` or a `benchmark` entry | that model's performance |
| `accuracy.py`, its requirements, or the `accuracy` entry | that model's accuracy |
| the DevKit runtime pin, the benchmarking workflow, or `benchmark_publication.py` | every performance and accuracy stage |
| a compile input of a model | that model's affected builds, then their DevKit stages |
| the `upstream.repository` basename of an LLiMa model | that model's builds, then their declared DevKit stages |
| the Model Compiler pin | every build, then every DevKit stage |
| `scripts/run_model_compilation.sh` | ONNX builds, then their DevKit stages |
| `scripts/run_llima_compilation.sh`, `artifact.py`, or `cli.py` | LLiMa builds, then their declared DevKit stages |

The keys are computed by `compile_key`, `performance_key` and `accuracy_key`
in `tools/model_registry/stage_keys.py`.

To apply a changed test at once, dispatch **Vulcan Model Orchestrator** with
`models` set to the model and `force_rebuild_all`: it recompiles the model's
builds, which runs the compile tests and remeasures every build. A changed
conversion test applies on the next conversion dispatch.

## 10. Write the model's tests

Each model carries its own tests in `tests/`, one file per stage, named by
what it checks. A test either exercises the model's scripts, or checks the real
output of its stage by taking that output's fixture from the root
`conftest.py`:

| File | Exercises | Stage-output fixture | That test runs in CI |
| --- | --- | --- | --- |
| `build/test_conversion.py` | `convert.py` | `converted`: the ONNX path | after conversion, before the ONNX upload |
| `build/test_compile.py` | `compile.py`, calibration, surgery, declared training code | `compiled`: the build's output directory | after compilation, before the build is published |
| `measure/test_accuracy.py` | `accuracy.py` | `accuracy`: the accuracy result | after the DevKit accuracy stage, before publication |
| `measure/test_performance.py` | `performance.py` | `performance`: one component's performance result | after the DevKit performance stage, before publication |

A test without a stage-output fixture runs in `python -m pytest` on every pull
request. Import a model script with the `load_script` fixture, as in
`load_script("compile")`; it stubs Model Compiler and DevKit modules the host
lacks. `build_profile` names the build profile of the given output.

A failing conversion or compile test stops the ONNX upload or the build's
publication. A failing measurement test publishes its result with test status
`failed`, then fails the run.

### Set up `tests/tests.yaml`

`tests/tests.yaml` has one entry per stage the model has: `conversion` when
the model has `scripts/convert.py`, `compile` always, `accuracy` when
`model.yaml` declares a `devkit_script` accuracy entry, and `performance` when
it declares a benchmark. `model-registry validate` rejects a missing or extra
entry.

Each entry sets `run_tests`. `true` runs the stage's test file. `false` skips
it and needs a `skip_reason`; a skipped measurement still publishes its result,
with test status `skipped`. Skip a stage only while its tests cannot run, for
example before the model has a published result to expect values from.

A running `accuracy` or `performance` entry also states what the model must
deliver on a DevKit in `expected`: the latest published value of every
measured build. For accuracy, that is every metric the accuracy entry
declares, per build profile; for performance, `throughput_fps`, per build
profile and component. Take the values from the model browser's latest
`develop` results, rounded to 4 decimal places for accuracy and 1 for
throughput.

A measured value passes when it falls at most 5% below its expected accuracy
or 10% below its expected throughput. For a metric whose result says lower is
better, it may rise by that much instead. Every model uses these limits:
`ACCURACY_ALLOWED_DROP` and `PERFORMANCE_ALLOWED_DROP` in
`tools/model_registry/model_tests.py`.

```yaml
compile:
  run_tests: true
accuracy:
  run_tests: true
  expected:
    w8a8:
      top1_accuracy: 0.7764
      top5_accuracy: 0.9356
performance:
  run_tests: true
  expected:
    w16a16:
      model:
        throughput_fps: 578.6
    w8a8:
      model:
        throughput_fps: 1491.4
```

`model-registry validate` requires an `expected` value for every build the
stage measures. The measurement test compares the result with these values
through one shared function. A performance test also checks that the result
records the execution plan the model browser draws:

```python
from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    plan = performance["core_report"]["execution_plan"]
    assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
    assert plan["nodes"] and plan["edges"]
```

Update `expected` when a reviewed change moves a published value, such as a
new compiler or a faster script, in the same pull request or right after its
results publish. No stage key reads `tests/`, so editing `tests.yaml` reruns
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
  scripts/run_model_tests.sh performance models/<model-id> <performance-result.json>
```

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

**Outputs.** <Each output tensor by its model.yaml name, and what it holds.>

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

Declare a complete language model as one component, with `interface.format: huggingface`,
`recipe: scripts/compile.sh`, and a pinned Hugging Face `file_set` whose entrypoint is
`config.json`. Include weights, tokenizer files, processor settings and the chat template;
each file retains its revision, byte size and SHA-256. The existing verified downloader
assembles the checkpoint directory. Do not list LLiMa's generated layers as components.

A build profile names one compiled configuration. Start with one profile per
model and set `max_num_tokens: 4096` explicitly in its component `parameters`,
alongside `language_group_size`. Add `input_height` and `input_width` only for
VLMs needing a resolution override. Omit both for text-only LLMs and VLMs whose
native resolution is fixed. Declare a VLM's actual resolution in its image
interface even when no override is needed.

The shared runner forwards these parameters as compiler options after the
checkpoint and output arguments; `compile.sh` must pass them to `llima-compile`.
Keep these values in `model.yaml` rather than duplicating them in the recipe.

The executable recipe takes two positional paths followed by compiler options:

```bash
#!/usr/bin/env bash
set -euo pipefail
llima-compile "$1" -o "$2" \
  "${@:3}" --jobs 6 --log_level INFO
```

Run it through the LLiMa runner, with Model Compiler and LLiMa activated. The
runner passes the selected profile to the recipe as `BUILD_PROFILE_ID`:

```bash
scripts/run_llima_compilation.sh models/qwen3_5_0_8b w4a16 model /path/to/output
```

The runner downloads the pinned files and passes the recipe a temporary symlink
named after `upstream.repository`, preserving the verified cache and giving
generated files the checkpoint's name. It then runs
`llima-deploy` into `<output-root>/<model>/<profile>/<component>/deployment/`.
This folder contains `devkit/`, `elf_files/`, and any `npy_files/`. Run
`model-registry verify-artifact <component-output> --write-manifest` to check
the deployment and write `deployment-manifest.json` beside it. The verifier
rejects incomplete configurations, missing executables, unsafe paths, and links.

`compile-outputs.json` uses `artifact_path: model/deployment-manifest.json`
for Qwen's `model` component and `artifact_sha256` for that index file's hash.
ONNX outputs retain their MPK fields. Publication registers the deployment's
files individually, with a JSON index named `llima`. A native full package is
the deployment folder with `manifest.json` at its root; its component uses
`format: llima`, `path: .`. Retrieval preserves the folder and verifies each
file. See the [manifest and metadata contract](metadata-schema.md#llima-checkpoint-and-deployment-formats).
No native stage creates a deployment ZIP.

The native compile key covers its runner, `tools/model_registry/artifact.py` and
`tools/model_registry/cli.py` for deployment verification. ONNX builds retain
their existing runner and compile keys.
README and hero changes remain outside compile identities.

Leave `calibrations`, `fidelity`, `accuracy`, and `benchmark` empty when no corresponding
stage is implemented. A manual Qwen build compiles and publishes only; performance
and accuracy are separate follow-ups [#294](https://github.com/sima-neat/models/issues/294)
and [#295](https://github.com/sima-neat/models/issues/295). The default Qwen profile
uses the default runner with six compile jobs, 4096 tokens, 448×448 input,
and language group size 128.
