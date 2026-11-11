# Qualify locally

This guide covers the complete local qualification loop: validate `model.yaml`,
fetch immutable inputs, compile each declared build, check its MPKs, run the
performance and accuracy stages on a Modalix DevKit, and review the evidence.
Commands assume the repository root as the current directory.

Validation, fetching, and matrix checks run on an ordinary development machine
with the contributor tool installed. Compilation requires the exact pinned
Model Compiler environment. The performance and accuracy stages require a
compatible Modalix DevKit. Treat those as distinct required environments and
complete the entire loop before submitting a model pull request.

## Local-first qualification loop

For each supported build, meaning each build profile, complete this sequence
before opening a pull request:

1. Validate the model directory and the complete catalog, and run the
   model's tests.
2. Fetch and checksum the declared source, calibration, and fidelity inputs.
3. Confirm the pinned compiler and the builds the matrix derives from
   `model.yaml`.
4. Compile the build in the pinned Model Compiler environment.
5. Validate the compile outputs and each MPK, and read the fidelity report when
   one is declared.
6. Run the performance stage on a DevKit when the model declares performance
   for the build profile.
7. Run the accuracy stage on a DevKit when the model declares accuracy for the
   build profile.
8. Review the results and keep them, and the logs, outside the repository.

## Install the contributor tool

Install the repository-pinned Model Compiler before installing
`model-registry`. The contributor package installs into that compiler
environment so validation and compilation use one Python environment:

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
model-registry --help
```

The installer finds an active Model Compiler virtual environment, a Model
Compiler installed with the NEAT SDK, or the standard standalone installation.
For a custom location, name the environment explicitly:

```bash
MODEL_COMPILER_HOME=/path/to/model-compiler \
  sima-cli neat install --stg models-utils/model-registry@develop:latest
```

`MODEL_COMPILER_PYTHON` may instead point directly to its Python executable.
The installer keeps the compiler environment's existing dependencies and
places `model-registry` beside its Python executable. Activate that environment
or call the executable from its `bin` directory if it is not on `PATH`.

This package holds only the author commands: validation, verified downloads,
the compile matrix, compile-output and MPK checks, and local package assembly.
It cannot publish models, taxonomy, or results; publication is CI-only.

Models that use calibration, fidelity, or accuracy data also need Dataset
Registry and working Artifactory access. Complete [Datasets for local model
development](datasets.md) before fetching those inputs.

## Validate one model

```bash
model-registry validate models/<model-id>
```

This checks the JSON Schema, taxonomy references, the immutable source policy,
the build profiles' components and calibration datasets, the `pipeline`, the
`numerical_fidelity`, `accuracy`, and `performance` sections and their
scripts, `reference_metrics`, the model folder layout, and the recipe. A declared `conversion` also needs `scripts/convert.py` and
`scripts/requirements-convert.txt`. Fix the reported field in `model.yaml`; do
not weaken a declaration just to pass validation.

Then validate every model:

```bash
model-registry validate-models --repository .
```

Run the model's tests from a virtual environment with this checkout's
development extra installed. The extra holds the test dependencies, such as
pytest and OpenCV, and matches what CI installs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m pytest models/<model-id>/tests
```

`.venv/bin/python -m pytest` with no path runs the whole repository suite.

Run a stage's tests on an output you produced with
`scripts/run_model_tests.sh <stage> models/<model-id> <output> [<profile-id>]`,
where the output is a build's compile output directory or a DevKit
`model-report.json`, the profile is the one that report measured, and
`REGISTRY_PYTHON` names that environment's Python. [Write the model's
tests](adding-a-model.md#10-write-the-models-tests) explains the files and
where each test runs.

## Fetch and verify immutable inputs

```bash
model-registry fetch models/<model-id> --component <component-id>
model-registry fetch-calibration models/<model-id> \
  --build-profile <profile-id> --component <component-id>
model-registry fetch-fidelity-dataset models/<model-id> \
  --build-profile <profile-id> --component <component-id>
```

`fetch` checks the exact byte size and SHA-256 before installing the source
into a checksum-addressed cache. `fetch-calibration` does the same for a
`ready` calibration, which belongs to the profile's first component, and
`fetch-fidelity-dataset` for the `numerical_fidelity` dataset. Both report
`kind: none` when the build needs no such input.

The default cache is `~/.cache/model-registry`. For an isolated cache, use
either form:

```bash
export MODEL_REGISTRY_CACHE="$PWD/.model-registry-cache"
model-registry fetch models/<model-id> --cache-dir "$PWD/.model-registry-cache"
```

`HF_TOKEN` is used when present and never printed. Do not
put tokens, signed URLs, or cache content in model metadata or Git.

## Inspect builds and fingerprints

Confirm the repository compiler pin and the builds `model.yaml` declares:

```bash
model-registry resolve-compiler --repository .
model-registry matrix --repository . --models <model-id>
```

`resolve-compiler` prints the pinned compiler `identity`, `label`, and
`version`. The matrix is deterministic JSON with one entry per build profile.
Each entry names its `compile_invocation_id`, which is the profile ID, its
`component_ids`, and the benchmark `stages` that measure it. Narrow it with
`--profiles` and `--components`.

Pass the compiler identity to see each build's compile fingerprint:

```bash
model-registry matrix --repository . --models <model-id> \
  --compiler-version <identity>
```

The fingerprint appears as `compile_fingerprint`. Compare it before and after
an edit: an unchanged fingerprint means CI reuses the published compilation.
[What a change reruns](adding-a-model.md#9-know-what-a-change-reruns) lists the
inputs of each fingerprint.

## Compile locally from `model.yaml`

Compilation needs the exact Model Compiler version that `model-registry
resolve-compiler` reports. A normal Python virtual environment is not that
compiler environment. Activate the compiler environment and check that its
Python APIs and compiler executable are available:

```bash
python -c 'import afe; print("Model Compiler Python APIs available")'
command -v mla-masm
```

Model Compiler does not read `model.yaml`. The model's recipe reads its build
profile from `model.yaml` and turns it into importer, quantization, surgery,
and compiler calls. Compilation is intentionally model-owned; `model-registry`
does not provide a public compile command.

`scripts/run_model_compilation.sh` runs one build the way CI does:

```bash
scripts/run_model_compilation.sh models/<model-id> <profile-id> <profile-id> <output-root>
```

The third argument is the compile invocation, which is named after the
profile. The script fetches the source, resolves the calibration and fidelity
datasets, runs `scripts/compile.py`, writes
`compile-outputs.json` for a single-component recipe that did not, and
validates every declared MPK. Results land in
`<output-root>/<model-id>/<profile-id>/`, with one directory per component.
Set `MODEL_COMPILER_PYTHON` when the compiler's Python is not `python3`, and
`MODEL_REGISTRY_CACHE_DIR` to choose the download cache.

To run the same steps by hand, take the verified paths from the fetch
commands:

```bash
source_path="$(model-registry fetch models/<model-id> \
  --component <component-id> --json | jq -r .path)"
calibration_path="$(model-registry fetch-calibration \
  models/<model-id> --build-profile <profile-id> --component <component-id> --json | jq -r .path)"
fidelity_path="$(model-registry fetch-fidelity-dataset \
  models/<model-id> --build-profile <profile-id> --component <component-id> --json | jq -r .path)"
```

For several components, fetch the source with the first component and
resolve the datasets for each component; the runner script does this and
rejects a profile whose components resolve different datasets.

A calibration resolves to a registered dataset archive, which the recipe
receives as `--dataset`. An empty path means the build needs no such input. Then invoke the
recipe once for the profile, with a new output directory:

```bash
recipe_args=()
if [[ -n "$calibration_path" ]]; then
  recipe_args=(--dataset "$calibration_path")
fi
if [[ -n "$fidelity_path" ]]; then
  recipe_args+=(--fidelity-dataset "$fidelity_path")
fi

python models/<model-id>/scripts/compile.py \
  --model "$source_path" \
  --output-dir <new-output-directory> \
  --build-profile <profile-id> \
  --compile-invocation <profile-id> \
  "${recipe_args[@]}"
```

In short, the interface is `compile.py --model <verified-entrypoint>
--output-dir <directory> --build-profile <profile-id> --compile-invocation
<profile-id>`, plus the optional dataset flags. The recipe runs once and
writes every component's MPK.

### Check the compile outputs

Each component directory must contain exactly one `*_mpk.tar.gz`. Validate an
MPK with:

```bash
model-registry verify-mpk <output-directory>/<component-id>
```

For a manual single-component compile, record the outputs before validating
them; a multi-component recipe writes its own `compile-outputs.json`, which
must also list each component's `derived_graphs` digest:

```bash
model-registry record-compile-outputs <output-directory> \
  --model-id <model-id> --build-profile-id <profile-id> \
  --compile-invocation-id <profile-id> --components <component-id>
model-registry validate-compile-outputs <output-directory>/compile-outputs.json \
  --output-root <output-directory> --model-id <model-id> \
  --build-profile-id <profile-id> --compile-invocation-id <profile-id> \
  --components <component-a>,<component-b>
```

When `model.yaml` declares `numerical_fidelity`, read
`<output-directory>/fidelity.json` and confirm its metrics are what the
section's status claims. See [Accuracy and
fidelity](local-accuracy-development.md#compile-time-fidelity).

To check a download package, assemble it from the compiled output:

```bash
model-registry assemble-package \
  --repository . \
  --compiled-dir <output-root> \
  --output-dir <package-output> \
  --model <model-id> \
  --build-profile <profile-id> \
  --package default
```

A single-component package is that component's MPK. Otherwise, confirm the
ZIP holds every component MPK at its top level. Package assembly does not replace MPK
validation or application testing.

Keep MPKs and compiler intermediates outside the repository or in an ignored
directory, and never commit them. If the pinned compiler is not installed, get
access to it before continuing; validation alone is not enough to submit.

## Benchmark locally on a Modalix DevKit

The performance stage runs the model's standalone `scripts/performance.py`
once per benchmarked build profile, with every component's MPK, and measures
the whole model. Its input is one image from the model's accuracy dataset in
a 1280x720 frame, which Neat preprocesses inside the measurement, or one sample
as tensors for a model whose inputs are not images (see
[Benchmark input](benchmarking.md#benchmark-input)); the stage does not
measure accuracy.

Complete the DevKit setup in [DevKit performance](benchmarking.md#set-up-a-devkit)
once. Then copy the exact compiled MPKs, the model folder, and
`tools/model_report` to the DevKit, extract the model's accuracy dataset as
for [accuracy](local-accuracy-development.md#run-it-on-a-devkit), and run the
script:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_mpk.tar.gz \
  --dataset imagenet-2012-val=/media/nvme/datasets/imagenet-2012-val \
  --output /workspace/performance 2>&1 | tee /workspace/performance.log
```

A multi-component model takes one `--mpk <component>=<path>` per component;
see [Multi-component models](benchmarking.md#multi-component-models). Confirm
a successful exit and check the saved `performance/model-report.json` for
`latency_avg_ms`, `throughput_fps`, and the power metrics under `metrics`, and
its `runtime.pyneat` version. For the script's settings, follow [DevKit
performance](benchmarking.md#the-performance-script).
Keep the report and log outside Git.

`scripts/run_model_performance.sh <model-dir> <profile-id> <output-root>
<result-directory>` runs the stage the way CI does against a local compile
output: it also resolves and extracts the dataset, sets the MLA and CPU
clocks, and runs `model-registry check-report`. It needs a full Model Registry
install from this checkout (`python -m pip install .`) with Dataset Registry,
named by `REGISTRY_PYTHON`, a mounted NVMe at `/media/nvme`, and passwordless
`sudo` for the clocks.

## Measure declared accuracy locally

Run this step when `model.yaml` declares `accuracy` for the build profile; the
matrix lists `accuracy` in the profile's `stages`.

`scripts/run_model_accuracy.sh` runs the accuracy stage on a DevKit against a
local compile output:

```bash
scripts/run_model_accuracy.sh models/<model-id> <profile-id> <output-root> <result-directory>
```

It needs a full Model Registry install from this checkout (`python -m pip
install .`) with Dataset Registry, named by `REGISTRY_PYTHON`, and PyNeat,
named by `PYNEAT_PYTHON`. It writes `model-report.json`, `accuracy.log`, and
`environment.json`. For a short smoke test, run `accuracy.py` directly with
`--limit`. [Accuracy and fidelity](local-accuracy-development.md#accuracy-on-the-devkit)
covers both forms and the report.

## Finish the local review

Before leaving the local workflow, confirm that:

- the model and the complete catalog validate, and the model's tests pass,
  including each stage's tests on its local output;
- every intended build appears in the matrix;
- every build compiles and its compile outputs and MPKs validate;
- each declared fidelity report has the expected metrics;
- the performance stage succeeds when declared, on those exact MPKs, with the
  required metrics;
- the accuracy stage succeeds when declared, with the expected sample count and
  required metrics; and
- fetched inputs, MPKs, reports, and logs remain outside the repository.

Summarize each stage in the pull request. A compact record such as `validate:
pass`, `compile w8a8/model: pass`, `performance: pass`, and `accuracy:
not declared` is clearer than a broad claim that local testing passed.

When this checklist passes, continue to [Submit for review](contributing.md).

## Common failures

| Failure | What to check |
| --- | --- |
| `id must match directory name` | Rename the directory or correct `id`; both use the same snake-case value. |
| Unknown or incompatible taxonomy ID | Select an ID from the canonical taxonomy or propose a reviewed taxonomy change. |
| `source.sha256` or size mismatch | Hash and measure the exact immutable bytes addressed by `source.uri`. |
| Source host or URL rejected | Use one of the [approved forms](adding-a-model.md#3-pin-every-input): a full-commit Hugging Face resource, or for a conversion checkpoint a tagged release file of a public GitHub repository. |
| Referenced file does not exist | Keep recipe and surgery paths under the model's `scripts/`; a declared `conversion` also needs `scripts/convert.py` and `scripts/requirements-convert.txt`. |
| Unexpected file in the model directory | A model directory holds only `model.yaml`, `README.md`, `hero.png`, `scripts/`, and `tests/`. |
| `set up exactly the stages this model has` or `unexpected test files` | Give `tests/tests.yaml` one entry per stage the model has, and keep only `tests.yaml`, the stage test files, and `fixtures/` under `tests/` ([write the model's tests](adding-a-model.md#10-write-the-models-tests)). |
| `tests run but tests/<group>/test_<stage>.py is missing` | Add the stage's test file, or set the stage's `run_tests: false` with a `skip_reason`. |
| `accuracy.expected.<profile> lacks` or `performance.expected.<profile> lacks` | Add the build profile's latest published value under `expected` ([set up `tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)). |
| Build missing from the matrix | Check the requested model, profile, and component IDs against `model.yaml`. |
| Unexpected MPK count | Start from an empty output directory. A build writes exactly one MPK per component. |


## Compile an LLM or VLM with LLiMa

For a Hugging Face language or vision-language model, use the LLiMa runner:

```bash
scripts/run_llima_compilation.sh models/<model-id> <profile-id> <profile-id> /path/to/output
model-registry verify-artifact /path/to/output/<model-id>/<profile-id>/<component-id> --write-manifest
```

Activate the matching LLiMa and pinned Model Compiler tools first. The recipe
calls `llima-compile` and saves its complete arguments in a standalone `compile.sh`.
The runner verifies the pinned checkpoint download,
calls that recipe, and runs `llima-deploy`. It writes
`<model-id>/<profile-id>/<component-id>/deployment/` with `devkit/`, `elf_files/`,
any `npy_files/`, and that script. Verification writes `deployment-manifest.json` beside
the folder. `compile-outputs.json` points to `<component-id>/deployment-manifest.json`
and hashes that index; it does not point to an archive.

A profile is a named compilation configuration in `model.yaml`. Its component
parameters explicitly set `max_num_tokens: 4096` and `language_group_size`.
Add `input_height` and `input_width` only for VLMs needing a resolution override;
omit them for LLMs and fixed-resolution VLMs. The runner forwards declared
settings to the recipe. Additional compiled configurations use separate profiles.

See [on-demand builds](adding-a-model.md#on-demand-builds)
for manual build and publication instructions.
See [LLiMa model definitions](adding-a-model.md#llima-language-and-vision-language-models)
for the source and package contracts.
