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

For each supported build, meaning each compile invocation of each build
profile, complete this sequence before opening a pull request:

1. Validate the model directory and the complete catalog.
2. Fetch and checksum the declared source, calibration, and fidelity inputs.
3. Confirm the pinned compiler and the builds the matrix derives from
   `model.yaml`.
4. Compile the build in the pinned Model Compiler environment.
5. Validate the compile outputs and each MPK, and read the fidelity report when
   one is declared.
6. Run the performance stage for every benchmarked component on a DevKit.
7. Run the accuracy stage on a DevKit when the build profile declares DevKit
   accuracy.
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
matching directory and model IDs, component and build-profile references,
compile invocations, packages, the `fidelity`, `accuracy`, and `benchmark`
entries, the model folder layout, and every referenced recipe and surgery
hook. A declared `conversion` also needs `scripts/convert.py` and
`scripts/requirements-convert.txt`. Fix the reported field in `model.yaml`; do
not weaken a declaration just to pass validation.

Then validate every model:

```bash
model-registry validate-models --repository .
```

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
`ready` calibration, and `fetch-fidelity-dataset` for the dataset of a
`fidelity` entry that applies to the build. Both report `kind: none` when the
build needs no such input.

The default cache is `~/.cache/model-registry`. For an isolated cache, use
either form:

```bash
export MODEL_REGISTRY_CACHE="$PWD/.model-registry-cache"
model-registry fetch models/<model-id> --cache-dir "$PWD/.model-registry-cache"
```

`HF_TOKEN` and `GITHUB_TOKEN` are used when present and never printed. Do not
put tokens, signed URLs, or cache content in model metadata or Git.

## Inspect builds and stage keys

Confirm the repository compiler pin and the builds `model.yaml` declares:

```bash
model-registry resolve-compiler --repository .
model-registry matrix --repository . --models <model-id>
```

`resolve-compiler` prints the pinned compiler `identity`, `label`, and
`version`. The matrix is deterministic JSON with one entry per build. Each
entry names its `compile_invocation_id`, `component_ids`, the
`benchmark_component_ids` of its performance stage, and the `accuracy_id` of
its accuracy stage, or `null`. Narrow it with `--profiles` and `--components`.

Pass the compiler identity to see each build's compile key:

```bash
model-registry matrix --repository . --models <model-id> \
  --compiler-version <identity>
```

The key appears as `build_fingerprint`. Compare it before and after an edit:
an unchanged key means CI reuses the published compilation. [What a change
reruns](adding-a-model.md#9-know-what-a-change-reruns) lists the inputs of each
stage key.

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
scripts/run_model_compilation.sh models/<model-id> <profile-id> <invocation-id> <output-root>
```

It fetches the source of the invocation's first component, resolves the
calibration and fidelity datasets, runs `scripts/compile.py`, writes
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

For a grouped invocation, fetch the source of its first component and resolve
the datasets for each component; the runner script does this and rejects an
invocation whose components resolve different datasets.

A calibration resolves to either a registered dataset archive or the
immutable file of a calibration `source`; either way the recipe receives it as
`--dataset`. An empty path means the build needs no such input. Then invoke the
recipe once per compile invocation, with a new output directory:

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
  --compile-invocation <invocation-id> \
  "${recipe_args[@]}"
```

In short, the interface is `compile.py --model <verified-entrypoint>
--output-dir <directory> --build-profile <profile-id> --compile-invocation
<invocation-id>`, plus the optional dataset flags. A grouped invocation runs
once and writes every component's MPK.

### Check the compile outputs

Each component directory must contain exactly one `*_mpk.tar.gz`. Validate an
MPK with:

```bash
model-registry verify-mpk <output-directory>/<component-id>
```

For a manual single-component compile, record the outputs before validating
them; a grouped recipe writes its own `compile-outputs.json`, which must also
list each component's `derived_graphs` digest and the `correctness` entry that
points at the result component's `fidelity.json`:

```bash
model-registry record-compile-outputs <output-directory> \
  --model-id <model-id> --build-profile-id <profile-id> \
  --compile-invocation-id <invocation-id> --components <component-id>
model-registry validate-compile-outputs <output-directory>/compile-outputs.json \
  --output-root <output-directory> --model-id <model-id> \
  --build-profile-id <profile-id> --compile-invocation-id <invocation-id> \
  --components <component-a>,<component-b>
```

When a `fidelity` entry applies, read `<result-component>/fidelity.json` and
confirm its metrics are what the entry's status claims. See [Accuracy and
fidelity](local-accuracy-development.md#compile-time-fidelity).

To check a download package, assemble it from the compiled output:

```bash
model-registry assemble-package \
  --repository . \
  --compiled-dir <output-root> \
  --output-dir <package-output> \
  --model <model-id> \
  --build-profile <profile-id> \
  --package <package-id>
```

Confirm the bundle contains every declared component MPK, resource path, and
`manifest.json`. Package assembly does not replace MPK validation or
application testing.

Keep MPKs and compiler intermediates outside the repository or in an ignored
directory, and never commit them. If the pinned compiler is not installed, get
access to it before continuing; validation alone is not enough to submit.

## Benchmark locally on a Modalix DevKit

The performance stage runs the model's `scripts/performance.py` once for each
component listed in the build's `benchmark_component_ids`. It measures the
compiled model with Neat Core's default synthetic workload; it does not
measure accuracy or application preprocessing and postprocessing.

Complete the DevKit setup in [DevKit performance](benchmarking.md#set-up-a-devkit)
once. Then copy the exact compiled MPK and the model's `scripts/performance.py`
to the DevKit and run:

```bash
~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /workspace/resnet_50_mpk.tar.gz --output /workspace/core-benchmark.json \
  2>&1 | tee /workspace/benchmark.log
```

The same command with `python models/resnet_50/scripts/performance.py` works
inside an activated PyNeat environment. Confirm a successful exit and check the
report's `core_version`, `latency_ms`, `fps`, and power fields.
Keep the report and log outside Git. Run each component of a multi-component
model separately.

`scripts/run_model_performance.sh` is the CI wrapper around this script. It
needs the registry publication record of a published compilation, so it does
not run on a local compile.

## Measure declared accuracy locally

Run this step when the build profile has an `accuracy` entry with `engine:
devkit_script`; the matrix shows it as `accuracy_id`.

`scripts/run_model_accuracy.sh` runs the accuracy stage on a DevKit against a
local compile output:

```bash
scripts/run_model_accuracy.sh models/<model-id> <profile-id> <output-root> <result-directory>
```

It needs a full Model Registry install from this checkout (`python -m pip
install .`) with Dataset Registry, named by `REGISTRY_PYTHON`, and PyNeat,
named by `PYNEAT_PYTHON`. It writes `script-output.json`, `accuracy.log`, and
`accuracy-result.json`. For a short smoke test, run `accuracy.py` directly with
`--limit`. [Accuracy and fidelity](local-accuracy-development.md#accuracy-on-the-devkit)
covers both forms and the result contract.

## Finish the local review

Before leaving the local workflow, confirm that:

- the model and the complete catalog validate;
- every intended build appears in the matrix;
- every build compiles and its compile outputs and MPKs validate;
- each declared fidelity report has the expected metrics;
- the performance stage succeeds for every benchmarked component of that exact
  MPK;
- the accuracy stage succeeds when declared, with the expected sample count and
  required metrics; and
- fetched inputs, MPKs, result JSON, and logs remain outside the repository.

Summarize each stage in the pull request. A compact record such as `validate:
pass`, `compile modalix_int8/model: pass`, `performance: pass`, and `accuracy:
not declared` is clearer than a broad claim that local testing passed.

When this checklist passes, continue to [Submit for review](contributing.md).

## Common failures

| Failure | What to check |
| --- | --- |
| `id must match directory name` | Rename the directory or correct `id`; both use the same snake-case value. |
| Unknown or incompatible taxonomy ID | Select an ID from the canonical taxonomy or propose a reviewed taxonomy change. |
| `source.sha256` or size mismatch | Hash and measure the exact immutable bytes addressed by `source.uri`. |
| Source host or URL rejected | Use a full-commit Hugging Face resource, an approved GitHub release asset, or another approved checksum-verified host. |
| Referenced file does not exist | Keep recipe and surgery paths under the model's `scripts/`; a declared `conversion` also needs `scripts/convert.py` and `scripts/requirements-convert.txt`. |
| Unexpected file in the model directory | A model directory holds only `model.yaml`, `README.md`, `hero.png`, and `scripts/`. |
| Build missing from the matrix | Check the component binding `status`, the requested IDs, and the build profile's `compile_invocations`. |
| Unexpected MPK count | Start from an empty output directory. An invocation writes exactly one MPK per component. |
