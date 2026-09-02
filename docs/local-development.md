# Local development and testing

This guide covers the complete local qualification loop: validate `model.yaml`,
fetch immutable inputs, compile a declared variant, measure declared accuracy,
benchmark the resulting MPK on a Modalix DevKit, and review the local evidence.
Commands assume the repository root as the current directory.

Schema, catalog, dependency, and matrix checks run on an ordinary development
machine with the contributor tool installed. Compilation requires the exact
pinned Model Compiler environment. Performance measurement requires a
compatible Modalix DevKit. Treat those as distinct required environments and
complete the entire loop before submitting a model pull request.

## Local-first qualification loop

For each supported variant, complete this sequence before opening a pull
request:

1. Validate the model directory and complete catalog.
2. Fetch and checksum the declared source and calibration snapshot.
3. Confirm the pinned compiler and YAML-derived compile matrix.
4. Compile the variant in the pinned Model Compiler environment.
5. Validate the generated MPK.
6. Run the declared dataset-backed accuracy evaluation when it applies.
7. Run the declared performance test on a compatible Modalix DevKit.
8. Review the machine-readable results and retain the results and logs outside the
   repository.

## Install the contributor tool

Install the repository-pinned Model Compiler before installing
`model-registry`. The contributor package deliberately installs into that
compiler environment so validation and compilation use one compatible Python
environment:

```bash
sima-cli neat install --stg models-utils/model-registry@develop:latest
model-registry --version
model-registry --help
```

The installer finds an active Model Compiler virtual environment, a Model
Compiler installed with the NEAT SDK, or the standard standalone installation.
For a custom location, identify the environment explicitly:

```bash
MODEL_COMPILER_HOME=/path/to/model-compiler \
  sima-cli neat install --stg models-utils/model-registry@develop:latest
```

`MODEL_COMPILER_PYTHON` may instead point directly to its Python executable.
The installer preserves the compiler environment's existing dependencies and
places `model-registry` beside its Python executable. Activate that environment
or invoke the executable from its `bin` directory if it is not already on
`PATH`.

This is an author-safe command surface for validation, verified downloads,
matrix inspection, and MPK checks. It cannot package or publish models,
taxonomy, or benchmark results; those operations remain CI-only.

## Validate one model

```bash
model-registry validate models/<model-id>
```

This checks the JSON Schema, taxonomy references and compatibility, immutable
source policy, matching directory and model IDs, unique variant IDs, and every
referenced recipe, surgery hook, and test file. Fix the first reported field
paths in `model.yaml`; do not weaken a declaration just to pass validation.

Validate the complete catalog before submitting:

```bash
model-registry validate-catalog --repository .
```

Run this after validating the individual model so repository-wide contracts are
checked too.

## Fetch and verify immutable inputs

```bash
model-registry fetch models/<model-id>
model-registry fetch-calibration models/<model-id>
```

`fetch` verifies exact byte size and SHA-256 before atomically installing the
source into a checksum-addressed cache. `fetch-calibration` applies the same
policy when calibration status is `ready`; other statuses report that no
calibration fetch is required.

The default cache is `~/.cache/model-registry`. For an isolated or disposable
cache, use either form:

```bash
export MODEL_REGISTRY_CACHE="$PWD/.model-registry-cache"
model-registry fetch models/<model-id>
model-registry fetch models/<model-id> --cache-dir "$PWD/.model-registry-cache"
```

`HF_TOKEN` is used when present and is never printed. Do not put tokens, signed
URLs, or cache content in model metadata or Git.

## Inspect compile selection

Confirm the repository compiler pin and the variants selected by a model-local
change:

```bash
model-registry resolve-compiler --repository .
model-registry matrix \
  --repository . \
  --changed-path models/<model-id>/model.yaml
```

The matrix output is deterministic JSON. It includes supported variants only.
To inspect a model or variant explicitly:

```bash
model-registry matrix --repository . --models <model-id>
model-registry matrix --repository . --models <model-id> --variants <variant-id>
```

Changes below `models/<model-id>/` select that model. Repository-wide compile
inputs such as `deps/` can select the full catalog. Some models also declare
additional build inputs outside their model directory.

## Compile locally from `model.yaml`

Full compilation is only possible in an environment containing the exact Model
Compiler version reported by `model-registry resolve-compiler`. A normal Python
virtual environment is not that compiler environment.

Model Compiler does not consume the registry's `model.yaml` directly. The
manifest declares the supported variants, their parameters, and the
model-local `recipe`. The recipe reads the selected variant from `model.yaml`
and translates that declarative contract into model-specific importer,
quantization, surgery, tessellation, and compiler calls.

Compilation is intentionally model-owned; `model-registry` does not provide a
public compile command. Use the model recipe directly:

```text
compile.py --model <verified-source> --output-dir <new-directory> --variant <variant-id>
```

In the pinned Model Compiler environment:

1. Resolve the compiler version and choose a supported variant from the matrix.
2. Activate that exact locally installed Model Compiler environment using its
   installation instructions, then verify that its Python APIs and compiler
   executable are available:

   ```bash
   python -c 'import afe; print("Model Compiler Python APIs available")'
   command -v mla-masm
   ```

3. Fetch the source and variant-specific calibration input as JSON, then take
   the returned verified paths:

   ```bash
   source_json="$(model-registry fetch models/<model-id> --json)"
   source_path="$(jq -r .path <<<"$source_json")"

   calibration_json="$(model-registry fetch-calibration \
     models/<model-id> --variant <variant-id> --json)"
   calibration_kind="$(jq -r .kind <<<"$calibration_json")"
   calibration_path="$(jq -r .path <<<"$calibration_json")"
   ```

   `calibration_kind` is `dataset` only when the selected variant needs the
   ready calibration declaration. Otherwise its path is empty and the recipe
   must not receive `--dataset`.

4. Invoke the recipe declared by the selected `compile_variants` entry, using a
   new output directory:

   ```bash
   recipe_args=()
   if [[ "$calibration_kind" == "dataset" ]]; then
     recipe_args=(--dataset "$calibration_path")
   fi

   python models/<model-id>/compile.py \
     --model "$source_path" \
     --output-dir <new-output-directory> \
     --variant <variant-id> \
     "${recipe_args[@]}"
   ```

5. Verify that the directory contains exactly one `*_mpk.tar.gz` and validate
   it:

   ```bash
   model-registry verify-mpk <new-output-directory>
   ```

Use a location outside the repository or an ignored build directory. Never
commit the MPK or compiler intermediates. If the pinned compiler environment is
not installed, obtain access to it before continuing; validation alone is not
sufficient for model submission.

## Measure declared accuracy locally

If the selected variant has an executable correctness declaration with
`engine: model_registry`, evaluate the saved `.sima` candidate produced by the
recipe before benchmarking the MPK. This compares the candidate graph with the
pinned reference model over the declared correctness dataset; it does not run
the MPK on the DevKit.

Follow [Local accuracy development](local-accuracy-development.md) for dataset
resolution, regular versus qualification profiles, the `evaluate` command, and
result interpretation. If `fetch-accuracy-dataset --variant <variant-id>`
returns `kind: none`, the declaration does not apply to that variant and both
local development and CI skip the accuracy step.

## Benchmark locally on a Modalix DevKit

Benchmark the exact MPK produced above using the performance declaration in
`models/<model-id>/tests/performance.yaml`. The benchmark measures the compiled
model route with deterministic synthetic inputs; it does not establish model
accuracy or measure application preprocessing and postprocessing.

### 1. Check the DevKit environment

The DevKit must match the versions from `deps/manifest.json`. When working from
a NEAT SDK, confirm that its DevKit runner is available and keep the MPK,
manifest, and result paths under `/workspace`:

```bash
bash -lic 'type dk || type devkit-run'
```

Non-Python programs launched through the SDK runner must be ARM64. The
benchmarker itself runs in the DevKit's installed `pyneat` environment. If the
SDK runner is unavailable, use a direct DevKit shell and copy the same assets
there explicitly.

### 2. Install and verify the benchmarker on the DevKit

```bash
sima-cli neat install --stg models-utils/benchmarker@develop:latest
source ~/pyneat/bin/activate
benchmarker --version
```

### 3. Create the local benchmark manifest

Copy the compiled `*_mpk.tar.gz` under `/workspace`, calculate its SHA-256, and
create `/workspace/benchmark-test.json` using the manifest shown in
[Synthetic model benchmarking](benchmarking.md#create-a-benchmark-manifest).
For a local run, `provenance.compile_run_id` may be omitted. Keep `model.id`,
`model.variant_id`, compatibility versions, sample counts, and any
bounded-runner geometry consistent with `model.yaml`, `deps/manifest.json`, and
the model's `tests/performance.yaml`.

The published contributor command intentionally does not expose CI's
`make-benchmark-manifest` helper. Creating the local manifest explicitly keeps
the MPK path, checksum, tested revision, and performance settings reviewable.

### 4. Run on the DevKit and retain evidence

From a DevKit shell:

```bash
source ~/pyneat/bin/activate
mkdir -p /workspace/results
benchmarker \
  --manifest /workspace/benchmark-test.json \
  --output /workspace/results/performance-result.json \
  2>&1 | tee /workspace/results/benchmark.log
```

From a NEAT SDK, prefer its `dk` or `devkit-run` mechanism to execute the same
files under `/workspace`; use direct SSH only when that runner is unavailable.

Confirm that the command exits successfully and that
`performance-result.json` reports a successful result with the expected model,
variant, platform, Neat version, and measured sample count. Retain the JSON and
log outside the repository, but do not commit either file.

See [Synthetic model benchmarking](benchmarking.md) for manifest fields,
bounded-runner behavior, power telemetry, and result semantics.

## Finish the local review

Before leaving the local workflow, confirm:

- the model and complete catalog validate;
- each intended supported variant appears in the matrix;
- each locally tested variant produces exactly one valid MPK;
- each applicable dataset-backed accuracy run records the expected variant,
  profile, and sample count;
- the DevKit benchmark succeeds for that exact MPK;
- the reported model, variant, versions, and sample count are correct; and
- fetched inputs, MPKs, result JSON, and logs remain outside the repository.

Summarize the result of each stage in the pull request. A compact record such
as `validate: pass`, `compile modalix_int8: pass`, `accuracy: not applicable`,
and `DevKit benchmark: pass` is clearer than a broad claim that local testing
passed.

When this checklist passes, continue to
[Contributing with Git and GitHub](contributing.md) for commit, push, pull
request, and submission guidance.

## Common failures

| Failure | What to check |
| --- | --- |
| `id must match directory name` | Rename the directory or correct `id`; both must use the same snake-case value. |
| Unknown or incompatible taxonomy ID | Select a sanctioned v2 ID or propose a separately reviewed taxonomy version. |
| `source.sha256` or size mismatch | Hash and measure the exact immutable bytes addressed by `source.uri`. |
| Source host or URL rejected | Use a full-commit Hugging Face resource or an approved GitHub release asset. |
| Referenced file does not exist | Keep recipe, surgery, and test paths relative to and inside the model directory. |
| Variant missing from the matrix | Check its `status`, requested IDs, and changed path. |
| More than one MPK produced | Clean the new output directory and make the recipe emit exactly one archive. |
