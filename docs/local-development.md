# Qualify locally

This guide covers the complete local qualification loop: validate `model.yaml`,
fetch immutable inputs, compile a declared component/profile pair, measure declared accuracy,
benchmark the resulting MPK on a Modalix DevKit, and review the local evidence.
Commands assume the repository root as the current directory.

Schema, catalog, dependency, and matrix checks run on an ordinary development
machine with the contributor tool installed. Compilation requires the exact
pinned Model Compiler environment. Performance measurement requires a
compatible Modalix DevKit. Treat those as distinct required environments and
complete the entire loop before submitting a model pull request.

## Local-first qualification loop

For each supported component/profile pair, complete this sequence before opening a pull
request:

1. Validate the model directory and complete catalog.
2. Fetch and checksum the declared source and calibration snapshot.
3. Confirm the pinned compiler and YAML-derived compile matrix.
4. Compile the component with the selected build profile in the pinned Model Compiler environment.
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
matrix inspection, MPK checks, and local retrieval-bundle assembly. It cannot
publish models, taxonomy, or benchmark results; those operations remain CI-only.

Models that require calibration or accuracy data also require Dataset Registry
and working private Artifactory access. Complete
[Datasets for local model development](datasets.md) before fetching those
inputs.

## Validate one model

```bash
model-registry validate models/<model-id>
```

This checks the JSON Schema, taxonomy references and compatibility, immutable
source policy, matching directory and model IDs, component/profile references, and every
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
model-registry fetch models/<model-id> --component <component-id>
model-registry fetch-calibration models/<model-id> \
  --build-profile <profile-id> --component <component-id>
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

Confirm the repository compiler pin and the component builds selected by a model-local
change:

```bash
model-registry resolve-compiler --repository .
model-registry matrix \
  --repository . \
  --changed-path models/<model-id>/model.yaml
```

The matrix output is deterministic JSON. It includes supported component
bindings and declared compile invocations. To inspect a model, profile, or
component explicitly:

```bash
model-registry matrix --repository . --models <model-id>
model-registry matrix --repository . --models <model-id> \
  --profiles <profile-id> --components <component-id>
```

Changes below `models/<model-id>/` select that model. Repository-wide compile
inputs such as `deps/` can select the full catalog. Some models also declare
additional build inputs outside their model directory.

## Compile locally from `model.yaml`

Full compilation is only possible in an environment containing the exact Model
Compiler version reported by `model-registry resolve-compiler`. A normal Python
virtual environment is not that compiler environment.

Model Compiler does not consume the registry's `model.yaml` directly. The
manifest declares the supported build profiles, component bindings, and the
component-local `recipe`. The recipe reads the selected binding from `model.yaml`
and translates that declarative contract into model-specific importer,
quantization, surgery, tessellation, and compiler calls.

Compilation is intentionally model-owned; `model-registry` does not provide a
public compile command. Use the model recipe directly:

```text
compile.py --model <verified-entrypoint> --output-dir <new-directory> \
  --build-profile <profile-id> --component <component-id>
```

For a grouped matrix entry, invoke its shared recipe once:

```text
compile.py --model <verified-entrypoint> --dataset <verified-dataset> \
  --correctness-dataset <verified-correctness-dataset> \
  --output-dir <new-directory> --build-profile <profile-id> \
  --compile-invocation <invocation-id>
```

The grouped output directory contains one subdirectory per component and a
root schema-v2 `compile-outputs.json`. It records each component MPK, both
derived graph hashes, and the declared correctness result. Validate that
contract before checking each MPK:

```bash
model-registry validate-compile-outputs <output>/compile-outputs.json \
  --output-root <output> --model-id <model-id> \
  --build-profile-id <profile-id> --compile-invocation-id <invocation-id> \
  --components <component-a>,<component-b>
```

In the pinned Model Compiler environment:

1. Resolve the compiler version and choose a supported component/profile pair from the matrix.
2. Activate that exact locally installed Model Compiler environment using its
   installation instructions, then verify that its Python APIs and compiler
   executable are available:

   ```bash
   python -c 'import afe; print("Model Compiler Python APIs available")'
   command -v mla-masm
   ```

3. Fetch the component source and its profile-specific calibration input as JSON, then take
   the returned verified paths:

   ```bash
   source_json="$(model-registry fetch models/<model-id> --component <component-id> --json)"
   source_path="$(jq -r .path <<<"$source_json")"

   calibration_json="$(model-registry fetch-calibration \
     models/<model-id> --build-profile <profile-id> --component <component-id> --json)"
   calibration_kind="$(jq -r .kind <<<"$calibration_json")"
   calibration_path="$(jq -r .path <<<"$calibration_json")"
   ```

   `calibration_kind` distinguishes three outcomes: `dataset` resolves a
   registered dataset archive, `source` resolves the immutable file declared
   by a named calibration `source`, and `none` has no calibration input or path. Keep a
   nonempty path from either resolved kind; do not make the recipe download the
   same input again.

4. Invoke the recipe declared by the selected component, using a
   new output directory:

   ```bash
   recipe_args=()
   case "$calibration_kind" in
     dataset)
       recipe_args=(--dataset "$calibration_path")
       unset CALIBRATION_PATH
       ;;
     source)
       export CALIBRATION_PATH="$calibration_path"
       ;;
     none)
       unset CALIBRATION_PATH
       ;;
     *)
       echo "Unknown calibration kind: $calibration_kind" >&2
       exit 1
       ;;
   esac

   python models/<model-id>/compile.py \
     --model "$source_path" \
     --output-dir <new-output-directory> \
     --build-profile <profile-id> \
     --component <component-id> \
     "${recipe_args[@]}"
   ```

   Dataset-based recipes receive `--dataset`. Source-based recipes read the
   already verified file from `CALIBRATION_PATH`; this environment variable is
   the standard recipe interface for a calibration source. A recipe must not
   fall back to a network download when that variable is present.

5. Verify that the directory contains exactly one `*_mpk.tar.gz` and validate
   it:

   ```bash
   model-registry verify-mpk <new-output-directory>
   ```

For a multi-component product, repeat these steps for every matrix entry. A
grouped invocation is one matrix entry even though it emits several MPKs. Place
the verified output directories under
`<compiled-root>/<model-id>/<profile-id>/<component-id>`. To exercise the same
bundle boundary users will retrieve, write each compile manifest with the
matching model, profile, component, target, and source digest, then assemble
the declared package:

```bash
model-registry verify-mpk \
  <compiled-root>/<model-id>/<profile-id>/<component-id> \
  --write-manifest \
  --model-id <model-id> \
  --build-profile-id <profile-id> \
  --component-id <component-id> \
  --target modalix \
  --source-sha256 "$(jq -r .sha256 <<<"$source_json")"

model-registry assemble-package \
  --repository . \
  --compiled-dir <compiled-root> \
  --output-dir <package-output> \
  --model <model-id> \
  --build-profile <profile-id> \
  --package <package-id>
```

Inspect and extract the resulting bundle. Confirm that it contains every
declared component MPK, resource path, and `manifest.json`. Package assembly
does not replace component MPK validation or end-to-end application testing.

Use a location outside the repository or an ignored build directory. Never
commit the MPK or compiler intermediates. If the pinned compiler environment is
not installed, obtain access to it before continuing; validation alone is not
sufficient for model submission.

## Measure declared accuracy locally

If correctness uses `engine: model_registry`, evaluate the saved `.sima`
candidate before benchmarking. For `engine: model_recipe`, pass the separately
resolved correctness dataset to the grouped recipe; the recipe compares the
full candidate pipeline and emits the declared numerical fidelity report.

Follow [Local accuracy development](local-accuracy-development.md) for dataset
resolution, regular versus qualification profiles, the `evaluate` command, and
result interpretation. If the following command returns `kind: none`, the
declaration does not apply and both local development and CI skip the accuracy
step:

```bash
model-registry fetch-accuracy-dataset models/<model-id> \
  --build-profile <profile-id> --component <component-id> --json
```

## Benchmark locally on a Modalix DevKit

Benchmark the exact MPK produced above using the performance declaration in
`models/<model-id>/tests/performance.yaml`. The benchmark measures the compiled
model route with deterministic synthetic inputs; it does not establish model
accuracy or measure application preprocessing and postprocessing.

Complete the installation and compatibility checks in [Synthetic model
benchmarking](benchmarking.md#install-on-a-devkit) once for the target DevKit.
For each qualified component/profile pair, follow this shorter sequence.

### 1. Prepare the exact MPK and manifest

Confirm that the NEAT SDK can reach the DevKit:

```bash
bash -lic 'type dk || type devkit-run'
```

Copy the compiled `*_mpk.tar.gz` under `/workspace`, calculate its SHA-256, and
create `/workspace/benchmark-test.json` from the [benchmark manifest
reference](benchmarking.md#create-a-benchmark-manifest). Keep its model,
component, build profile, compatibility, and test settings consistent with the checked-in
contracts. For a local run, `provenance.compile_run_id` may be omitted.

The published contributor command intentionally does not expose CI's
`make-benchmark-manifest` helper. Creating the local manifest explicitly keeps
the MPK path, checksum, tested revision, and performance settings reviewable.

### 2. Run on the DevKit and retain evidence

From a DevKit shell:

```bash
source ~/pyneat/bin/activate
mkdir -p /workspace/results
benchmarker \
  --manifest /workspace/benchmark-test.json \
  --output /workspace/results/performance-result.json \
  2>&1 | tee /workspace/results/benchmark.log
```

Confirm that the command exits successfully and that
`performance-result.json` reports a successful result with the expected model,
component, build profile, platform, Neat version, and measured sample count. Retain the JSON and
log outside the repository, but do not commit either file.

See [Synthetic model benchmarking](benchmarking.md) for runner installation,
manifest fields, bounded-runner behavior, power telemetry, and result
semantics.

## Finish the local review

Before leaving the local workflow, confirm:

- the model and complete catalog validate;
- each intended supported component/profile pair appears in the matrix;
- each locally tested component/profile pair produces exactly one valid MPK;
- each applicable dataset-backed accuracy run records the expected component,
  profile, and sample count;
- the DevKit benchmark succeeds for that exact MPK;
- the reported model, component, build profile, versions, and sample count are correct; and
- fetched inputs, MPKs, result JSON, and logs remain outside the repository.

Summarize the result of each stage in the pull request. A compact record such
as `validate: pass`, `compile modalix_int8: pass`, `accuracy: not applicable`,
and `DevKit benchmark: pass` is clearer than a broad claim that local testing
passed.

When this checklist passes, continue to
[Submit for review](contributing.md) for commit, push, pull
request, and submission guidance.

## Common failures

| Failure | What to check |
| --- | --- |
| `id must match directory name` | Rename the directory or correct `id`; both must use the same snake-case value. |
| Unknown or incompatible taxonomy ID | Select a sanctioned v3 ID or propose a separately reviewed taxonomy version. |
| `source.sha256` or size mismatch | Hash and measure the exact immutable bytes addressed by `source.uri`. |
| Source host or URL rejected | Use a full-commit Hugging Face resource, an approved GitHub release asset, or another approved checksum-verified upstream host. |
| Referenced file does not exist | Keep recipe, surgery, and test paths relative to and inside the model directory. |
| Build missing from the matrix | Check the component binding `status`, requested IDs, and changed path. |
| Unexpected MPK count | Clean the output directory. A normal invocation emits one archive; a grouped invocation emits exactly its declared component set. |
