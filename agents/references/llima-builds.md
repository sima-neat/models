# LLiMa builds

How CI compiles, deploys, verifies, publishes, and fingerprints a model whose
`compilation.format` is `huggingface`: an LLM, a VLM, or Whisper. Read it before you
change `scripts/run_llima_compilation.sh` or the LLiMa branches of the shared tooling,
and when a LLiMa compile or publish job fails.

The author's steps and the manifest fields are in `docs/`:

- [LLiMa language and vision-language models](../../docs/adding-a-model.md#llima-language-and-vision-language-models)
- [Compile an LLM or VLM with LLiMa](../../docs/local-development.md#compile-an-llm-or-vlm-with-llima)
- [LLiMa checkpoint and deployment formats](../../docs/metadata-schema.md#llima-checkpoint-and-deployment-formats)

## How CI recognizes a LLiMa build

- `load_and_validate_model` in `tools/model_registry/model.py` requires exactly one
  component and a file-set source whose `main_file` is a `config.json`.
- `build_matrix` in `compile_matrix.py` sets `runner_script:
  scripts/run_llima_compilation.sh` when the first component's `interface.format` is
  `huggingface`.
- The `Compile` job of `.github/workflows/vulcan-model-compile.yml` runs that script in
  place of `scripts/run_model_compilation.sh`.

A LLiMa model that sets `build-on-demand: true` publishes as a candidate. See
[on-demand builds](on-demand-builds.md).

## What the runner does

`scripts/run_llima_compilation.sh <model_dir> <build_profile> <compile_invocation>
<output_root>` needs `llima-compile` and `llima-deploy` on `PATH`. It runs these steps:

1. Resolve the invocation's component with `model-registry matrix`. Fail unless the
   invocation has exactly one component.
2. Download the pinned file set with `model-registry fetch`, which verifies each file's
   SHA-256 and size. `MODEL_REGISTRY_CACHE_DIR` sets the cache.
3. Delete `<output_root>/<model>/<profile>` and create a temporary work directory beside
   it. The script removes the work directory on exit.
4. Link the checkpoint directory under the last path segment of `upstream.repository`.
   LLiMa names its output files after the checkpoint directory.
5. Turn the component's `parameters` into compiler options, as listed below.
6. Run `bash models/<model>/scripts/compile.sh <checkpoint> <output> <options>` with
   `BUILD_PROFILE_ID` in the environment. The recipe only compiles.
7. Save the reproduction files into the compiled output with `save_llima_recipe` in
   `artifact.py`.
8. Run `llima-deploy <compiled> <output_root>/<model>/<profile>/<component>/deployment`.
9. Copy the saved `compile.sh`, `compile.py`, and `config.py` into `deployment/`.
10. Run `model-registry verify-artifact <component directory> --write-manifest`.
11. Write `compile-timing.json` for the profile, with scope `compile_recipe`.
12. Write and validate `compile-outputs.json` with `record-compile-outputs` and
    `validate-compile-outputs`.

### Compiler options

The runner forwards each parameter as `--<name> <value>`. A boolean becomes `--<name>`
or `--no-<name>`. These parameters are exceptions:

| Parameter | Handling |
|---|---|
| `precision`, `speculative_method`, `draft_precision` | not forwarded |
| `language_group_size` | not forwarded when it is 128 |
| `draft_model_path` | forwarded as the path of the linked draft checkpoint |

### Reproduction files

The runner saves the reproduction `compile.sh`. The model's recipe does not.
`save_llima_recipe` writes a new `compile.sh` that a reader can run outside CI:

- One comment per checkpoint gives its download command, `hf download <repository>
  --local-dir <name>`. A source that is not a pinned Hugging Face model fails here.
- The command is `llima-compile <checkpoint> -o <output>`, with `-c config.py` when the
  model has `scripts/config.py`. A model with `scripts/compile.py` gets
  `python3 compile.py <checkpoint> <output>` instead.
- The resolved options from the step above follow.
- The process count is four: `--jobs 4`, or `--num_processes 4` for `compile.py`. CI
  keeps the count the model's own recipe sets. Extra arguments pass through as `"$@"`.
- The output name is the checkpoint name without a trailing `-Safetensors`. A checkpoint
  without that suffix gets `-<precision>` appended.

`save_llima_recipe` also copies the model's `compile.py` and `config.py` beside it.

## The deployment folder

```text
<output_root>/<model>/<profile>/
  compile-outputs.json            artifact_path: <component>/deployment-manifest.json
  compile-timing.json
  <component>/
    deployment/                   what llima-deploy wrote, plus the reproduction files
      devkit/  elf_files/  npy_files/ (when present)
      compile.sh  compile.py  config.py (when the model has them)
    deployment-manifest.json      {"format": "llima", "files": [...]}
    compile-result.json           format: llima, layout: directory
```

`verify_artifact` in `artifact.py` treats a component directory with a `deployment/`
folder as a LLiMa build. With `--write-manifest` it writes `deployment-manifest.json` and
`compile-result.json`. Without the flag it fails when the folder and the manifest differ.

`verify_llima` checksums every file and rejects a deployment that has any of these:

- a symbolic link, or an entry that is neither a file nor a directory
- a path part that contains `:` or `\`, or ends in a dot or a space
- a top-level `manifest.json`
- two paths that differ only in letter case
- no `.elf` file under an `elf_files` directory

`llima_compiled_parameters` then reads `devkit/vlm_config.json`. When a configuration
exists, it requires exactly one target configuration and returns the compiled
`max_num_tokens`, `language_group_size`, image size, and speculative method. Publication
records them as `compiled_parameters`. It fails with `compiled <name> does not match
model.yaml` when one differs from the same parameter in the build profile
(`_preflight_compilations` in `registry.py`).

`compile-outputs.json` names the manifest in `artifact_path` and its SHA-256 in
`artifact_sha256`. `compile-result.json` holds that same SHA-256, and its `size_bytes`
is the sum of the deployment's file sizes. No stage creates a ZIP of the deployment.

In CI the `Compile` job runs `verify-artifact --write-manifest` a second time with the
provenance fields. That call records the installed LLiMa compiler version as
`llima_version` when the `sima-lmm` package is present (`_verify_mpk` in `cli.py`).

## How publication registers the files

| Run | Artifact name | Type | Content |
|---|---|---|---|
| `model_compile` | `file-<sha256 of the relative path>`, one per file | `model-file` | the deployment file |
| `model_compile` | `llima` | `model-pack` | `deployment-manifest.json` |
| `model_package` | `file-<sha256 of the relative path>`, one per file | `model-file` | the same files |
| `model_package` | `package` | `model-package` | the package `manifest.json` |

- `publish_compilations` in `registry.py` registers the compile run. Its test result is
  `compile/deployment-validation`. Its parameters carry `compiled_parameters` and
  `llima_version`.
- `assemble_package` in `package.py` copies the deployment folder to
  `packages/<package>/deployment/` and writes `manifest.json` at its root. The component
  entry uses `format: llima` and `path: .`. A LLiMa package holds no `runtime_files`.
- `publish_packages` registers every file again on the package run, so each manifest
  entry names an artifact on its own run.
- The root artifact of each run (`llima` or `package`) is a JSON index with media type
  `application/json`. Its metadata carries `layout: directory`, `storage_backend: s3`,
  `file_count`, and `deployment_size_bytes`, which the model browser reads.
- `_publish_deployment_files` stores a file at `<prefix>/files/<sha256>/<file name>`. The
  prefix is `models`, or the candidate's `upload_prefix`. Compile and package
  publication share the key, so each file uploads once. Four uploads run at a time.
- `S3ArtifactStore.ensure_native_sha256` refuses a file larger than 5 GiB. The size of
  the whole deployment has no limit.
- `stage_catalog_build` stages only the deployment manifest and the list of its files.
  On the DevKit, `download_build` in `fetch.py` downloads each file by its compile run
  and checks its size and SHA-256.

### Publish job runner

The `Publish` job of `vulcan-model-compile.yml` runs on a self-hosted runner for a LLiMa
build, with the labels `self-hosted`, `linux`, `arm64`, the run's runner environment,
`ubuntu`, and `default`. Every other build publishes on `ubuntu-latest`.

## Compile fingerprint

`compile_fingerprint` in `fingerprints.py` takes a separate branch for a `huggingface`
component. It hashes exactly these inputs:

| Input | Source |
|---|---|
| compiler identity | `model-compiler` in `deps/manifest.json` (`resolve_model_compiler`) |
| target | the build profile's `target` |
| checkpoint content | `source_identity`: each file's path, SHA-256, size, and media type, and the `main_file` |
| graph parameters | the component's `parameters` in the build profile |

A change to anything else keeps the fingerprint and recompiles nothing:

- every file under the model's `scripts/`, including `compile.sh`, `compile.py`, and
  `config.py`
- `scripts/run_llima_compilation.sh` and the `COMPILE_TOOLING` files
- the model, build profile, and compile invocation IDs
- `runner_class` and `runner_market`
- `numerical_fidelity`, `resources`, and calibration
- the `llima` pin in `deps/manifest.json`, which enters the model's benchmark fingerprints
  only
- the README and the hero

A graph-changing option therefore belongs in the component's `parameters`. An option set
only in `compile.sh` or `config.py` changes the build without changing its fingerprint.

## Speculative pairs

A speculative pair is a target checkpoint and a draft checkpoint compiled, deployed, and
published as one component.

- The source file set holds both checkpoints in named subfolders. `main_file` is the
  target's `config.json`, and `parameters.draft_model_path` names the draft subfolder.
- `load_and_validate_model` fails with `draft_model_path must name a separate checkpoint
  in the source file set` unless `<draft>/config.json` is a pinned file other than
  `main_file`.
- The runner links the draft folder beside the target and passes `--draft_model_path`.
- The reproduction `compile.sh` lists a download command for each checkpoint. Its output
  name ends in `-DFlash` when `speculative_method` is `dflash`, and in `-MTP` otherwise.
- `llima_compiled_parameters` reads each checkpoint's `<name>/devkit/vlm_config.json`. It
  requires one target and one draft configuration, each with ELF files.

## `benchmark_baseline`

A build profile may name an ordinary model and build profile as its `benchmark_baseline`,
so a speculative profile is measured against normal decoding.

- `load_and_validate_model` requires the baseline model and profile to exist and to
  differ from the profile that declares them.
- The baseline enters the performance fingerprint with its own compile fingerprint
  (`benchmark_fingerprint` in `fingerprints.py`). It never enters a compile fingerprint.
- `stage_benchmark_baseline` in `registry.py` stages the baseline deployment for a
  performance run and writes `benchmark-baseline.json`. It fails with `a speculative
  benchmark baseline must be a native LLiMa deployment` for any other baseline.
- On the DevKit, `download-build` reads `benchmark-baseline.json` and downloads the
  baseline's files beside the target's (`download_build` in `fetch.py`).

## Benchmarks

Only declared stages run. No push benchmarks an on-demand LLiMa build. Dispatch
**Vulcan Model Benchmarking** for the model and profile
([manual dispatch](ci-flow.md#manual-dispatch)). The workloads and report fields are in
[LLiMa text and image performance](../../docs/benchmarking.md#llima-text-and-image-performance).
A LLiMa benchmark fingerprint holds each component's compile run ID and no file digest.
