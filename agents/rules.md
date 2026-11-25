# Repository rules

These rules hold for every change in this repository. [coding-style.md](coding-style.md)
owns how model scripts and model tests read, and [change-map.md](change-map.md) lists the
files that change together.

## Stages and fingerprints

| Stage | Venue | Model entry point | Runner script |
|---|---|---|---|
| compile (ONNX) | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` |
| compile (LLiMa) | Model Compiler | `scripts/compile.sh` | `scripts/run_llima_compilation.sh` |
| performance | DevKit | `scripts/performance.py` + `requirements-performance.txt` | `scripts/run_model_performance.sh` |
| accuracy | DevKit | `scripts/accuracy.py` + `requirements-accuracy.txt` | `scripts/run_model_accuracy.sh` |

Conversion comes before the stages when a model declares it: `scripts/convert.py`,
run by `scripts/run_model_conversion.sh`, produces the ONNX that `compile.py`
receives as `--model`.

Each stage's registry run carries its fingerprint, and CI runs a stage only
when no completed run carries the current fingerprint. A change therefore
reruns exactly the stages whose inputs it touches. `compile_fingerprint` and
`benchmark_fingerprint` in `tools/model_registry/fingerprints.py` and
`conversion_fingerprint` in `tools/model_registry/conversion.py` compute them.
[What a change reruns](../docs/adding-a-model.md#9-know-what-a-change-reruns)
lists their inputs. Read the code and that table before you change what a stage
depends on.

- A tooling change reruns a stage only when the changed file is in that stage's
  tooling list in `fingerprints.py`. No shared file is a conversion input.
- The ONNX opset, `onnx.opset` in `deps/manifest.json`, is a conversion input and
  an ONNX compile input. Changing it means Vulcan Model Convert for every
  conversion-backed model, and every ONNX build then recompiles.
- Raise `FINGERPRINT_VERSION` in `conversion.py` in the change that makes shared
  tooling convert the same inputs to a different model. Planning then fails
  until Vulcan Model Convert reruns for every conversion-backed model.
- For an ONNX model, every `.py` and `.sh` file under `models/<id>/scripts/` is a
  compile input, except `convert.py`, `performance.py`, `accuracy.py`, and
  `fp32_baseline_accuracy.py` (`NON_COMPILE_SCRIPTS`). Editing a comment in a
  compile input recompiles and then remeasures the model.
- No fingerprint reads a model's `README.md`, `hero.png`, `tests/`, its `license`
  field, the taxonomy, `docs/`, `agents/`, `skills/`, or `AGENTS.md`. Editing them
  reruns nothing. The `license` resource pin is a compile input, like every
  resource.

## Before a push

A push reruns each build whose fingerprint changed. Some files are fingerprint inputs of every
model: `scripts/run_model_compilation.sh`, the compile helpers in
`tools/model_registry/compile_script.py`, the image listing in `tools/model_inputs/images.py`,
`scripts/unpack_benchmark_datasets.sh`, and the other entries of `COMPILE_TOOLING`,
`BOARD_STAGE_TOOLING`, `PERFORMANCE_TOOLING`, and `ACCURACY_TOOLING` in `fingerprints.py`, the
compiler pin, the Core runtime target, and `FINGERPRINT_VERSION` in `conversion.py`. Three
more lists there each cover part of the models: `FIDELITY_TOOLING` recompiles each model that
declares `numerical_fidelity`, `FRAME_TOOLING`, the benchmark frame in
`tools/model_inputs/frame.py`, remeasures the performance of each model PyNeat measures, and
`LLIMA_PERFORMANCE_TOOLING` that of each model MoLE measures. An edit to one of them, even a
comment made while working on one model, recompiles or remeasures models the pull request has
nothing to do with.

- Before each push, list the changed files and check each against those inputs.
- When one matches, say so before pushing, in these terms: the file, the stages it reruns, and
  that it reruns all models or how many, beyond the models the pull request is about. Push
  after the user approves, or move that file to its own pull request.
- The pull request description states the same under `Reruns`.

## Rules no check catches

Pull request CI passes when one of these is broken, and the failure shows up
later or never. Apply them while you write.

### Sources

- **Pin sources by content.** Prefer a Hugging Face checkpoint pinned to a full
  repository commit. Use an official upstream GitHub release asset only as a
  fallback, with its SHA-256 and byte count, and explain the fallback in a
  comment in `model.yaml`.

### Conversion

- **`convert.py` exports, `surgery.py` rewrites.** `convert.py` exports the
  upstream forward pass at the shipped input shape, from its input to the final
  outputs the package keeps. `onnxsim.simplify`, including constant folding, is
  allowed there when it preserves the upstream inputs, outputs, and computation.
  Every other change for the compiler belongs in `surgery.py`, even an exact
  one. That includes a cut: an export that ends at an inner tensor, such as raw
  heads before the upstream decode, or starts after an upstream step
  ([what `convert.py` exports](../docs/conversion-artifacts.md#what-convertpy-exports)).
- **Read conversion output names from `model.yaml`.** When a model declares
  `conversion.outputs`, `convert.py`, `compile.py`, and the build tests read
  the names with `conversion_outputs()`, so renaming a file changes only
  `model.yaml`. The benchmark scripts take their paths as arguments, because
  they run without `model_registry`.

### Compile

- **Hide an unready variant by commenting it out.** A commented-out profile in
  `model.yaml` is neither compiled nor benchmarked, and its tests skip. A promoted
  build of it stays promoted, and `check-main-merge` blocks the merge into `main`
  while the catalog of `develop` or `main` promotes a profile `model.yaml` no
  longer declares, so a maintainer first withdraws it on each of those branches
  ([when a promoted profile leaves
  `model.yaml`](references/promotion-operations.md#when-a-promoted-profile-leaves-modelyaml)).
- **Calibrate on at most 20 samples.** Keep each component's `calibration_samples` at 20 or
  fewer, because each sample adds compile time to every build and nothing validates the
  parameter.
- **A shared helper holds what every recipe does the same way, never a recipe step.**
  `tools/model_registry/compile_script.py` holds the command line, one component's
  parameters, the MPK manifest, and the compiler log level. `tools/model_registry/fidelity.py`
  runs samples through the ONNX source and the quantized graph and writes `fidelity.json`.
  `tools/model_inputs/images.py` lists and reads a dataset's images. Quantization, the
  compile call, preprocessing, which samples calibrate the model and which measure
  fidelity, graph surgery, and the MPK audit stay in the model's scripts, because they
  show how the model reaches its MPK. A script whose step differs from a helper keeps its
  own code, because a per-model branch or flag in a helper recompiles every ONNX build
  when one model changes.
- **Every model exports at the opset `deps/manifest.json` sets.** An export passes
  `onnx_opset_version()` as its opset, and `validate_export` calls
  `require_onnx_opset_version(model)`, both from `model_registry.dependencies`. A model
  script or test writes no opset number, so one edit to the manifest moves every model.
- **A recipe takes its compiler log level from `compiler_log_level()`.** It passes the
  result to every `quantize` and `compile` call as `log_level`, so CI keeps the full log
  of a failed compile and a local compile stays readable. A hard-coded level loses one of
  the two.
- **Name a shared module once.** A fingerprint hashes a shared file's path with its
  content, so renaming or moving `fidelity.py`, `images.py`, or `frame.py` recompiles or
  remeasures every model that reads it.
- **`tools/model_inputs/__init__.py` holds only its docstring.** `calibration.py` imports
  the package where OpenCV may be absent, and no fingerprint or test reads that file, so
  nothing reruns or catches an import added there.
- **An LLM or VLM is one component.** Declare the complete model as one
  component with one 4K profile (`max_num_tokens: 4096` and
  `language_group_size`). Keep graph-changing options in component
  `parameters`. Leave generated layers out of `components`, and ship the
  deployment folder as files, never as a nested archive. A speculative pair is
  still one component: deploy and publish the target and its draft together.

### Benchmark

- **Board-only code stays in `performance.py` or `accuracy.py`.** A helper
  module under `scripts/` is a compile input. Put maintainer explanations in
  `model.yaml` comments or in the docstrings of `convert.py`, `performance.py`,
  and `accuracy.py`.
- **Benchmark scripts need only `tools/model_report` and `tools/model_inputs`.**
  `performance.py` and `accuracy.py` run with `tools/` on `PYTHONPATH` and import only those
  two packages from it, without `model_registry` imports, YAML configuration, registry
  credentials, or downloads. They take verified local paths: `--mpk`, or `--deployment` for
  a build packaged as a deployment folder, `--dataset`, `--output`, and `--resources` when
  the model declares supporting files. `accuracy.py` also takes `--limit`, the number of
  samples to score. `model_report` needs only the Python standard library,
  and `model_inputs` only what PyNeat's environment has, so a customer copies both to a
  DevKit beside the model folder.
- **The benchmark tooling asks who measures a model, not how it is packaged.** A speech
  model is a deployment folder like a language model, and PyNeat measures it like a
  vision model. Shared benchmark code that must tell them apart calls `benchmark_runner`
  in `benchmark_declarations.py`, which returns `mole` only for a deployment that
  generates text.

### Tests

- **DevKit tests import lazily.** Tests under `tests/benchmark/` also run on the
  DevKit, in the registry tooling environment the job installs with
  `pip install '.[tests]'`. It has NumPy and pytest, and no `pyneat`, `cv2`,
  Pillow, `onnx`, `onnxruntime`, or `torch`. Import such
  a package inside the test with `pytest.importorskip("onnx")`. A module-level
  import stops collection on the DevKit, and the stage publishes no result
  ([the tests step](../docs/adding-a-model.md#10-write-the-models-tests)).
- **Expected values follow published results.** When a reviewed change moves a
  published value, set `expected` in `tests/tests.yaml` to the new published
  result. Leave `ACCURACY_ALLOWED_DROP` and `PERFORMANCE_ALLOWED_DROP` alone:
  every model shares them. When the new accuracy is more than
  `ACCURACY_ALLOWED_REFERENCE_DROP` below the reference score, comment the profile out
  instead ([the reference rule](../docs/adding-a-model.md#comment-out-a-variant-far-below-its-reference-score)),
  because `model-registry validate` fails that `expected` value.
- **Comment out a profile that fails the reference test on `develop` before the merge into
  `main`.** The DevKit accuracy test fails a measured accuracy more than
  `ACCURACY_ALLOWED_REFERENCE_DROP` below the reference score, but the stage still publishes
  its report. `check-main-merge` (`find_main_merge_blockers` in `stage_plan.py`) checks
  that each profile has a completed result, not its test outcome, so it lets the merge carry
  the profile to customers.

### Publication and approval

- **Approval comes from the server.** Read `approved_deployments` and
  `branch_deployments` as Vulcan returns them. Never write or infer one.

### Workflows

- **Workflow YAML only orchestrates.** It holds triggers, jobs, permissions,
  inputs, and the order of steps. A step that needs more than one command calls
  a named, executable script with a usage header, under `scripts/` or beside the
  code it serves. Registry logic belongs in a `model-registry` command.

### Schemas

- **Schemas reject unknown fields.** Published reports keep their
  `schema_version` (model report 1, numerical fidelity 1), because the model
  browser and `sima-cli` choose a renderer by it. Documents only this
  repository reads (`model.yaml`, compile outputs, package manifests) carry no
  version. Change their readers and writers together.

## Limits

The code holds each limit, and this table names where. A change to a limit updates its row.

| Limit | Held by | Value |
|---|---|---|
| A committed file is neither excluded by `.gitignore` nor larger than this, unless `ALLOWED_FILES` lists it | `MAX_FILE_BYTES` in `tests/test_target_branch_checks.py` | 2 MiB |
| How far an accuracy measurement may be worse than its `expected` value in `tests/tests.yaml`, relative to that value, and still pass | `ACCURACY_ALLOWED_DROP` in `tools/model_registry/model_tests.py` | 0.05 |
| How far a performance measurement may be worse than its `expected` value, relative to that value, and still pass | `PERFORMANCE_ALLOWED_DROP` in `model_tests.py` | 0.10 |
| How far a profile's expected or measured accuracy may be worse than the source model's reference score, relative to that score, before `model-registry validate`, `model-registry validate-models`, or the DevKit accuracy test fails | `ACCURACY_ALLOWED_REFERENCE_DROP` in `model_tests.py` | 0.08 |
| Samples a model's numerical fidelity compares during each compile | `maximum` of `numerical_fidelity.selection.samples` in `schemas/model.schema.json` | 5 |
| The conversion fingerprint's version, raised as [Stages and fingerprints](#stages-and-fingerprints) says | `FINGERPRINT_VERSION` in `tools/model_registry/conversion.py` | 1 |
| Words in a model README after the hero | `MAX_README_WORDS` in `tools/model_registry/documentation.py` | 400 |
| Words in the README's Overview, in one or two paragraphs | the check in `load_model_documentation` in `documentation.py` | 50 to 120 |
| Size of a model README | `MAX_README_BYTES` in `documentation.py` | 128 KiB |
| Size of a model's `hero.png` | `MAX_HERO_BYTES` in `documentation.py` | 2 MiB |
| Pixel size of a model's `hero.png` | `HERO_WIDTH` and `HERO_HEIGHT` in `documentation.py` | 1200 by 480 |
| Clock cycles and seconds the MLA stage simulation runs before it stops | `CYCLE_LIMIT` and `TIMEOUT_S` in `tools/model_registry/mla_stage_simulation.py` | 2,000,000,000 cycles, 600 s |
| Roles and required names in one Vulcan candidate | the check in `tools/model_registry/candidates.py` | 32 roles, 4096 names |
| Lines in `AGENTS.md`, which every agent session loads | `AGENTS_LINE_BUDGET` in `tests/test_docs.py` | 150 |

## Where a lesson goes

Every pull request feeds what its work taught back into the file that owns the topic:

- **Check.** Before a change rewrites a graph, sets a quantization or compile option, or
  changes how a build is measured, read what `skills/sima-compile-model/` or
  `skills/sima-benchmark-model/` says about it. Compare that with the change's compile
  output or measurement.
- **Lesson.** A finding the work established goes where its topic belongs:

  | The lesson is about | It goes in |
  |---|---|
  | A graph surgery pattern, an operator fact, or a quantization or compile option | `skills/sima-compile-model/references/` |
  | A way to measure performance or accuracy | `skills/sima-benchmark-model/` |
  | How model scripts or model tests are written | [coding-style.md](coding-style.md) |
  | A step an agent's workflow in this repository lacked or got wrong | the skill under `agents/skills/` that runs it |
  | A rule every change in this repository follows | this file |
  | A change that is incomplete without another file | [change-map.md](change-map.md) |
  | How tooling, CI, or publication works | `agents/references/` |
  | The contributor contract for people | `docs/` |

- **Generalize.** Write a lesson in one or two sentences, as a rule with its condition: when X,
  do Y, because Z. It names the condition, not the model where it was found, and holds for any model that meets
  the condition. A fact true of only one model stays in that model's folder. A fact about
  one operator or option is stated exactly, because it holds wherever that operator or
  option appears.
- **Evidence.** Each lesson cites a compile output or a measurement from the pull request.
- **Correct.** A statement the work proves wrong is fixed in the same pull request, wherever
  it lives, even when the pull request is about something else.
- **Merge.** Extend or generalize an existing statement before adding a new one, so each
  fact lives in one place.

The files `skills/sima-compile-model/source.json` lists started as copies from
`sima-neat/model-compiler`, at the commit it names. This repository owns them from then on, so
a lesson edits them here. A later sync from the compiler repository is a manual merge that
updates `source.json`.

`skills/` ships alone, so a skill links only files inside its own folder and public URLs. It
reaches `docs/` and other files of this repository only by their public URL, and it never
names an `agents/` path or links a URL into `agents/` (tested). General compiler knowledge
is written once, in `skills/`, and `agents/` points to it. A skill keeps routing in its
`SKILL.md` and detail in its `references/`.

Every pull request description lists each change under `Rule, skill, and knowledge updates`,
or writes `Rule, skill, and knowledge updates: none` with the reason.
