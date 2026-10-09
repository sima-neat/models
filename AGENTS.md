# SiMa Neat Models

This repository is the source of truth for model definitions and for how each
model is compiled, measured, and published. `model.yaml` declares a model, the
model's `scripts/` hold code specific to that model, and `tools/model_registry`
holds the shared lifecycle.

## Stages and stage keys

A **build** is one compile invocation of one build profile of one model. Each
build passes through up to three **stages**. Each stage has a **stage key**, a
SHA-256 of everything its result depends on. The registry run of a stage
carries its key, and CI runs a stage only when no completed run carries the
current key. A change therefore reruns exactly the stages whose inputs it
touches.

| Stage | Venue | Model entry point | Runner script | Key inputs |
|---|---|---|---|---|
| compile | Model Compiler | `scripts/compile.py` | `scripts/run_model_compilation.sh` | the invocation's `model.yaml` declarations, `fidelity:` entries, every `.py`/`.sh` in the model's `scripts/` except the stage and conversion scripts below, declared training code, compiler version |
| performance | DevKit | `scripts/performance.py` | `scripts/run_model_performance.sh` | the compile run and MPK digest, `benchmark:` entries, `performance.py`, shared performance tooling, DevKit runtime pin |
| accuracy | DevKit | `scripts/accuracy.py` + `requirements-accuracy.txt` | `scripts/run_model_accuracy.sh` | the compile runs and MPK digests of its components, the `accuracy:` entry, the script and its requirements, shared accuracy tooling, DevKit runtime pin |

The keys are computed by `compile_fingerprint` in
`tools/model_registry/catalog.py` and by `performance_key` and `accuracy_key`
in `tools/model_registry/stages.py`; read them before changing what a stage
depends on. Conversion (`scripts/convert.py`, run by
`scripts/run_model_conversion.sh`) produces the ONNX that `model.yaml` pins as
the source; only that pin enters the compile key.

What a change reruns:

| Change | Reruns |
|---|---|
| registry tooling outside a stage's tooling, a README, `convert.py` | nothing |
| a model's `performance.py` | that model's performance |
| a model's `accuracy.py` | that model's accuracy |
| the DevKit runtime pin | every performance and accuracy stage |
| a model's compile inputs or the compiler pin | that model's builds (all builds for the compiler pin), then their board stages |

A board key hashes only its own stage script. A helper module that
`performance.py` or `accuracy.py` imports from `scripts/` is a compile input,
so editing it recompiles the model and then remeasures it. Keep board-only
code in the stage script itself.

## CI

- **Pull request.** `Vulcan Model Compile` validates the catalog, taxonomy, and
  compiler pin and prints an offline stage plan. Contributor code never runs on
  self-hosted runners for a pull request. `Model Registry Contributor Tool`
  runs `pytest tests/unit` and builds the `model-registry` package.
- **Push.** The `plan` job runs `model-registry plan-stages`, which reads the
  branch catalog (the base branch's when the branch has none) and resolves each
  stage key against the registry. `build` calls
  `.github/workflows/Vulcan-Model-Build.yml` once per build to compile,
  publish, and start the build's board stages. `publish-catalog` writes the
  final catalog snapshot, and `measure` starts the board stages a reused build
  still lacks.
- **Board stages.** `Vulcan-Model-Benchmarking.yml` runs the requested
  `stages` on a DevKit and publishes performance (`model_benchmark` runs) and
  accuracy (`model_accuracy` runs) separately.
- A newer push cancels the branch's running compile workflow; builds that
  already published stay and are reused.

## Contributing

1. Declare the change in `model.yaml` and the model's `scripts/`, following
   [adding a model](docs/adding-a-model.md) and the
   [metadata schema](docs/metadata-schema.md).
2. Run `pytest tests/unit` and `model-registry validate models/<model_id>`.
3. Compile each changed build in the pinned Model Compiler with
   `scripts/run_model_compilation.sh` and check every MPK with
   `model-registry verify-mpk` ([local development](docs/local-development.md)).
4. On a DevKit, run each changed component's `scripts/performance.py --mpk
   --output` and, for builds that declare accuracy,
   `scripts/run_model_accuracy.sh` on the compile output.
   `scripts/run_model_performance.sh` needs a registered build, so CI runs it
   ([benchmarking](docs/benchmarking.md),
   [local accuracy](docs/local-accuracy-development.md)).
5. Report which stages you ran and where; CI publishes the results.

Conventions the code relies on:

- Model scripts receive verified local paths (`--model`, `--dataset`,
  `--fidelity-dataset`, `--mpk`); data enters only through the dataset registry
  declared in `model.yaml`. Board stage scripts run without registry
  credentials.
- A CI step calls a named script under `scripts/` or a `model-registry`
  command; keep workflow YAML to orchestration.
- Schemas under `schemas/` reject unknown fields. Published results keep their
  `schema_version` (performance 5, accuracy 2, numerical fidelity 1) because
  the artifact browser and `sima-cli` choose a renderer by it. Registry
  documents that only this repository's tooling reads (`model.yaml`, compile
  outputs, package manifests) carry no version; change their readers and
  writers together instead.
- When a published result's shape changes, update `artifact-browser/` in the
  same change and keep it rendering results already in the registry.

## Model README requirements

Every top-level registry entry under `models/<model_id>` must contain a
`README.md`. The README begins with the following structure:

1. `# <display name>`
2. An optional `![<display name> model hero](hero.png)`
3. `## Overview`
4. `## Usage`
5. `## Model interface`
6. `## Available builds`
7. `## Quality and performance`
8. `## Source and license`
9. `## Reproducing the artifact`

The H1 must exactly match `display_name` in `model.yaml`. Keep the listed H2
headings even when a section is brief so model documentation remains
predictable across the registry. Third-level headings may organize conversion,
training, calibration, compilation, or validation details under
`## Reproducing the artifact`.

The Model Browser and `sima-cli` use `## Overview` as the model overview. It
must be the first H2, immediately after the required hero image, and must:

- contain one or two prose paragraphs totaling 50 to 120 words;
- explain what the model does and its intended use;
- summarize the expected input and meaningful output or application boundary;
- state a material limitation, qualification caveat, or deployment constraint
  when one exists;
- avoid tables, lists, commands, subheadings, hashes, workflow instructions,
  compiler internals, and other maintainer-only details.

Write the remaining sections for model users first and maintainers second:

- `Usage` provides the normal installation or download command and the shortest
  supported path to using the model.
- `Model interface` describes preprocessing, inputs, outputs, and required
  application-side postprocessing.
- `Available builds` explains the purpose and deployment differences of the
  declared build profiles and components.
- `Quality and performance` distinguishes authoritative qualification or
  accuracy results from diagnostic fidelity and benchmark measurements.
- `Source and license` identifies the upstream model, meaningful derivation,
  model card, and redistribution restrictions.
- `Reproducing the artifact` contains conversion, training, calibration,
  compilation, validation, and CI details needed by maintainers.

Treat `model.yaml` as authoritative for IDs, tensor declarations, source pins,
checksums, licenses, build profiles, and component structure. Do not copy large
metadata blocks into the README. When prose mentions one of these values,
verify it against `model.yaml` and the current scripts so the documentation
cannot silently drift. Commands must be runnable from the repository root and
must use current paths and supported CLI syntax.

These requirements apply to top-level model READMEs consumed by the registry.
Nested documentation such as training or QAT guides may use a structure suited
to that workflow, but the top-level README should link to it from the relevant
standard section.

When supplied, the hero must be a regular (non-symlinked) 1200x480 PNG no
larger than 2 MiB. The registry's strict PNG profile permits `IHDR`, optional
`PLTE`, `IDAT`, `IEND`, and optional validated `tIME` or `tRNS` ancillary
chunks. Keep it legible in light and dark themes, avoid embedding essential
instructions in the pixels, and use the exact accessible Markdown alt text
shown above. The registry publishes the image as a checksum-bound Vulcan
artifact alongside the README; do not use a placeholder, remote image URL, or
an image outside the model directory.

When changing model behavior, interfaces, supported builds, qualification
evidence, installation steps, or redistribution terms, update the owning
README and any existing hero image in the same change. A README- or hero-only edit is
documentation or catalog metadata; it must not change compilation, validation,
benchmark, or artifact fingerprints.
