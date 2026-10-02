# SiMa Neat Models

This repository is the source of truth for model definitions, reproducible
compilation, validation, and publication. Keep model metadata declarative,
model-specific preparation inside the owning model directory, and shared
lifecycle behavior in `tools/model_registry`.

## Model README requirements

Every top-level registry entry under `models/<model_id>` must contain a
`README.md` with the following section order:

1. `# <display name>`
2. `## Overview`
3. `## Usage`
4. `## Model interface`
5. `## Available builds`
6. `## Quality and performance`
7. `## Source and license`
8. `## Reproducing the artifact`

The H1 must exactly match `display_name` in `model.yaml`. Keep the listed H2
headings even when a section is brief so model documentation remains
predictable across the registry. Third-level headings may organize conversion,
training, calibration, compilation, or validation details under
`## Reproducing the artifact`.

The Model Browser and `sima-cli` use `## Overview` as the model overview. It
must be the first H2, immediately after the H1, and must:

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

When changing model behavior, interfaces, supported builds, qualification
evidence, installation steps, or redistribution terms, update the owning
README in the same change. A README-only edit is documentation or catalog
metadata; it must not change compilation, validation, benchmark, or artifact
fingerprints.
