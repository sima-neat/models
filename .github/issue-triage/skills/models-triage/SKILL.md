---
name: models-triage
description: Classify sima-neat/models GitHub issues and identify missing evidence using the supplied issue context.
---

# Models issue triage

Models owns model definitions (`models/<model_id>/model.yaml` and each model's `scripts/` and `tests/`), how each build is compiled, benchmarked, and published, the shared lifecycle in `tools/`, the CI workflows, the model browser in `artifact-browser/`, the public docs in `docs/`, and the customer skills in `skills/`. Read the issue body and discussion together. Compare it with related reports in the available context before suggesting overlap. Flag a likely duplicate for maintainer review and explain the evidence; do not use a duplicate label.

## Classification

Choose one label from this table and set `category` to the closest schema value.

| Label | When to use it |
| --- | --- |
| `bug` | A model fails to download, convert, compile, or run, a published result is wrong, or a documented command fails |
| `feature` | A new model, a new build of a model, or a new capability is requested |
| `enhancement` | Existing tooling, a model's performance or accuracy, or the model browser needs improvement |
| `documentation` | A doc page, a model README, or a customer skill needs correcting |
| `ci` | A workflow, runner, or publication job needs work |

Classify by the requested outcome. A model request uses `feature`. A slower or less accurate build than its published result uses `bug`. A request to make an existing build faster uses `enhancement`. A failure inside the Model Compiler or a Neat package that Models only calls is still labeled by its effect on the model; say in the comment which component the evidence points to.

Add `needs-repro` only when a missing model ID, build profile, command, version, or error output blocks investigation. Triage only adds labels and never removes one, so when an issue already carries `needs-repro` and the discussion now supplies enough information, say in the comment that a maintainer can remove it.

## Actionability

An actionable issue can have empty template sections. Ask one focused question only when its answer would change the investigation. For a bug, look for the model ID and build profile, the failing stage and command, the Model Compiler or Palette SDK version, and the host or DevKit. For a model request, look for a link to the official model and its license, the exact variant, and the precision needed. A documentation report may need only a page link and the wrong statement.

## Safety and comment

Models is public. If an issue names a customer, contains customer data or private assets, or looks security-sensitive, set `needs_human_review`, keep the public comment to a neutral acknowledgement, and request no extended analysis. Ask only for public model links, redacted output, or synthetic inputs, never for credentials, private URLs, or proprietary models.

Set `area` to the affected surface, such as `model-definition`, `compile`, `performance`, `accuracy`, `model-browser`, `download`, `docs`, `customer-skills`, or `ci`.

Keep the comment short. State the likely area and whether the issue is actionable, mention concrete evidence, and ask the one question needed to move forward. Ground any root-cause claim in the supplied evidence or inspected source.
