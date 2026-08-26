# Model categorization taxonomy

`taxonomies/model-taxonomy-v1.yaml` is the canonical source of searchable model
categories. It owns stable machine IDs, display labels, architecture paradigms,
and model-type compatibility rules. JSON Schema validates the shape of category
references; the Model Registry semantic validator resolves those references
against this file.

Do not copy sanctioned lists into model definitions, browser code, API code, or
database migrations. Individual schema-version 2 models declare references:

```yaml
taxonomy_version: 1
model_type: vision
tasks: [image_classification]
modalities:
  inputs: [image]
  outputs: [class_scores]
architecture:
  id: resnet_50
  variant: imagenet_1k
```

The facets are intentionally independent:

- model type describes the broad product class, such as Vision, LLM, or VLM;
- task describes what the model does;
- modality describes semantic inputs and outputs;
- architecture identifies the sanctioned family/version and derives its
  technical paradigm, such as CNN or decoder transformer;
- tags are normalized free-form secondary descriptors.

## Versioning and rollout

The taxonomy is immutable once its version is published. Every additive or
corrective change requires a new taxonomy version, including display-label
corrections, so historical records remain deterministic. Adding sanctioned IDs
therefore evolves taxonomy data without requiring another model-schema or
database-schema redesign.

Model schema v1 remains supported during migration and has no categorization.
Schema v2 requires all authoritative category facets. ResNet-50 is the initial
v2 validation model; other models remain explicitly uncategorized until their
definitions are migrated and rebuilt.

Packaging emits `taxonomy.json`, records its SHA-256 in schema-version 2
`latest.json`, and projects canonical category IDs into `latest.json` and
`catalog.json`. Registry publication uploads the immutable taxonomy before a
categorized run. Existing v1 run records are not guessed or heuristically
backfilled.

## Search semantics

The registry read API supports model type, task, architecture, architecture
paradigm, input/output modality, and optional tag filters. Comma-separated
values use OR semantics within one array facet; different facets combine with
AND semantics. Pagination cursors are bound to the complete active filter set.
