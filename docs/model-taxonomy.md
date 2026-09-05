# Model categorization taxonomy

`taxonomies/model-taxonomy-v4.yaml` is the current canonical source of searchable model
categories. It owns stable machine IDs, display labels, architecture paradigms,
and model-type compatibility rules. JSON Schema validates the shape of category
references; the Model Registry semantic validator resolves those references
against this file.

Do not copy sanctioned lists into model definitions, browser code, API code, or
database migrations. Individual schema-version 2 models declare references:

```yaml
taxonomy_version: 4
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

Model schema v2 and all authoritative category facets are mandatory. Catalog
validation rejects a missing field, an unknown sanctioned ID, or an incompatible
model type before any compilation job can be scheduled. Taxonomy v1 remains in
the repository as immutable publication history; taxonomy v2 adds the SSD and
YOLO architecture families needed by the fully categorized catalog.

Registry publication uploads the immutable taxonomy before a categorized run
and records canonical category IDs with that run. Existing records are not
guessed or heuristically backfilled.

## Search semantics

The registry read API supports model type, task, architecture, architecture
paradigm, input/output modality, and optional tag filters. Comma-separated
values use OR semantics within one array facet; different facets combine with
AND semantics. Pagination cursors are bound to the complete active filter set.
