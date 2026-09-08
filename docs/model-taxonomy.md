# Model categorization taxonomy

`taxonomies/model-taxonomy.yaml` is the canonical source of searchable model
categories. It owns stable machine IDs, display labels, architecture paradigms,
and model-type compatibility rules. JSON Schema validates the shape of category
references; the Model Registry semantic validator resolves those references
against this file.

Do not copy sanctioned lists into model definitions, browser code, API code, or
database migrations. Individual schema-version 2 models declare references:

```yaml
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

## Publication identity

The taxonomy does not have an independently managed release version. Each
complete catalog publication records its full Git commit as
`catalog_revision` and the canonical taxonomy SHA-256 as an integrity checksum.
Several progress snapshots may share a Git revision; each has its own canonical
SHA-256 identity. Historical catalog snapshots remain immutable and therefore retain the exact
taxonomy used to interpret their model metadata.

Model schema v2 and all authoritative category facets are mandatory. Catalog
validation rejects a missing field, an unknown sanctioned ID, or an incompatible
model type before any compilation job can be scheduled.

Registry publication validates the complete taxonomy and model catalog before
atomically promoting a catalog snapshot. Immutable compile runs contain build
provenance; current categories and display metadata come from the current
catalog snapshot.

## Renames and removals

Canonical IDs are stable. To rename a concept, first add the replacement ID and
retain the old entry as the compatibility alias while model declarations are
migrated. Its display name may mark it deprecated; there is no separate alias
registry or taxonomy release number. Remove the old entry only after no model
in the complete catalog references it. Full-catalog validation makes an early
removal fail before publication.

To remove a model or compile invocation, omit it from the next complete catalog
snapshot. It disappears from current search and browser views, while historical
snapshots, runs, and MPKs remain available under their existing retention rules.

## Search semantics

The registry read API supports model type, task, architecture, architecture
paradigm, input/output modality, and optional tag filters. Comma-separated
values use OR semantics within one array facet; different facets combine with
AND semantics. Pagination cursors are bound to the complete active filter set.
