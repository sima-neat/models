# Model categorization taxonomy

`taxonomies/model-taxonomy.yaml` is the canonical source of searchable model
categories. It owns stable machine IDs, display labels, architecture paradigms,
and model-type compatibility rules. JSON Schema validates the shape of category
references; the Model Registry semantic validator resolves those references
against this file.

Do not copy sanctioned lists into model definitions, browser code, API code, or
database migrations. Each model declares references:

```yaml
model_type: vision
tasks: [image_classification]
modalities:
  inputs: [image]
  outputs: [class_scores]
architecture:
  id: resnet
```

The facets are intentionally independent:

- model type describes the broad product class, such as Vision, LLM, or VLM;
- task describes what the model does;
- modality describes semantic inputs and outputs;
- `architecture.id` names the model family, the upstream model line without its
  size: `yolo26_det`, not `yolo26n_det`, and `resnext`, not `resnext_50`. The
  model's `id` and `display_name` carry the size. The browser labels this filter
  Model family;
- the family's `paradigm` (CNN or Transformer) is derived, not declared. The
  browser labels it Architecture;
- `tags` are optional search terms that no facet covers, such as a training
  dataset (`coco`) or a deployment trait (`one_class`). Write each as a
  lowercase ID with underscores, and do not repeat a task, family, or model
  type.

To categorize a model, take every ID from the taxonomy. When none fits, add the
entry to `taxonomies/model-taxonomy.yaml` in the same pull request.

Each architecture entry requires a short `description` for the family card,
alongside its existing `display_name`, `paradigm` and `allowed_model_types`:

```yaml
architectures:
  yolov8_obb:
    display_name: YOLOv8-obb
    description: Detects objects with rotated bounding boxes in aerial imagery.
    paradigm: cnn
    allowed_model_types: [vision]
```

Describe the whole family, without size-specific claims. Keep this text in the
taxonomy; individual `model.yaml` files do not repeat it. The browser derives
family membership, tasks and the representative hero from available models.
Cards group and sort by the existing architecture ID and use `display_name`
as their title. Detection, segmentation, pose and OBB keep their separate IDs.

## Publication identity

The taxonomy does not have an independently managed release version. Each
complete catalog publication records its full Git commit as
`catalog_revision` and the canonical taxonomy SHA-256 as an integrity checksum.
Historical catalog snapshots remain immutable and therefore retain the exact
taxonomy used to interpret their model metadata.

Family descriptions travel in the full catalog taxonomy and change its checksum,
but do not enter conversion, compile or benchmark fingerprints. Existing builds
and results remain reusable. Historical catalogs without descriptions still
render; their family cards omit the description.

All authoritative category facets are mandatory. Catalog
validation rejects a missing field, an unknown sanctioned ID, or an incompatible
model type before any compilation job can be scheduled.

Registry publication validates the complete taxonomy and model catalog before
atomically promoting a catalog snapshot. Immutable compile runs contain build
provenance; current categories and display metadata come from the current
catalog snapshot.

## Renames and removals

Canonical IDs are stable. To rename or remove a concept, change its taxonomy
entry and every model that references it in one pull request; there is no
alias, deprecated entry, or taxonomy release number. Full-catalog validation
rejects a model that still references a removed ID before publication.

To remove a model or compile invocation, omit it from the next complete catalog
snapshot. It disappears from current search and browser views, while historical
snapshots, runs, and MPKs remain available under their existing retention rules.

## Search semantics

The registry read API supports model type, task, architecture, architecture
paradigm, input/output modality, and optional tag filters. Comma-separated
values use OR semantics within one array facet; different facets combine with
AND semantics. Pagination cursors are bound to the complete active filter set.
