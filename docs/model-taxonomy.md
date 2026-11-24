# Model categorization taxonomy

`taxonomies/model-taxonomy.yaml` is the canonical source of searchable model
categories. It owns stable machine IDs, display labels, architecture paradigms,
and model-type compatibility rules. JSON Schema validates the shape of category
references; the `model-registry` semantic validator resolves those references
against this file.

Do not copy sanctioned lists into model definitions, browser code, API code, or
database migrations. Each model declares references:

```yaml
model_type: vision
tasks: [image_classification]
modalities:
  inputs: [image]
  outputs: [class_scores]
architecture: resnet
```

The facets are intentionally independent:

- model type describes the broad product class, such as Vision, LLM, or VLM;
- task describes what the model does;
- modality describes semantic inputs and outputs;
- `architecture` names the model family, the upstream model line without its
  size: `yolo26_det`, not `yolo26n_det`, and `resnext`, not `resnext_50`. The
  model's `id` and `display_name` carry the size. The browser labels this filter
  Model family;
- the family's `paradigm` (CNN or Transformer) is derived, not declared. The
  browser labels it Architecture.

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

The taxonomy has no release version of its own. CI publishes it with every
catalog, so each published catalog keeps the exact taxonomy that describes its
models.

No conversion, compile, or benchmark fingerprint includes the taxonomy. A
taxonomy change, such as a new family description, therefore reruns nothing,
and existing builds and results stay reusable.

All authoritative category facets are mandatory. Validation rejects a missing
field, an unknown sanctioned ID, or an incompatible model type before CI
compiles anything.

## Renames and removals

Canonical IDs are stable. To rename or remove a concept, change its taxonomy
entry and every model that references it in one pull request; there is no
alias, deprecated entry, or taxonomy release number. Full-catalog validation
rejects a model that still references a removed ID before publication.

To remove a model or a build profile, delete it from the repository. After the
next catalog publication, the model browser no longer lists it. A promoted build
of a removed profile stays listed until a maintainer withdraws it on each branch
([withdraw a promoted build](model-promotion.md#withdraw-a-promoted-build)).
