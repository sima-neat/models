# `model.yaml` schema v1

`schemas/model.schema.json` is the authoritative contract. YAML is parsed with
the safe loader and validated with JSON Schema Draft 2020-12.

The contract requires identity and ownership, license policy, immutable source
provenance, ONNX inputs and outputs, preprocessing and postprocessing,
calibration status, surgery behavior, compile variants, tests, and release
maturity. Remote sources require SHA-256 and byte size.

Calibration is explicit even before a dataset is approved. `status: pending`
keeps an experimental entry honest and blocks a future release gate; `status:
ready` requires a checksummed source declaration.

Each compile variant declares a semantic ID, target, model-local Python recipe,
and canonical parameter map. The recipe path is validated to exist inside the
model directory. Variant IDs must be unique within a model. The exact compiler
version is repository-wide and comes from `deps/manifest.json`, preventing
variants from silently selecting different Model Compiler environments.

Variants are supported by default. A combination proven unusable with the
repository compiler may be retained as `status: unsupported` together with a
non-empty `reason`. Unsupported variants remain part of the documented model
contract, but matrix generation and release packaging exclude them. This keeps
known limitations visible without publishing an artifact that failed its
qualification evidence.

Validation errors use paths such as `source.sha256` and include the model
directory, making both local and CI failures actionable.
