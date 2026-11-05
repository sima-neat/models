# Converted ONNX artifacts

Models with an upstream checkpoint can use this declaration:

```yaml
conversion:
  source:
    uri: hf://models/owner/repository@0123456789012345678901234567890123456789/model.pt
    sha256: <upstream checksum>
    size_bytes: <upstream size>
compilation:
  format: onnx
  source: conversion
```

Keep upstream pins and exporter dependencies in Git. Do not copy the output URL,
checksum, size, or run ID into `model.yaml`. The folder identifies the model and
the single conversion produces `model.onnx`. Direct file and file-set sources
continue to work.

## Publication and selection

Run **Vulcan Model Convert** in the selected environment. The explicit
`staging`/`production` input selects the matching GitHub environment and runner;
the shared configuration resolver returns `stg`/`prod` aliases for infrastructure
lookup, which must not be used as GitHub environment names. It installs the model's
exporter requirements, runs conversion and model tests, checks the ONNX, and
publishes a deterministic bundle under `models/conversions/`. The bundle contains
`model.onnx`, any referenced external tensor files, and a checked file manifest.
Skipped or failed model tests do not publish successful conversions.

Vulcan stores ordinary `model_conversion` runs with upstream inputs, parameters,
validation and the `onnx` artifact.

### When to run Convert again

Run **Vulcan Model Convert** again after changing anything that can change the ONNX:

- the upstream pins under `conversion.source`
- `scripts/convert.py`, `scripts/requirements-convert.txt`, or a module in `scripts/`
  that `convert.py` imports
- any other file under `scripts/` that is not a Python module beside `convert.py`,
  except `requirements-compile.txt` and `requirements-accuracy.txt`
- shared export code, listed in `CONVERSION_TOOLING` in
  `tools/model_registry/conversion.py`

Everything else leaves the conversion valid, including tests, compile and benchmark
code, and the taxonomy. A changed conversion test runs the next time Convert runs.

Use a normal `import` for helpers, because a module loaded dynamically is not
detected. Pin exporter requirements too, because packages they pull in indirectly
are not tracked.

Catalog planning resolves conversion declarations for the requested models (all
models when no `--models` filter is supplied), including on-demand models,
so their build requirements have concrete content identities. This does not
schedule compilation of an on-demand model. During targeted catalog publication,
unselected conversion-backed models keep their previous build/package entries
and on-demand requirements, along with any existing pending build records;
documentation receipts are refreshed for the current
catalog revision. Their conversions are not re-resolved or
marked pending. A new unselected conversion-backed model enters the catalog when
it has a selected conversion plan. Before deciding compile reuse, planning resolves the latest completed matching
run and checks its artifact. The plan uses ordinary `model_compile_plan` run
records and versioned inputs. Its identity includes the GitHub workflow run,
commit, model and conversion fingerprint. Rerunning that workflow resumes its
saved selection, including after interrupted publication. A new workflow can
select a newer compatible conversion. Missing or unavailable output fails with
an instruction to run conversion; it never falls back to an older artifact.

The selected metadata travels as `MODEL_REGISTRY_CONVERSIONS` to compile,
publication and catalog jobs. It contains exact object versions, not signed
URLs. Downloads request fresh URLs for that exact artifact and verify both the
bundle checksum and member checksums. Compile fingerprints use output content;
another run or environment with identical bundle bytes does not change them.
Compile publication records the conversion as a versioned registry input.

For explicit local reproduction, save the original plan's `conversion_sources`
JSON map and use it with the same checkout and compiler:

```bash
export MODEL_REGISTRY_CONVERSIONS="$(cat conversion-sources.json)"
model-registry fetch models/<model-id> --json
scripts/run_model_compilation.sh models/<model-id> <profile> <profile> /tmp/compiled
```

The plan is also recoverable through its run's `/provenance` endpoint: its
`conversion` input records producer/artifact IDs and content metadata. Retained
compile inputs record the same selection. Use the original environment's reader;
do not substitute a current resolve result when reproducing an older build.

## Retention and rollout

Deploy Vulcan's conversion validation and migration
`019_model_conversion_retention` before enabling registry sources. Plans protect
selected conversions for 24 hours, covering the normal six-hour compile limit
and queue time. Abandoned plans do not create permanent holds. After publication,
retained compile inputs protect conversion bytes and provenance, including when
the conversion branch is deleted. After the plan window, an unpublished build
has no retention guarantee; unavailable outputs require a new workflow.

Completed conversions without consumers follow normal branch retention. Newer
conversion publication alone does not delete older outputs. The legacy S3
sweeper skips the shared `models/conversions/` prefix because it has no branch
`latest.tag`; registry cleanup owns these versioned objects.

To move a pinned model to `compilation.source: conversion`, publish its conversion in
each environment that needs it, then change the source. Conversions are not promoted
between environments, and nothing runs Convert automatically.

### Private production reads

When a selection uses the IAM API, compile fetches sign artifact metadata requests
with SigV4, including cache hits. The compile job assumes the registry role with
an inline session policy restricted to `GET /v1/runs/*/artifacts/onnx`; the session
cannot publish, write objects, or access KMS directly. These credentials are
passed only to the registry reader. Immediately after compilation, the job assumes
a fresh session with the same restricted policy for provenance verification, so a
long compilation cannot leave cached-source verification using an expired session.
Dataset access retains the runner credentials. Vulcan must grant that GET
route to the role before enabling this path. Public metadata reads stay unsigned,
and the artifact body uses the version-specific signed download URL. Exact version,
checksum, size, and validation checks apply to both paths.

The offline browser preview supplies deterministic synthetic conversion identities
for its fixture catalog. It does not resolve real conversions or require registry
credentials; production planning still requires validated registry selections.
