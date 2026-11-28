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

The checkpoint is a pinned Hugging Face file or a release file of a public
GitHub repository. [Pin every input](adding-a-model.md#3-pin-every-input)
lists the approved forms.

## What `convert.py` exports

`convert.py` exports the official model as it is: its forward pass at the input shape
the package ships, from the model's input to the final outputs the package keeps. It may
simplify the ONNX with `onnxsim.simplify`, including constant folding, when that keeps
the same inputs, outputs, and computation.

Every model exports at one ONNX opset, which
[`deps/manifest.json`](../deps/manifest.json) sets as `onnx.opset`. The export passes
`onnx_opset_version()` from `model_registry.dependencies`, and `validate_export` checks
the written file with `require_onnx_opset_version(model)`. A `surgery.py` that exports
a graph does the same.

Every other change made for the compiler belongs in `surgery.py`, even one that leaves
the results exactly the same. That includes a cut: an export that ends at an inner
tensor, such as the raw heads before the official decode step, or one that starts after
an official step. This split keeps the converted ONNX comparable with the official model.

## What to keep in Git

Keep upstream pins and exporter dependencies in Git. Do not copy the output URL,
checksum, size, or run ID into `model.yaml`. The folder identifies the model and
the conversion produces `model.onnx`, unless `conversion.outputs` names other
files. Direct file and file-set sources continue to work.

## Publish other files

To publish several ONNX models, or a PyTorch checkpoint, list the files under
`conversion.outputs`:

```yaml
conversion:
  source:
    uri: hf://models/owner/repository@0123456789012345678901234567890123456789/model.pt
    sha256: <upstream checksum>
    size_bytes: <upstream size>
  outputs: [vector_estimator.onnx, vocoder.onnx]
compilation:
  format: onnx
  source: conversion
  components: [vector_estimator, vocoder]
```

Each name is a file name you choose, ending in `.onnx`, `.pt` or `.pth`, without
a folder. Order matters: `compile.py` receives the first file as `--model`, and
the others sit beside it. The default is `[model.onnx]`, so a model with one ONNX
leaves the field out.

Scripts read the names from `model.yaml` and never hard-code them. Renaming a
file then takes one edit in `model.yaml`, followed by a new conversion:

```python
from model_registry.conversion import conversion_outputs

# convert.py
estimator, vocoder = conversion_outputs(MODEL_DIR)
export_estimator(args.output_dir / estimator)
export_vocoder(args.output_dir / vocoder)

# compile.py
_, vocoder = conversion_outputs(MODEL_DIR)
vocoder_path = args.model.with_name(vocoder)
```

`compile.py` and `surgery.py` can also pick a file for each build profile by
branching on `--build-profile`.

For graph surgery in PyTorch, publish the checkpoint with `outputs: [weights.pth]`.
`compile.py` receives `weights.pth`, and `surgery.py` edits the model and exports
the ONNX to compile, so `compilation.format` stays `onnx`.

Convert fails when a listed file is missing, and runs the ONNX checker on every
`.onnx` file. Nothing shared checks a `.pt` or `.pth` file, so the model's
conversion tests must do that. Their `converted` fixture is the first file, and
the others sit beside it.

## Publish a conversion

No push runs a conversion. In **Actions**, run **Vulcan Model Convert** on the branch
that holds the model. Give it the model's ID and the registry environment, `staging`
or `production`.

The workflow installs the model's exporter requirements and runs `convert.py` and the
model's conversion tests. It then publishes the declared outputs, with any external
tensor files the ONNX files reference. A failed or skipped conversion test publishes
nothing.

A conversion belongs to the environment it ran in. To move a pinned model to
`compilation.source: conversion`, run Convert in each environment that builds the
model, then change the source.

CI resolves `compilation.source: conversion` to the latest published conversion that
matches the model's current conversion inputs. When none matches, CI stops with
`<model-id>: no matching validated conversion; run Vulcan Model Convert first`. The
compile fingerprint uses the content of the converted files, so a conversion that
publishes the same bytes again recompiles nothing.

### When to run Convert again

Run **Vulcan Model Convert** again after changing anything that can change the
converted files:

- the checkpoint pin under `conversion.source`, or `conversion.outputs`
- `scripts/convert.py`, or a module beside it that `convert.py` imports, directly or
  through another such module
- any other file directly in `scripts/` that is not a `.py` file, such as
  `requirements-convert.txt` or an exporter patch
- any file in a subfolder of `scripts/`
- the ONNX opset, `onnx.opset` in `deps/manifest.json`, which changes every
  conversion-backed model at once

Three files never count, wherever they sit: `requirements-compile.txt`,
`requirements-accuracy.txt`, and `requirements-performance.txt`.

Everything else leaves the conversion valid, including shared tooling, tests,
compile and benchmark code, and the taxonomy. A changed conversion test runs the
next time Convert runs.

No shared file is on the list because the checkpoint is verified against its
pinned SHA-256: where it is hosted and how it is fetched cannot change the output.
When a shared change does alter converted output, such as a new Python version in
`scripts/run_model_conversion.sh` or a new bundle layout, raise
`FINGERPRINT_VERSION` in `tools/model_registry/conversion.py` in the same change.
Every conversion-backed model then needs Convert again.

Leave a successful conversion alone. A repeated Convert whose output bytes differ
recompiles that model on every branch.

Import helpers with a plain `import helper` or `from helper import name`. The check
does not detect a module that `convert.py` loads dynamically or through a relative
import, so a change to it does not ask for a new conversion. Pin exporter
requirements too, because the check does not track packages they pull in indirectly.
