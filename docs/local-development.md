# Local development and testing

This guide covers checks that a contributor can run before opening a pull
request. Commands assume the repository root as the current directory.

## Set up the environment

Python 3.10 or newer is required; Python 3.12 matches the current GitHub
workflow. Create an isolated environment and install the registry plus
development dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
model-registry --help
```

The editable install exposes the `model-registry` command while keeping changes
to `tools/model_registry/` live. Re-run the install if dependency metadata
changes.

## Validate one model

```bash
model-registry validate models/<model-id>
```

This checks the JSON Schema, taxonomy references and compatibility, immutable
source policy, matching directory and model IDs, unique variant IDs, and every
referenced recipe, surgery hook, and test file. Fix the first reported field
paths in `model.yaml`; do not weaken a declaration just to pass validation.

Validate the complete catalog before submitting:

```bash
model-registry validate-catalog --repository .
```

This is the same catalog validation entry point used before CI considers a
compile matrix.

## Fetch and verify immutable inputs

```bash
model-registry fetch models/<model-id>
model-registry fetch-calibration models/<model-id>
```

`fetch` verifies exact byte size and SHA-256 before atomically installing the
source into a checksum-addressed cache. `fetch-calibration` applies the same
policy when calibration status is `ready`; other statuses report that no
calibration fetch is required.

The default cache is `~/.cache/model-registry`. For an isolated or disposable
cache, use either form:

```bash
export MODEL_REGISTRY_CACHE="$PWD/.model-registry-cache"
model-registry fetch models/<model-id>
model-registry fetch models/<model-id> --cache-dir "$PWD/.model-registry-cache"
```

`HF_TOKEN` is used when present and is never printed. Do not put tokens, signed
URLs, or cache content in model metadata or Git.

## Inspect compile selection

Confirm the repository compiler pin and the variants selected by a model-local
change:

```bash
model-registry resolve-compiler --repository .
model-registry matrix \
  --repository . \
  --changed-path models/<model-id>/model.yaml
```

The matrix output is deterministic JSON. It includes supported variants only.
To inspect a model or variant explicitly:

```bash
model-registry matrix --repository . --models <model-id>
model-registry matrix --repository . --models <model-id> --variants <variant-id>
```

Changes below `models/<model-id>/` select that model. Repository-wide compile
inputs such as `deps/` can select the full catalog. Some models also declare
additional build inputs outside their model directory. CI combines changed-path
selection with build fingerprints and the previous published index, so an
unchanged fingerprint can reuse an immutable build.

## Run host-side checks

Run lint and the complete unit suite:

```bash
python -m ruff check .
python -m pytest
```

For a quick documentation/model iteration, useful focused checks are:

```bash
python -m pytest tests/unit/test_docs.py
python -m pytest tests/unit/test_model.py tests/unit/test_catalog.py tests/unit/test_cli.py
```

These tests do not compile a model or exercise Modalix hardware. They validate
host-side contracts with fixtures and fakes.

## Optional local compilation

Full compilation is only possible in an environment containing the exact Model
Compiler version reported by `model-registry resolve-compiler`. A normal Python
virtual environment is not that compiler environment.

When the pinned environment is available:

1. Run `model-registry fetch ... --json` and take the returned verified `path`.
2. Choose a supported variant from the matrix.
3. Invoke the model recipe with a new output directory:

   ```bash
   python models/<model-id>/compile.py \
     --model <verified-source-path> \
     --output-dir <new-output-directory> \
     --variant <variant-id>
   ```

4. Verify that the directory contains exactly one `*_mpk.tar.gz` and validate
   it:

   ```bash
   model-registry verify-mpk <new-output-directory>
   ```

Use a location outside the repository or an ignored build directory. Never
commit the MPK or compiler intermediates. If the pinned compiler environment is
not available, record compilation as not run in the pull request rather than
claiming local qualification.

## What CI adds

Pull requests repeat catalog validation and matrix construction on a
GitHub-hosted runner. They do not run contributor-controlled recipes on Vulcan.
After a trusted push, selected variants run in the pinned self-hosted Model
Compiler environment. The workflow verifies source and calibration inputs,
executes the declared recipe, requires exactly one MPK, validates it, packages
provenance, and can dispatch staging benchmarks to a Modalix DevKit.

See [Contributing with Git and GitHub](contributing.md) for the review checklist
and [Architecture](architecture.md) for the full trust boundary.

## Common failures

| Failure | What to check |
| --- | --- |
| `id must match directory name` | Rename the directory or correct `id`; both must use the same snake-case value. |
| Unknown or incompatible taxonomy ID | Select a sanctioned v2 ID or propose a separately reviewed taxonomy version. |
| `source.sha256` or size mismatch | Hash and measure the exact immutable bytes addressed by `source.uri`. |
| Source host or URL rejected | Use a full-commit Hugging Face resource or an approved GitHub release asset. |
| Referenced file does not exist | Keep recipe, surgery, and test paths relative to and inside the model directory. |
| Variant missing from the matrix | Check its `status`, requested IDs, changed path, and whether CI can reuse its build fingerprint. |
| More than one MPK produced | Clean the new output directory and make the recipe emit exactly one archive. |
