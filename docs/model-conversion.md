# Manual ONNX conversion

Convert a pinned checkpoint once, then use the published ONNX for compilation.
Conversion runs on a generic Vulcan Ubuntu ARM64 runner in staging. It uses its
own Python environment and does not activate Model Compiler.

A conversion-enabled model has three required files:

- `convert.yaml` declares verified source files, the Python version and dependency lock.
- `convert.py` loads the model, prepares export, writes `model.onnx`, and validates it against the original framework model.
- The dependency lock pins every installed package and wheel hash inside the model folder.

There is one ONNX graph per conversion. External tensor-data files stay beside
it with their relative paths intact. The model ID comes from the model directory;
`model.onnx` is the fixed output filename. Existing ONNX-source models, including
ResNet and MobileNet, do not need conversion files.

## Local use

Run the checkout's Model Registry CLI from a tooling environment:

```sh
model-registry validate-conversion models/rf_detr_small --json
model-registry convert models/rf_detr_small \
  --python python3.12 --work-dir /tmp/rfdetr-conversion \
  --cache-dir /tmp/model-conversion-cache --json
```

The work directory must be new. The CLI fetches and verifies the declared files,
creates a venv without system packages, installs the hash-locked dependencies,
and calls:

```sh
<venv>/bin/python -E -s models/rf_detr_small/convert.py \
  --source <verified-checkpoint> --output-dir <work-dir>/export
```

The converter must return nonzero on failure. A successful output contains
`model.onnx` and `validation.json` with `status: passed`; the CLI adds
`provenance.json`. Conversion logs and failure reports remain in the work
directory. RF-DETR also preserves framework inputs/outputs in `reference.npz`.
Export validation does not establish surgery or AFE compatibility.

The RF-DETR lock targets Python 3.12 on Linux ARM64, including the SDK. It is not
a macOS lock. The CLI caches downloaded dependencies by OS, architecture,
Python version and lock identity. The manual workflow restores this cache from
GitHub Actions storage onto NVMe; each conversion still receives a fresh venv.
Compilation only downloads the produced ONNX and never needs this venv.

## RF-DETR Hugging Face test

The default declaration uses the pinned Roboflow `.pth` checkpoint. The optional
`convert-hf.yaml` selects the official SafeTensors checkpoint and its configuration:

```sh
model-registry convert models/rf_detr_small --config convert-hf.yaml \
  --python python3.12 --work-dir /tmp/rfdetr-hf-conversion \
  --cache-dir /tmp/model-conversion-cache --json
```

The same `convert.py` selects the loader by checkpoint suffix. The Hugging Face
branch freezes positional encodings for 512×512 on an export copy and compares
against the unmodified model. Both branches check opset 17, exact tensor names,
shapes and float types, and numerical agreement on two deterministic synthetic
images and a pinned real image at `atol=rtol=0.001`.

The Hugging Face experiment currently fails the real-image numerical check and
its graph does not match the existing Roboflow surgery. It must not be published
as a validated compiler input. Do not reorder proposals or loosen tolerances to
turn that failure into a passing export. QAT is outside this workflow.

## Run on Vulcan

In GitHub Actions, select **Vulcan Model Convert**, choose the branch, and provide
the model directory ID. Leave `config` as `convert.yaml` for the ordinary route;
use `convert-hf.yaml` for the RF-DETR experiment. The workflow pins the branch SHA
across its jobs. Conversion only runs through manual dispatch, using staging's
default conversion capacity and NVMe storage.

The workflow validates the declaration before allocating the conversion runner,
restores the dependency cache, runs conversion, and publishes only passing
exports through the pinned Vulcan bundle helper. It verifies the published
bundle using Model Registry's own download path before emitting a source pin.
The job summary and downloadable diagnostics contain the proposed declaration:

```yaml
source:
  type: onnx_bundle
  uri: https://artifacts.stg.neat.sima.ai/models-onnx/<model>/<fingerprint>/<run-attempt>/manifest.json
  sha256: <completion-manifest-sha256>
  size_bytes: <completion-manifest-byte-size>
```

Use the exact generated values in `model.yaml`. The immutable manifest pins the
bundle parts, external files, hashes and sizes. Model Registry verifies downloads
and cached files, rejects unsafe archives, and returns the extracted `model.onnx`
to the normal recipe. A missing or corrupt member fails before compilation.
Successful publication does not automatically change the source declaration.

The existing `sima-cli download <bundle-url> --dest <directory>` can retrieve a
bundle object listed in that manifest. Bundles larger than 1 GiB have several
parts; Model Registry fetch assembles those parts and performs the integrity
checks. There is no mutable `latest` reference or automatic model-name download
catalog added by this change.

Deliberate re-export creates a new run/attempt version. Completed versions are
retained by Vulcan; failure diagnostics have a 14-day Actions retention period.
An interrupted runner may leave only the logs already uploaded by Actions.

## Dependency updates

Edit the model-local `requirements-convert.in`, resolve it on Python 3.12 Linux
ARM64 with `uv pip compile --generate-hashes`, and download the resolution with
`pip download --require-hashes --only-binary=:all:`. Record each downloaded
wheel's package version and SHA-256 in `requirements-convert.lock`. Re-run export
and numerical validation after every dependency change. The lock intentionally
contains only wheels for the supported export platform, with all transitive
packages pinned. Do not update Model Compiler's packages to make export work.
