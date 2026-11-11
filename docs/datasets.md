# Datasets for local model development

Model development is not complete without the data used to calibrate and
measure the model. A model contribution must resolve the same registered dataset
identity locally that trusted automation will use later. Do not replace it with
an untracked directory, a convenient download, or a private archive whose
contents cannot be reproduced.

The responsibilities are split across two repositories:

| Repository | Owns |
| --- | --- |
| `sima-neat/dataset` | Dataset IDs and variants, Artifactory source locations, governance, deterministic preparation, and immutable publication policy. |
| `sima-neat/models` | The exact dataset ID, variant, archive SHA-256, and size a model uses, plus the model's preprocessing, sample selection, and metrics. |

Dataset payloads belong in approved data storage, not Git. Model repositories
contain only the contract needed to resolve and verify those bytes.

## Set up Artifactory access first

Local calibration, fidelity, and accuracy work requires read access to the SiMa
Artifactory host at `artifacts.eng.sima.ai`. Before starting model work:

1. Connect to the SiMa private network or VPN.
2. Obtain read access to the dataset repository through the normal SiMa
   Artifactory access process.
3. Configure one supported authentication method without committing the
   credential or placing it in documentation, metadata, or shell scripts.
4. Install Dataset Registry into the same pinned Model Compiler environment as
   Model Registry.

```bash
sima-cli neat install --stg dataset/dataset-registry@main:latest
activate-model-compiler
dataset-registry --version
```

Dataset Registry checks Artifactory credentials in this order:

1. `JFROG_TOKEN` or `ARTIFACTORY_TOKEN` bearer token;
2. `ARTIFACTORY_API_KEY`;
3. `ARTIFACTORY_USER` and `ARTIFACTORY_PASSWORD`;
4. a host-specific entry for `artifacts.eng.sima.ai` in `~/.netrc`; or
5. anonymous access when the artifact permits it.

Prefer a credential supplied by the approved local secret-management flow.
Never paste a token into `model.yaml`, a model script, a pull-request log,
or a command that will be retained in shell history. If you use `~/.netrc`,
restrict it to the current user:

```bash
chmod 600 ~/.netrc
```

## Prove that the registered dataset resolves locally

Resolve the exact dataset and variant required by the model before compiling.
The result is JSON containing the verified local path and provider:

```bash
dataset_json="$(dataset-registry resolve imagenet-2012 \
  --variant mini \
  --cache-dir "$PWD/.dataset-cache")"
dataset_path="$(jq -r .path <<<"$dataset_json")"
jq -e '.provider == "artifactory"' <<<"$dataset_json"
test -f "$dataset_path"
```

The resolver prefers the registered Artifactory archive. It may fall back to
the archive's registered private S3 mirror when Artifactory is unavailable, but
model developers are expected to establish working Artifactory access for local
development rather than depend on CI credentials or an incidental cache.

Use a cache location with enough free space for the compressed archive and any
compiler intermediates. Dataset Registry verifies the provider metadata and
stores a verified local copy; do not rename a different archive into the cache
or bypass a checksum failure.

## Use the dataset through the model contract

`model.yaml` pins each registered archive where a stage uses it, with a
dataset ID, variant, SHA-256, and byte size. The archive goes by its Registry
source ID, `<id>-<variant>` such as `coco-2017-val`. Three declarations hold
pins:

| Declaration | Used by | Resolved with |
| --- | --- | --- |
| a `calibrations` entry, named by a build profile's `calibration_dataset` | `compile.py --dataset` | `model-registry fetch-calibration` |
| `numerical_fidelity.dataset` | `compile.py --fidelity-dataset` | `model-registry fetch-fidelity-dataset` |
| each pin in `accuracy.datasets` | `accuracy.py --dataset <source id>=<path>` | `scripts/run_model_accuracy.sh` |

Resolve the compile inputs for one component of a build:

```bash
calibration_json="$(model-registry fetch-calibration \
  models/<model-id> --build-profile <profile-id> --component <component-id> --json)"
fidelity_json="$(model-registry fetch-fidelity-dataset \
  models/<model-id> --build-profile <profile-id> --component <component-id> --json)"
```

These commands delegate registered dataset resolution to Dataset Registry and
then check the SHA-256 and size committed with the model.
`scripts/run_model_compilation.sh` runs both and passes the paths to the
recipe. On a DevKit, `scripts/run_model_accuracy.sh` resolves each accuracy
dataset into a shared cache and extracts it for `accuracy.py`. Follow [Qualify
locally](local-development.md) for the complete command sequence and
[Accuracy and fidelity](local-accuracy-development.md) for the entry
contracts.

## Required accuracy metrics

A dataset that models evaluate accuracy on has a file `datasets/<registry
id>.yaml` in this repository. It lists, per model task, the metrics every
accuracy report on that dataset must hold:

```yaml
# datasets/coco-2017.yaml
required_metrics:
  object_detection: [coco_map]
  instance_segmentation: [coco_mask_map]
  pose_estimation: [coco_keypoint_map]
```

A model must report, on each of its accuracy datasets, the metrics listed for
any of its `tasks`. Validation fails when a dataset has no file or the file
lists nothing for the model's tasks.
[`dataset-requirements.schema.json`](https://github.com/sima-neat/models/blob/1078eca263c1d9a017f1666c4e7659685bb78305/schemas/dataset-requirements.schema.json)
defines the file, and each metric must be an accuracy metric of
[`tools/model_report/metrics.json`](https://github.com/sima-neat/models/blob/1078eca263c1d9a017f1666c4e7659685bb78305/tools/model_report/metrics.json).

A change to a task's entry reruns accuracy only for the models with that task
that evaluate the dataset. A new file or an entry for a task no such model has
reruns nothing.

To start evaluating on a dataset that has no file yet, add the file with the
metrics the task needs, in the same change as the model or its new metrics.

## Introduce a dataset or variant that does not exist

Do not add a one-off download path to the model recipe. Introduce the data in
the Dataset Registry first, then consume the reviewed identity from the Models
repository.

### 1. Check the Dataset Registry catalog

Inspect
[`catalog/datasets.yaml`](https://github.com/sima-neat/dataset/blob/main/catalog/datasets.yaml)
for an existing dataset ID and suitable `artifactory_sources` variant. Reuse a
registered variant only when its contents, split layout, license, and intended
use actually match the model.

### 2. Establish provenance and permission

Record the official source, terms, license identifier, commercial-use and
redistribution status, intended use, owner, and required approval. Publicly
downloadable data is not automatically approved for SiMa calibration or
evaluation. Start unresolved access or license fields as `review-required`.

### 3. Place the archive through the authorized Artifactory process

An authorized dataset owner or Artifactory operator places the archive in the
approved SiMa dataset repository. The Models and Dataset Registry tools do not
grant Artifactory write access and must not receive upload credentials. Record
the stable HTTPS download URL and choose a lowercase kebab-case variant name.

### 4. Add or update the Dataset Registry entry

Open a change in `sima-neat/dataset`. Add the dataset inventory fields and an
`artifactory_sources` entry to `catalog/datasets.yaml`:

```yaml
- id: example-dataset
  priority: extension
  tasks: [image-classification]
  access: review-required
  license: review-required
  artifactory_sources:
    - id: example-dataset-mini
      variant: mini
      artifactory_url: https://artifacts.eng.sima.ai/artifactory/<approved-repository>/<archive>.tgz
      vulcan_key: datasets/artifactory/example-dataset/mini/<archive>.tgz
```

Add a `lifecycle` block and deterministic preparation implementation when the
dataset needs a canonical prepared release rather than only a registered legacy
archive. Keep model-specific resize, normalization, tensor layout, and output
decoding in the model contract; they do not belong in canonical dataset
preparation.

From a Dataset Registry checkout, validate the complete catalog and run its
tests using synthetic fixtures only:

```bash
dataset-registry validate --all --repository . --json
python -m pytest
```

Do not add the dataset archive, samples, credentials, cookies, or signed URLs
to the pull request. Dataset pull-request CI is intentionally credential-free
and does not download private data.

### 5. Wait for review, mirroring, and resolution

After the Dataset Registry change is reviewed and merged, protected automation
mirrors the registered Artifactory archive into private Vulcan storage. That
mirror is a transport fallback, not governance approval or an immutable
lifecycle release. Confirm the new identity resolves locally from Artifactory
before referencing it from a model.

### 6. Bind the reviewed dataset to the model

Add the exact dataset ID, variant, archive SHA-256, and size to the
calibration, fidelity, or accuracy declaration that uses it. Then run the full
required local workflow: resolve the registered archive, compile every
submitted build, verify its MPKs, and run its declared performance and
accuracy stages on a compatible Modalix DevKit.

## Troubleshooting access

| Symptom | What to fix |
| --- | --- |
| DNS failure or connection timeout | Connect to the SiMa private network or VPN and confirm `artifacts.eng.sima.ai` is reachable. |
| HTTP 401 or 403 | Refresh the supported credential and confirm it has read permission for the registered Artifactory repository. |
| Unknown dataset or variant | Update/install the current Dataset Registry package and check the authoritative catalog. If the identity is still absent, introduce it through the Dataset Registry workflow above. |
| Artifactory fails and S3 also fails | Fix local Artifactory access first; do not borrow CI credentials. The resolver reports both provider failures. |
| Size or checksum mismatch | Stop. Confirm the registered archive was not replaced and update the dataset through review; never weaken model verification. |
| Insufficient disk space | Use a larger cache location and remove only disposable, unreferenced local cache entries. |

For the authoritative catalog, lifecycle, governance, and publication rules,
see the
[`sima-neat/dataset` documentation](https://github.com/sima-neat/dataset#readme).
