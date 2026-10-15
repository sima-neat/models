# Accuracy and fidelity

`model.yaml` declares two kinds of quality evidence for a build:

| Section | Question it answers | Measured by | Stage |
| --- | --- | --- | --- |
| `fidelity` | Does the compiled graph produce the same numbers as the source model? | `scripts/compile.py` | compile |
| `accuracy` | How well does the compiled model do its task? | `scripts/accuracy.py` on a DevKit, or recorded results | accuracy |

Fidelity is a numerical diagnostic, not task accuracy. Neither is calibration:
calibration data feeds quantization and proves nothing about quality.

Complete [Datasets for local model development](datasets.md) first. Fidelity
and accuracy datasets resolve through Dataset Registry and need working
Artifactory access.

## Compile-time fidelity

A `fidelity` entry asks the recipe to compare the compiled candidate with the
source model on a fixed sample of a registered dataset:

```yaml
fidelity:
  - id: logits_fidelity
    scope: {type: package, package: default}
    build_profiles: [modalix_bf16, modalix_int8]
    status: diagnostic
    result_component: model
    dataset: {id: imagenet-2012, variant: mini, sha256: <archive sha256>, size_bytes: <bytes>}
    selection:
      method: seeded_random_without_replacement
      seed: 42
      offset: 0
      samples: 50
    outputs: [logits]
```

The rules:

- `scope` must be a package, and `result_component` must be one of its
  components.
- `outputs` names outputs of the result component's entrypoints.
- `status` is `diagnostic`, `provisional`, or `ready`.
- At most one fidelity entry covers a component in a build profile.

The fidelity entries of a build profile are compile inputs: changing one
recompiles the affected builds.

`scripts/run_model_compilation.sh` resolves the dataset with
`model-registry fetch-fidelity-dataset` and passes it to the recipe as
`--fidelity-dataset`. The recipe selects the declared samples, runs the
reference and the candidate, and writes `fidelity.json` beside the result
component's MPK. A grouped invocation also lists that file as the
`correctness` entry of its `compile-outputs.json`.

The report follows
[`numerical-fidelity.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/numerical-fidelity.schema.json).
For each declared output it records cosine similarity, normalized RMSE, and
mean and maximum absolute error. CI rejects a compilation whose result
component lacks the report, or whose other components carry one. It publishes
the report on the compile run as a `numerical-fidelity` artifact, a
`correctness/numerical-fidelity` test result, and measurements of each metric.

## Accuracy on the DevKit

Use DevKit accuracy when a task metric needs the model's own decoding and
scoring, such as top-1 accuracy for a classifier or COCO mAP for a detector.
The model's `scripts/accuracy.py` runs the compiled MPKs on a DevKit with
PyNeat. Declare it as an `accuracy` entry with `engine: devkit_script`:

```yaml
accuracy:
  - id: coco_accuracy
    scope: {type: package, package: default}
    build_profiles: [modalix_int8_bf16]
    engine: devkit_script
    status: ready
    result_component: transformer
    dataset: {id: coco-2017, variant: mini, sha256: <archive sha256>, size_bytes: <bytes>}
    metrics: [coco_map, coco_ap50, coco_ap75]
    required_metrics: [coco_map]
```

Record the source model's published scores in the top-level
[`published_metrics`](metadata-schema.md#published-metrics), not in the
accuracy entry, so the model page shows them before the first DevKit run.

The rules:

- A build profile has at most one `devkit_script` entry.
- `result_component` must be in the scope.
- Every scoped component must compile in one compile invocation of each listed
  build profile, so the accuracy stage runs on the MPKs of one build.
- `status` is `ready` or `provisional`.
- `required_metrics` must be listed in `metrics`.

Optional `scripts/requirements-accuracy.txt` pins extra wheels. They install
into a separate folder beside PyNeat's environment, never into it.

### The accuracy script

`scripts/accuracy.py` receives:

- `--mpk <component>=<path>` once for each scoped component;
- `--dataset <directory>`, the extracted dataset archive; and
- `--output <file>`.

It writes:

```json
{
  "core_version": "3.0.0",
  "dataset": {"split": "val2017", "evaluated_samples": 5000, "available_samples": 5000},
  "metrics": {"coco_map": {"value": 0.469, "unit": "ratio", "direction": "higher"}},
  "details": {"inference_seconds": 187.7}
}
```

Every metric must be declared in the entry, and every required metric must be
present. Existing scripts also accept `--limit <images>` to evaluate a subset
for a quick local check; CI never passes it.

### Run it on a DevKit

`scripts/run_model_accuracy.sh` runs the stage the way CI does, against the
output of `scripts/run_model_compilation.sh`:

```bash
REGISTRY_PYTHON=<python with model-registry and dataset-registry> \
PYNEAT_PYTHON=~/pyneat/bin/python \
  scripts/run_model_accuracy.sh models/<model-id> <build-profile> <compile-output-root> <result-directory>
```

`<compile-output-root>` holds `<model-id>/<build-profile>/compile-outputs.json`
and the MPKs it lists. The script:

1. resolves the declared dataset into `DATASET_CACHE` (default
   `/media/nvme/datasets/artifacts`), holding a lock so concurrent runs share
   one download;
2. extracts it under `WORK_DIR` (default `/media/nvme/neat-tests/`) and
   removes that directory when it exits;
3. installs `requirements-accuracy.txt` beside PyNeat;
4. runs `accuracy.py` and writes `script-output.json` and `accuracy.log`; and
5. checks the output against the entry and writes `accuracy-result.json`.

`REGISTRY_PYTHON` needs the full Model Registry tooling from this checkout
(`python -m pip install .`) and Dataset Registry, because the dataset
resolution and result checks are not in the author-only `model-registry`
package. It exits without running anything when the build profile declares no
DevKit accuracy.

For a quicker check, resolve the dataset with Dataset Registry, extract it, and
run the script directly:

```bash
~/pyneat/bin/python models/resnet_50/scripts/accuracy.py \
  --mpk model=<path to MPK> --dataset <extracted dataset> \
  --output /tmp/resnet50-accuracy.json --limit 500
```

A model whose `accuracy.py` imports another file under `scripts/`, such as
ResNet-50's `calibration.py`, needs that file beside it.

### Result and publication

`accuracy-result.json` follows
[`accuracy-result.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/accuracy-result.schema.json).
It wraps the script output with the model, build profile, entry, the SHA-256 of
each MPK, the dataset identity, the SHA-256 of `accuracy.py`, and the source
revision.

On the DevKit, `scripts/run_model_tests.sh` runs the model's
`tests/measure/test_accuracy.py` on the result, which compares each metric
with the build's expected value in `tests/tests.yaml` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)). CI publishes the result as
a `model_accuracy` registry run under the result component's compile run. The
run records the accuracy stage key and carries an `accuracy-result` artifact,
an `accuracy/<entry-id>` test result whose status is the test outcome
(`passed`, `failed`, or `skipped`), and one `model.accuracy.<metric>`
measurement per metric. A failed test still publishes the result, then fails
the workflow run. The model browser shows the
first four metrics as tiles and lists the rest in a table, each with its
published score when the entry records one, and the evaluated sample count.

## Recorded accuracy

An `accuracy` entry without `engine` records results measured outside the
registry stages. It needs only `id`, `scope`, and `build_profiles`; other
fields are free-form. No stage runs it and it has no stage key. SSD-MobileNet
uses such an entry to record its quantization evaluation.

A recipe that reads settings from a recorded entry, as the SSD-MobileNet
recipes read `compiler_raw_head_thresholds`, does not recompile when those
settings change, because `accuracy` entries are not compile inputs.
