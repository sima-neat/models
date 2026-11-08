# Accuracy and fidelity

`model.yaml` declares two kinds of quality evidence for a build profile's
whole model:

| Section | Question it answers | Measured by | Stage |
| --- | --- | --- | --- |
| `numerical_fidelity` | Does the compiled model produce the same numbers as the source model? | `scripts/compile.py` | compile |
| `accuracy` | How well does the compiled model do its task? | `scripts/accuracy.py` on a DevKit | accuracy |

Fidelity is a numerical diagnostic, not task accuracy. Neither is calibration:
calibration data feeds quantization and proves nothing about quality.

Complete [Datasets for local model development](datasets.md) first. Fidelity
and accuracy datasets resolve through Dataset Registry and need working
Artifactory access.

## Compile-time fidelity

`numerical_fidelity` asks the recipe to compare every build profile's compiled
model with the source model on a fixed sample of a registered dataset:

```yaml
numerical_fidelity:
  status: diagnostic
  dataset: {id: imagenet-2012, variant: cal, sha256: e6c18b28…, size_bytes: 112710756}
  selection: {method: seeded_random_without_replacement, seed: 42, offset: 0, samples: 50}
  outputs: [logits]
```

`dataset` pins a Dataset Registry archive, `outputs` names outputs of the model, and
`status` is `diagnostic`, `provisional`, or `ready`; see
[`numerical_fidelity`](metadata-schema.md#numerical_fidelity). The section is
a compile input: changing it recompiles the model's builds.

`scripts/run_model_compilation.sh` resolves the dataset with
`model-registry fetch-fidelity-dataset` and passes it to the recipe as
`--fidelity-dataset`. The recipe selects the declared samples, runs the
reference and the candidate, and writes `fidelity.json` in the profile's
output directory, beside the component directories.

The report follows
[`numerical-fidelity.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/numerical-fidelity.schema.json).
For each declared output it records cosine similarity, normalized RMSE, and
mean and maximum absolute error. Package assembly carries the report into the
profile's `default` package, and CI rejects a package that lacks it when
`numerical_fidelity` is declared, or carries one when it is not. CI publishes
the report on the profile's package run as a `numerical-fidelity` artifact, a
`correctness/numerical-fidelity` test result, and measurements of each metric.

## Accuracy on the DevKit

Use DevKit accuracy when a task metric needs the model's own decoding and
scoring, such as top-1 accuracy for a classifier or COCO mAP for a detector.
The model's `scripts/accuracy.py` runs every compiled MPK of a build profile
on a DevKit with PyNeat, on each dataset `accuracy` names:

```yaml
accuracy:
  datasets:
  - {id: imagenet-2012, variant: val, sha256: 018bf95e…, size_bytes: 6666407650}
```

Each `datasets` entry pins a Dataset Registry archive, which goes by its
source ID, here `imagenet-2012-val`. The metrics the report must hold come from
`datasets/<registry id>.yaml` for the model's `tasks`; see
[Required accuracy metrics](datasets.md#required-accuracy-metrics).
Without `build_profiles`, every build profile is measured; see
[`accuracy`](metadata-schema.md#accuracy). Record the source model's published
scores in the top-level [`reference_metrics`](metadata-schema.md#reference-metrics),
not in `accuracy`, so the model page shows them before the first DevKit run.

Optional `scripts/requirements-accuracy.txt` pins extra wheels. They install
into a separate folder beside PyNeat's environment, never into it.

### The accuracy script

`scripts/accuracy.py` receives:

- `--mpk <component>=<path>` once for each component, `model=<path>` for a
  single-component model;
- `--dataset <source id>=<directory>` for each dataset `accuracy` pins, the
  extracted archive; and
- `--output <directory>`.

It scores the whole model on each dataset and hands the scores to
`tools/model_report`, keyed by the dataset source IDs. `model_report` writes
`model-report.json` into the output directory:

```python
dataset_id, dataset_dir = args.dataset.split("=", 1)  # imagenet-2012-val, <directory>
...
report = model_report.from_accuracy({
    dataset_id: {
        "metrics": {"top1_accuracy": top1 / len(names), "top5_accuracy": top5 / len(names)},
        "split": VALIDATION_SPLIT,
        "evaluated_samples": len(names),
        "available_samples": len(available),
        "details": {"inference_seconds": round(time.monotonic() - started, 1)},
    },
})
report.save(args.output)
```

How the script evaluates each dataset is up to the model. Every score is
named as an accuracy metric in
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json) and
given in that metric's unit: a ratio from 0 to 1 for most metrics, a
percentage from 0 to 100 for `d1_all`, and pixels for `epe`. Each dataset's
result must hold every metric its requirements file lists for
the model's tasks. Other known accuracy metrics are allowed. Existing scripts also accept
`--limit <images>` to evaluate a subset for a quick local check; CI never
passes it.

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

1. resolves each declared dataset into `DATASET_CACHE` (default
   `/media/nvme/datasets/artifacts`), holding a lock so concurrent runs share
   one download;
2. lists each component's MPK with `model-registry model-component-mpks`;
3. extracts each dataset under `WORK_DIR` (default `/media/nvme/neat-tests/`)
   and removes that directory when it exits;
4. installs `requirements-accuracy.txt` beside PyNeat;
5. runs `accuracy.py` with `--dataset <source id>=<directory>` for each dataset
   and `PYTHONPATH` including `tools`, which writes
   `model-report.json`, and saves its output to `accuracy.log`; and
6. runs `model-registry check-report`, which validates the report against
   [`model-report.schema.json`](../schemas/model-report.schema.json), requires
   a result for exactly the declared datasets, each with its required metrics,
   and writes the board's installation to
   `environment.json`.

`REGISTRY_PYTHON` needs the full Model Registry tooling from this checkout
(`python -m pip install .`) and Dataset Registry, because the dataset
resolution and report checks are not in the author-only `model-registry`
package. It exits without running anything when the build profile is not
measured.

For a quicker check, resolve the dataset with Dataset Registry, extract it, and
run the script directly:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/resnet_50/scripts/accuracy.py \
  --mpk model=<path to MPK> --dataset imagenet-2012-val=<extracted dataset> \
  --output /tmp/resnet50-accuracy --limit 500
```

A model whose `accuracy.py` imports another file under `scripts/`, such as
ResNet-50's `calibration.py`, needs that file beside it.

### FP32 baseline

When no published score matches the package's setup, such as SuperGlue's 512
keypoints from a 640x480 image, the model page can show an FP32 baseline instead:
what the unquantized source model scores on the same inputs and protocol. A
standalone `scripts/fp32_baseline_accuracy.py` measures it on a host. It runs the
converted ONNX with ONNX Runtime and scores it with the model's own `accuracy.py`,
so only the model execution differs from the DevKit stage:

```bash
PYTHONPATH=tools python models/superglue_outdoor/scripts/fp32_baseline_accuracy.py \
  --onnx <converted model.onnx> --dataset yfcc100m-keypoints-val=<extracted dataset> \
  --output fp32-baseline.json
```

Copy its metrics into a `reference_metrics` entry with `origin: fp32_baseline`
([reference metrics](metadata-schema.md#reference-metrics)). No stage runs the
script and no fingerprint includes it, so adding or editing it reruns nothing.

### Report and publication

`model-report.json` follows
[`model-report.schema.json`](../schemas/model-report.schema.json): the stage,
the PyNeat version and build, and `datasets`, which gives for each dataset source
ID its evaluated split and sample counts, one number per metric, and free-form
`details`. The model, build profile, MPK
digests, and board are not report fields; publication records them on the run.

On the DevKit, `scripts/run_model_tests.sh` runs the model's
`tests/benchmark/test_accuracy.py` on the report, which compares each dataset's
metrics with the build profile's expected values for that dataset in
`tests/tests.yaml` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)). CI publishes the
report with `model-registry publish-report` as one `model_accuracy` registry
run under the build profile's `model_package` run. Its `run_fingerprint` is the
accuracy benchmark fingerprint, and the run carries one `compiled-model-<component>`
input per component, the `model-report` and log artifacts, an `accuracy/model`
test result whose summary is the whole report and whose status is the test
outcome (`passed`, `failed`, or `skipped`), and one measurement per dataset
and metric, with a `dataset` dimension. A
failed test still publishes the report, then fails the workflow run. See [CI
and registry publication](benchmarking.md#ci-and-registry-publication) for the
shared flow. The model browser shows each dataset's result in the accuracy
section: its first four metrics as tiles and the rest in a table, each with its
reference score when `reference_metrics` records one for the same dataset, and
the evaluated sample count. The catalog and the Compare page use the first
declared dataset.
