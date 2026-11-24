# Accuracy and fidelity

`model.yaml` declares two kinds of quality evidence for a build profile's
whole model:

| Section | Question it answers | Measured by | Stage |
| --- | --- | --- | --- |
| `numerical_fidelity` | Does the quantized graph produce the same numbers as the source model? | `scripts/compile.py` | compile |
| `accuracy` | How well does the compiled model do its task? | `scripts/accuracy.py` on a DevKit | accuracy |

Fidelity is a numerical diagnostic, not task accuracy. Neither is calibration:
calibration data feeds quantization and proves nothing about quality.

Complete [Datasets for local model development](datasets.md) first. Fidelity
and accuracy datasets resolve through Dataset Registry and need working
Artifactory access.

## Compile-time fidelity

`numerical_fidelity` asks the recipe to compare every build profile's quantized
graph, run in the Model Compiler, with the ONNX source. The MPK does not run in
this measurement. The recipe compares on a fixed sample of a registered dataset:

```yaml
numerical_fidelity:
  status: diagnostic
  dataset: {id: imagenet-2012, variant: cal, sha256: e6c18b28…, size_bytes: 112710756}
  selection: {method: seeded_random_without_replacement, seed: 42, offset: 0, samples: 5}
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
mean and maximum absolute error. The profile's `default` package carries the
report, and CI publishes it with the build. A build must write the report when
`model.yaml` declares `numerical_fidelity`, and must not write one when it
does not. CI rejects the package otherwise.

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
  extracted archive;
- `--resources <directory>` when the compile declares supporting `runtime_files`,
  the verified profile root with the same relative asset paths as the package; and
- `--output <directory>`.

It scores the whole model on each dataset and hands the scores to
`tools/model_report`, keyed by the dataset source IDs. `model_report` writes
`model-report.json` into the output directory:

```python
dataset_id, dataset_dir = args.dataset.split("=", 1)  # imagenet-2012-val, <directory>
...
model = pyneat.Model(str(parse_mpks(args.mpk)))
runner = None
...
for count, (name, image) in enumerate(zip(names, images), start=1):
    tensor = pyneat.Tensor.from_numpy(image, copy=False)
    if runner is None:
        runner = model.build([tensor])
    logits = runner.run([tensor], timeout_ms=120000)[0].to_numpy(copy=False).reshape(-1)
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

The example is ResNet-50's script. It builds the runner once, from the first tensor. Each
`image` is one preprocessed float32 array in height, width, channel order, without a
batch axis.

How the script evaluates each dataset is up to the model. Every score is
named as an accuracy metric in
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json) and
given in that metric's unit: a ratio from 0 to 1 for most metrics, a
nonnegative ratio without an upper bound for `abs_rel`, `wer`, `wer_as_written` and `cer`, a percentage from 0 to 100
for `d1_all`, and pixels for `epe`. Each dataset's
result must hold every metric its requirements file lists for
the model's tasks. Other known accuracy metrics are allowed. Existing scripts also accept
`--limit <images>` to evaluate a subset for a quick local check.

The script prints progress with `log()` from `model_report`, which writes to standard
error. It prints one JSON line with the scores as the last line of standard output.

CI passes `--limit` only when `model.yaml` sets `accuracy.samples`
([accuracy](metadata-schema.md#accuracy)). The report then records how many
samples the script evaluated. The model's reference score is still the one for
the whole split, and the model tests compare the capped score with it.

A speech recognition model scores its transcripts with
[`tools/model_report/speech.py`](../tools/model_report/speech.py), so every
model is measured the same way:

```python
from model_report.speech import char_error_rate, word_error_rate, word_error_rate_as_written

metrics = {  # FLEURS raw transcriptions, in the transcripts' order
    "wer": word_error_rate(references, transcripts, "en"),
    "wer_as_written": word_error_rate_as_written(references, transcripts),
    "cer": char_error_rate(references, transcripts, "en"),
}
```

`wer` and `cer` normalize both texts for the spoken language first. English
uses Whisper's normalizer, the one published English error rates are commonly
scored with: "15" matches "fifteen", "it's" matches "it is", and "travellers"
matches "travelers". Unlike Whisper's, it keeps the words inside brackets, which
FLEURS speakers read aloud. Any other language uses a plain rule: Unicode NFKC,
lowercase, punctuation and symbols replaced by spaces, and whitespace collapsed.
`wer_as_written` applies the plain rule in every language, so "15" and "fifteen"
count as an error there. A published score is comparable only when its source
used the same normalizer.

A rate is a corpus total, the edits of every recording over the reference words
of every recording, and can exceed 1 when a model inserts many words.
`char_error_rate` scores characters with whitespace removed, for languages
written without spaces. Strip the model's own special tokens, such as a language
tag, before scoring. A model whose accuracy datasets require `wer`,
`wer_as_written` or `cer` hashes the scorer (`SPEECH_ACCURACY_TOOLING`) into its
accuracy fingerprint, so a scoring change reruns only those models.

### Run it on a DevKit

`scripts/run_model_accuracy.sh` runs the stage the way CI does, against the
output of `scripts/run_model_compilation.sh`:

```bash
REGISTRY_PYTHON=<python with model-registry and dataset-registry> \
PYNEAT_PYTHON=~/pyneat/bin/python \
  scripts/run_model_accuracy.sh models/<model-id> <build-profile> <compile-output-root> <result-directory>
```

`<compile-output-root>` holds `<model-id>/<build-profile>/compile-outputs.json`
and the MPKs it lists. The script does this:

1. It unpacks each declared dataset into `DATASET_DIR`, by default
   `WORK_DIR/datasets`, with `scripts/unpack_benchmark_datasets.sh`
   ([benchmark input](benchmarking.md#benchmark-input)).
2. It lists each component's MPK with `model-registry model-component-mpks`.
3. It finds the build's supporting files with `model-registry
   model-runtime-directory`. When the compile declared `runtime_files`, it
   passes the verified profile root to `accuracy.py` as `--resources`.
4. It installs `requirements-accuracy.txt` beside PyNeat, in `WORK_DIR`, which
   it removes when it exits with the datasets unpacked there. The default
   `WORK_DIR` is a folder for this run under `/media/nvme/neat-tests/`, which a
   local run names `model-accuracy-local-1`. Set `WORK_DIR` before the command
   to use another folder.
5. It runs `accuracy.py` with `--dataset <source id>=<directory>` for each
   dataset, `--limit` when `model.yaml` sets `accuracy.samples`, and
   `PYTHONPATH` including `tools`. The script writes `model-report.json`, and
   the runner saves its output to `accuracy.log`.
6. It runs `model-registry check-report`, which validates the report against
   [`model-report.schema.json`](../schemas/model-report.schema.json), requires
   a result for exactly the declared datasets, each with its required metrics,
   and writes the board's installation to `environment.json`.

`REGISTRY_PYTHON` needs the full `model-registry` tooling from this checkout
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

Copy the whole model folder, `tools/model_report`, and `tools/model_inputs` to the
DevKit, because `accuracy.py` imports `calibration.py`, which imports
`model_inputs.images`. Install `requirements-accuracy.txt` first, as
[step 8 of Getting started](getting-started.md#8-measure-accuracy) does.

### FP32 baseline

When no published score matches the package's setup, such as SuperGlue's 512
keypoints from a 640x480 image, the model page can show an FP32 baseline instead:
what the unquantized source model scores on the same inputs and protocol. A
standalone `scripts/fp32_baseline_accuracy.py` measures it by hand on a host. It runs an
FP32 ONNX with ONNX Runtime and scores it with the model's own `accuracy.py`,
so only the model execution differs from the DevKit stage. Its docstring names the ONNX
it reads: the output of the model's own `convert.py`, such as DETR's, or that output
after `surgery.py`, such as SuperGlue's:

```bash
PYTHONPATH=tools python models/superglue_outdoor/scripts/fp32_baseline_accuracy.py \
  --onnx <model.onnx from surgery.py> --dataset yfcc100m-keypoints-val=<extracted dataset> \
  --output fp32-baseline.json
```

Copy its metrics into a `reference_metrics` entry with `origin: fp32_baseline`
([reference metrics](metadata-schema.md#reference-metrics)), and give the command and
its output in the pull request. No stage runs the script and no fingerprint includes it,
so adding or editing it reruns nothing, and the ONNX you convert locally need not match
the sha256 of the file Vulcan Model Convert published.

### Report and publication

`model-report.json` follows
[`model-report.schema.json`](../schemas/model-report.schema.json): the stage,
the PyNeat version and build, and `datasets`, which gives for each dataset source
ID its evaluated split and sample counts, one number per metric, and free-form
`details`. The model, build profile, MPK digests, and board are not report
fields. CI records them when it publishes the report.

On the DevKit, `scripts/run_model_tests.sh` runs the model's
`tests/benchmark/test_accuracy.py` on the report, which compares each dataset's
metrics with the build profile's expected values for that dataset in
`tests/tests.yaml` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)). It also compares
them with the model's reference score for that dataset ([comment out a variant
far below its reference score](adding-a-model.md#comment-out-a-variant-far-below-its-reference-score)). CI then
publishes the report with the test outcome: `passed`, `failed`, or `skipped`.
A failed test still publishes the report, then fails the workflow run. See [CI
and registry publication](benchmarking.md#ci-and-registry-publication) for the
shared flow.

The model browser shows each dataset's result in the accuracy section. The
first four metrics appear as tiles and the rest in a table. Each metric shows
its reference score when `reference_metrics` records one for the same dataset.
The section also shows the evaluated sample count. The catalog and the Compare page use the first
declared dataset.
