# Local accuracy development

Accuracy measurement is a shared Model Registry capability. Model directories
declare datasets, profiles, preprocessing, executors, and metrics in a
`validation` entry of `model.yaml` with `engine: model_registry`; they must not
carry a model-specific evaluation loop. The one exception is
[accuracy on the DevKit](#accuracy-on-the-devkit), for task metrics the shared
evaluator cannot compute.

## Environment

Run AFE-backed evaluation inside the Neat SDK after activating the resolved
Model Compiler environment. From a repository or worktree mounted at
`/workspace/<repo>`:

```bash
source /sdk-extensions/model-compiler/bin/activate
cd /workspace/<repo>
export PYTHONPATH="$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
python -m model_registry evaluate --help
```

Using `PYTHONPATH` runs the tooling from the current checkout without replacing
the Model Compiler environment's AFE, JAX, ONNX Runtime, NumPy, or Pillow
packages.

Complete [Datasets for local model development](datasets.md) before continuing.
That guide owns Dataset Registry installation, Artifactory authentication,
proof that the exact dataset variant resolves, and the process for introducing missing
data.

## Compile and evaluate locally

Resolve the immutable inputs first:

```bash
source_json="$(python -m model_registry fetch models/resnet_50 --component model --json)"
source_path="$(jq -r .path <<<"$source_json")"
dataset_json="$(python -m model_registry fetch-accuracy-dataset models/resnet_50 \
  --build-profile modalix_int8 --component model --json)"
dataset_path="$(jq -r .path <<<"$dataset_json")"
```

Accuracy dataset resolution is independent of calibration. The `validation`
entry in `model.yaml` scopes evaluation to named build profiles. ResNet-50
currently measures INT8 only, so BF16 resolution
returns `kind: none` and CI skips its accuracy step.

Compile the component/profile pair. Compilation consumes calibration samples declared by the
validation entry but does not contain accuracy logic:

```bash
python models/resnet_50/scripts/compile.py \
  --model "$source_path" \
  --dataset "$dataset_path" \
  --build-profile modalix_int8 \
  --component model \
  --output-dir /tmp/resnet50-int8
```

Run the regular deterministic 500-image profile against the saved `.sima`
graph:

```bash
python -m model_registry evaluate models/resnet_50 \
  --build-profile modalix_int8 \
  --component model \
  --reference-model "$source_path" \
  --candidate-model /tmp/resnet50-int8/resnet_50_modalix_int8.sima \
  --dataset "$dataset_path" \
  --profile regular \
  --workers 4 \
  --output /tmp/resnet50-int8/accuracy.json
```

Omit `--reference-model` or `--dataset` to let Model Registry and Dataset
Registry resolve the pinned artifacts. Use `--profile qualification` for the
full configured mini validation split. `--samples` and `--workers` are explicit
development/CI overrides; they do not alter the checked-in profile.

Four workers fit the local SDK container used during development. The Vulcan
`c8gd.4xlarge` workflow uses six workers. Increase workers only after checking
memory: every worker loads an independent ONNX Runtime session and AFE graph.

## Report interpretation

Every JSON report records:

- model, component, build profile, source, candidate, and dataset hashes;
- evaluation profile and configuration status;
- evaluated and available sample counts;
- deterministic selection hash and sampling method;
- executor types, worker count, elapsed time, and throughput;
- reference and candidate top-k accuracy, agreement, and deltas.

A `sampled_accuracy_smoke` result is a regression signal, not definitive model
accuracy. Use the qualification profile on demand before release. An entry with
`status: provisional` may be used for engineering feedback but must not be used
as release evidence.

The current engine evaluates the saved AFE `.sima` graph on the host. It does
not execute the compiled MPK on a DevKit; MPK/device correctness remains a
separate runtime test.

## Vulcan reporting

When CI publishes a compilation containing `accuracy.json`, Model Registry
publishes the report on the same Vulcan compile run as three forms of evidence:

- an immutable `accuracy-result` JSON artifact with dataset, profile, scope,
  sample-count, and candidate top-k metadata;
- an `accuracy/<report-scope>` test result containing the full report;
- normalized `model.accuracy.*` ratio measurements for candidate top-1/top-5,
  reference top-1/top-5, agreement, and deltas.

The model browser reads the test-result summary and always displays the
evaluated sample count and provisional or qualification status beside the accuracy
numbers. A missing `accuracy.json` remains valid for component/profile pairs without an
executable dataset-backed validation entry.

The browser requests `GET /v1/runs/<compile-run-id>/accuracy/latest`. Vulcan
returns the latest schema-versioned accuracy summary, its test-result identity,
and a short-lived URL for the immutable raw report. During rollout, the browser
also accepts the same summary from the generic run test-results response.

A regular run over a deterministic subset is labeled provisional. A run over
the complete configured validation split is labeled qualification and is used
for final accuracy reporting. The status help control in the browser explains
the coverage represented by each label.

## Adding an executable accuracy validation entry

Add a `validation` entry to `model.yaml` with:

- `engine: model_registry` and `task: image_classification`;
- an explicit `build_profiles` list and component `scope`;
- immutable dataset identity and archive loader/split definitions;
- regular and qualification profiles;
- ONNX Runtime reference and AFE `.sima` candidate executors;
- allowlisted metric definitions.

`model-registry validate` checks the entry against `schemas/model.schema.json`.
Preprocessing is read from the component in `model.yaml` and is restricted to
allowlisted operations. The entry cannot name a Python entrypoint to execute.

Run the local contract checks before submitting changes:

```bash
PYTHONPATH="$PWD/tools" MODEL_REGISTRY_ROOT="$PWD" \
  python -m model_registry validate models/resnet_50
PYTHONPATH="$PWD/tools" MODEL_REGISTRY_ROOT="$PWD" \
  pytest -q tests/unit/test_accuracy.py tests/unit/test_cli.py
```

## Accuracy on the DevKit

Use this when a model's task metric needs its own decoding and scoring, such as
COCO mAP for a detector whose pipeline spans several MPKs. The model's
`scripts/accuracy.py` runs the benchmarked MPKs on the DevKit with PyNeat right
after the benchmark, so the published accuracy and performance describe the
same archives.

Declare it as a `validation` entry with `engine: devkit_script`:

```yaml
validation:
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

A build profile has at most one such entry. Every component in its scope must
be benchmarked, and `result_component` names the component whose benchmark run
carries the result. Optional `scripts/requirements-accuracy.txt` pins extra
wheels; they install beside PyNeat's environment, not into it.

`scripts/accuracy.py` receives `--mpk <component>=<path>` for each scoped
component, `--dataset <extracted archive>`, and `--output <file>`. It writes:

```json
{
  "core_version": "3.0.0",
  "dataset": {"split": "val2017", "evaluated_samples": 5000, "available_samples": 5000},
  "metrics": {"coco_map": {"value": 0.469, "unit": "ratio", "direction": "higher"}},
  "details": {"inference_seconds": 187.7}
}
```

Metric names must be declared in the entry's `metrics`. CI wraps the output
with the model, MPK digests, dataset, and source revision as an
[`accuracy-result-v2`](https://github.com/sima-neat/models/blob/develop/schemas/accuracy-result-v2.schema.json)
document. A missing or mismatched result blocks the benchmark publication. The
result is published on the result component's benchmark run as an
`accuracy-result` artifact, an `accuracy/<entry-id>` test result, and one
`model.accuracy.<metric>` measurement per metric. The model browser shows one
tile per metric.

To run it on a board, benchmark each scoped component first so that
`<benchmark-root>/<component>/performance-test.json` exists, then run from the
repository root:

```bash
REGISTRY_PYTHON=<registry venv python> \
  scripts/run_model_accuracy.sh models/<model-id> <build-profile> <benchmark-root> <output-dir>
```

The script extracts the dataset under `/media/nvme/neat-tests/` and removes
that scratch directory when it exits; set `WORK_DIR` to use another location.

For a quicker check, run the script directly with `--limit <images>`. A
limited run evaluates a seeded sample and shows as provisional.
