# Local accuracy development

Accuracy measurement is a shared Model Registry capability. Model directories
declare datasets, profiles, preprocessing through `model.yaml`, executors, and
metrics in a correctness YAML file with `engine: model_registry`; they must not
carry a model-specific evaluation loop.

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
packages. `dataset-registry` must be installed in the same environment when the
dataset path is not passed explicitly.

Install the published `main` branch tool artifact when it is not already
present:

```bash
sima-cli neat install --stg dataset/dataset-registry@main:latest
```

## Compile and evaluate locally

Resolve the immutable inputs first:

```bash
source_json="$(python -m model_registry fetch models/resnet_50 --json)"
source_path="$(jq -r .path <<<"$source_json")"
dataset_json="$(python -m model_registry fetch-accuracy-dataset models/resnet_50 --json)"
dataset_path="$(jq -r .path <<<"$dataset_json")"
```

Accuracy dataset resolution is independent of calibration. BF16 evaluation
uses the correctness dataset even though BF16 compilation does not consume
representative calibration samples.

Compile the variant. Compilation consumes calibration samples declared by the
correctness YAML but does not contain accuracy logic:

```bash
python models/resnet_50/compile.py \
  --model "$source_path" \
  --dataset "$dataset_path" \
  --variant modalix_int8 \
  --output-dir /tmp/resnet50-int8
```

Run the regular deterministic 500-image profile against the saved `.sima`
graph:

```bash
python -m model_registry evaluate models/resnet_50 \
  --variant modalix_int8 \
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

- model, variant, source, candidate, and dataset hashes;
- evaluation profile and configuration status;
- evaluated and available sample counts;
- deterministic selection hash and sampling method;
- executor types, worker count, elapsed time, and throughput;
- reference and candidate top-k accuracy, agreement, and deltas.

A `sampled_accuracy_smoke` result is a regression signal, not definitive model
accuracy. Use the qualification profile on demand before release. A config with
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
numbers. A missing `accuracy.json` remains valid for variants without an
executable dataset-backed correctness specification.

The browser requests `GET /v1/runs/<compile-run-id>/accuracy/latest`. Vulcan
returns the latest schema-versioned accuracy summary, its test-result identity,
and a short-lived URL for the immutable raw report. During rollout, the browser
also accepts the same summary from the generic run test-results response.

A regular run over a deterministic subset is labeled provisional. A run over
the complete configured validation split is labeled qualification and is used
for final accuracy reporting. The status help control in the browser explains
the coverage represented by each label.

## Adding an executable correctness specification

Add a correctness case to `model.yaml`, then create its YAML with:

- `engine: model_registry` and `task: image_classification`;
- immutable dataset identity and archive loader/split definitions;
- regular and qualification profiles;
- ONNX Runtime reference and AFE `.sima` candidate executors;
- allowlisted metric definitions.

`model-registry validate` validates executable correctness YAML against
`schemas/accuracy.schema.json`. Preprocessing is read from `model.yaml` and is
restricted to allowlisted operations. Arbitrary Python entrypoints are not
executed from accuracy YAML.

Run the local contract checks before submitting changes:

```bash
PYTHONPATH="$PWD/tools" MODEL_REGISTRY_ROOT="$PWD" \
  python -m model_registry validate models/resnet_50
PYTHONPATH="$PWD/tools" MODEL_REGISTRY_ROOT="$PWD" \
  pytest -q tests/unit/test_accuracy.py tests/unit/test_cli.py
```
