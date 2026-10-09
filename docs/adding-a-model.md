# Define the model contract

A Model Registry contribution describes a product that developers can find,
build, test, and download reproducibly. For a simple model, that product is one
ONNX input and one compiled MPK. A larger product can contain several compiled
components, runtime resources, a composition, and a download package.
`model.yaml` joins those pieces, and the model's `scripts/` directory holds the
code that compiles and measures them.

This guide ends when the contract validates. Then follow [Qualify
locally](local-development.md) to compile and measure every build before you
open a pull request.

## 1. Decide what the product contains

Before creating files, identify:

- the product and its responsible owner;
- each independently compiled ONNX component;
- whether a component source is one file or a set of related files;
- the public entrypoints, tensors, shapes, state, and preprocessing;
- runtime files such as tokenizers, labels, or configuration;
- the target-wide build profiles and component-specific compiler parameters;
- how the components compose and what users download;
- the calibration, fidelity, and accuracy data each build needs.

Do not add a product whose source, ownership, or license is ambiguous.
`experimental` maturity may describe unfinished qualification; it does not
permit mutable provenance.

## 2. Create the directory

The directory name and the `id` in `model.yaml` are the same lowercase
snake-case identifier. A model directory may contain only these entries:

```text
models/<model-id>/
  README.md
  hero.png                      # optional
  model.yaml                    # the model's only configuration file
  scripts/
    compile.py                  # compile stage: writes one MPK per component
    performance.py              # performance stage: Core benchmark of one MPK on a DevKit
    accuracy.py                 # accuracy stage, when model.yaml declares DevKit accuracy
    requirements-accuracy.txt   # optional wheels for accuracy.py
    convert.py                  # only when model.yaml declares a conversion
    requirements-convert.txt    # required with convert.py
    <helper>.py                 # model-specific surgery, calibration, or shared code
```

Put every model-specific setting in `model.yaml`, not in separate
configuration files. The README follows the structure in
[`AGENTS.md`](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).

Keep board-only code inside `performance.py` or `accuracy.py`. Any other
`.py` or `.sh` file under `scripts/` is a compile input, so a helper that
`accuracy.py` imports recompiles the model when it changes. See [step
9](#9-know-what-a-change-reruns).

## 3. Pin every input

Each compiled component owns its source. Use `type: file` for one artifact and
`type: file_set` when the recipe needs several related files in a fixed
relative layout. Every file gets its own immutable URI, SHA-256, byte count,
and media type.

Hugging Face sources use a full 40-character repository commit. Approved HTTPS
sources include pinned Hugging Face resources and GitHub release assets.
Mutable branches, credentials, query parameters, fragments, and unapproved
hosts are rejected.

Record hashes from the exact bytes the registry will fetch:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

Do not commit downloaded inputs. Once they are declared, check them:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id> --component <component-id>
```

`--component` may be omitted when the model has one component.

When upstream publishes a checkpoint instead of ONNX, declare it under
`conversion` and provide `scripts/convert.py` with
`scripts/requirements-convert.txt`. `convert.py` receives `--conversion
<entry>` and `--output-dir <directory>` and writes `model.onnx` there.
`scripts/run_model_conversion.sh` runs it in a fresh Python 3.12 environment:

```bash
scripts/run_model_conversion.sh models/<model-id> default /tmp/<model-id>-export
```

The **Vulcan Model Convert** workflow runs the same script, publishes
`model.onnx`, and prints the `source` block to pin in `model.yaml`. Only that
pin reaches the compile stage; editing `convert.py` recompiles nothing.

## 4. Describe components and interfaces

Declare one `components` entry for each independently compiled unit. A
component owns:

- its role and immutable source;
- one or more named ONNX entrypoints;
- tensor, state, streaming, and concrete shape-profile contracts;
- preprocessing and postprocessing pipelines;
- optional model-specific surgery;
- its `recipe`, `scripts/compile.py`;
- optional dependencies on other components.

Names and tensor details must match the pinned inputs. Inspect the ONNX files;
do not copy them from a model card.

## 5. Add resources and calibration

Declare non-compiled files under `resources` and give each its package
`install_path`. Examples include tokenizers, vocabularies, labels, and runtime
configuration. Declare a separate license when a resource has different
distribution terms.

Declare calibration inputs under `calibrations`. A ready calibration references
either an immutable `source` or a registered dataset. A pending entry states
what is missing. A build-profile component binding selects a calibration by ID.

Calibration is compiler input for quantization. It is not accuracy evidence.
Calibration, fidelity, and accuracy datasets resolve through Dataset Registry
and need working Artifactory access. See [Datasets for local model
development](datasets.md).

## 6. Define build profiles and compile invocations

A build profile is one product-wide target configuration, such as
`modalix_int8`. Bind every participating component and put its compiler
parameters on the binding:

```yaml
build_profiles:
  - id: modalix_int8
    target: modalix
    components:
      - id: model
        calibration: default
        parameters: {precision: int8}
```

By default each supported binding is its own compile invocation, named after
its component. Components that share preparation or intermediate data compile
together in one declared invocation:

```yaml
    compile_invocations:
      - id: detector
        components: [backbone, transformer]
```

Grouped components must share one source, one effective runner class, and at
most one calibration, set on the first listed component. A package and a DevKit
accuracy entry must each compile all their components in one invocation.

A binding proven incompatible may stay as `status: unsupported` with a reason.
Do not use it for untested work.

Every recipe has one interface, for single and grouped invocations alike:

```text
scripts/compile.py --model <verified-source> --output-dir <new-directory> \
  --build-profile <profile-id> --compile-invocation <invocation-id> \
  [--dataset <verified-calibration>] [--fidelity-dataset <verified-fidelity-dataset>]
```

`--model` is the verified source of the invocation's first component. The
runner passes `--dataset` when the binding selects a ready calibration and
`--fidelity-dataset` when a `fidelity` entry applies. The recipe:

- reads its build profile from `model.yaml`;
- writes exactly one `*_mpk.tar.gz` under `<output-dir>/<component-id>/` for
  each component of the invocation;
- writes `fidelity.json` beside the MPK of the fidelity entry's
  `result_component` when a `fidelity` entry applies;
- for a grouped invocation, also writes `<output-dir>/compile-outputs.json`
  with `outputs` (each component's MPK path and SHA-256), `derived_graphs`
  (each component's derived graph SHA-256), and `correctness` (the result
  component's `fidelity.json` and its SHA-256); and
- never downloads undeclared data.

For a single-component invocation, `scripts/run_model_compilation.sh` writes
`compile-outputs.json` when the recipe does not. Shared download, checksum, MPK
validation, and publication code belongs in `tools/`, not in model code.

Elevated `runner_class` requests need approval. See [Compile runner class
requests and approvals](runner-class-requests.md).

## 7. Define composition and packages

A one-component product uses a `single` composition naming its component and
entrypoint. A multi-component product uses a `declarative` composition whose
`components`, `connections`, `inputs`, and `outputs` are written inline in
`model.yaml`; see [Compositions and
packages](metadata-schema.md#compositions-and-packages).

Then declare one or more packages. A package selects:

- the components whose MPKs it contains;
- the runtime resources to install;
- the compatible build profiles;
- the composition that explains how to use them.

Exactly one package is `default: true`. CI assembles each package from the
build that compiles its components, so a user downloads the complete product
as one archive.

## 8. Declare fidelity, accuracy, and benchmark entries

Each entry names its `scope` (a component or a package) and the
`build_profiles` it applies to. IDs are unique within each section.

| Section | Measured by | Runs in |
| --- | --- | --- |
| `fidelity` | `compile.py`, comparing the compiled graph with the source model | compile stage |
| `accuracy` with `engine: devkit_script` | `accuracy.py`, running the compiled MPKs | accuracy stage |
| `accuracy` without `engine` | nothing; the entry records results measured elsewhere | no stage |
| `benchmark` | `performance.py`, once per component | performance stage |

Fidelity is diagnostic numerical agreement, not task accuracy. A build profile
has at most one `devkit_script` accuracy entry. Every component with a
`benchmark` entry in a build profile gets its own performance result.

Follow [Accuracy and fidelity](local-accuracy-development.md) for the fidelity
and accuracy contracts and [DevKit performance](benchmarking.md) for the
performance script.

## 9. Know what a change reruns

Each stage has a stage key, a SHA-256 of everything its result depends on. CI
compiles a build only when the branch catalog, or the base branch's catalog for
a new branch, records no build with its current compile key. It runs a DevKit
stage only when no completed registry run carries that stage's current key.

| Stage | Key inputs |
| --- | --- |
| compile | the invocation's components and bindings in `model.yaml` (source, interface, surgery, recipe, dependencies, parameters, shape profile, calibration), the build profile's target and compatibility, the `fidelity` entries for the profile, every `.py` and `.sh` file in `scripts/` except `performance.py`, `accuracy.py`, `requirements-accuracy.txt`, `convert.py`, and `requirements-convert.txt`, declared training code, and the Model Compiler version |
| performance | the compile run and MPK digest, the component's `benchmark` entries, `performance.py`, the shared performance tooling, and the DevKit runtime pin |
| accuracy | the compile runs and MPK digests of its components, the `accuracy` entry, `accuracy.py` and `requirements-accuracy.txt`, the shared accuracy tooling, and the DevKit runtime pin |

A new compile run changes the performance and accuracy keys too, so a
recompiled build is always remeasured.

| Change | Reruns |
| --- | --- |
| README, display metadata, `convert.py`, registry tooling outside a stage's tooling | nothing |
| `performance.py` or a `benchmark` entry | that model's performance |
| `accuracy.py`, its requirements, or the `accuracy` entry | that model's accuracy |
| the DevKit runtime pin | every performance and accuracy stage |
| a compile input of a model | that model's affected builds, then their DevKit stages |
| the Model Compiler pin | every build, then every DevKit stage |

The keys are computed by `compile_fingerprint` in
`tools/model_registry/catalog.py` and by `performance_key` and `accuracy_key`
in `tools/model_registry/stages.py`.

## 10. Validate the contract

```bash
model-registry validate models/<model-id>
model-registry validate-catalog --repository .
```

These checks establish that the contract is coherent, not that it compiles or
runs. Continue with [Qualify locally](local-development.md) and complete the
full loop for every build before submitting.
