# Define the model contract

A Model Registry contribution describes a product that developers can find,
build, test, and retrieve reproducibly. For a simple model, that product is one
ONNX input and one compiled MPK. More involved products can contain several
ONNX components, runtime resources, an explicit composition, and a combined
download package. `model.yaml` is the source-controlled contract joining those
pieces together.

This guide ends when the contract validates. You must then follow [Qualify
locally](local-development.md) and compile and test every supported build
before opening a pull request.

## 1. Decide what the product contains

Before creating files, identify:

- the logical product and its responsible owner;
- each independently compiled ONNX component;
- whether a component source is one file or an ONNX file set;
- the public entrypoints, tensors, shapes, state, and preprocessing behavior;
- runtime files such as tokenizers, labels, or configuration;
- the target-wide build profiles and component-specific compiler parameters;
- how the components compose and what users should receive in each package;
- the calibration and correctness data required by each build;
- the component-level and end-to-end tests needed for qualification.

Do not add a product whose source, ownership, or license remains ambiguous.
`experimental` maturity may describe unfinished qualification; it does not
permit mutable provenance.

## 2. Create the directory

The directory name and `model.yaml` `id` must be the same lowercase snake-case
identifier.

```text
models/<model-id>/
  README.md
  model.yaml                  # the only configuration file: every stage reads its section
  scripts/
    compile.py                # one recipe, optionally shared by a compile group
    benchmark.py              # Core benchmark for a compiled MPK on a DevKit
    export.py                 # only for an upstream checkpoint source
    surgery.py                # only for model-specific preparation
```

Model-specific settings, such as validation thresholds or a multi-component
composition, go into `model.yaml`, not separate YAML files.

The README records upstream context, unusual compiler decisions, local
qualification notes, and known limitations. Requirements that tooling needs to
enforce must also appear in `model.yaml`.

## 3. Pin every input

Each compiled component owns its source. Use `type: file` for a standalone
artifact. Use `type: file_set` when the recipe needs several related files in a
particular relative layout. Every file gets its own immutable
URI, SHA-256, byte count, and media type.

Hugging Face sources use a full 40-character repository commit. Approved HTTPS
sources include pinned Hugging Face resources and GitHub release assets.
Mutable branches, credentials, query parameters, fragments, and unapproved
hosts are rejected.

Record hashes from the exact bytes the registry will fetch:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

Do not commit downloaded inputs. Once they are declared, verify them by
component:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id> --component <component-id>
```

If the model has only one component, `--component` may be omitted.

When upstream publishes a checkpoint instead of ONNX, declare the checkpoint
and its pinned exporter package as one file set. CI fetches and verifies both.
The model's `scripts/export.py` converts the local checkpoint to ONNX; it must not
download an undeclared model. The compiler runner must already provide the
exporter's dependencies. CI installs the verified exporter package without
resolving dependencies from the network. Generated ONNX files remain temporary.

## 4. Describe components and interfaces

Declare one `components` entry for each independently compiled unit. A
component owns:

- its role and immutable source;
- one or more named ONNX entrypoints;
- tensor, state, streaming, and concrete shape-profile contracts;
- preprocessing and postprocessing pipelines;
- optional model-specific surgery;
- the recipe that produces its MPK;
- optional dependencies on other components.

Names and tensor details must match the pinned inputs. Inspect the ONNX files;
do not infer them from a mutable model card. Use separate components when files
must compile independently, as in an encoder/decoder system. Use one file-set
source when several files jointly form one compiler input.

## 5. Add resources and calibration

Declare non-compiled files under `resources` and give each its package
`install_path`. Examples include tokenizers, vocabularies, speaker embeddings,
labels, and runtime configuration. Declare a separate license when a resource
has different distribution terms.

Declare reusable calibration inputs under `calibrations`. A ready calibration
references either an immutable source or a registered dataset. A pending entry
states what is missing. Build-profile component bindings select calibration by
ID only when required.

Calibration is compiler input for quantization; correctness or accuracy tests
measure behavior. Do not treat calibration data as accuracy evidence. Any
calibration or correctness dataset must resolve through Dataset Registry and
requires working Artifactory access. See [Datasets for local model
development](datasets.md).

## 6. Define build profiles

A build profile represents one compatible product-wide target configuration,
such as `modalix_int8`. Within it, bind every participating component and put
component-specific compiler parameters on that binding:

```yaml
build_profiles:
  - id: modalix_int8
    target: modalix
    components:
      - id: encoder
        calibration: speech_calibration
        parameters: {precision: int8}
      - id: decoder
        parameters: {precision: int8}
```

By default, the matrix creates one compile job per supported component binding.
Components that must share preparation or intermediate calibration data can be
placed in one compile invocation:

```yaml
compile_invocations:
  - id: detector
    components: [backbone, transformer]
    python_packages:
      - path: exporter-1.2.3-py3-none-any.whl
        imports: [exporter]
```

Every component in a compile invocation must share one source, recipe, and
runner class. The recipe runs once and writes one MPK per component plus
`compile-outputs.json`. Declare any verified source-set Python package and the
imports CI must check under `python_packages`; CI never discovers dependencies
from filenames. The manifest maps each component to its MPK and records derived
graph hashes and the one declared validation result.
CI accepts the bundle only when every declared output is present and its hash
matches. Existing models do not need this field. When the group needs
calibration data, put its single calibration reference on the first listed
component; CI resolves that verified input once for the shared recipe.

For model-owned numerical comparison, declare a package-scoped `validation`
entry with `engine: model_recipe`, write `fidelity.json` next to its owning MPK,
name that component with `result_component`, and reference the report from the
grouped output manifest. Numerical fidelity is diagnostic, not task accuracy;
CI checks report integrity and dataset identity, not similarity thresholds.

Each model build is independent. A failed compile invocation must not prevent
other invocations from publishing or benchmarking. Within a grouped invocation,
every declared MPK must be present and verified before any component publishes.

A component binding proven incompatible may be retained as
`status: unsupported` with a reason. Do not use unsupported for untested work.

Each component recipe has this interface:

```text
scripts/compile.py --model <verified-entrypoint> --output-dir <new-directory> \
  --build-profile <profile-id> --component <component-id>
```

The recipe reads only its selected profile binding, consumes already verified
inputs, uses the repository-pinned compiler, creates exactly one MPK, and never
downloads undeclared data. Shared download, checksum, MPK validation, and
publication behavior belongs in `tools/`, not model code.

A grouped recipe uses this interface instead:

```text
scripts/compile.py --model <verified-entrypoint> --dataset <verified-dataset> \
  --correctness-dataset <verified-correctness-dataset> \
  --output-dir <new-directory> --build-profile <profile-id> \
  --compile-invocation <invocation-id>
```

Elevated `runner_class` requests require Vulcan approval. See [Compile runner
class requests and approvals](runner-class-requests.md).

## 7. Define composition and retrieval packages

A one-component product uses a `single` composition naming its component and
entrypoint. A product with several components uses a separately versioned
declarative composition file to describe connections and product-level
behavior.

Then declare one or more packages. A package selects:

- the components whose MPKs it contains;
- the runtime resources to install;
- the compatible build profiles;
- the composition that explains how to use them.

Exactly one package is `default: true`. Trusted assembly places the intact MPKs
and resources into one deterministic archive, so a user can retrieve and
extract the complete product without downloading each artifact separately.

## 8. Scope validation and benchmarks

Every `validation` and `benchmark` entry in `model.yaml` states which build
profiles it applies to and whether it exercises a component or a package. Use
component scope for one compiled unit and package scope for an end-to-end
composed product. Keep the settings in the entry itself and make missing
qualification work visible.

For executable accuracy validation, follow [Local accuracy
development](local-accuracy-development.md). For DevKit benchmarks, follow
[Synthetic model benchmarking](benchmarking.md).

## 9. Validate the contract

```bash
model-registry validate models/<model-id>
model-registry validate-catalog --repository .
```

These checks establish contract coherence, not successful compilation or
execution. The local qualification guide owns matrix inspection and build
selection.

Continue with [Qualify locally](local-development.md) and complete the full
loop for every matrix entry before submitting the contribution.
