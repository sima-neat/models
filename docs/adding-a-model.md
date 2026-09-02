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
  model.yaml
  compile.py                  # or one recipe per component
  composition.yaml           # multi-component products only
  surgery.py                  # only for model-specific preparation
  tests/
    correctness.yaml
    performance.yaml
```

The README records upstream context, unusual compiler decisions, local
qualification notes, and known limitations. Requirements that tooling needs to
enforce must also appear in `model.yaml`.

## 3. Pin every input

Each compiled component owns its source. Use `type: file` for a standalone
ONNX file. Use `type: file_set` when an ONNX entrypoint needs external data or
other files in a particular relative layout. Every file gets its own immutable
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

The matrix creates one compile job per supported component binding. This keeps
compatible component builds together without inventing every possible
cross-product. A component binding proven incompatible may be retained as
`status: unsupported` with a reason. Do not use unsupported for untested work.

Each component recipe has this interface:

```text
compile.py --model <verified-entrypoint> --output-dir <new-directory> \
  --build-profile <profile-id> --component <component-id>
```

The recipe reads only its selected profile binding, consumes already verified
inputs, uses the repository-pinned compiler, creates exactly one MPK, and never
downloads undeclared data. Shared download, checksum, MPK validation, and
publication behavior belongs in `tools/`, not model code.

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

## 8. Scope the tests

Every correctness and performance test states which build profiles it applies
to and whether it exercises a component or a package. Use component scope for
one compiled unit and package scope for an end-to-end composed product. Keep
test configuration inside the model directory and make missing qualification
work visible.

For executable accuracy tests, follow [Local accuracy
development](local-accuracy-development.md). For device performance tests,
follow [Synthetic model benchmarking](benchmarking.md).

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
