# Compile runner class requests and approvals

> **Status: planned contract.** The current schema does not accept
> `runner_class`, and the current compilation workflow uses the platform default.
> Do not add this field to a model manifest until the schema, workflow, and Vulcan
> admission check are implemented together.

Model compilation requirements vary substantially. Most variants should use the
default runner, while a small number may need more CPU, memory, or time. The
Model Registry should let a variant request a logical runner class without
letting a model contribution authorize additional infrastructure cost.

This document defines the contract between a model manifest, the compilation
workflow, and the Vulcan scaler.

## Manifest contract

`runner_class` is an optional property of each compile variant:

```yaml
compile_variants:
  - id: modalix_bf16
    target: modalix
    recipe: compile.py
    runner_class: expensive
    parameters:
      precision: bf16
      input_shape: [1, 3, 1536, 1536]
```

The field is a **resource request, not an approval**.

- If the field is absent, the variant requests `default`.
- Values must come from a centrally defined logical-class allowlist.
- A model manifest must never name an EC2 instance type, Kubernetes node type,
  memory quantity, or scaler implementation detail.
- The class is declared per variant because resource requirements can differ
  between variants of the same model.
- A class above `default` requires a matching approval in the Vulcan scaler
  specification.
- `status: unsupported` variants are not scheduled and therefore do not
  consume a runner approval.

The platform owns the mapping from logical classes such as `default` and
`expensive` to infrastructure. That mapping can change without modifying
model manifests.

## Approval contract

An elevated request must be approved outside the Models repository. The approval
is recorded in the Vulcan specification used by the scaler and is reviewed by
the infrastructure owner. A model pull request cannot approve its own resource
request.

The Vulcan approval must identify, at minimum:

| Field | Purpose |
| --- | --- |
| repository | Prevents an approval from being reused by another registry. |
| model ID | Identifies the model requesting elevated resources. |
| variant ID | Limits approval to the variant with measured need. |
| environment | Separates development, staging, and production authority. |
| maximum runner class | Caps the class that the scaler may provision. |
| approver and approval time | Provides an auditable decision record. |
| reason and evidence | Records measurements such as peak memory, timeout, or a benchmark run. |
| expiry or review condition | Prevents exceptional capacity from becoming permanent without review. |

An illustrative Vulcan record is:

```yaml
model_runner_approvals:
  - repository: sima-neat/models
    model_id: example_detector
    variant_id: modalix_bf16
    environment: staging
    maximum_runner_class: expensive
    approved_by: platform-team
    approved_at: "2026-09-01T16:00:00Z"
    reason: Default runner exceeded its memory limit during compilation.
    evidence: https://github.com/sima-neat/models/actions/runs/123456789
    expires_at: "2026-12-01T00:00:00Z"
```

This example communicates the required semantics; the Vulcan specification
schema is authoritative for its exact field names and formats.

## Admission and scheduling

Approval enforcement must happen before GitHub queues a self-hosted job and
before the scaler provisions an instance:

```text
model.yaml
    |
    v
schema and semantic validation
    |
    v
Vulcan approval lookup
    |
    v
approved logical runner label
    |
    v
job scheduling and scaler provisioning
```

The admission check matches repository, model ID, variant ID, environment, and
requested class against the Vulcan record. It returns only an approved logical
runner label.

The workflow must fail closed when:

- the requested class is unknown;
- an elevated request has no matching approval;
- the approval is expired, malformed, or for another environment;
- the requested class exceeds the approved maximum; or
- the approval service or specification cannot be verified.

Do not silently downgrade an unapproved request to `default`. A downgrade can
turn a policy error into a long-running compile failure and makes build behavior
non-deterministic.

Checking approval inside the compile job is too late: the expensive runner may
already have been provisioned. The workflow definition must also avoid accepting
an arbitrary runner label directly from untrusted event input.

## Model author workflow

1. Start with the default runner and capture evidence from the compile attempt.
2. Reduce avoidable resource use in the recipe before requesting a larger class.
3. Add the smallest sufficient `runner_class` to the affected variant only.
4. Explain the request and link the evidence in the model pull request.
5. Obtain a matching Vulcan approval through the infrastructure review process.
6. Confirm admission succeeds before merging or scheduling the elevated build.
7. Re-evaluate or remove the request when the compiler, recipe, or model changes.

Changes that increase runner class should be conspicuous in review. A variant
rename or model rename invalidates the old approval unless the Vulcan policy
explicitly migrates it.

## Build provenance

Compilation results should record:

- the requested runner class;
- the approved runner class;
- the Vulcan specification revision or approval identifier;
- the actual infrastructure selected by the scaler; and
- the model revision, variant ID, compiler version, duration, and outcome.

This allows cost, reliability, and compile-time analysis without coupling the
model manifest to a particular EC2 instance type.
