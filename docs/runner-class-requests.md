# Compile runner class requests and approvals

The schema and compilation workflow support this contract. Elevated requests
remain unusable until a matching approval is deployed through the Vulcan
environment specification.

Model compilation requirements vary substantially. Most component builds should use the
default runner, while a small number may need more CPU, memory, or time. The
Model Registry lets a build-profile component binding request a logical runner class without
letting a model contribution authorize additional infrastructure cost.

This document defines the contract between a model manifest, the compilation
workflow, and the Vulcan scaler.

## Manifest contract

`runner_class` is optional on a build profile and may be overridden by a
component binding:

```yaml
build_profiles:
  - id: w16a16
    target: modalix
    runner_class: expensive
    components:
      - id: detector
        parameters:
          precision: bf16
          input_shape: [1, 3, 1536, 1536]
```

The field is a **resource request, not an approval**.

- If the field is absent at both levels, the build requests `default`.
- Values must come from a centrally defined logical-class allowlist.
- A model manifest must never name an EC2 instance type, Kubernetes node type,
  memory quantity, or scaler implementation detail.
- A binding override is useful when components in one profile have different
  resource requirements.
- A class above `default` requires a matching approval in the Vulcan scaler
  specification.
- `status: unsupported` component bindings are not scheduled and therefore do not
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
| build profile ID | Limits approval to one target-wide configuration. |
| component ID | Limits approval to the independently compiled unit with measured need. |
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
    build_profile_id: w16a16
    component_id: detector
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

The admission check matches repository, model ID, build profile ID, component
ID, environment, and
requested class against the Vulcan record. It returns only an approved logical
runner label.

During the v2 migration, an existing `variant_id` approval without the new
fields is interpreted as that build profile for the conventional single
component named `model`. It cannot authorize a newly introduced named
component. Replace legacy approvals with explicitly scoped records when they
are next reviewed.

The workflow and scaler must fail closed when:

- the requested class is unknown;
- an elevated request has no matching approval;
- the approval is expired, malformed, or for another environment;
- the requested class exceeds the approved maximum; or
- the approval service or specification cannot be verified.

Do not silently downgrade an unapproved request to `default`. A downgrade can
turn a policy error into a long-running compile failure and makes build behavior
non-deterministic.

The Models workflow authorizes the complete matrix in an Ubuntu-hosted admission
job before GitHub expands the self-hosted compile matrix. Vulcan repeats the
approval check in the scaler before provisioning as a defense-in-depth boundary.
The workflow definition must also avoid accepting an arbitrary runner label
directly from untrusted event input.

Pull-request checks validate the requested class against the schema, but they
do not read the protected Vulcan specification or run admission. The first
trusted push after merge performs that lookup. A missing, expired, or mismatched
approval fails before the self-hosted compile job is scheduled.

## Model author workflow

1. Start with the default runner and capture evidence from the compile attempt.
2. Reduce avoidable resource use in the recipe before requesting a larger class.
3. Add the smallest sufficient `runner_class` to the affected component binding only.
4. Explain the request and link the evidence in the model pull request.
5. Obtain a matching Vulcan approval through the infrastructure review process
   and link its review evidence in the pull request.
6. After merge, confirm the trusted admission job authorizes the request before
   expecting the compile job to be scheduled.
7. Re-evaluate or remove the request when the compiler, recipe, or model changes.

Changes that increase runner class should be conspicuous in review. A model,
component, or build-profile rename invalidates the old approval unless the Vulcan policy
explicitly migrates it.

## Build provenance

Compilation results should record:

- the requested runner class;
- the approved runner class;
- the Vulcan specification revision or approval identifier;
- the actual infrastructure selected by the scaler; and
- the model revision, component ID, build profile ID, compiler version, duration, and outcome.

This allows cost, reliability, and compile-time analysis without coupling the
model manifest to a particular EC2 instance type.
