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
| maximum runner class | Caps the class that the scaler may provision. |
| owner | Identifies responsibility; Git history records the decision. |
| reason and evidence | Records measurements such as peak memory, timeout, or a benchmark run. |

Vulcan authors these grants in `config/model-runner-approvals.json`. Groups share
owner, evidence, and maximum class across a list of exact model folder IDs.
A model grant covers all current and future build profiles/components. Grants
may be precreated before a model is registered; no build-target fields are needed.
Staging and production consume the same catalog. Grants have no expiration or
environment scope; removal or modification in source control changes authority
when deployed. Compilation grants do not authorize ONNX conversion.

An illustrative Vulcan record is:

```yaml
model_runner_approvals_schema_version: 2
model_runner_approvals:
  - repository: sima-neat/models
    model_id: example_detector
    maximum_runner_class: expensive
    approved_by: platform-team
    reason: Default runner exceeded its memory limit during compilation.
    evidence: https://github.com/sima-neat/models/actions/runs/123456789
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

The version 2 admission check matches repository, model folder ID, and requested
class against the Vulcan record. It returns only an approved logical
runner label.

For legacy published schema version 1, an existing `variant_id` approval without the new
fields is interpreted as that build profile for the conventional single
component named `model`. It cannot authorize a newly introduced named
component. Migrate legacy approvals to model-folder grants when adopting version 2.

The workflow and scaler must fail closed when:

- the requested class is unknown;
- an elevated request has no matching approval;
- the approval is malformed or the specification version is unsupported;
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
trusted push after merge performs that lookup. A missing or mismatched
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

Changes that increase runner class should be conspicuous in review. A model-folder rename requires a catalog update. Profile and component changes
remain covered by the same model grant.

## Build provenance

Compilation results should record:

- the requested runner class;
- the approved runner class;
- the Vulcan specification revision or approval identifier;
- the actual infrastructure selected by the scaler; and
- the model revision, component ID, build profile ID, compiler version, duration, and outcome.

This allows cost, reliability, and compile-time analysis without coupling the
model manifest to a particular EC2 instance type.


## Shared-policy rollout compatibility

The admission reader accepts published approval schema version 2 (shared,
durable grants) and legacy version 1 (including its environment and timestamp
restrictions) so the reader can be released before Vulcan changes its policy.
A missing version means legacy version 1. Unknown versions fail closed. Version
2 rejects legacy environment, timestamp, profile, component, and variant fields
rather than ignoring them. Version 2 approval IDs are `repository:model_id`;
all builds under that model record the same approval ID and policy revision.
Grouped compilation records the single shared model grant in `runner_approval_ids`.

Release this reader to every active workflow revision before deploying Vulcan's
version 2 specification. Do not roll back to an older reader while version 2 is
published. Legacy support can be removed once all environments have migrated.
