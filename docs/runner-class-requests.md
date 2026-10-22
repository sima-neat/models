# Compile runner class and market requests

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
          precision: w16a16
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

The admission check matches repository, model folder ID, and requested class
against the Vulcan record. It returns only an approved logical runner label.
Readers accept published approval schema versions 2 and 3: one grant per model
folder, the same in every environment. Version 3 additionally requires a boolean
`allow_on_demand`; version 2 never permits an explicit on-demand request. Unknown
versions or fields fail closed. The approval ID is `repository:model_id`, and every build of the model
records it with the policy revision.

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

## Runner market requests

`runner_market` is independent of `runner_class`. It defaults to `spot`, may be
set to `on-demand` on a build profile, and may be overridden by a component
binding. Every component in a grouped compile must resolve to the same market;
conflicts fail before jobs queue. This is unrelated to `build-on-demand`, which
controls when a build is requested.

```yaml
build_profiles:
  w8a8:
    runner_class: default
    runner_market: on-demand
    components:
      model: {parameters: {}}
```

This fragment requires a separately reviewed Vulcan market grant, including at
default size. A size approval alone never permits on-demand. A market grant
permits a request but never selects on-demand automatically, triggers a build,
or upgrades an interrupted Spot job. Unapproved requests fail before queuing.

Request approval through a Vulcan policy PR with exact model folder IDs, owner,
reason, observed compile durations, interruption evidence (a canceled GitHub job
alone is not proof of Spot reclamation), expected build frequency, and the cost
tradeoff. The shared grant covers current and future profiles/components of those
models; consider that scope when estimating cost. Update the owning model README
when changing its request. No model requests or approvals change in this feature.

New jobs request `runner-market-spot` or `runner-market-on-demand`. Vulcan checks
the grant again and registers only the selected market label. The job verifies
EC2 instance lifecycle through IMDSv2 and records requested/selected/actual market,
runner name, instance ID/type, approval identity and policy revision in provenance
and its summary. Market and approval metadata are outside compile/artifact identity;
changing a model's market request does not invalidate an otherwise reusable compile.

On-demand avoids Spot reclamation, but not workflow timeouts, cancellation, or
other failures. The supported job timeout is 360 minutes. Vulcan's compiler
profile uses a seven-hour hard lifetime: six hours plus 15 minutes for startup and
45 minutes for cleanup. Pending/canceled/completed cleanup still applies earlier.

## Market rollout order

1. Deploy the compatible Vulcan scaler while retaining catalog v1 / published
   policy v2. It understands explicit market labels and retains configured
   behavior for older unlabelled jobs.
2. Release the Models reader/workflow to active branches. New jobs explicitly
   request Spot by default; on-demand still fails closed under v2.
3. In a reviewed Vulcan policy PR, migrate the grouped catalog to v2 (published
   v3), setting `allow_on_demand: false` for existing size-only grants and `true`
   only for separately approved exact model IDs. A default-size market grant
   uses `maximum_runner_class: default`. Deploy to staging, validate, then production.
4. Enable the approved model request with its README update. Verify a staging job's
   actual EC2 lifecycle, GitHub runner matching, provenance and cleanup before wider use.

Do not roll readers back to v2-only while v3 is published. Live deployment and
staging acceptance are tracked in [Vulcan #280](https://github.com/sima-neat/vulcan/issues/280).
