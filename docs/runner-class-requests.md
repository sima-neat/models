# Request expensive or on-demand compile runners

Use this guide when your model needs more compilation capacity or more predictable
completion than the default Spot runner provides. It explains what makes a model
eligible for an exception, what to submit for approval, and how to use the approved
runner. You request resources in `model.yaml`; the Vulcan infrastructure owner
reviews and deploys the permission. Start with the default size on Spot.

## What qualifies as a special model?

A special model is one with a demonstrated compilation need that justifies an
exception. It is not a schema category or an automatic entitlement for a model
family. The following are reasons to request review, not automatic approvals:

| Demonstrated need | Examples worth investigating | Appropriate request |
| --- | --- | --- |
| Default capacity cannot compile the model reliably because of memory or resource pressure. | Large LLM/VLM checkpoints, long-context builds, or large vision graphs with measured peak-memory pressure or out-of-memory failures. | The smallest sufficient elevated class, initially on Spot. |
| A larger runner materially improves an otherwise impractical compile. | CPU-heavy compilation or expensive graph optimization, with measured runtime and cost comparisons. | Elevated class supported by the comparison; more cores alone do not establish a speedup. |
| Long, costly-to-repeat compilation makes Spot reclamation unacceptable for the use case. | A long detector, language, vision-language or audio build with lost work, verified interruptions, or a documented completion deadline and reliability requirement. | On-demand at the existing size unless separate evidence supports a size upgrade. |
| Both capacity and completion reliability are limiting. | A resource-heavy compile that also loses substantial work to Spot interruptions. | Elevated class plus on-demand, with evidence for each permission. |

A model name, parameter count, large input resolution, or membership in the LLM,
VLM, audio or RF-DETR families is not sufficient on its own. Existing grants are
for exact folder IDs, not whole families. A new model can request review before
registration using representative measurements or documented resource estimates;
label estimates clearly and explain what still needs measurement.

There is no universal parameter-count or compile-duration cutoff in this policy.
Explain why the default runner is unsuitable for this build and why the added
cost is justified. First rule out invalid model inputs, compiler errors, avoidable
recipe work, and queue delays: on-demand does not fix those. A canceled GitHub job
is not evidence of Spot reclamation unless corroborated by EC2 events. If there
have been no observed interruptions, say so and justify the reliability need
without claiming a measured interruption rate.

## Choose size and market independently

| Request | Purpose | Vulcan approval |
| --- | --- | --- |
| `runner_class: default`, `runner_market: spot` | Normal compilation; both fields may be omitted. | No model-specific grant. |
| `runner_class: expensive`, `runner_market: spot` | More compile capacity while retaining Spot pricing. | Maximum class of `expensive` or `most-expensive`. |
| `runner_class: most-expensive`, `runner_market: spot` | Highest supported logical compile class. | Maximum class of `most-expensive`. |
| `runner_class: default`, `runner_market: on-demand` | Avoid Spot reclamation without increasing runner size. | Published policy v3 with `allow_on_demand: true`. |
| `runner_class: expensive`, `runner_market: on-demand` | More capacity and avoidance of Spot reclamation. | Both a sufficient maximum class and `allow_on_demand: true`. |

Vulcan owns the mapping from class to EC2 instance types. Do not put instance
types in the model manifest. Permission does not select a larger runner or a
market automatically, trigger a build, or retry an interrupted Spot job on
on-demand. Neither class nor market changes invalidate a reusable compile.

`runner_market: on-demand` selects EC2 purchasing. The separate top-level
`build-on-demand: true` controls scheduling: it excludes the model from normal
push builds and requires an explicit manual selection. Either purchasing market
can be used with that scheduling setting.

## Declare the request

Set the request on the existing profile in `models/<model_id>/model.yaml`.
These are partial examples; retain the model's real parameters and calibration.
For a build approved for both an expensive runner and on-demand purchasing:

```yaml
build_profiles:
  w8a8:
    runner_class: expensive
    runner_market: on-demand
    components:
      model:
        parameters: {precision: w8a8}
```

If only on-demand is justified, keep the default size:

```yaml
runner_class: default
runner_market: on-demand
```

For an expensive Spot build, use `runner_class: expensive` with
`runner_market: spot` (or omit the market). Request only the combination approved
for your model.

Place both fields directly under the profile, alongside `components`, not inside
`parameters`. A component binding can override either profile value. A missing
market inherits the profile value, then defaults to `spot`; a missing class
inherits the profile value, then defaults to `default`.

A profile's components compile together on one runner. Prefer a profile-level
class and market for that shared invocation; do not use component overrides to
try to allocate separate machines. All effective markets must agree or validation
rejects the grouped build. For example, a profile requesting on-demand with one
component overriding it to Spot conflicts unless every component resolves to Spot.

## Obtain access through a Vulcan policy PR

Before enabling the request on a branch that runs trusted compilation:

1. Measure the existing build. For a larger class, provide peak-memory or runtime
   evidence and explain why recipe improvements are insufficient. For on-demand,
   provide observed durations and interruption evidence; a canceled GitHub job
   alone does not establish Spot reclamation.
2. Open a PR against the shared
   [Vulcan approval catalog](https://github.com/sima-neat/vulcan/blob/develop/config/model-runner-approvals.json).
   Give exact model folder IDs, an owner, reason, evidence links, expected build
   frequency, the requested class/market, and the cost tradeoff. Measure size
   upgrades separately instead of assuming a larger runner is cheaper overall.
3. Have the infrastructure owner review and deploy the grant. A model PR cannot
   approve itself. Existing size grants do not permit on-demand; an on-demand
   grant needs published policy v3 with `allow_on_demand: true`. Ask the owner
   to confirm that compatible Models workflows and Vulcan scaler code are deployed
   before enabling it; a merged policy PR alone does not prove it is active.
4. Link the Vulcan PR in the Models PR, update the owning README to explain the
   runner request, and validate the manifest. Enable only the reviewed request.

Use this checklist in the approval request so the owner can review it without
reconstructing the build history:

```text
Model folder ID(s):
Owner:
Profiles/components measured:
Requested class and market:
Why the default Spot runner is unsuitable:
Compiler version, current runner type, and build/run links:
Observed duration and peak memory (or clearly labeled estimates):
Interruption evidence and lost work, or the required completion window:
Recipe/resource reductions already considered:
Expected builds per week/month, including retries:
Estimated current versus requested cost per build and per month:
Why this is the smallest sufficient exception:
Validation plan and links to the Models/Vulcan PRs:
```

Estimate compute cost as hourly price × billed hours, including startup and
retries; state the region, pricing date, assumptions, and excluded charges.
On-demand can be worthwhile for predictable completion without being cheaper.
For a size upgrade, compare runtime as well as hourly price. Do not claim a
speedup or cost saving that has not been measured.

You do not need to operate Terraform or edit deployed specifications to use this
feature. The infrastructure owner maintains the shared approval catalog and
publishes it to both admission and provisioning. For size-only access, the grant
sets `maximum_runner_class` high enough for the request. For on-demand it must
also explicitly allow the market, including when the maximum class is `default`.
The owner handles any required policy-version migration; existing size-only
grants must not silently acquire on-demand permission. Infrastructure details
belong in [Vulcan's approval documentation](https://github.com/sima-neat/vulcan/blob/develop/docs/model-runner-approvals.md).

The grant identity is `sima-neat/models:<model_id>`, where `model_id` is the exact
folder name, without `models/`. It covers every current and future profile and
component in that folder. Staging and production share the source-controlled
catalog; there are no wildcard, expiration, or environment-scoped grants. A
rename needs a policy update. Grants may precede model registration. Revocation
affects future admission after deployment; it does not cancel an existing job.
Compile grants do not authorize ONNX conversion runners.

## Validate and run

With contributor tooling installed, run from the repository root, replacing
`example_detector` and `w8a8` with the exact model/profile IDs:

```bash
model-registry validate models/example_detector
model-registry matrix --repository . --models example_detector --profiles w8a8
```

The matrix reports `requested_runner_class` and `requested_runner_market`.
These local commands validate and describe the request; they do not grant access
or read the protected Vulcan policy. PR checks likewise do not prove admission.
Trusted automation authorizes the compile matrix before queuing self-hosted jobs,
and Vulcan checks it independently before launching capacity.

A normal trusted push builds missing fingerprints. To explicitly select a build,
use **Vulcan Model Orchestrator** with the model ID and profile ID. The build
uses `VULCAN_ENV` for its runner and publication environment. For example, from
this repository with GitHub CLI access:

```bash
gh workflow run vulcan-model-orchestrator.yml --ref develop \
  -f models=example_detector -f profiles=w8a8
```

The selected ref must contain the request and compatible workflow. A market-only
change may reuse an existing result and launch no runner. For a deliberate fresh
verification in the selected environment, add `-f force_rebuild_all=true` to that narrowly selected
dispatch; this incurs another compile and may run its declared benchmarks. Manual
selection also permits models with `build-on-demand: true`.

## Verify the result and diagnose failures

The compile job's **Compile runner** summary shows requested, selected and actual
market, runner name, EC2 instance ID/type, approval identity and policy revision.
The same identity fields persist in the registry's `model_compile` run parameters
and the package manifest's `provenance`, also published in the `model_package`
run parameters, beyond workflow-artifact retention.
They describe the runner that actually compiled the result, not a later runner
that reuses it. Older builds can omit these fields. Runner audit metadata does
not change the compile fingerprint, compile-run fingerprint or compiled file
checksums; the package manifest's own digest reflects its provenance content. A default Spot
request needs no approval identity, but still records the policy revision. Link
the first staging run and its summary in your Models PR, and ask the infrastructure
owner to confirm EC2 lifecycle and cleanup. Local validation or admission alone
does not prove a runner was launched or that compilation succeeded.

| Symptom | Action |
| --- | --- |
| Missing approval or requested class exceeds the maximum | Check the exact folder ID and the deployed grant, not only whether its PR merged. |
| No Vulcan on-demand approval | Check runtime policy v3 and `allow_on_demand: true`; a size grant is insufficient. |
| Conflicting markets | Make all bindings in the grouped invocation resolve to one market. |
| Unsupported schema or malformed grant | Ask the infrastructure owner to check reader compatibility and policy publication. Do not bypass admission. |
| Job waits for a runner | Check that the deployed scaler supports the requested market label and that pool capacity is available. |
| Runner market mismatch | Stop and investigate runner registration versus EC2 lifecycle; the job fails before recording identity. |
| No new compile job | Check fingerprint reuse and `build-on-demand`; an approval never forces a rebuild. |

There is no silent downgrade. On-demand avoids Spot reclamation but not
cancellation, compiler errors, workflow timeouts, or Vulcan cleanup. The job timeout
is 360 minutes; the compiler profile's hard lifetime is seven hours, including
startup and cleanup margin. A larger class does not extend these limits.

Revisit the exception when the compiler, recipe or model changes. Remove an
unneeded manifest request and ask the infrastructure owner to reduce or revoke
the grant. Policy changes affect future admission, not a job already running.
