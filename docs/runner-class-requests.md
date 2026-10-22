# Request expensive or on-demand compile runners

Model authors request compile resources in `model.yaml`; Vulcan grants access
through a separately reviewed policy. Start with the default size on Spot and
request additional resources only when measurements justify them. This guide
covers the author workflow. Platform maintainers should also follow the
[runner integration guide](runner-integration.md).

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
For an expensive Spot build:

```yaml
build_profiles:
  w8a8:
    runner_class: expensive
    runner_market: spot
    components:
      model:
        parameters: {precision: w8a8}
```

For on-demand at the default size, change those two fields to:

```yaml
runner_class: default
runner_market: on-demand
```

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
   grant needs the v3 runtime contract described in the integration guide.
4. Link the Vulcan PR in the Models PR, update the owning README to explain the
   runner request, and validate the manifest. Enable only the reviewed request.

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
use **Vulcan Model Orchestrator** with environment `staging`, the model ID and
profile ID. For example, from this repository with GitHub CLI access:

```bash
gh workflow run vulcan-model-orchestrator.yml --ref develop \
  -f environment=staging -f models=example_detector -f profiles=w8a8
```

The selected ref must contain the request and compatible workflow. A market-only
change may reuse an existing result and launch no runner. For a deliberate fresh
staging verification, add `-f force_rebuild_all=true` to that narrowly selected
dispatch; this incurs another compile and may run its declared benchmarks. Manual
selection also permits models with `build-on-demand: true`.

## Verify the result and diagnose failures

The compile job's **Compile runner** summary shows requested, selected and actual
market, runner name, EC2 instance ID/type, approval identity and policy revision.
The same identity fields are written into artifact provenance. A default Spot
request needs no approval identity, but still records the policy revision.

| Symptom | Action |
| --- | --- |
| Missing approval or requested class exceeds the maximum | Check the exact folder ID and the deployed grant, not only whether its PR merged. |
| No Vulcan on-demand approval | Check runtime policy v3 and `allow_on_demand: true`; a size grant is insufficient. |
| Conflicting markets | Make all bindings in the grouped invocation resolve to one market. |
| Unsupported schema or malformed grant | Have the integrator check reader compatibility and policy publication. Do not bypass admission. |
| Job waits for a runner | Check that the deployed scaler supports the requested market label and that pool capacity is available. |
| Runner market mismatch | Stop and investigate runner registration versus EC2 lifecycle; the job fails before recording identity. |
| No new compile job | Check fingerprint reuse and `build-on-demand`; an approval never forces a rebuild. |

There is no silent downgrade. On-demand avoids Spot reclamation but not
cancellation, compiler errors, workflow timeouts, or Vulcan cleanup. The job timeout
is 360 minutes; the compiler profile's hard lifetime is seven hours, including
startup and cleanup margin. A larger class does not extend these limits.
