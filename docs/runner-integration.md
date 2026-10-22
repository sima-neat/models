# Integrator's guide: compile runner access

This guide is for maintainers connecting Models' compile workflow to Vulcan's
approval policy and scaler. Model developers should start with
[requesting expensive or on-demand runners](runner-class-requests.md). Deploy
compatible code before granting access; merging a Models manifest cannot authorize
cloud spending.

## Contract and ownership

| Layer | Responsibility |
| --- | --- |
| Models schema and normalization | Accept profile/binding `runner_class` and `runner_market`; default to `default`/`spot`; reject grouped market conflicts. |
| Matrix admission | Match exact repository/model identity and independently authorize size and market against the protected environment specification. |
| Compile workflow | Request the authorized class and explicit market label; verify actual EC2 lifecycle; write provenance and summary. |
| Vulcan catalog and Terraform | Review grants, expand groups into exact model records, and publish the same policy/revision to the workflow specification and scaler. |
| Vulcan scaler | Recheck policy, provision the requested market, register the matching runner label, enforce concurrency and cleanup. |

New jobs carry `runner-market-spot` or `runner-market-on-demand`. Older unlabelled
jobs retain the scaler's configured market during migration. Explicit markets
are supported for `sima-neat/models` jobs on the Model Compiler profile. A grant
covers all profiles/components of the exact model folder; profile/component labels
identify jobs but do not narrow the grant.

## Policy versions and examples

There are two version numbers: the grouped catalog's authoring schema and the
expanded policy's runtime schema.

| Catalog | Published runtime | Capability |
| --- | --- | --- |
| v1 | v2 | Existing elevated-size grants; no explicit on-demand permission. |
| v2 | v3 | Every group requires boolean `allow_on_demand`; maximum class may be `default`, `expensive`, or `most-expensive`. |

Compatible Models and Vulcan readers accept runtime v2 and v3. Older v2-only
Models readers reject the v3 field. Do not publish v3 until active workflow
revisions and scalers understand it.

In Vulcan's `config/model-runner-approvals.json`, an illustrative catalog v2 group
for default-size on-demand looks like this. Replace every example identity and
evidence value with reviewed information; this is not an existing approval:

```json
{
  "schema_version": 2,
  "approval_groups": [{
    "id": "example-long-compile",
    "operation": "compile",
    "repository": "sima-neat/models",
    "maximum_runner_class": "default",
    "allow_on_demand": true,
    "owner": "platform-team",
    "reason": "Measured long compile; predictable completion justifies the documented premium.",
    "evidence": ["URL of reviewed duration, interruption and cost evidence"],
    "model_ids": ["example_detector"]
  }]
}
```

When migrating a real catalog, retain existing grants and add
`allow_on_demand: false` to every size-only group. Do not replace the catalog with
this example. Use `maximum_runner_class: expensive` with false for expensive Spot
access, or with true when both permissions have been approved. Each exact model
must occur once across all groups: update or split an existing group instead of
adding a duplicate grant. Keep defaults at Spot in Models.

The module expands each grant to `repository`, `model_id`,
`maximum_runner_class`, `allow_on_demand` (v3 only), `approved_by`, `reason`, and
`evidence`. The protected environment specification publishes
`model_runner_approvals_schema_version`, `model_runner_approvals`, and
`model_runner_approvals_revision`. The revision hashes the normalized record list;
compare it with the scaler's packaged policy rather than hand-editing either copy.

## Roll out in dependency order

1. Deploy Vulcan's compatible reader, provisioning code and label registration
   under the existing catalog v1/runtime v2 policy. Verify the deployed compiler
   profile has `RUNNER_MAX_AGE_SECONDS = 25200`, retaining its existing pool cap.
2. Release the compatible Models reader and workflow to every branch/ref that
   will run builds. Its default requests now include an explicit Spot label.
   Keep runtime v2 until v2-only readers are retired; on-demand remains denied.
3. Review the catalog v2/runtime v3 migration with exact model IDs, owner, reason,
   compile durations, interruption evidence, build frequency and cost estimates.
   Deploy to staging and check both published copies agree. The shared policy
   source is the same for production, but deployment is a separate operation.
4. Enable one approved model/profile request in Models with its README update.
   Run the staging acceptance below before extending use or deploying policy to
   production. Size permission alone is never a market approval.

The staging deployment of [Vulcan #281](https://github.com/sima-neat/vulcan/pull/281)
was verified on 22 October 2026 with the seven-hour compiler limit, runtime v2 and
65 existing grants. That checkpoint does not enable on-demand or prove a live
on-demand compile. Recheck deployed state before continuing. Rollout acceptance
is tracked in [Vulcan #280](https://github.com/sima-neat/vulcan/issues/280).

## Validate admission without launching EC2

From the Models repository root, with registry tooling installed and an authorized
local copy of the protected environment specification:

```bash
model-registry matrix --repository . --models example_detector --profiles w8a8 \
  > /tmp/compile-matrix.json
model-registry authorize-runner-matrix \
  --matrix /tmp/compile-matrix.json \
  --vulcan-spec /secure/path/environment.json \
  --repository-name sima-neat/models
```

Use real model/profile IDs and the operator's securely retrieved specification.
Do not commit the private specification or credentials. Authorization returns
requested/selected class and market, approval ID and policy revision; it queues
nothing. Test that the approved on-demand request succeeds at default size and
that an unapproved request fails. An elevated request must also stay within the
approved maximum. Runtime v2 must reject explicit on-demand.

In the Vulcan repository root, run the policy and scaler tests before deployment:

```bash
terraform -chdir=modules/model-runner-approvals init -backend=false
terraform -chdir=modules/model-runner-approvals test
python -m pytest -q tests/test_model_runner_approvals.py \
  tests/test_runner_safe_termination.py tests/test_staging_model_compiler_capacity.py
```

Inspect the Terraform plan for the intended staging changes before applying it.
Verify deployed Lambda code hashes/status and the packaged policy; successful
unit tests alone do not show which configuration is live.

## Staging acceptance

Use one reviewed model and profile, within existing concurrency limits. Follow the
[developer dispatch instructions](runner-class-requests.md#validate-and-run), using
`force_rebuild_all=true` only when a fresh compile is intentionally required.

- Confirm admission returns the requested class/market and expected policy revision.
- Confirm the queued job and registered runner share the exact market label.
- Read the job's **Compile runner** summary. Compare requested, selected and actual
  market, instance ID/type, approval ID and policy revision with artifact provenance.
- Independently inspect that instance in the staging AWS account. On-demand instances
  do not have `InstanceLifecycle=spot`; the job's IMDS lifecycle must be `on-demand`.
- Confirm completion or cancellation cleans up the instance. Pending/stalled jobs
  remain subject to cleanup; the hard seven-hour compiler limit applies even when
  a runner reports busy or GitHub status cannot be read.
- Record the run URL, instance identity, policy revision, lifecycle and cleanup
  evidence in the rollout issue. Do not describe admission-only tests as live
  scheduling, successful compilation or hardware qualification.

For example, using a configured staging AWS profile and the summary's instance ID:

```bash
aws --profile vulcan-stg --region us-west-2 ec2 describe-instances \
  --instance-ids i-REPLACE-WITH-SUMMARY-ID \
  --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,Lifecycle:InstanceLifecycle,State:State.Name}'
```

On-demand does not remove the 360-minute workflow timeout, cancellation or compile
failure. The seven-hour cap budgets 15 minutes for startup and 45 for cleanup
around the full job timeout. It is a bound, not a completion guarantee.

## Revoke or roll back

To stop future on-demand launches, remove the model request or set its policy
permission to false and deploy. Existing jobs are not canceled by policy edits.
Keep the compatible reader/scaler while runtime v3 remains published. To return
to v2-only code, first remove explicit on-demand requests, review the loss of
market grants, publish a valid v2 policy to every consuming environment, and
verify that no active workflow still requires the newer labels or contract.
Do not roll back the scaler while Models jobs require its market labels.
