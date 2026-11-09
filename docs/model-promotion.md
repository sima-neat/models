# Promoting a model to Develop or Main

The **Promote model** workflow selects an existing finalized candidate in
the selected Vulcan environment's target catalog. It never compiles, copies binaries, or changes source
provenance. Develop is available in staging and production; Main is available
only in production and keeps the same S3 artifacts. Hugging Face publication is a
separate release operation. Develop acceptance remains tracked in
[Models #574](https://github.com/sima-neat/models/issues/574).

## Enable the protected gate

`VULCAN_ENV` selects `staging` or `production`. GitHub's repository variable
overrides the organization variable. Models has no repository override; verify
the resolved value in the configure job rather than assuming an environment.
There is no implicit default. The configuration job selects that GitHub environment and
reads its `VULCAN_ARTIFACT_REGISTRY_API_URL` (authenticated writer origin) and
`VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` (public catalog reader). Candidate and
promotion operations use the signed writer client; catalog snapshots are fetched
without AWS credentials from the public reader.
Each subsequent job uses the resolved environment and URLs, with publisher roles
from that environment and approver credentials from `<VULCAN_ENV>-promote-<target_branch>`.
Both URLs are configuration, not embedded in Python. For source CLI use, export all three
variables explicitly. Missing/invalid configuration fails before any registry call.

The request plan saves the environment and both URLs. A retry with a different environment or URL
fails before a registry call; use the original configuration to finish the old
request, or start a new dispatch in the new environment with its own candidate.
Environment-local candidate and promotion IDs must never be moved between registries.

Before normal use, a repository administrator and Vulcan operator must:

1. Create `<VULCAN_ENV>-promote-<target_branch>` (`staging-promote-develop`,
   `production-promote-develop`, or `production-promote-main`) with required reviewers, prevent self-review,
   and restrict deployment branches to `develop`. Verify the repository plan
   supports these protection rules, and restrict administrator bypass according
   to the team's approval policy. GitHub requires one eligible listed reviewer,
   not all listed reviewers.
2. Configure Vulcan's `promotion_approvers` entry for `sima-neat/models` with that
   exact environment and a separate role authorized only for that target's request
   inspection/approval/rejection. Scope its OIDC trust to that environment.
3. Set environment variable `VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN` in
   the selected protected environment to the provisioned role ARN. Keep the publisher
   role and both registry URL variables in the corresponding `staging` or
   `production` environment. Set each URL to its corresponding writer or reader origin.
4. Protect Develop and review changes to this workflow and its tooling. All jobs
   check out the workflow run's exact trusted commit, never the candidate ref.
5. Make `.github/workflows/model-promotion.yml` available on the repository default
   branch (`main`) so GitHub discovers its [manual dispatch trigger](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow). Dispatch with
   `develop` selected; execution from Main is intentionally excluded. This is
   workflow discovery; the selected `target_branch` determines the target catalog.
6. Run disposable fixture acceptance below. Provisioning the environment or a
   successful unit test does not establish working OIDC or backend authorization.

The workflow fails without its role configuration. GitHub may create a missing
referenced environment automatically; that does not configure protection. Do not
provision approver trust until the required environment rules are in place.

## Production setup and acceptance

`production-promote-develop` is configured with required `owners-models` review,
self-review prevention, administrator bypass disabled, and only the `develop`
deployment branch. Its `VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN` points to
`arn:aws:iam::069171140053:role/NeatProductionModelsRegistryPromotionRole`.
Vulcan's validator, exact-environment role trust and Develop-only backend grant
were deployed from [Vulcan #433](https://github.com/sima-neat/vulcan/pull/433).
The publisher role and writer/public-reader URLs stay in the separate
`production` environment.

For production, supply `deployment_id` explicitly as well as `model_id`,
`build_profile` and a nonempty reason with evidence links. Use that production
build's finalized `candidate.json`; staging receipts cannot be reused. The
candidate's model/profile must still be declared as build-on-demand on the target branch,
with matching `build_requirements`. A finalized candidate alone does not imply
promotion eligibility. If the model becomes an automatic build or its requirements
change, review the current target instead of changing policy to force approval.

A single-file MPK package stays an MPK; bundled archives stay ZIPs. LLiMa packages
are directory manifests whose runtime files must each be downloaded and verified.
Promotion does not compile or copy these artifacts. Provisioning the role,
finalizing a build and approving its Develop selection are separate milestones.

See the [production acceptance record](evidence/production-promotion-2026-11-08.md)
for the tested revision, candidate inventory, workflow outcome and remaining
acceptance. Do not treat source tests or configuration deployment as a successful
protected promotion. Production automatic cleanup is still disabled: explicitly
cancel/reject abandoned pending requests before expiry and arrange controlled
cleanup of expired holds.

## Production Main selection

Run **Promote model** from **develop** and choose `target_branch: main` with
production selected. The workflow always executes trusted Develop code, even
when selecting Main. The configure job rejects Main in staging before obtaining
publisher credentials. The protected gate is `production-promote-main`, with
its own `NeatProductionModelsRegistryMainPromotionRole`; the Develop role cannot
inspect or decide Main requests, and the Main role cannot inspect or decide
Develop requests. Both gates restrict deployment branches to `develop` because
that is the workflow's execution ref.

The Main environment requires `owners-models` review, self-review prevention,
administrator bypass disabled, and its own
`VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN`. Activate that IAM role only after
those controls exist. Vulcan migration `021_main_candidate_promotions` must be
applied before deploying the updated API package. Deployments preserve existing
Develop pending requests, approved selections and retention holds.

Run **Publish Main model metadata** from **develop** to publish Main's metadata
catalog before requesting selection. It checks out Main, installs that revision's
matching producer and publishes only definitions/documentation, without compiling
or approving artifacts. It records the exact checked-out Main SHA and does not
inherit Develop's definitions. The Main model/profile must declare matching on-demand
`build_requirements`, even if it has not been compiled on Main. A model present
only on Develop needs a reviewed source/metadata release to Main first. The
workflow does not copy Develop's definitions into Main or silently change the
target requirements to fit a candidate. Missing Main metadata fails explicitly.

Before Main can select a candidate, that exact candidate and manifest digest
must have a completed Develop approval. Main's request pins its Develop promotion
UUID and exposes the original workflow evidence for review. A later Develop
selection does not invalidate that historical approval or retarget the Main
request. Main still requires its own protected approval and matching target
requirements.

Main selection reuses the same finalized production candidate and exact S3
versions. It neither recompiles nor copies model payloads. Develop and Main
have independent catalog snapshots and approval retention holds; replacing one
selection does not release the other's hold. This production registry operation
does not upload to Hugging Face or change downloads to a different host.

The saved plan pins the target as well as the environment and URLs. Select the
same target for status/cancel/reject, CLI operations and retries. For CLI use,
add `--target-branch main` to every `model-registry promotion` operation;
omitting it keeps the Develop default. A target change requires a fresh request
and protected review. The default branch's discovery workflow must expose the
same input schema. Copy both manual workflow files to Main for discovery; their
Develop execution guards keep the implementation on trusted Develop code.

## Build, request, review, consume

1. Run **Vulcan Model Orchestrator** with the on-demand model and profile selected.
   Candidates may originate on a feature branch or Develop. Compile
   success is not runtime qualification; review the relevant evidence first.
2. Run **Promote model** from **develop**, choose the target (`develop` by default,
   or production `main`) and `promote`, enter `model_id` and
   `build_profile`, and a reason including validation evidence links. The model ID
   is the folder name under `models/` (also the top-level `id` in `model.yaml`);
   the profile is an `id` under `build_profiles` in that file. For Qwen3.5 0.8B,
   use `qwen3_5_0_8b` and `w4a16`. Leave promotion ID blank.
   In staging, leave `deployment_id` blank to select the newest matching candidate receipt by creation time
   across source branches, verifies its identity and finalized state with Vulcan,
   and pins the UUID before review. Missing or expired receipts fail with build
   instructions; it never silently falls back to an older receipt. Check the
   resolved source commit and candidate in the review summary. Retries restore
   the saved plan instead of resolving a newer build. Source CLI users may still
   supply an exact UUID with `--deployment-id`.
3. The prepare job reads the candidate and current target canonical snapshot,
   creates a correlated request issue, and uploads `promotion-plan`. At this point
   the issue is only a preparation record: no pending hold is promised yet.
4. The request job submits the saved body/key, verifies backend pending protection,
   and writes the review details to its summary and the issue **before** the
   approval job waits. Inspect source commit, candidate digest, model/profile,
   previous/proposed selections, evidence, and expiry. Pending holds last seven
   days. The complete backend JSON is shown to avoid hiding review fields.
5. A different eligible maintainer reviews the protected environment deployment.
   The gated job inspects the exact saved request and approves it using the
   separate role. Only Vulcan's `promoted` receipt and result snapshot establish
   success. The final report inspects current backend state; later catalog
   publication may already have advanced beyond the promotion receipt's snapshot.
6. Download the selected model through a reader supporting `approved_deployments`.
   Directory roots are JSON file manifests, not ZIPs; download and verify every
   required file. Never treat compile success as proof of model runtime quality.

## Status, cancellation, rejection, and retries

Dispatch the same workflow from Develop with `status`, `cancel`, or `reject`, an
existing promotion UUID, and a reason. `status` and `cancel` use publisher
credentials; `reject` requires the protected approver gate. Cancellation cannot
undo a completed promotion. Management dispatches produce their own summaries;
they do not edit the original request issue. Use its request UUID to check state.

Re-running jobs restores the original `promotion-plan` from the same run. The
request body, target snapshot, and idempotency key stay unchanged. Receipts are
saved separately per attempt. After a lost response, an exact POST replay recovers
the request; inspection reports current state. Decision retries inspect the same
request and recognize an already completed matching transition.

If issue updating fails after request creation, the request remains protected and
the gate does not proceed; rerun the failed job to recover the same request and
retry the issue update. If preparation fails before its plan is uploaded, start
a new dispatch. On a rerun, a missing/expired plan fails closed rather than taking
a fresh target. Artifact records are retained for 30 days; pending holds still
expire after seven days.

A target conflict, expired request, or changed review content requires a new
workflow dispatch and review. There is no automatic rebase. Explicitly cancel an
obsolete pending request where useful; changing workflows does not cancel it.

Rejecting/cancelling a GitHub job is not backend rejection/cancellation. The report
job attempts inspection after a rejected gate, but a cancelled workflow may never
run it. Use `status`, then explicitly `cancel` if appropriate; otherwise the
pending hold expires. Other retention holds remain effective.

## CLI operations

These commands are CI-only and excluded from the author-safe installed package.
Use a source checkout with `python -m pip install .` and the appropriate temporary
AWS role credentials. Commands run from the repository root. Example identifiers
must be replaced with actual UUIDs.

```sh
model-registry promotion prepare --record plan.json \
  --deployment-id "$CANDIDATE_ID" --reason 'Validated candidate; evidence URL' \
  --identity "$GITHUB_RUN_ID" --run-url "$WORKFLOW_URL"
# Durably save plan.json before request. Never regenerate it on a retry.
model-registry promotion request --record plan.json --output request.json
model-registry promotion inspect --promotion-id "$PROMOTION_ID" --record status.json
# Separate approver role, only after review of request.json:
model-registry promotion approve --record request.json --output decision.json \
  --identity "$GITHUB_RUN_ID" --reason 'Reviewed candidate and evidence'
# Publisher role; reject instead requires approver credentials:
model-registry promotion cancel --record request.json --output cancellation.json \
  --identity "$CANCEL_OPERATION_ID" --reason 'Request superseded'
```

## Acceptance and remaining rollout work

Local tests cover request/decision recovery, snapshot conflicts, immutable review
content, terminal states, and workflow credential/job boundaries. They do not
prove deployed OIDC, S3 retention, model compatibility, or consumer installation.
Before routine enablement, record an actual protected staging demonstration with
separate requester/reviewer identities and tiny Models-produced MPK and native
LLiMa fixtures (component and full directory package, and ZIP if supported).
Verify every download/checksum, publisher-denied approval, wrong-scope denial,
metadata preservation, source purge/supersession protection, and terminal/retry
behavior. Retain workflow URLs, commits/schema versions, candidate/request IDs,
snapshots, and outcomes. Verify required evidence/documentation retention:
current candidate publication binds compile/package runs and their inline fidelity;
ordinary README/hero publication is branch-retained, not automatically bound into
candidate protection. Close any required source-documentation binding gap before
claiming it survives source-branch deletion. Test sima-cli consumption separately
from browser coverage.

[Vulcan #264](https://github.com/sima-neat/vulcan/issues/264) is separate work.
Until it lands, first approval and same-requirements replacement work, but changing
an approved profile's requirements blocks ordinary catalog publication. Use
disposable fixture profiles for acceptance; do not prematurely approve Qwen's
shared evolving definition. Routine upgrades will preserve the old approved build
until a future approval replaces it; no withdrawal-first workflow is introduced.

Tracking: [Models #315](https://github.com/sima-neat/models/issues/315), under
[master plan #306](https://github.com/sima-neat/models/issues/306). Backend contract:
[Vulcan integration guide](https://github.com/sima-neat/vulcan/pull/253#issuecomment-6210178603).

## Approval and outcome notifications

After Vulcan confirms a pending promotion, the request job adds a comment
mentioning `@sima-neat/owners-models` to the evidence issue. It posts that comment
once per promotion, including on retries. It then sends a Slack reminder with
the model/profile, environment, promotion ID, evidence issue, and approval URL.
This notification does not approve the request.

Reuse `SLACK_VULCAN_EVENT_CHANNEL_ID`, the shared channel variable used by
other Vulcan notifications. It must be available to the selected environment
through environment, repository, or organization variables. Make
`SLACK_BOT_TOKEN` available to the request job; the bot needs `chat:write` and
access to the destination channel. The Slack reminder identifies the GitHub
team by name; it does not assume a corresponding Slack user group.

Only pending promote requests send reminders. Retrying the request job may send
another Slack reminder; the GitHub team comment is deduplicated. Slack delivery
failure is visible as a failed notification step but does not block approval.

## Environment-local candidate selection

Automatic model/profile discovery uses legacy build receipts, which are staging-only:
candidate publication currently rejects other environments. Production dispatches
must provide an exact `deployment_id` from the production registry. They fail
before searching GitHub artifacts when that ID is omitted. The UUID is also an
optional override in staging, allowing selection of a specific validated build.
Find it in the build's `candidate.json` receipt, using `deployment_id`.

Model ID and build profile remain required and are verified against the candidate
returned by the selected registry. A staging UUID cannot be used to promote in
production. This does not enable production candidate publication; that producer
path still requires separate implementation and acceptance testing.

After the approval/decision job finishes, the report job sends a second Slack
message to the same channel. It includes the model/profile, environment, promotion
ID, workflow and evidence links, approval-job result, and freshly read backend
state. Only `promoted` with a result snapshot is described as completed.
Rejected, cancelled, expired, still-pending, and unreadable backend states are
reported explicitly. Status-only dispatches do not send outcome notifications.
The outcome message is prepared before updating the issue, so an issue-update
failure does not suppress Slack delivery. Reporting requires the saved plan and
request receipt; failures before those are available cannot send this outcome.
Retries may resend the outcome notification.
