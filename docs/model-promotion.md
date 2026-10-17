# Promoting a model to Develop

The **Promote model** workflow selects an existing finalized candidate in
the selected Vulcan environment's Develop catalog. It never compiles, copies binaries, or changes source
provenance. Main/Hugging Face release is not supported. Production use still requires its own
approver configuration and acceptance testing.

## Enable the protected gate

`VULCAN_ENV` selects `staging` or `production`. GitHub's repository variable
overrides the organization variable; Models currently sets `staging`. There is
no implicit default. The configuration job selects that GitHub environment and
reads its `VULCAN_ARTIFACT_REGISTRY_API_URL` (authenticated writer origin) and
`VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` (public catalog reader). Candidate and
promotion operations use the signed writer client; catalog snapshots are fetched
without AWS credentials from the public reader.
Each subsequent job uses the resolved environment and URLs, with publisher roles
from that environment and approver credentials from `<VULCAN_ENV>-promote-develop`.
Both URLs are configuration, not embedded in Python. For source CLI use, export all three
variables explicitly. Missing/invalid configuration fails before any registry call.

The request plan saves the environment and both URLs. A retry with a different environment or URL
fails before a registry call; use the original configuration to finish the old
request, or start a new dispatch in the new environment with its own candidate.
Environment-local candidate and promotion IDs must never be moved between registries.

Before normal use, a repository administrator and Vulcan operator must:

1. Create `<VULCAN_ENV>-promote-develop` (`staging-promote-develop` or
   `production-promote-develop`) with required reviewers, prevent self-review,
   and restrict deployment branches to `develop`. Verify the repository plan
   supports these protection rules, and restrict administrator bypass according
   to the team's approval policy. GitHub requires one eligible listed reviewer,
   not all listed reviewers.
2. Configure Vulcan's `promotion_approvers` entry for `sima-neat/models` with that
   exact environment and a separate role authorized only for Develop request
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
   workflow discovery, not Main artifact promotion.
6. Run disposable fixture acceptance below. Provisioning the environment or a
   successful unit test does not establish working OIDC or backend authorization.

The workflow fails without its role configuration. GitHub may create a missing
referenced environment automatically; that does not configure protection. Do not
provision approver trust until the required environment rules are in place.

## Build, request, review, consume

1. Run **Vulcan Model Orchestrator** with the on-demand model and profile selected.
   Candidates may originate on a feature branch or Develop. Compile
   success is not runtime qualification; review the relevant evidence first.
2. Run **Promote model** from **develop**, choose `promote`, enter `model_id` and
   `build_profile`, and a reason including validation evidence links. The model ID
   is the folder name under `models/` (also the top-level `id` in `model.yaml`);
   the profile is an `id` under `build_profiles` in that file. For Qwen3.5 0.8B,
   use `qwen3_5_0_8b` and `w4a16`. Leave promotion ID blank.
   The workflow selects the newest matching candidate receipt by creation time
   across source branches, verifies its identity and finalized state with Vulcan,
   and pins the UUID before review. Missing or expired receipts fail with build
   instructions; it never silently falls back to an older receipt. Check the
   resolved source commit and candidate in the review summary. Retries restore
   the saved plan instead of resolving a newer build. Source CLI users may still
   supply an exact UUID with `--deployment-id`.
3. The prepare job reads the candidate and current Develop canonical snapshot,
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

Temporary acceptance testing also permits `codex/promotion-catalog-reader` when
`VULCAN_ENV=staging`. It uses the same approval gate and targets Develop in staging.
Remove this exact-branch exception before merging the implementation.
