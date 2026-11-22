# Promotion operations

The maintainer runbook for the **Promote model** workflow: set up its protected gate,
promote a candidate to develop or main, check, cancel, reject, or withdraw a request, and
recover a failed run. Read it before you operate or change
`.github/workflows/model-promotion.yml`, `scripts/model_promotion.py`, or
`tools/model_registry/promotions.py`. The requester's steps are in
[Promote a model](../../docs/model-promotion.md). How a candidate is built is in
[on-demand builds](on-demand-builds.md).

## What promotion does

Promotion selects one finalized candidate in the target branch's catalog, in one Vulcan
environment. It does not compile, copy files, or change the candidate's source provenance.

- Develop is a target in staging and production. Main is a target in production only
  (`configuration` in `promotions.py`).
- The workflow runs only from `develop` of this repository (`configure` job condition,
  `main` in `scripts/model_promotion.py`). Every job checks out the run's own commit,
  never the candidate's ref.
- Only a Vulcan record in state `promoted` with a `result_snapshot` means the promotion
  succeeded. A submitted request or an approved GitHub job does not.
- Promotion retains the candidate. It does not retain the source branch's README or hero,
  which each branch's catalog publishes for itself.
- The workflow never uploads to Hugging Face. That is a separate release operation.

## Jobs and credentials

| Job | GitHub environment | AWS role | What it does |
|---|---|---|---|
| `configure` | selected Vulcan environment | none | resolves the environment, target, and both URLs |
| `prepare` | selected Vulcan environment | publisher | builds or restores the request plan, creates the request issue |
| `request` | selected Vulcan environment | publisher | submits, inspects, or cancels the request, then updates the issue |
| `approve` | `<environment>-promote-<target_branch>` | promotion | approves, rejects, or withdraws after a reviewer approves the gate |
| `report` | selected Vulcan environment | publisher | reads the final state, refreshes benchmark references, notifies |

The `operation` input decides which job changes the request:

| Operation | Job that acts | Needs the gate |
|---|---|---|
| `promote` | `request` submits, `approve` approves | yes |
| `status` | `request` inspects | no |
| `cancel` | `request` cancels | no |
| `reject` | `approve` rejects | yes |
| `withdraw` | `approve` withdraws | yes |

## Configuration

Each value lives in GitHub settings, so this repository cannot check it.

| Value | Where it is configured |
|---|---|
| `VULCAN_ENV` | repository or organization variable. The workflow's `environment` input overrides it for one run. |
| `VULCAN_ARTIFACT_REGISTRY_API_URL` | variable of the `staging` or `production` environment. The signed writer origin. |
| `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL` | variable of the same environment. The public catalog reader. |
| `VULCAN_ARTIFACT_REGISTRY_PUBLISHER_ROLE_ARN` | variable of the same environment. The publisher role. |
| `VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN` | variable of the `<environment>-promote-<target_branch>` environment. The promotion role for that target. |
| `SLACK_VULCAN_EVENT_CHANNEL_ID` | variable visible to the selected environment. The channel other Vulcan notifications use. |
| `SLACK_BOT_TOKEN` | secret visible to the `request` and `report` jobs. |

- `configuration` in `promotions.py` fails unless `VULCAN_ENV` is `staging` or
  `production` and both URLs are HTTPS origins without a path, query, or credentials. The
  failure comes before any registry call. Read the resolved values in the `configure`
  job's outputs.
- Candidate and promotion requests use the signed writer client. Catalog snapshots come
  from the public reader without AWS credentials (`prepare` in `promotions.py`).
- The saved plan holds the environment, both URLs, and the target. A later job or retry
  with a different value fails with `Saved promotion environment/URL/target differs; use
  its original configuration`.
- Candidate and promotion IDs belong to one registry. A staging ID cannot be used in
  production.

## Set up a protected gate

A repository administrator and a Vulcan operator do this once per environment and target.
The gates are `staging-promote-develop`, `production-promote-develop`, and
`production-promote-main`. Steps 1 to 5 are GitHub and Vulcan settings that nothing in
this repository enforces or checks.

1. Create the GitHub environment `<VULCAN_ENV>-promote-<target_branch>`. Require
   reviewers from the model owners team, prevent self-review, and disable administrator
   bypass as the team's approval policy requires.
2. Restrict the environment's deployment branches to `develop`. This holds for the main
   gate too, because `develop` is the ref the workflow runs from.
3. Ask the Vulcan operator to add a `promotion_approvers` entry for this repository with
   that exact environment name. The entry names a separate role limited to that target.
   The `approve` job uses it to inspect, approve, reject, and withdraw. Its OIDC trust is
   scoped to that environment.
4. Set `VULCAN_ARTIFACT_REGISTRY_PROMOTION_ROLE_ARN` in the new environment to that role.
   Leave the publisher role and both URLs in the `staging` or `production` environment.
5. Protect `develop`, and review every change to the workflow and its tooling.
6. Keep `model-promotion.yml` and `model-main-catalog.yml` on the default branch, `main`,
   with the same inputs as on `develop`. GitHub lists a manual trigger only from the
   default branch. A dispatch from any ref other than `develop` skips every job.
7. Run the [acceptance](#accept-a-new-gate) below.

Keep to this order. Create the environment rules before the Vulcan operator activates the
role. GitHub creates a missing environment on first use without any protection, and the
workflow fails when the role variable is missing.

GitHub asks for one eligible reviewer from the list, not all of them. Vulcan's role
grants keep the develop role from deciding main requests and the main role from deciding
develop requests.

## Promote a build

1. Build the candidate with **Vulcan Model Orchestrator**, on a feature branch or on
   `develop`. A passed compile is not a runtime qualification. Review the DevKit evidence
   first.
2. Check the target. Vulcan accepts the request only when the model still sets
   `build-on-demand: true` on the target branch and the target catalog's
   `build_requirements` match the candidate. When the model became an automatic build or
   its requirements changed, review the target as it is. Do not change policy to force
   the approval.
3. Dispatch **Promote model** from `develop` with `operation=promote`, the `model_id`, the
   `build_profile`, the target, and a `reason` of 1 to 2,000 characters with evidence
   links. Leave `promotion_id` blank.
4. Set `deployment_id` from the build's `candidate.json`. In production it is required.
   In staging a blank value selects the newest receipt, as described under
   [Staging receipt lookup](#staging-receipt-lookup).
5. Wait for `prepare`. It reads the candidate from Vulcan, reads the target's current
   catalog snapshot, creates the request issue, and uploads `promotion-plan`. The issue
   is only a preparation record at this point. No hold exists yet.
6. Wait for `request`. It submits the saved body and key, confirms the request is
   `pending` with `pending_protection`, and writes the review details to its summary and
   the issue. A pending hold lasts seven days.
7. Review as a different eligible maintainer. Check the source commit, the candidate
   digest, the model and profile, the previous and proposed selections, the evidence, and
   the expiry. Then approve the environment deployment.
8. Wait for `approve`. It inspects the same request again, fails with `Promotion review
   content changed` when a reviewed field differs, and approves with the promotion role.
9. Read the `report` job. It inspects the current state, so a later catalog publication
   may have moved past the promotion's snapshot. The issue closes on `promoted`,
   `rejected`, `cancelled`, or `expired`.
10. Download the selected build through the target catalog and verify it. A LLiMa package
    is a JSON manifest of files, not a ZIP. Download and verify every file it lists.
11. Benchmark the promoted build. See
    [Benchmarks](on-demand-builds.md#benchmarks).

### Staging receipt lookup

`candidate_receipt` in `scripts/model_promotion.py` runs only in staging and only when
`deployment_id` is blank.

- It lists the repository's Actions artifacts named
  `candidate-<model_id>-<build_profile>-<attempt>` across all source branches and takes
  the newest by creation time.
- It fails when no receipt exists or when the newest one has expired. It never falls back
  to an older receipt. The build workflow keeps receipts for 14 days.
- The build workflow uploads receipts with `if: always()`, so the newest one may come
  from a failed publication.
- `prepare` then asks Vulcan for that candidate. It fails unless the candidate is
  finalized and matches the requested model and profile, and it does not try an older
  receipt. On success it pins the UUID in the plan.
- A retry restores the saved plan. It does not resolve a newer build.

### Benchmark references

`report` runs for every operation once `request` passes. When the request is `promoted`
and still selected, it checks out the target catalog's source revision and calls
`refresh_benchmarks` in `promotions.py`. That publishes the benchmark fingerprints of the
promoted build to the target catalog without starting a measurement. A `status` dispatch
therefore refreshes them too. It does nothing when a later promotion replaced the
selection.

When it fails with `Promotion target revision changed`, rerun the `report` job. No new
approval is needed.

### When the target's requirements change

A target branch may publish new `build_requirements` while its previous approved build
stays selected. Vulcan then requires a replacement candidate to match the new
requirements and to get its own approval.

## Promote the same candidate to main

1. Complete the develop promotion of the exact candidate first, in production. Vulcan
   enforces that a main request names a candidate and manifest digest develop approved.
2. Release the model definition to `main`. A model that exists only on `develop` needs a
   reviewed source release first.
3. Dispatch **Publish Main model metadata** from `develop`
   (`.github/workflows/model-main-catalog.yml`). It checks out `main`, installs that
   revision's `model-registry`, and runs `scripts/publish_main_catalog_metadata.sh`.
4. Confirm the summary names the expected `main` revision. The run uploads
   `main-catalog-metadata-<run id>` with the receipt and keeps it for 30 days.
5. Dispatch **Promote model** from `develop` with `environment=production`,
   `target_branch=main`, and the same `deployment_id`. The `configure` job rejects main
   in staging before any job assumes a role.
6. Have a maintainer approve the `production-promote-main` gate.
7. Check the selected download in the main catalog.

The metadata run publishes main's definitions and documentation with `publish-catalog
--branch main --base-branches main`. It compiles, benchmarks, and approves nothing, and
it never copies develop's definitions. The script fails when the `main` checkout has
uncommitted changes. When it is the first publication of a `main` revision, it lists
develop's finished automatic builds, as an orchestrator run on `main` does
([merge into main](branch-catalogs.md#merge-into-main)).

The run sets no `MODEL_REGISTRY_CONVERSIONS`, so `_catalog_models` in `registry.py` defers
every conversion-backed model. Such a model keeps its previous catalog entry with its old
`build_requirements`, and one without a previous entry does not enter the catalog. Only
an orchestrator run on `main` with `VULCAN_ENV` set to `production` rebuilds that entry.

Vulcan requires main's catalog to hold matching `build_requirements` for the profile,
even when main never compiled it. The main request pins the develop promotion it relies
on (`source_promotion_id` in the request record). Main selects the same candidate and
the same stored files. Per the Vulcan contract, a later develop selection does not
retarget the main request, and each target keeps its own retention hold.

## Check, cancel, reject, or withdraw

Dispatch **Promote model** from `develop` with the operation, the existing
`promotion_id`, the original environment and `target_branch`, and a reason.

| Operation | Effect |
|---|---|
| `status` | Reads the request and updates the original request issue. |
| `cancel` | Cancels a pending request. It cannot undo a completed promotion. |
| `reject` | Rejects a pending request at the protected gate. |
| `withdraw` | Removes a promoted build from its target catalog at the protected gate. |

- `withdraw` posts to the promotion's `withdraw` endpoint and requires a
  `result_snapshot` in the reply (`withdraw` in `promotions.py`). Vulcan keeps the request
  in state `promoted` and decides what the catalog then shows for the profile.
- Rejecting or cancelling the GitHub job does not reject or cancel the request in Vulcan.
  Run `status`, then `cancel` when the request should end. Otherwise the pending hold
  stays until it expires.
- Cancel an abandoned pending request explicitly. Changing the workflow does not cancel
  it.

### When a promoted profile leaves `model.yaml`

When a change comments out or removes a profile that has a promoted build, `withdraw` each
of its promotions, on `develop` and on `main`, with that promotion's environment and
`target_branch`. Do it before the customer portal is deployed, because the customer view
lists `main`'s promoted builds and nothing else takes this one out:

- The repository never writes or removes an approval. A catalog snapshot carries no
  `approved_deployments` or `branch_deployments` of its own (`SERVER_OWNED_SELECTIONS` in
  `registry.py`), and the registry fills them in.
- A run that defers a conversion-backed model, such as the Main metadata run, keeps that
  model's previous catalog entry ([promote the same candidate to
  main](#promote-the-same-candidate-to-main)).
- The browser lists every profile whose approval the registry returns, declared or not
  (`deploymentCatalog` in `artifact-browser/branch-builds.js`).
- `check-main-merge` does not ask whether a promoted profile is still declared
  (`find_main_merge_blockers` in `stage_plan.py`), so it does not block the merge into
  `main` on it.

What the registry does with an approval whose profile is no longer declared is not
verified here. Check the target catalog after the withdrawal.

## Recover a failed run

| Situation | Action |
|---|---|
| A job failed after `promotion-plan` was uploaded | Rerun the failed jobs. `prepare` restores the plan from the same run, so the body, target snapshot, and key stay unchanged. |
| A response was lost | Rerun. The same request replays, and inspection reports the current state. A decision retry recognizes a transition that already completed. |
| The issue update failed after the request was created | Rerun the failed job. The request stays protected and the gate does not open until the job passes. |
| `prepare` failed before it uploaded the plan | Start a new dispatch. |
| `Saved plan is missing` or `Saved plan expired` on a rerun | Start a new dispatch and a new review. The rerun never takes a fresh target. |
| The target catalog changed, the request expired, or the review content changed | Start a new dispatch and a new review. Nothing rebases the request. |

Each attempt saves its own `promotion-record-<attempt>` and
`promotion-decision-<attempt>`. GitHub keeps these and `promotion-plan` for 30 days. The
pending hold still expires after seven days.

## Notifications

- After Vulcan confirms a pending promote request, `request_review` in
  `scripts/model_promotion.py` comments on the request issue and mentions the model
  owners team. It posts that comment once per promotion.
- The same step sends a Slack reminder with the model, profile, environment, promotion
  ID, issue, and approval link. A retry of the `request` job may send it again. The
  reminder names the GitHub team in plain text and approves nothing.
- `report` sends a second Slack message for `promote`, `cancel`, and `reject`, with the
  state it read from Vulcan. Only `promoted` with a result snapshot reads as completed.
  `status` and `withdraw` send no outcome message.
- `report` prepares the outcome message before it updates the issue, so a failed issue
  update does not suppress it. A failure before the plan and the request record exist
  sends no outcome.
- A failed Slack delivery shows as a failed step and does not block approval. The bot
  needs `chat:write` and access to the channel.

## CLI operations

The `promotion` command is excluded from the installed author package
(`packaging/model-registry/build.sh`). Use a source checkout, `python -m pip install .`,
temporary credentials for the right role, and these variables: `VULCAN_ENV`,
`VULCAN_ARTIFACT_REGISTRY_API_URL`, and `VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL`.

```sh
model-registry promotion prepare --record plan.json \
  --deployment-id "$CANDIDATE_ID" --reason 'Validated candidate, evidence URL' \
  --identity "$GITHUB_RUN_ID" --run-url "$WORKFLOW_URL"
# Save plan.json before the request. Never regenerate it on a retry.
model-registry promotion request --record plan.json --output request.json
model-registry promotion inspect --promotion-id "$PROMOTION_ID" --record status.json
# Promotion role, only after review of request.json:
model-registry promotion approve --record request.json --output decision.json \
  --identity "$GITHUB_RUN_ID" --reason 'Reviewed candidate and evidence'
# Publisher role. reject needs the promotion role instead:
model-registry promotion cancel --record request.json --output cancellation.json \
  --identity "$CANCEL_OPERATION_ID" --reason 'Request superseded'
```

- Add `--target-branch main` to every command for a main request. Without it the target
  is `PROMOTION_TARGET_BRANCH`, or `develop`.
- `prepare` refuses to overwrite an existing plan file.
- Keep `--identity` unchanged across retries of one operation. It seeds the idempotency
  key.
- The CLI has no `withdraw` action. Withdraw through the workflow.

## Accept a new gate

`tests/test_promotions.py` and `tests/test_promotion_workflow.py` cover request and
decision recovery, snapshot conflicts, unchanged review content, terminal states, and the
credential boundary between jobs. They do not prove deployed OIDC trust, S3 retention,
model compatibility, or a consumer's install. A provisioned environment proves none of
these either.

Before routine use of a new gate, run one protected promotion with separate requester and
reviewer identities and small fixtures: an MPK build and a LLiMa build, as a component
and as a full package. Check each of these:

- every download and its checksum
- the publisher role is denied approval, and a role scoped to another target is denied
- catalog metadata survives the promotion
- the candidate stays protected after its source branch is purged and after a newer
  candidate supersedes it
- a terminal request stays terminal, and a retried job resumes the same request
- `sima-cli` downloads the selected build, tested apart from the model browser

Keep the record of runs, commits, candidate and request IDs, snapshots, and outcomes
outside this repository.
