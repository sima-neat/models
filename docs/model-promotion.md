# Promote a model

Promotion makes a tested build available for download from the Develop or Main
catalog. It reuses the existing files; it does not compile the model again.

Build-on-demand models need substantial compute resources and time to build.
They have `build-on-demand: true` in `model.yaml`, so CI builds them only when
someone requests a build. Pushing or merging their source updates the model
definition but does not start a build.

Use this promotion process to make a tested build of one of these models
available for download. Models that build automatically follow the normal
pull request and CI process. A successful compile alone does not show that a
model works correctly: test it on the target device before requesting approval.

## 1. Build and test

Run **Vulcan Model Orchestrator** on your feature branch. Fill in both `models`
and `profiles`, using the model folder name and a profile ID from `model.yaml`.
For example, Qwen3.5 0.8B uses `models=qwen3_5_0_8b` and `profiles=w4a16`.

Wait for compilation and publication to finish. Test the published build and
save links to the results. Keep the `deployment_id` from that build's
`candidate.json` receipt. This ID identifies the exact build you will promote.

## 2. Merge the model definition

Open a pull request to `develop` with your model changes and test results.
After it merges, wait for the Develop workflow to publish the updated model
page and definition. The model must still be marked as build-on-demand, and
its build requirements must match your tested build.

If you change build inputs after testing, build and test that version before
requesting promotion. Keep your feature branch until you have checked the
promoted download.

## 3. Request Develop promotion

Open [**Promote model** in GitHub Actions](https://github.com/sima-neat/models/actions/workflows/model-promotion.yml)
and choose **Run workflow**. Fill in these fields:

| Field | What to enter |
| --- | --- |
| Branch to run the workflow | `develop`, including for Main promotion and recovery actions |
| `target_branch` | `develop` |
| `operation` | `promote` |
| `model_id` | The folder name under `models/`, such as `qwen3_5_0_8b` |
| `build_profile` | The profile ID from `model.yaml`, such as `w4a16` |
| `deployment_id` | The ID from the tested build's `candidate.json` |
| `promotion_id` | Leave blank for a new request |
| `reason` | **Required.** Explain why the build is ready and include links to your test results. |

The `deployment_id` is required. Use the ID from the successfully published
build you tested.

Check the source commit, model, profile and build ID in the review
summary. A different eligible maintainer reviews and approves the request.
Wait for the final report to show **`promoted`**. Submitting a request or
approving a GitHub job is not proof that promotion finished.

## 4. Check the download

Open the model page in the Develop catalog. Download
the selected build from that page or with `sima-cli`, and check that it is the
build you tested. For a directory package, download and verify every required
file.

You can delete the feature branch after its source changes are merged, the
Develop model page includes its README and image, and the promoted download
has been checked. Promotion retains the build independently of the source
branch; it does not by itself retain the branch's published README or image.

## Promote the same build to Main

First complete Develop approval for the exact build you want to release.

The model definition must also be released to `main`. Run **Publish Main model
metadata** from `develop` and wait for it to publish Main's definitions. The
Main model and profile must be build-on-demand with requirements matching the
candidate. This step publishes metadata; it does not copy Develop's definitions
or approve a build.

Run **Promote model** again from `develop`, using `target_branch=main` and the
same `deployment_id`. Fill in the model, profile and required reason,
and leave `promotion_id` blank. Main requires its own maintainer approval.
After the report shows `promoted`, check the selected download in Main.

Main promotion reuses the same files. Publishing to Hugging Face is a separate
release step.

## If a request does not finish

Run **Promote model** from `develop` with `operation=status`, the existing
`promotion_id`, the same `target_branch`, and a reason.

- **A job failed:** rerun the failed jobs if the saved plan and request are still valid. The rerun keeps the original build and request.
- **The request expired, the target catalog changed, or the saved plan is missing or expired:** start a new workflow run and obtain a new review.
- **You no longer need a pending request:** use `operation=cancel` with its promotion ID. Cancelling the GitHub run does not cancel the promotion request.

Cancellation cannot undo a completed promotion. Keep the feature branch while
you resolve an unfinished request.

For workflow setup, approval permissions, CLI commands and acceptance
records, see [Promotion operations](model-promotion-operations.md).
