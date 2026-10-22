# Promoting a model to Develop

## When you need promotion

Use promotion for models with `build-on-demand: true` in `model.yaml`.
These models are costly to build, so they compile only when you request a
build. Merging their source into `develop` publishes the model definition,
but does not build it or approve a build for download.

Promotion makes an existing, validated build available in the Develop
catalog without compiling it again. Models that build automatically use the
normal pull request and CI process and do not need this workflow.

This guide covers promotion to Develop in staging. Release to Main is a
separate process.

## Build and validate on your branch

1. In GitHub Actions, run **Vulcan Model Orchestrator** on your feature branch. Select both the model and the build profile you want to build.
2. Wait for the build and publication to finish. Test the result on your target device and keep links to your validation results. A successful compile alone does not show that the model works correctly.
3. Open a pull request to `develop` with the model changes and validation links.

For example, Qwen3.5 0.8B uses `models=qwen3_5_0_8b` and
`profiles=w4a16`. See [on-demand builds](adding-a-model.md#on-demand-builds)
for more about requesting a build.

## Merge before requesting promotion

Merge the pull request once review and the required checks pass. Then wait
for the Develop workflow to publish the updated model definition. The
published definition must match the build you intend to promote.

**Keep the feature branch until promotion is complete.** If GitHub offers to
delete it after merging, leave it in place for now.

If you change build inputs after validation, build and validate that version
before requesting promotion. Promotion reuses the selected build; it does
not rebuild it from the merged source.

## Request promotion

Open [**Promote model** in GitHub Actions](https://github.com/sima-neat/models/actions/workflows/model-promotion.yml)
and choose **Run workflow**:

- Select the `develop` branch and the `promote` operation.
- Set `model_id` to the model folder name under `models/` and `build_profile` to the profile ID in its `model.yaml`.
- Add a short reason and links to your validation results.
- Leave `promotion_id` blank. Leave `deployment_id` blank to use the newest matching build in staging, or enter the `deployment_id` from your build's `candidate.json` to select that exact build.

Check the source commit and selected build in the workflow summary. Automatic
selection looks across branches, so confirm that it found the build you tested.
A maintainer reviews the request and approves it. Wait for the final report
to show **`promoted`**; submitting the request is not completion.

## Delete your branch after checking Develop

Delete the original feature branch only after:

- Its source changes are merged into `develop` and the updated model page is published, including its README and image.
- The promotion report shows `promoted` for the intended build.
- You can download the intended build from the Develop model page or with `sima-cli`.

The promoted build is retained independently of the original branch. Merging
also preserves the source and model documentation; promotion alone does not
preserve the branch's published README or image.

If promotion fails or expires, keep the branch and follow the workflow report
to retry or submit a new request. Cancelling a GitHub workflow does not cancel
its promotion request: use **Promote model** with `status` to check it, then
`cancel` with its promotion ID if you no longer need it.
