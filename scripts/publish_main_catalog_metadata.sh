#!/usr/bin/env bash
# Usage: publish_main_catalog_metadata.sh <clean-main-checkout>
# CI-only metadata publication. Install the producer from that Main checkout first.
set -euo pipefail

main_checkout="${1:?Usage: publish_main_catalog_metadata.sh <clean-main-checkout>}"
: "${VULCAN_ENV:?Select production}"
[[ "${VULCAN_ENV}" == production ]] || { echo 'Main metadata is production-only' >&2; exit 2; }
: "${VULCAN_ARTIFACT_REGISTRY_API_URL:?Missing production writer URL}"
: "${VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL:?Missing production reader URL}"
: "${VULCAN_ARTIFACT_REGISTRY_BUCKET:?Missing production artifact bucket}"
: "${AWS_REGION:?Missing production region}"
: "${GITHUB_REPOSITORY:?Run from GitHub Actions}"
: "${GITHUB_RUN_ID:?Run from GitHub Actions}"
: "${GITHUB_STEP_SUMMARY:?Run from GitHub Actions}"

main_revision="$(git -C "${main_checkout}" rev-parse HEAD)"
git -C "${main_checkout}" diff --exit-code --quiet
git -C "${main_checkout}" diff --cached --exit-code --quiet
args=(
  publish-catalog --repository "${main_checkout}"
  --api-url "${VULCAN_ARTIFACT_REGISTRY_API_URL}"
  --read-api-url "${VULCAN_ARTIFACT_REGISTRY_PUBLIC_READ_URL}"
  --aws-region "${AWS_REGION}"
  --bucket "${VULCAN_ARTIFACT_REGISTRY_BUCKET}"
  --environment production --branch main --base-branches main
  --git-sha "${main_revision}"
  --github-run-url "https://github.com/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
  --json
)
if [[ -n "${VULCAN_ARTIFACT_REGISTRY_KMS_KEY_ID:-}" ]]; then
  args+=(--kms-key-id "${VULCAN_ARTIFACT_REGISTRY_KMS_KEY_ID}")
fi
mkdir -p _main-catalog
model-registry "${args[@]}" > _main-catalog/receipt.json
{
  echo '## Production Main metadata'
  echo "Source revision: ${main_revision}"
  echo 'Published the Main definitions and their declared requirements using the matching producer.'
  echo 'This operation does not compile models or approve artifact selections.'
} >> "${GITHUB_STEP_SUMMARY}"
