#!/usr/bin/env bash
# Usage: AWS_PROFILE=<production operator profile> PRODUCTION_AWS_ACCOUNT_ID=<account> \
#   PRODUCTION_DOCS_BUCKET=<bucket> PRODUCTION_CLOUDFRONT_DISTRIBUTION_ID=<distribution> \
#   scripts/deploy-production-model-browser.sh developer|customer
# The customer view also needs MODEL_BROWSER_CATALOG_SNAPSHOT=<catalog_revision>:<canonical_sha256>.
# Build, test and publish only the production Developer Center models/ prefix.
# The script refuses to publish unless the active credentials belong to the given
# account and the bucket is encrypted with a key of that account.
set -euo pipefail
mode="${1:?Usage: $0 developer|customer}"
case "$mode" in developer|customer) ;; *) echo 'Expected developer or customer' >&2; exit 2;; esac
: "${PRODUCTION_AWS_ACCOUNT_ID:?PRODUCTION_AWS_ACCOUNT_ID is required}"
: "${PRODUCTION_DOCS_BUCKET:?PRODUCTION_DOCS_BUCKET is required}"
: "${PRODUCTION_CLOUDFRONT_DISTRIBUTION_ID:?PRODUCTION_CLOUDFRONT_DISTRIBUTION_ID is required}"
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo 'Commit or remove working-tree changes before deploying a reproducible revision.' >&2
  exit 1
fi
[[ "$(aws sts get-caller-identity --query Account --output text)" == "$PRODUCTION_AWS_ACCOUNT_ID" ]]
kms_key="$(aws s3api get-bucket-encryption --bucket "$PRODUCTION_DOCS_BUCKET" \
  --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.KMSMasterKeyID' --output text)"
[[ "$kms_key" == "arn:aws:kms:us-west-2:$PRODUCTION_AWS_ACCOUNT_ID:key/"* ]]
export MODEL_BROWSER_ENVIRONMENT=production MODEL_BROWSER_UI_MODE="$mode"
export DOCS_SITE_URL=https://developer.sima.ai DOCS_BASE_URL=/models/
unset MODEL_REGISTRY_API_URL MODEL_HERO_CDN_BASE_URL GITHUB_SHA
npm ci --prefix website
npm run typecheck --prefix website
node --test tests/browser/*.test.mjs
npm run build:portal --prefix website
npm run smoke:portal --prefix website
# Keep prior fingerprinted assets for cached pages and rollback. Switch the entry
# point only after all assets are available; never sync/delete unrelated docs.
aws s3 cp website/portal-build/ "s3://$PRODUCTION_DOCS_BUCKET/models/" \
  --recursive --exclude index.html --sse aws:kms --sse-kms-key-id "$kms_key" --cache-control 'public,max-age=300' --only-show-errors
aws s3 cp website/portal-build/index.html "s3://$PRODUCTION_DOCS_BUCKET/models/index.html" \
  --sse aws:kms --sse-kms-key-id "$kms_key" --content-type text/html --cache-control 'public,max-age=60' --only-show-errors
aws cloudfront create-invalidation --distribution-id "$PRODUCTION_CLOUDFRONT_DISTRIBUTION_ID" --paths '/models' '/models/*'
git rev-parse HEAD
