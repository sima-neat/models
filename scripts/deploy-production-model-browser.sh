#!/usr/bin/env bash
# Usage: AWS_PROFILE=vulcan-prod scripts/deploy-production-model-browser.sh developer|customer
# Build, test and publish only the production Developer Center models/ prefix.
set -euo pipefail
mode="${1:?Usage: $0 developer|customer}"
case "$mode" in developer|customer) ;; *) echo 'Expected developer or customer' >&2; exit 2;; esac
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo 'Commit or remove working-tree changes before deploying a reproducible revision.' >&2
  exit 1
fi
[[ "$(aws sts get-caller-identity --query Account --output text)" == 069171140053 ]]
export MODEL_BROWSER_ENVIRONMENT=production MODEL_BROWSER_UI_MODE="$mode"
export DOCS_SITE_URL=https://developer.sima.ai DOCS_BASE_URL=/models/
unset MODEL_REGISTRY_API_URL MODEL_HERO_CDN_BASE_URL GITHUB_SHA
npm ci --prefix website
npm run typecheck --prefix website
node --experimental-default-type=module --test tests/browser/*.test.mjs
npm run build:portal --prefix website
npm run smoke:portal --prefix website
# Keep prior fingerprinted assets for cached pages and rollback. Switch the entry
# point only after all assets are available; never sync/delete unrelated docs.
aws s3 cp website/portal-build/ s3://sima-neat-docs-production/models/ \
  --recursive --exclude index.html --cache-control 'public,max-age=300' --only-show-errors
aws s3 cp website/portal-build/index.html s3://sima-neat-docs-production/models/index.html \
  --content-type text/html --cache-control 'public,max-age=60' --only-show-errors
aws cloudfront create-invalidation --distribution-id E3UTCTPN35EZTK --paths '/models' '/models/*'
git rev-parse HEAD
