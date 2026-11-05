#!/usr/bin/env bash
# Usage: ARTIFACT_BUCKET=<bucket> [KMS_KEY_ID=<key>] scripts/publish_registry_state.sh
# Run from the repository root in the main-branch GitHub Actions workflow.
set -euo pipefail

if [[ "${GITHUB_REF:-}" != refs/heads/main ]]; then
  echo "Registry state may only be published from main." >&2
  exit 1
fi
: "${ARTIFACT_BUCKET:?ARTIFACT_BUCKET is required}"
jq -e '
  type == "object" and
  (keys == ["current_state", "default_branch"]) and
  (.current_state == "GA" or .current_state == "Beta" or .current_state == "Alpha") and
  (.default_branch == "develop" or .default_branch == "main")
' state.json >/dev/null

encryption=(--sse aws:kms)
if [[ -n "${KMS_KEY_ID:-}" ]]; then
  encryption+=(--sse-kms-key-id "$KMS_KEY_ID")
fi
aws s3 cp state.json "s3://$ARTIFACT_BUCKET/models/state.json" \
  --content-type application/json \
  --cache-control no-store \
  "${encryption[@]}"
published_state="$(mktemp)"
trap 'rm -f "$published_state"' EXIT
aws s3 cp "s3://$ARTIFACT_BUCKET/models/state.json" "$published_state"
diff -u state.json "$published_state"
