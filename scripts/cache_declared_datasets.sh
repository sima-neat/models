#!/usr/bin/env bash
# Resolve each component's declared dataset into the persistent NVMe cache.

set -euo pipefail

MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
COMPONENT_IDS="$3"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"
DATASET_ROOT="${DATASET_ROOT:-$NVME_MOUNT/datasets}"
LOCK_WAIT_SECONDS="${LOCK_WAIT_SECONDS:-600}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

prepare_cache_root() {
  if ! mountpoint -q "$NVME_MOUNT"; then
    echo "$NVME_MOUNT is not mounted; refusing to cache datasets on the root filesystem" >&2
    exit 1
  fi
  mkdir -p "$DATASET_ROOT/artifacts" "$DATASET_ROOT/locks"
}

# One resolution at a time per board, so concurrent jobs never download the same dataset twice.
resolve_dataset() {
  local component="$1"
  flock --verbose --wait "$LOCK_WAIT_SECONDS" --close "$DATASET_ROOT/locks/prepare.lock" \
    timeout --verbose --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" \
    "$REGISTRY_PYTHON" -m model_registry fetch-accuracy-dataset "$MODEL_DIR" \
      --build-profile "$BUILD_PROFILE_ID" \
      --component "$component" \
      --cache-dir "$DATASET_ROOT/artifacts" \
      --json
}

# Dataset Registry rewrites its receipt only when it downloads the archive.
summary_row() {
  local component="$1" result="$2" started="$3"
  local path receipt dataset cache="hit"
  if [[ "$(jq -er '.kind' <<< "$result")" == "none" ]]; then
    echo "| $component | none | | | | |"
    return
  fi
  path="$(jq -er '.path' <<< "$result")"
  receipt="$(dirname "$path")/resolution-receipt.json"
  dataset="$(jq -er '"\(.dataset_id)/\(.variant)"' "$receipt")"
  if [[ "$receipt" -nt "$started" ]]; then
    cache="downloaded"
  fi
  echo "| $component | $dataset | \`$path\` | \`$(jq -er '.sha256' <<< "$result")\` | $(jq -er '.size_bytes' <<< "$result") | $cache |"
}

main() {
  local components component result row
  prepare_cache_root
  {
    echo "### Cached datasets"
    echo
    echo "| Component | Dataset | Path | SHA-256 | Bytes | Cache |"
    echo "|---|---|---|---|---|---|"
  } >> "$SUMMARY"
  IFS=',' read -ra components <<< "$COMPONENT_IDS"
  for component in "${components[@]}"; do
    result="$(resolve_dataset "$component")"
    row="$(summary_row "$component" "$result" "$STARTED")"
    echo "$row"
    echo "$row" >> "$SUMMARY"
  done
}

# Receipts newer than this marker were written by a download in this run.
STARTED="$(mktemp)"
trap 'rm -f "$STARTED"' EXIT
main
