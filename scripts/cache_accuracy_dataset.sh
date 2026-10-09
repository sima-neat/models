#!/usr/bin/env bash
# Resolve a build profile's DevKit accuracy dataset into the persistent NVMe cache.
#
# Usage: cache_accuracy_dataset.sh <model_dir> <build_profile>
#
# Runs with dataset credentials and no model code, so run_model_accuracy.sh later reads the archive from the
# cache without them.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <model_dir> <build_profile>" >&2
  exit 2
fi
MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"
DATASET_CACHE="${DATASET_CACHE:-$NVME_MOUNT/datasets/artifacts}"
DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"
DATASET_LOCK_SECONDS="${DATASET_LOCK_SECONDS:-1200}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

if ! mountpoint -q "$NVME_MOUNT"; then
  echo "$NVME_MOUNT is not mounted; refusing to cache datasets on the root filesystem" >&2
  exit 1
fi
mkdir -p "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
# Receipts newer than this marker were written by a download in this run.
started="$(mktemp)"
trap 'rm -f "$started"' EXIT
# The board-wide dataset lock run_model_accuracy.sh also takes, so concurrent jobs never download one archive twice.
result="$(flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK" \
  timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry resolve-devkit-accuracy "$MODEL_DIR" \
  --build-profile "$BUILD_PROFILE_ID" --cache-dir "$DATASET_CACHE")"
# The DevKit has no jq; read the resolution with the registry Python.
field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$result" "$1"
}
if [[ "$(field kind)" == "none" ]]; then
  echo "$(basename "$MODEL_DIR") declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi
path="$(field path)"
cache="hit"
if [[ "$(dirname "$path")/resolution-receipt.json" -nt "$started" ]]; then
  cache="downloaded"
fi
row="| $(field accuracy_id) | \`$path\` | \`$(field sha256)\` | $(field size_bytes) | $cache |"
echo "$row"
{
  echo "### Accuracy dataset"
  echo
  echo "| Accuracy | Path | SHA-256 | Bytes | Cache |"
  echo "|---|---|---|---|---|"
  echo "$row"
} >> "$SUMMARY"
