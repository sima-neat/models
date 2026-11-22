#!/usr/bin/env bash
# Give a build profile's DevKit accuracy datasets as directories from the board's store of unpacked datasets.
#
# Usage: unpack_benchmark_datasets.sh <model_dir> <build_profile> [--first]
#
# DATASET_STORE keeps one unpacked copy of each pinned archive in <sha256>/dataset, named by the archive's SHA-256.
# A copy is complete once <sha256>/complete exists, and a job that finds one reads no archive. A missing copy comes
# from resolve-accuracy-dataset: from the DevKit pool's shared dataset archive share at SHARED_DATASET_ARCHIVES when
# it holds the pin and gives a listing within SHARE_LISTING_SECONDS, else from the Dataset Registry into
# DATASET_CACHE, verified against the pin either way. The unpacked tree is read-only, so a model script that writes
# into its dataset fails instead of changing a later job's input. Each use touches the marker, and the script
# removes each copy unused for 30 days.
#
# Prints one JSON line per dataset: its devkit-accuracy-datasets line plus "directory", looking through a single
# top-level directory, and "source": board, share, or registry. --first prints only the first dataset, which the
# performance stage reads. Prints nothing when the profile measures no accuracy.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != --first ) ]]; then
  echo "usage: $0 <model_dir> <build_profile> [--first]" >&2
  exit 2
fi
MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
FIRST_ONLY="${3:-}"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"
DATASET_STORE="${DATASET_STORE:-$NVME_MOUNT/datasets/unpacked}"
DATASET_CACHE="${DATASET_CACHE:-$NVME_MOUNT/datasets/artifacts}"
DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"
DATASET_LOCK_SECONDS="${DATASET_LOCK_SECONDS:-1200}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
SHARED_DATASET_ARCHIVES="${SHARED_DATASET_ARCHIVES:-/mnt/dataset-private/current}"
SHARE_LISTING_SECONDS="${SHARE_LISTING_SECONDS:-20}"
# A copy unused for 30 days is removed.
STORE_RETENTION_MINUTES=$((30 * 24 * 60))
# An archive copied from the share waits here for its unpack; removed on exit.
SHARE_COPY="$DATASET_STORE/.share-copy"

# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

pins="$("$REGISTRY_PYTHON" -m model_registry devkit-accuracy-datasets "$MODEL_DIR" --build-profile "$BUILD_PROFILE_ID")"
if [[ -z "$pins" ]]; then
  exit 0
fi
if [[ -n "$FIRST_ONLY" ]]; then
  pins="${pins%%$'\n'*}"
fi

mkdir -p "$DATASET_STORE" "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
# Concurrent jobs on one board take turns until each exits, so none unpacks an archive twice or removes a copy
# another job reads.
exec 9> "$DATASET_LOCK"
flock --wait "$DATASET_LOCK_SECONDS" 9 ||
  { echo "Waited $DATASET_LOCK_SECONDS s for $DATASET_LOCK without getting it." >&2; exit 1; }
trap 'rm -rf "$SHARE_COPY"' EXIT

while IFS= read -r pin; do
  entry="$DATASET_STORE/$(json_field sha256 <<< "$pin")"
  origin=board
  if [[ ! -f "$entry/complete" ]]; then
    # A folder without the marker is an interrupted unpack, whose tree may already be read-only.
    if [[ -d "$entry" ]]; then
      chmod -R u+w "$entry"
    fi
    rm -rf "$entry" "$SHARE_COPY"
    mkdir -p "$entry/dataset" "$SHARE_COPY"
    # A stale share blocks every read of it. One that gives no listing in time is left out, so the archive comes
    # from the Dataset Registry.
    shared_archives="$SHARED_DATASET_ARCHIVES"
    listing=0
    timeout --signal=TERM --kill-after=5s "$SHARE_LISTING_SECONDS" ls "$shared_archives" > /dev/null 2>&1 || listing=$?
    if [[ $listing -eq 124 || $listing -eq 137 ]]; then
      echo "The shared dataset archive share gave no listing in $SHARE_LISTING_SECONDS s." \
        "Resolving through the Dataset Registry." >&2
      shared_archives="$SHARE_COPY/no-share"
    fi
    # The resolution gets no stdin, which holds the remaining pins.
    archive="$(timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry \
      resolve-accuracy-dataset "$MODEL_DIR" --dataset "$(json_field name <<< "$pin")" --cache-dir "$DATASET_CACHE" \
      --shared-archives "$shared_archives" --copy-dir "$SHARE_COPY" < /dev/null)"
    origin="$(json_field source <<< "$archive")"
    tar -xzf "$(json_field path <<< "$archive")" -C "$entry/dataset"
    chmod -R a-w "$entry/dataset"
  fi
  # Created only after a finished unpack. Its time records the last use, which the removal below reads.
  touch "$entry/complete"
  directory="$entry/dataset"
  entries=("$directory"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    directory="${entries[0]}"
  fi
  "$REGISTRY_PYTHON" -c 'import json, sys
print(json.dumps({**json.loads(sys.argv[1]), "directory": sys.argv[2], "source": sys.argv[3]}, sort_keys=True))' \
    "$pin" "$directory" "$origin"
done <<< "$pins"

# Remove each copy whose marker is older than the retention, and each interrupted unpack. Its tree is read-only, so
# restore write permission first.
for entry in "$DATASET_STORE"/*; do
  if [[ -d "$entry" && -z "$(find "$entry/complete" -mmin "-$STORE_RETENTION_MINUTES" 2>/dev/null)" ]]; then
    chmod -R u+w "$entry"
    rm -rf "$entry"
  fi
done
