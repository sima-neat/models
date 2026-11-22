#!/usr/bin/env bash
# Unpack a build profile's DevKit accuracy datasets into a job's dataset folder, once per job.
#
# Usage: unpack_benchmark_datasets.sh <model_dir> <build_profile> <dataset_dir> [--first]
#
# Each dataset is unpacked into <dataset_dir>/<name>, and <name>.unpacked beside it records the archive's SHA-256 and
# source after a complete unpack. A later call with the same <dataset_dir> finds that marker and reads no archive.
# The archive comes from the DevKit pool's shared dataset archive share at SHARED_DATASET_ARCHIVES, read in place, when
# the share's index lists the pin and the share gives a listing within SHARE_LISTING_SECONDS. Otherwise the Dataset
# Registry resolves it into DATASET_CACHE, one resolution at a time per board. Either source is checked against the
# pin's SHA-256 and size. The caller removes <dataset_dir> when the job ends.
#
# Prints one JSON line per dataset: its devkit-accuracy-datasets line plus "directory", looking through a single
# top-level directory, and "source": share or registry. --first unpacks and prints only the first dataset, which the
# performance stage reads. Prints nothing when the profile measures no accuracy.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ( $# -eq 4 && "$4" != --first ) ]]; then
  echo "usage: $0 <model_dir> <build_profile> <dataset_dir> [--first]" >&2
  exit 2
fi
MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
DATASET_DIR="$3"
FIRST_ONLY="${4:-}"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"
DATASET_CACHE="${DATASET_CACHE:-$NVME_MOUNT/datasets/artifacts}"
DATASET_LOCK="${DATASET_LOCK:-$NVME_MOUNT/datasets/locks/prepare.lock}"
DATASET_LOCK_SECONDS="${DATASET_LOCK_SECONDS:-1200}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
SHARED_DATASET_ARCHIVES="${SHARED_DATASET_ARCHIVES:-/mnt/dataset-private/current}"
SHARE_LISTING_SECONDS="${SHARE_LISTING_SECONDS:-20}"

# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

# Unpack dataset $1 into folder $2 from the shared dataset archive share, or fail. A stale share blocks every read of
# it, so each access sits under a timeout, and a share that gives no listing in time is left out.
unpack_from_share() {
  local status=0
  timeout --signal=TERM --kill-after=5s "$SHARE_LISTING_SECONDS" ls "$SHARED_DATASET_ARCHIVES" > /dev/null 2>&1 ||
    status=$?
  if (( status == 124 || status == 137 )); then
    echo "The shared dataset archive share gave no listing in $SHARE_LISTING_SECONDS s." \
      "Resolving through the Dataset Registry." >&2
    return 1
  fi
  status=0
  # The unpack gets no stdin, which holds the remaining pins.
  timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry \
    unpack-shared-dataset "$MODEL_DIR" --dataset "$1" --shared-archives "$SHARED_DATASET_ARCHIVES" \
    --destination "$2" < /dev/null || status=$?
  if (( status == 124 || status == 137 )); then
    echo "The shared dataset archive share did not give $1 in $RESOLVE_TIMEOUT_SECONDS s." \
      "Resolving it through the Dataset Registry." >&2
  fi
  return "$status"
}

pins="$("$REGISTRY_PYTHON" -m model_registry devkit-accuracy-datasets "$MODEL_DIR" --build-profile "$BUILD_PROFILE_ID")"
if [[ -z "$pins" ]]; then
  exit 0
fi
if [[ -n "$FIRST_ONLY" ]]; then
  pins="${pins%%$'\n'*}"
fi

while IFS= read -r pin; do
  name="$(json_field name <<< "$pin")"
  sha256="$(json_field sha256 <<< "$pin")"
  target="$DATASET_DIR/$name"
  # Written only after a complete unpack, so a failed one leaves no marker.
  marker="$target.unpacked"
  unpacked="" origin=""
  if [[ -f "$marker" ]]; then
    read -r unpacked origin < "$marker"
  fi
  if [[ "$unpacked" != "$sha256" ]]; then
    rm -rf "$target" "$marker"
    origin=share
    if ! unpack_from_share "$name" "$target"; then
      origin=registry
      rm -rf "$target"
      mkdir -p "$target" "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
      # Concurrent jobs on one board wait for a single download instead of fetching the same archive twice.
      archive="$(flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK" \
        timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry \
        resolve-accuracy-dataset "$MODEL_DIR" --dataset "$name" --cache-dir "$DATASET_CACHE" < /dev/null)"
      tar -xzf "$archive" -C "$target"
    fi
    echo "$sha256 $origin" > "$marker"
  fi
  directory="$target"
  entries=("$target"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    directory="${entries[0]}"
  fi
  "$REGISTRY_PYTHON" -c 'import json, sys
print(json.dumps({**json.loads(sys.argv[1]), "directory": sys.argv[2], "source": sys.argv[3]}, sort_keys=True))' \
    "$pin" "$directory" "$origin"
done <<< "$pins"
