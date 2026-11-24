#!/usr/bin/env bash
# Unpack each DevKit accuracy dataset of a build profile into the job's dataset folder, and report each one's source.
#
# Usage: cache_accuracy_dataset.sh <model_dir> <build_profile> <dataset_dir>
#
# Runs with dataset credentials and no model code. It unpacks through unpack_benchmark_datasets.sh, so the stage
# scripts later find each dataset in <dataset_dir> and need no credentials. The job summary names each dataset's
# source: the shared dataset archive share or the Dataset Registry.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <dataset_dir>" >&2
  exit 2
fi
MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
DATASET_DIR="$3"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
NVME_MOUNT="${NVME_MOUNT:-/media/nvme}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"

if ! mountpoint -q "$NVME_MOUNT"; then
  echo "$NVME_MOUNT is not mounted; refusing to cache datasets on the root filesystem" >&2
  exit 1
fi
# One JSON line per dataset; none when the profile measures no accuracy.
datasets="$("$(dirname "${BASH_SOURCE[0]}")/unpack_benchmark_datasets.sh" "$MODEL_DIR" "$BUILD_PROFILE_ID" \
  "$DATASET_DIR")"
if [[ -z "$datasets" ]]; then
  echo "$(basename "$MODEL_DIR") declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi
# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}
{
  echo "### Accuracy datasets"
  echo
  echo "| Dataset | SHA-256 | Bytes | Source | Directory |"
  echo "|---|---|---|---|---|"
} >> "$SUMMARY"
while IFS= read -r dataset; do
  case "$(json_field source <<< "$dataset")" in
    share) origin="shared dataset archive share" ;;
    registry) origin="Dataset Registry" ;;
  esac
  row="| $(json_field name <<< "$dataset") | \`$(json_field sha256 <<< "$dataset")\` | \
$(json_field size_bytes <<< "$dataset") | $origin | \`$(json_field directory <<< "$dataset")\` |"
  echo "$row"
  echo "$row" >> "$SUMMARY"
done <<< "$datasets"
