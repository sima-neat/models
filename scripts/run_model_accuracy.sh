#!/usr/bin/env bash
# Measure a build profile's whole-model accuracy on the DevKit with the model's scripts/accuracy.py.
#
# Usage: run_model_accuracy.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
#
# <compiled_dir> holds <model_id>/<build_profile>/compile-outputs.json and the MPKs it lists. Each declared
# dataset resolves into DATASET_CACHE, one resolution at a time per board, and the script takes them all in one
# run. It writes model-report.json into <output_dir> with tools/model_report; check-report then holds it to
# model.yaml and records the board in environment.json.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <compiled_dir> <output_dir>" >&2
  exit 2
fi
MODEL_DIR="$(cd "$1" && pwd)"
BUILD_PROFILE_ID="$2"
COMPILED_DIR="$3"
OUTPUT_DIR="$4"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
PYNEAT_PYTHON="${PYNEAT_PYTHON:-$HOME/pyneat/bin/python}"
DATASET_CACHE="${DATASET_CACHE:-/media/nvme/datasets/artifacts}"
DATASET_LOCK="${DATASET_LOCK:-/media/nvme/datasets/locks/prepare.lock}"
DATASET_LOCK_SECONDS="${DATASET_LOCK_SECONDS:-1200}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
# Scratch space for the extracted dataset and accuracy wheels; removed on exit. NVMe keeps it off the root filesystem.
WORK_DIR="${WORK_DIR:-/media/nvme/neat-tests/model-accuracy-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"

SCRIPT="$MODEL_DIR/scripts/accuracy.py"
REQUIREMENTS="$MODEL_DIR/scripts/requirements-accuracy.txt"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"

# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

mkdir -p "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
# Concurrent jobs on one board wait for a single download instead of fetching the same archive twice.
# One JSON line per dataset; none when the profile measures no accuracy.
resolutions="$(flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK" \
  timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry resolve-devkit-accuracy "$MODEL_DIR" \
  --build-profile "$BUILD_PROFILE_ID" --cache-dir "$DATASET_CACHE")"
if [[ -z "$resolutions" ]]; then
  echo "$MODEL_ID declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi

# The script takes each MPK as <component>=<path>.
listed="$("$REGISTRY_PYTHON" -m model_registry model-component-mpks "$COMPILED_DIR" \
  --repository "$REPOSITORY" --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID")"
mpks=()
while IFS= read -r input; do mpks+=(--mpk "$input"); done <<< "$listed"

# Unpack each cached archive, looking through a single top-level directory. The script takes each dataset as
# <name>=<path>, named as in model.yaml.
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$OUTPUT_DIR"
datasets=()
while IFS= read -r resolution; do
  name="$(json_field name <<< "$resolution")"
  dataset="$WORK_DIR/datasets/$name"
  rm -rf "$dataset"
  mkdir -p "$dataset"
  tar -xzf "$(json_field path <<< "$resolution")" -C "$dataset"
  entries=("$dataset"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    dataset="${entries[0]}"
  fi
  datasets+=(--dataset "$name=$dataset")
done <<< "$resolutions"

# Accuracy packages go in a folder beside PyNeat's environment, never into it, so its numpy stays in place.
site="$WORK_DIR/site-packages"
if [[ -f "$REQUIREMENTS" ]]; then
  "$PYNEAT_PYTHON" -m pip install --quiet --disable-pip-version-check --no-deps --only-binary=:all: \
    --target "$site" -r "$REQUIREMENTS"
fi

PYTHONPATH="$REPOSITORY/tools:$site" PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$SCRIPT" "${mpks[@]}" \
  "${datasets[@]}" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/accuracy.log"
"$REGISTRY_PYTHON" -m model_registry check-report "$OUTPUT_DIR" --repository "$REPOSITORY" \
  --stage accuracy --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID"
