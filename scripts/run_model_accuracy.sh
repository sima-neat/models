#!/usr/bin/env bash
# Run a model's scripts/accuracy.py on the DevKit with its benchmarked MPKs and record an accuracy result.

set -euo pipefail

MODEL_DIR="$1"
BUILD_PROFILE_ID="$2"
# Benchmark result root holding each component's performance-test.json.
BENCHMARK_ROOT="$3"
OUTPUT_DIR="$4"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
PYNEAT_PYTHON="${PYNEAT_PYTHON:-$HOME/pyneat/bin/python}"
DATASET_CACHE="${DATASET_CACHE:-/media/nvme/datasets/artifacts}"
# Scratch space for the extracted dataset and accuracy wheels; removed on exit. NVMe keeps it off the root filesystem.
WORK_DIR="${WORK_DIR:-/media/nvme/neat-tests/model-accuracy-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"

SCRIPT="$MODEL_DIR/scripts/accuracy.py"
REQUIREMENTS="$MODEL_DIR/scripts/requirements-accuracy.txt"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"

# Print one field of the JSON on stdin; lists print comma-separated.
json_field() {
  "$REGISTRY_PYTHON" -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    value = value[key]
print(",".join(value) if isinstance(value, list) else value)
' "$1"
}

resolution="$("$REGISTRY_PYTHON" -m model_registry resolve-devkit-accuracy "$MODEL_DIR" \
  --build-profile "$BUILD_PROFILE_ID" --cache-dir "$DATASET_CACHE")"
if [[ "$(json_field kind <<< "$resolution")" == "none" ]]; then
  echo "$MODEL_ID declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi

mpks=()
IFS=',' read -ra components <<< "$(json_field components <<< "$resolution")"
for component in "${components[@]}"; do
  manifest="$BENCHMARK_ROOT/$component/performance-test.json"
  if [[ ! -f "$manifest" ]]; then
    echo "DevKit accuracy needs the benchmarked $component MPK, but $manifest is missing" >&2
    exit 1
  fi
  mpks+=(--mpk "$component=$(json_field model.path < "$manifest")")
done

# Unpack the cached archive, looking through a single top-level directory.
trap 'rm -rf "$WORK_DIR"' EXIT
dataset="$WORK_DIR/dataset"
rm -rf "$dataset"
mkdir -p "$dataset" "$OUTPUT_DIR"
tar -xzf "$(json_field path <<< "$resolution")" -C "$dataset"
entries=("$dataset"/*)
if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
  dataset="${entries[0]}"
fi

# Accuracy packages go in a folder beside PyNeat's environment, never into it, so its numpy stays in place.
site="$WORK_DIR/site-packages"
if [[ -f "$REQUIREMENTS" ]]; then
  "$PYNEAT_PYTHON" -m pip install --quiet --disable-pip-version-check --no-deps --only-binary=:all: \
    --target "$site" -r "$REQUIREMENTS"
fi

PYTHONPATH="$site" PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$SCRIPT" "${mpks[@]}" \
  --dataset "$dataset" --output "$OUTPUT_DIR/script-output.json" \
  2>&1 | tee "$OUTPUT_DIR/accuracy.log"

"$REGISTRY_PYTHON" -m model_registry import-accuracy \
  --repository "$REPOSITORY" \
  --model-id "$MODEL_ID" \
  --build-profile-id "$BUILD_PROFILE_ID" \
  --script-output "$OUTPUT_DIR/script-output.json" \
  --result-root "$BENCHMARK_ROOT" \
  --output "$OUTPUT_DIR/accuracy-result-v2.json"
