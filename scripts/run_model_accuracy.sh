#!/usr/bin/env bash
# Run a model's scripts/accuracy.py on the DevKit with its compiled MPKs and record an accuracy result.
#
# Usage: run_model_accuracy.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
#
# <compiled_dir> holds <model_id>/<build_profile>/compile-outputs.json and the MPKs it lists. The declared
# dataset resolves into DATASET_CACHE, one resolution at a time per board.

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
SOURCE_SHA="${SOURCE_SHA:-$(git -C "$REPOSITORY" rev-parse HEAD)}"
OUTPUTS="$COMPILED_DIR/$MODEL_ID/$BUILD_PROFILE_ID/compile-outputs.json"

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

mkdir -p "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
# Concurrent jobs on one board wait for a single download instead of fetching the same archive twice.
resolution="$(flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK" \
  timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry resolve-devkit-accuracy "$MODEL_DIR" \
  --build-profile "$BUILD_PROFILE_ID" --cache-dir "$DATASET_CACHE")"
if [[ "$(json_field kind <<< "$resolution")" == "none" ]]; then
  echo "$MODEL_ID declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi

mpks=()
IFS=',' read -ra components <<< "$(json_field components <<< "$resolution")"
for component in "${components[@]}"; do
  mpk="$("$REGISTRY_PYTHON" -c '
import json, sys
from pathlib import Path

outputs = Path(sys.argv[1])
paths = [item["mpk_path"] for item in json.loads(outputs.read_text())["outputs"] if item["component_id"] == sys.argv[2]]
if len(paths) != 1:
    raise SystemExit(f"DevKit accuracy needs the compiled {sys.argv[2]} MPK, but {outputs} does not list it")
print(outputs.parent / paths[0])
' "$OUTPUTS" "$component")"
  mpks+=(--mpk "$component=$mpk")
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
  --compiled-dir "$COMPILED_DIR" \
  --git-sha "$SOURCE_SHA" \
  --output "$OUTPUT_DIR/accuracy-result.json"
