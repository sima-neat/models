#!/usr/bin/env bash
# Measure a build profile's whole-model accuracy on the DevKit with the model's scripts/accuracy.py.
#
# Usage: run_model_accuracy.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
#
# <compiled_dir> holds <model_id>/<build_profile>/compile-outputs.json and the MPKs it lists.
# unpack_benchmark_datasets.sh unpacks each declared dataset into DATASET_DIR, or reuses the folder an earlier step of
# the job unpacked there. The script takes them all in one run, with --limit when model.yaml sets accuracy.samples.
# It writes model-report.json into <output_dir> with tools/model_report; check-report then holds it to model.yaml
# and records the board in environment.json.

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
# Scratch space for accuracy wheels; removed on exit. NVMe keeps it off the root filesystem.
WORK_DIR="${WORK_DIR:-/media/nvme/neat-tests/model-accuracy-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"
# The unpacked datasets, inside WORK_DIR unless the caller shares them between steps.
DATASET_DIR="${DATASET_DIR:-$WORK_DIR/datasets}"

SCRIPT="$MODEL_DIR/scripts/accuracy.py"
REQUIREMENTS="$MODEL_DIR/scripts/requirements-accuracy.txt"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"

# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

trap 'rm -rf "$WORK_DIR"' EXIT
# One JSON line per dataset with its directory; none when the profile measures no accuracy.
resolutions="$("$(dirname "${BASH_SOURCE[0]}")/unpack_benchmark_datasets.sh" "$MODEL_DIR" "$BUILD_PROFILE_ID" \
  "$DATASET_DIR")"
if [[ -z "$resolutions" ]]; then
  echo "$MODEL_ID declares no DevKit accuracy for $BUILD_PROFILE_ID."
  exit 0
fi

# The script takes each MPK as <component>=<path>.
listed="$("$REGISTRY_PYTHON" -m model_registry model-component-mpks "$COMPILED_DIR" \
  --repository "$REPOSITORY" --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID")"
mpks=()
while IFS= read -r input; do mpks+=(--mpk "$input"); done <<< "$listed"

# Supporting files travel with the compiled package and are verified before the model script reads them.
resources=()
runtime_dir="$("$REGISTRY_PYTHON" -m model_registry model-runtime-directory "$COMPILED_DIR" \
  --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID")"
if [[ -n "$runtime_dir" ]]; then resources+=(--resources "$runtime_dir"); fi

# The script takes each dataset as <name>=<path>, named as in model.yaml.
mkdir -p "$OUTPUT_DIR"
datasets=()
while IFS= read -r resolution; do
  datasets+=(--dataset "$(json_field name <<< "$resolution")=$(json_field directory <<< "$resolution")")
done <<< "$resolutions"

# accuracy.samples in model.yaml caps the samples the script evaluates; unset, it evaluates every sample.
samples="$("$REGISTRY_PYTHON" -c 'import json, sys; print(json.loads(sys.argv[1]).get("samples", ""))' \
  "${resolutions%%$'\n'*}")"
limit=()
if [[ -n "$samples" ]]; then
  limit=(--limit "$samples")
fi

# Accuracy packages go in a folder beside PyNeat's environment, never into it, so its numpy stays in place.
site="$WORK_DIR/site-packages"
if [[ -f "$REQUIREMENTS" ]]; then
  "$PYNEAT_PYTHON" -m pip install --quiet --disable-pip-version-check --no-deps --only-binary=:all: \
    --target "$site" -r "$REQUIREMENTS"
fi

PYTHONPATH="$REPOSITORY/tools:$site" PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$SCRIPT" "${mpks[@]}" \
  "${datasets[@]}" "${resources[@]}" "${limit[@]}" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/accuracy.log"
"$REGISTRY_PYTHON" -m model_registry check-report "$OUTPUT_DIR" --repository "$REPOSITORY" \
  --stage accuracy --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID"
