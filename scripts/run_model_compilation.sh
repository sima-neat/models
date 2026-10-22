#!/usr/bin/env bash
# Compile one compile invocation of a model with its scripts/compile.py and record the MPKs it produced.
#
# Usage: run_model_compilation.sh <model_dir> <build_profile> <compile_invocation> <output_root>
#
# Run inside an activated Model Compiler environment with model-registry installed. Outputs land in
# <output_root>/<model_id>/<build_profile>/: one directory per component, compile-outputs.json and
# compile-timing.json.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <compile_invocation> <output_root>" >&2
  exit 2
fi
MODEL_DIR="$(cd "$1" && pwd)"
BUILD_PROFILE_ID="$2"
COMPILE_INVOCATION_ID="$3"
OUTPUT_ROOT="$4"
PYTHON="${MODEL_COMPILER_PYTHON:-python3}"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"
PROFILE_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_ID/$BUILD_PROFILE_ID"
cache_args=()
if [[ -n "${MODEL_REGISTRY_CACHE_DIR:-}" ]]; then
  cache_args=(--cache-dir "$MODEL_REGISTRY_CACHE_DIR")
fi

# Print one field of the JSON on stdin; lists print comma-separated.
json_field() {
  "$PYTHON" -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    value = value[key]
print(",".join(value) if isinstance(value, list) else value)
' "$1"
}

# Set DATASET to the one dataset a fetch command resolves across the invocation's components, or empty.
resolve_single_dataset() {
  local command="$1" result candidate
  DATASET=""
  for component in "${components[@]}"; do
    result="$("$PYTHON" -m model_registry "$command" "$MODEL_DIR" --build-profile "$BUILD_PROFILE_ID" \
      --component "$component" ${cache_args[@]+"${cache_args[@]}"} --json)"
    candidate="$(json_field path <<< "$result")"
    if [[ -n "$candidate" && -n "$DATASET" && "$candidate" != "$DATASET" ]]; then
      echo "$command resolves different datasets within $COMPILE_INVOCATION_ID; compile.py takes one" >&2
      exit 1
    fi
    DATASET="${candidate:-$DATASET}"
  done
}

invocation="$("$PYTHON" -m model_registry matrix --repository "$REPOSITORY" \
  --models "$MODEL_ID" --profiles "$BUILD_PROFILE_ID" | "$PYTHON" -c '
import json, sys
matches = [entry for entry in json.load(sys.stdin)["include"] if entry["compile_invocation_id"] == sys.argv[1]]
if len(matches) != 1:
    raise SystemExit(f"no compile invocation {sys.argv[1]!r} in this build profile")
print(json.dumps(matches[0]))
' "$COMPILE_INVOCATION_ID")"
COMPONENT_IDS="$(json_field component_ids <<< "$invocation")"
IFS=',' read -ra components <<< "$COMPONENT_IDS"

# compile.py receives the first component's source; multi-component recipes derive the rest from it.
source_path="$("$PYTHON" -m model_registry fetch "$MODEL_DIR" --component "${components[0]}" \
  ${cache_args[@]+"${cache_args[@]}"} --json | json_field path)"
dataset_args=()
resolve_single_dataset fetch-calibration
if [[ -n "$DATASET" ]]; then
  dataset_args+=(--dataset "$DATASET")
fi
resolve_single_dataset fetch-fidelity-dataset
if [[ -n "$DATASET" ]]; then
  dataset_args+=(--fidelity-dataset "$DATASET")
fi

# compile.py creates the output directory; some recipes refuse one that already exists.
rm -rf "$PROFILE_OUTPUT_DIR"
mkdir -p "$(dirname "$PROFILE_OUTPUT_DIR")"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" "$MODEL_DIR/scripts/compile.py" \
  --model "$source_path" \
  --output-dir "$PROFILE_OUTPUT_DIR" \
  --build-profile "$BUILD_PROFILE_ID" \
  --compile-invocation "$COMPILE_INVOCATION_ID" \
  ${dataset_args[@]+"${dataset_args[@]}"}
if [[ ! -d "$PROFILE_OUTPUT_DIR" ]]; then
  echo "$MODEL_DIR/scripts/compile.py wrote no output directory $PROFILE_OUTPUT_DIR" >&2
  exit 1
fi
printf '{"scope": "compile_recipe", "started_at": "%s", "finished_at": "%s"}\n' \
  "$started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PROFILE_OUTPUT_DIR/compile-timing.json"

identity=(--model-id "$MODEL_ID" --build-profile-id "$BUILD_PROFILE_ID"
  --compile-invocation-id "$COMPILE_INVOCATION_ID" --components "$COMPONENT_IDS")
if [[ ! -f "$PROFILE_OUTPUT_DIR/compile-outputs.json" ]]; then
  "$PYTHON" -m model_registry record-compile-outputs "$PROFILE_OUTPUT_DIR" "${identity[@]}" >/dev/null
fi
"$PYTHON" -m model_registry validate-compile-outputs "$PROFILE_OUTPUT_DIR/compile-outputs.json" \
  --output-root "$PROFILE_OUTPUT_DIR" "${identity[@]}"
