#!/usr/bin/env bash
# Compile and deploy one LLiMa model into a complete registry artifact.
# Usage: run_llima_compilation.sh <model_dir> <build_profile> <compile_invocation> <output_root>
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <compile_invocation> <output_root>" >&2
  exit 2
fi
MODEL_DIR="$(cd "$1" && pwd)"
BUILD_PROFILE_ID="$2"
COMPILE_INVOCATION_ID="$3"
MODEL_ID="$(basename "$MODEL_DIR")"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
PROFILE_OUTPUT_DIR="$4/$MODEL_ID/$BUILD_PROFILE_ID"
PYTHON="${MODEL_COMPILER_PYTHON:-python3}"

command -v llima-compile >/dev/null || { echo "LLiMa is missing from the Model Compiler environment" >&2; exit 1; }
command -v llima-deploy >/dev/null || { echo "llima-deploy is missing from the Model Compiler environment" >&2; exit 1; }

component_id="$("$PYTHON" -m model_registry matrix --repository "$REPOSITORY" \
  --models "$MODEL_ID" --profiles "$BUILD_PROFILE_ID" | "$PYTHON" -c '
import json, sys
entries = [entry for entry in json.load(sys.stdin)["include"] if entry["compile_invocation_id"] == sys.argv[1]]
if len(entries) != 1 or len(entries[0]["component_ids"]) != 1:
    raise SystemExit("LLiMa requires one component per compile invocation")
print(entries[0]["component_id"])
' "$COMPILE_INVOCATION_ID")"
cache_args=()
if [[ -n "${MODEL_REGISTRY_CACHE_DIR:-}" ]]; then
  cache_args=(--cache-dir "$MODEL_REGISTRY_CACHE_DIR")
fi
checkpoint="$("$PYTHON" -m model_registry fetch "$MODEL_DIR" --component "$component_id" \
  "${cache_args[@]}" --json | "$PYTHON" -c '
import json, sys
from pathlib import Path
print(Path(json.load(sys.stdin)["path"]).parent)
')"

rm -rf "$PROFILE_OUTPUT_DIR"
mkdir -p "$(dirname "$PROFILE_OUTPUT_DIR")"
work="$(mktemp -d "$(dirname "$PROFILE_OUTPUT_DIR")/.llima-XXXXXXXX")"
trap 'rm -rf "$work"' EXIT
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BUILD_PROFILE_ID="$BUILD_PROFILE_ID" bash "$MODEL_DIR/scripts/compile.sh" "$checkpoint" "$work/compiled"
llima-deploy "$work/compiled" "$work/deployed"
"$PYTHON" -m model_registry pack-llima "$work/deployed" \
  "$PROFILE_OUTPUT_DIR/$component_id/${MODEL_ID}_llima.zip"
printf '{"scope": "compile_recipe", "started_at": "%s", "finished_at": "%s"}\n' \
  "$started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PROFILE_OUTPUT_DIR/compile-timing.json"

identity=(--model-id "$MODEL_ID" --build-profile-id "$BUILD_PROFILE_ID"
  --compile-invocation-id "$COMPILE_INVOCATION_ID" --components "$component_id")
"$PYTHON" -m model_registry record-compile-outputs "$PROFILE_OUTPUT_DIR" "${identity[@]}" >/dev/null
"$PYTHON" -m model_registry validate-compile-outputs "$PROFILE_OUTPUT_DIR/compile-outputs.json" \
  --output-root "$PROFILE_OUTPUT_DIR" "${identity[@]}"
