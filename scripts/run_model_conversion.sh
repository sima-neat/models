#!/usr/bin/env bash
# Convert a model to ONNX with its scripts/convert.py in a fresh environment built from requirements-convert.txt.
#
# Usage: run_model_conversion.sh <model_dir> <conversion> <output_dir>
#
# The conversion is an entry in the model's model.yaml. The environment is created at $CONVERSION_VENV
# (default <output_dir>-venv) with $CONVERSION_PYTHON (default python3.12); pip uses $PIP_CACHE_DIR when set.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <model_dir> <conversion> <output_dir>" >&2
  exit 2
fi
MODEL_DIR="$(cd "$1" && pwd)"
CONVERSION="$2"
OUTPUT_DIR="$3"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
CONVERSION_PYTHON="${CONVERSION_PYTHON:-python3.12}"
CONVERSION_VENV="${CONVERSION_VENV:-${OUTPUT_DIR%/}-venv}"

for file in convert.py requirements-convert.txt; do
  if [[ ! -f "$MODEL_DIR/scripts/$file" ]]; then
    echo "$MODEL_DIR/scripts/$file is required for conversion." >&2
    exit 1
  fi
done

# --isolated ignores pip environment variables, so the cache directory is passed explicitly.
pip_install=("$CONVERSION_VENV/bin/python" -E -s -m pip --isolated install)
if [[ -n "${PIP_CACHE_DIR:-}" ]]; then
  pip_install+=(--cache-dir "$PIP_CACHE_DIR")
fi

"$CONVERSION_PYTHON" -E -s -m venv "$CONVERSION_VENV"
"${pip_install[@]}" "$REPOSITORY"
"${pip_install[@]}" --only-binary=:all: -r "$MODEL_DIR/scripts/requirements-convert.txt"
"$CONVERSION_VENV/bin/python" -E -s "$MODEL_DIR/scripts/convert.py" \
  --conversion "$CONVERSION" --output-dir "$OUTPUT_DIR"
