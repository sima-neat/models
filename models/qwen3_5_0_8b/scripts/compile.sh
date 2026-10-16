#!/usr/bin/env bash
# Compile the prequantized checkpoint with the selected profile's settings.
# Usage: compile.sh <checkpoint_directory> <output_directory> [llima-compile options...]
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <checkpoint_directory> <output_directory> [llima-compile options...]" >&2
  exit 2
fi
llima-compile "$1" -o "$2" \
  "${@:3}" \
  --jobs 6 --log_level INFO
