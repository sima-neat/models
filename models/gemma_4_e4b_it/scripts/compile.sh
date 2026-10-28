#!/usr/bin/env bash
# Compile the prequantized checkpoint with the selected profile's settings.
# Usage: compile.sh <checkpoint_directory> <output_directory> [llima-compile options...]
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <checkpoint_directory> <output_directory> [llima-compile options...]" >&2
  exit 2
fi
arguments=("${@:3}" --jobs 20)
llima-compile "$1" -o "$2" "${arguments[@]}"

# Keep the executed options in a standalone recipe for this deployment.
checkpoint_name="$(basename "$1")"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n\n'
  printf 'MODEL_PATH=%q\n' "$checkpoint_name"
  printf 'OUTPUT_PATH=%q\n\n' "${checkpoint_name%-Safetensors}-compiled"
  printf '# hf download simaai/%s --local-dir "${MODEL_PATH}"\n\n' "$checkpoint_name"
  printf 'llima-compile "${MODEL_PATH}" -o "${OUTPUT_PATH}"'
  printf ' %q' "${arguments[@]}"
  printf ' "$@"\n'
} > "$2/compile.sh"
