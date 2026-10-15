#!/usr/bin/env bash
# Compile Qwen3.5-2B with 4K context and 448x448 image input.
# Usage: compile.sh <checkpoint_directory> <output_directory>
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <checkpoint_directory> <output_directory>" >&2
  exit 2
fi
llima-compile "$1" -o "$2" \
  --max_num_tokens 4096 \
  --language_group_size 128 \
  --input_height 448 --input_width 448 \
  --jobs 12 --log_level INFO
