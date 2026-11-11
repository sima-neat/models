#!/usr/bin/env bash
# Usage: compile.sh <checkpoint_directory> <output_directory> --draft_model_path <directory> [llima-compile options...]
llima-compile "$1" -o "$2" -c "$(dirname "$0")/config.py" --jobs 20 "${@:3}"
