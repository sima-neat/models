#!/usr/bin/env bash
# Usage: compile.sh <checkpoint_directory> <output_directory> [Whisper options...]
python3 "$(dirname "$0")/compile.py" "$1" "$2" --num_processes 20 "${@:3}"
