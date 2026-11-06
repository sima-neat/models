#!/usr/bin/env bash
# Usage: compile.sh <checkpoint_directory> <output_directory> [llima-compile options...]
llima-compile "$1" -o "$2" --jobs 20 "${@:3}"
