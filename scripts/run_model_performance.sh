#!/usr/bin/env bash
# Measure a build profile's whole-model performance on the DevKit with the model's scripts/performance.py.
#
# Usage: run_model_performance.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
#
# <compiled_dir> holds <model_id>/<build_profile>/compile-outputs.json and the MPKs it lists. The model's first
# accuracy dataset, already in DATASET_CACHE, supplies the benchmark input. A model whose CPU work reads its package
# resources gets them as --resources, and scripts/requirements-performance.txt adds the wheels that work needs. The
# script writes model-report.json into <output_dir> with tools/model_report; check-report then holds it to model.yaml
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
LLIMA_MOLE_PYTHON="${LLIMA_MOLE_PYTHON:-$HOME/sima-mole-venv/bin/python}"
DATASET_CACHE="${DATASET_CACHE:-/media/nvme/datasets/artifacts}"
DATASET_LOCK="${DATASET_LOCK:-/media/nvme/datasets/locks/prepare.lock}"
DATASET_LOCK_SECONDS="${DATASET_LOCK_SECONDS:-1200}"
RESOLVE_TIMEOUT_SECONDS="${RESOLVE_TIMEOUT_SECONDS:-600}"
# Scratch space for the extracted dataset, resources and wheels; removed on exit. NVMe keeps it off the root filesystem.
WORK_DIR="${WORK_DIR:-/media/nvme/neat-tests/model-performance-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}}"
REQUIREMENTS="$MODEL_DIR/scripts/requirements-performance.txt"
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"
trap 'rm -rf "$WORK_DIR"' EXIT

# A single-component model takes its MPK; a multi-component model takes each as <component>=<path>.
listed="$("$REGISTRY_PYTHON" -m model_registry model-component-artifacts "$COMPILED_DIR" \
  --repository "$REPOSITORY" --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID")"
mapfile -t inputs <<< "$listed"
mpks=()
if [[ ${#inputs[@]} -eq 1 ]]; then
  mpks=(--mpk "${inputs[0]#*=}")
else
  for input in "${inputs[@]}"; do mpks+=(--mpk "$input"); done
fi

# Print one field of the JSON on stdin; the DevKit has no jq.
json_field() {
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

backend="$("$REGISTRY_PYTHON" -c '
import sys
from model_registry.model import load_model
print("llima" if load_model(sys.argv[1])["components"][0]["interface"]["format"] == "huggingface" else "core")
' "$MODEL_DIR")"

# The script takes the first accuracy dataset as <name>=<path>, unpacked and looking through a single top-level
# directory, and picks its input from it. A model without an accuracy dataset gets none.
dataset=()
if [[ "$backend" == core ]]; then
mkdir -p "$DATASET_CACHE" "$(dirname "$DATASET_LOCK")"
resolutions="$(flock --wait "$DATASET_LOCK_SECONDS" --close "$DATASET_LOCK" \
  timeout --signal=TERM --kill-after=30s "$RESOLVE_TIMEOUT_SECONDS" "$REGISTRY_PYTHON" -m model_registry resolve-devkit-accuracy "$MODEL_DIR" \
  --build-profile "$BUILD_PROFILE_ID" --cache-dir "$DATASET_CACHE")"
dataset=()
if [[ -n "$resolutions" ]]; then
  resolution="${resolutions%%$'\n'*}"
  name="$(json_field name <<< "$resolution")"
  directory="$WORK_DIR/datasets/$name"
  rm -rf "$directory"
  mkdir -p "$directory"
  tar -xzf "$(json_field path <<< "$resolution")" -C "$directory"
  entries=("$directory"/*)
  if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
    directory="${entries[0]}"
  fi
  dataset=(--dataset "$name=$directory")
fi

# The model's pinned resources, laid out as its package installs them. A model without resources gets none.
resources=()
resource_dir="$("$REGISTRY_PYTHON" -m model_registry fetch-resources "$MODEL_DIR" \
  --output-dir "$WORK_DIR/resources" --cache-dir "$WORK_DIR/sources")"
if [[ -n "$resource_dir" ]]; then
  resources=(--resources "$resource_dir")
fi

# Performance packages go in a folder beside PyNeat's environment, never into it, so its numpy stays in place.
site="$WORK_DIR/site-packages"
if [[ -f "$REQUIREMENTS" ]]; then
  "$PYNEAT_PYTHON" -m pip install --quiet --disable-pip-version-check --no-deps --only-binary=:all: \
    --target "$site" -r "$REQUIREMENTS"
fi
fi

# Throughput and power depend on the MLA clock, while outputs do not. Raise a lower clock to the 1 GHz
# maximum, and stop before measuring if the board does not run at it.
MLA_DEVFREQ=/sys/class/devfreq/5000000.mla
MLA_FREQUENCY_HZ=1000000000
if (( $(cat "$MLA_DEVFREQ/cur_freq") < MLA_FREQUENCY_HZ )); then
  echo "$MLA_FREQUENCY_HZ" | sudo -n tee "$MLA_DEVFREQ/userspace/set_freq" >/dev/null ||
    { echo "Could not set the MLA clock to $MLA_FREQUENCY_HZ Hz." >&2; exit 1; }
fi
mla_frequency_hz="$(cat "$MLA_DEVFREQ/cur_freq")"
if (( mla_frequency_hz != MLA_FREQUENCY_HZ )); then
  echo "The MLA clock is $mla_frequency_hz Hz; benchmarks require $MLA_FREQUENCY_HZ Hz." >&2
  exit 1
fi

# The CPU feeds the MLA and runs postprocessing, so its clock also bounds throughput. One cpufreq policy covers
# every Modalix core. Set any other clock to the 1.4 GHz maximum, and stop before measuring if it does not hold.
CPU_CPUFREQ=/sys/devices/system/cpu/cpufreq/policy0
CPU_FREQUENCY_KHZ=1400000
if (( $(cat "$CPU_CPUFREQ/scaling_cur_freq") != CPU_FREQUENCY_KHZ )); then
  echo "$CPU_FREQUENCY_KHZ" | sudo -n tee "$CPU_CPUFREQ/scaling_setspeed" >/dev/null ||
    { echo "Could not set the CPU clock to $CPU_FREQUENCY_KHZ kHz." >&2; exit 1; }
fi
cpu_frequency_khz="$(cat "$CPU_CPUFREQ/scaling_cur_freq")"
if (( cpu_frequency_khz != CPU_FREQUENCY_KHZ )); then
  echo "The CPU clock is $cpu_frequency_khz kHz; benchmarks require $CPU_FREQUENCY_KHZ kHz." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$backend" == llima ]]; then
  deployment="$(dirname "${inputs[0]#*=}")/deployment"
  export HF_HOME="${HF_HOME:-/media/nvme/huggingface}"
  baseline=()
  if [[ -f "$COMPILED_DIR/benchmark-baseline.json" ]]; then
    baseline=(--baseline "$COMPILED_DIR/benchmark-baseline.json")
  fi
  PYTHONPATH="$REPOSITORY/tools" PYTHONDONTWRITEBYTECODE=1 "$LLIMA_MOLE_PYTHON" "$MODEL_DIR/scripts/performance.py" \
    --deployment "$deployment" "${baseline[@]}" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/performance.log"
else
  PYTHONPATH="$REPOSITORY/tools:$site" PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py" \
    "${mpks[@]}" "${dataset[@]}" "${resources[@]}" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/performance.log"
fi
"$REGISTRY_PYTHON" -m model_registry check-report "$OUTPUT_DIR" --repository "$REPOSITORY" \
  --stage performance --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID"
