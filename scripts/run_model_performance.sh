#!/usr/bin/env bash
# Measure a build profile's whole-model performance on the DevKit with the model's scripts/performance.py.
#
# Usage: run_model_performance.sh <model_dir> <build_profile> <compiled_dir> <output_dir>
#
# <compiled_dir> holds <model_id>/<build_profile>/compile-outputs.json and the MPKs it lists. The script
# writes model-report.json into <output_dir> with tools/model_report; check-report then holds it to
# model.yaml and records the board in environment.json.

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
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"
MODEL_ID="$(basename "$MODEL_DIR")"

# A single-component model takes its MPK; a multi-component model takes each as <component>=<path>.
listed="$("$REGISTRY_PYTHON" -m model_registry model-component-mpks "$COMPILED_DIR" \
  --repository "$REPOSITORY" --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID")"
mapfile -t inputs <<< "$listed"
mpks=()
if [[ ${#inputs[@]} -eq 1 ]]; then
  mpks=(--mpk "${inputs[0]#*=}")
else
  for input in "${inputs[@]}"; do mpks+=(--mpk "$input"); done
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
PYTHONPATH="$REPOSITORY/tools" PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py" \
  "${mpks[@]}" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/performance.log"
"$REGISTRY_PYTHON" -m model_registry check-report "$OUTPUT_DIR" --repository "$REPOSITORY" \
  --stage performance --model "$MODEL_ID" --build-profile "$BUILD_PROFILE_ID"
