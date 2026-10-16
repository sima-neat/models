#!/usr/bin/env bash
# Benchmark compiled components on the DevKit with the model's scripts/performance.py and record performance results.
#
# Usage: run_model_performance.sh <model_dir> <build_profile> <compiled_dir> <output_dir> [components]
#
# <compiled_dir> holds <model_id>/<build_profile>/<component>/ with the MPK, compile-result.json and
# registry-publication.json. [components] is comma-separated and defaults to every component the build
# profile benchmarks. Each component's results land in <output_dir>/<component>/.

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 <model_dir> <build_profile> <compiled_dir> <output_dir> [components]" >&2
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
SOURCE_SHA="${SOURCE_SHA:-$(git -C "$REPOSITORY" rev-parse HEAD)}"
SOURCE_BRANCH="${SOURCE_BRANCH:-$(git -C "$REPOSITORY" rev-parse --abbrev-ref HEAD)}"

if [[ $# -eq 5 ]]; then
  COMPONENT_IDS="$5"
else
  COMPONENT_IDS="$("$REGISTRY_PYTHON" -m model_registry matrix --repository "$REPOSITORY" \
    --models "$MODEL_ID" --profiles "$BUILD_PROFILE_ID" | "$REGISTRY_PYTHON" -c '
import json, sys
print(",".join(c for entry in json.load(sys.stdin)["include"] for c in entry["benchmark_component_ids"]))
')"
fi
IFS=',' read -ra components <<< "$COMPONENT_IDS"
if [[ ${#components[@]} -eq 0 ]]; then
  echo "$MODEL_ID benchmarks no component in $BUILD_PROFILE_ID." >&2
  exit 1
fi

SIMA_PLATFORM_VERSION="$("$REGISTRY_PYTHON" -m model_registry resolve-benchmark-runtime --repository "$REPOSITORY" |
  "$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["devkit"]["platform_version"])')"
export SIMA_PLATFORM_VERSION

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

for component in "${components[@]}"; do
  output="$OUTPUT_DIR/$component"
  mkdir -p "$output"
  "$REGISTRY_PYTHON" -m model_registry make-benchmark-manifest \
    --repository "$REPOSITORY" \
    --compiled-dir "$COMPILED_DIR" \
    --output "$output/performance-test.json" \
    --branch "$SOURCE_BRANCH" \
    --git-sha "$SOURCE_SHA" \
    --model "$MODEL_ID" \
    --build-profile "$BUILD_PROFILE_ID" \
    --component "$component" >/dev/null
  archive="$("$REGISTRY_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"]["path"])' \
    "$output/performance-test.json")"

  PYTHONDONTWRITEBYTECODE=1 "$PYNEAT_PYTHON" "$MODEL_DIR/scripts/performance.py" \
    --mpk "$archive" --output "$output" 2>&1 | tee "$output/benchmark.log"

  # Record the workload Core benchmarked beside its saved report. A script that configures the model
  # reports its selection and postprocessing in its context; otherwise the MPK default applies. The
  # context's sample count is what energy per inference divides by.
  "$PYNEAT_PYTHON" - "$output/performance-test.json" "$output/benchmark-context.json" <<'PY'
import json
import sys
from pathlib import Path

import pyneat

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
context = json.loads(Path(sys.argv[2]).read_text())
model = pyneat.Model(manifest["model"]["path"])
options = pyneat.BenchmarkOptions()
info = model.info()
manifest["configuration"] = {
    "samples": context["num_samples"],
    "min_duration_ms": context["min_duration_ms"],
    "include_plugin_latency": options.include_plugin_latency,
    "original_width": None, "original_height": None, "resize_mode": None,
}
manifest["workload"] = {
    "kind": "model.synthetic", "input_generation": "deterministic",
    "selection": "mpk_default", "batch_size": model.compiled_batch_size(),
    "inputs": [{"shape": spec.shape, "dtype": spec.dtypes[0].name} for spec in model.input_specs()],
    "preprocess": info.selection.preprocess_graph,
    "postprocess": info.selection.selected_post_kind,
    **context.get("workload", {}),
}
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

  "$REGISTRY_PYTHON" -m model_registry import-core-benchmark \
    --manifest "$output/performance-test.json" \
    --report "$output/benchmark-report.json" \
    --context "$output/benchmark-context.json" \
    --schema "$REPOSITORY/schemas/performance-result.schema.json" \
    --output "$output/performance-result.json"
done
