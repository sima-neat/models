#!/usr/bin/env bash
# Run one stage's tests of a model on that stage's output and print the outcome: passed, failed or skipped.
#
# Usage: run_model_tests.sh <stage> <model_dir> <stage_output>
#
# <stage> is conversion, compile, accuracy or performance, and <stage_output> is what that stage produced:
# the converted ONNX model, one build's compile output directory, accuracy-result.json, or one component's
# performance-result.json. The tests are <model_dir>/tests/<build|measure>/test_<stage>.py, and only those
# that take the output run. The outcome is skipped when tests/tests.yaml sets run_tests: false. A
# failed outcome exits 1 after printing it; any other error exits without an outcome. pytest reports on
# stderr. REGISTRY_PYTHON must have pytest and model_registry installed.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <conversion|compile|accuracy|performance> <model_dir> <stage_output>" >&2
  exit 2
fi
STAGE="$1"
MODEL_DIR="$(cd "$2" && pwd)"
STAGE_OUTPUT="$3"
REGISTRY_PYTHON="${REGISTRY_PYTHON:-python3}"
# The model's own revision supplies its tests and the pytest configuration and fixtures they use.
REPOSITORY="$(cd "$MODEL_DIR/../.." && pwd)"

case "$STAGE" in
  conversion) TEST_FILE="build/test_conversion.py" OPTION="--converted" ;;
  compile) TEST_FILE="build/test_compile.py" OPTION="--compiled" ;;
  accuracy) TEST_FILE="measure/test_accuracy.py" OPTION="--accuracy" ;;
  performance) TEST_FILE="measure/test_performance.py" OPTION="--performance" ;;
  *)
    echo "unknown stage: $STAGE; expected conversion, compile, accuracy or performance" >&2
    exit 2
    ;;
esac
if [[ ! -e "$STAGE_OUTPUT" ]]; then
  echo "$STAGE output does not exist: $STAGE_OUTPUT" >&2
  exit 2
fi

reason="$("$REGISTRY_PYTHON" -c '
import sys
from model_registry.model_tests import load_model_tests, skip_reason
print(skip_reason(load_model_tests(sys.argv[1]), sys.argv[2]) or "")
' "$MODEL_DIR" "$STAGE")"
if [[ -n "$reason" ]]; then
  echo "$STAGE tests of $(basename "$MODEL_DIR") are skipped in tests/tests.yaml: $reason" >&2
  echo skipped
  exit 0
fi
if [[ ! -f "$MODEL_DIR/tests/$TEST_FILE" ]]; then
  echo "$MODEL_DIR/tests/$TEST_FILE is missing; add it or set $STAGE run_tests: false in tests/tests.yaml" >&2
  exit 2
fi

status=0
"$REGISTRY_PYTHON" -m pytest -c "$REPOSITORY/pyproject.toml" --rootdir "$REPOSITORY" -p no:cacheprovider \
  "$MODEL_DIR/tests/$TEST_FILE" "$OPTION=$STAGE_OUTPUT" >&2 || status=$?
case "$status" in
  0) echo passed ;;
  5)
    # pytest deselected every test: none in the file takes this stage's output.
    echo "no $STAGE test of $(basename "$MODEL_DIR") takes the $STAGE output" >&2
    echo passed
    ;;
  1)
    echo failed
    exit 1
    ;;
  *)
    echo "pytest could not run the $STAGE tests (exit $status)" >&2
    exit "$status"
    ;;
esac
