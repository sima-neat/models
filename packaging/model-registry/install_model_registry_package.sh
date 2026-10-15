#!/usr/bin/env bash

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVER="${PACKAGE_DIR}/resolve_model_compiler_python.sh"

if [[ ! -f "${RESOLVER}" ]]; then
  echo "Missing Model Compiler environment resolver: ${RESOLVER}" >&2
  exit 1
fi
# shellcheck source=resolve_model_compiler_python.sh
source "${RESOLVER}"

wheel_count="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | wc -l | tr -d ' ')"
wheel_path="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | sort | head -n 1)"
if [[ "${wheel_count}" != "1" || -z "${wheel_path}" ]]; then
  echo "Expected exactly one model-registry wheel, found ${wheel_count}" >&2
  exit 1
fi

if ! MODEL_COMPILER_PYTHON_BIN="$(resolve_model_compiler_python)"; then
  echo "Model Compiler Python was not found." >&2
  echo "Install or activate Model Compiler in NEAT SDK or standalone mode." >&2
  echo "For a custom layout, set MODEL_COMPILER_HOME or MODEL_COMPILER_PYTHON." >&2
  exit 1
fi

# Model Compiler owns this environment's dependency versions. Installing the
# contributor tool with --no-deps prevents pip from upgrading or replacing
# compiler dependencies.
"${MODEL_COMPILER_PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${wheel_path}"

"${MODEL_COMPILER_PYTHON_BIN}" - <<'PY'
import certifi  # noqa: F401
import yaml  # noqa: F401
from jsonschema import Draft202012Validator, FormatChecker  # noqa: F401
PY

MODEL_REGISTRY_BIN="$(dirname "${MODEL_COMPILER_PYTHON_BIN}")/model-registry"
help_text="$("${MODEL_REGISTRY_BIN}" --help)"
module_help_text="$("${MODEL_COMPILER_PYTHON_BIN}" -m model_registry --help)"
if [[ "${help_text}" != "${module_help_text}" ]]; then
  echo "Author console and module command surfaces do not match" >&2
  exit 1
fi
for command in validate validate-models fetch fetch-calibration fetch-fidelity-dataset matrix resolve-compiler verify-mpk verify-artifact pack-llima; do
  grep -q "${command}" <<< "${help_text}" || {
    echo "Installed author CLI is missing command: ${command}" >&2
    exit 1
  }
done
for command in package publish publish-benchmark publish-accuracy publish-catalog stage-registry-build; do
  if "${MODEL_REGISTRY_BIN}" "${command}" --help >/dev/null 2>&1; then
    echo "Installed author CLI unexpectedly exposes CI command: ${command}" >&2
    exit 1
  fi
done

"${MODEL_REGISTRY_BIN}" --version
echo "Installed author-safe model-registry into Model Compiler: ${MODEL_REGISTRY_BIN}"
