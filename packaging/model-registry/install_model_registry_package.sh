#!/usr/bin/env bash

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

wheel_count="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | wc -l | tr -d ' ')"
wheel_path="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | sort | head -n 1)"
if [[ "${wheel_count}" != "1" || -z "${wheel_path}" ]]; then
  echo "Expected exactly one model-registry wheel, found ${wheel_count}" >&2
  exit 1
fi

is_model_compiler_python() {
  local candidate="$1"
  [[ -x "${candidate}" ]] || return 1
  "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
import afe  # noqa: F401

raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_model_compiler_python() {
  local candidate=""

  if [[ -n "${MODEL_COMPILER_PYTHON:-}" ]]; then
    if is_model_compiler_python "${MODEL_COMPILER_PYTHON}"; then
      printf '%s\n' "${MODEL_COMPILER_PYTHON}"
      return
    fi
    echo "MODEL_COMPILER_PYTHON does not identify a Python 3.10+ environment with afe installed: ${MODEL_COMPILER_PYTHON}" >&2
    return 1
  fi

  for candidate in \
    "${VIRTUAL_ENV:-}/bin/python" \
    "/sdk-extensions/model-compiler/bin/python" \
    "/sdk-add-on/model-compiler/bin/python" \
    "${HOME}/sdk-extensions/model-compiler/bin/python" \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"; do
    if is_model_compiler_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

if ! MODEL_COMPILER_PYTHON_BIN="$(find_model_compiler_python)"; then
  echo "Model Compiler Python was not found." >&2
  echo "Install or activate Model Compiler first, or set MODEL_COMPILER_PYTHON to its bin/python path." >&2
  exit 1
fi

# Model Compiler owns this environment's dependency versions. In particular,
# sima-accelerator-mode pins jsonschema. Installing the contributor tool with
# --no-deps prevents pip from upgrading or replacing compiler dependencies.
"${MODEL_COMPILER_PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${wheel_path}"

"${MODEL_COMPILER_PYTHON_BIN}" - <<'PY'
import certifi  # noqa: F401
from importlib.metadata import version
import yaml  # noqa: F401

jsonschema_version = version("jsonschema")
if jsonschema_version != "4.18.4":
    raise SystemExit(f"Model Compiler requires jsonschema 4.18.4, found {jsonschema_version}")
PY

MODEL_REGISTRY_BIN="$(dirname "${MODEL_COMPILER_PYTHON_BIN}")/model-registry"
help_text="$("${MODEL_REGISTRY_BIN}" --help)"
module_help_text="$("${MODEL_COMPILER_PYTHON_BIN}" -m model_registry --help)"
if [[ "${help_text}" != "${module_help_text}" ]]; then
  echo "Author console and module command surfaces do not match" >&2
  exit 1
fi
for command in validate validate-catalog fetch fetch-calibration matrix resolve-compiler verify-mpk; do
  grep -q "${command}" <<< "${help_text}" || {
    echo "Installed author CLI is missing command: ${command}" >&2
    exit 1
  }
done
for command in package publish publish-benchmark publish-taxonomy; do
  if grep -Eq "(^|[,{[:space:]])${command}([,}[:space:]]|$)" <<< "${help_text}"; then
    echo "Installed author CLI unexpectedly exposes CI command: ${command}" >&2
    exit 1
  fi
done

"${MODEL_REGISTRY_BIN}" --version
echo "Installed author-safe model-registry into Model Compiler: ${MODEL_REGISTRY_BIN}"
