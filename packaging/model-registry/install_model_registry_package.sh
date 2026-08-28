#!/usr/bin/env bash

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${MODEL_REGISTRY_INSTALL_ROOT:-${HOME}/.local/share/sima-neat/model-registry}"
BIN_DIR="${MODEL_REGISTRY_BIN_DIR:-${HOME}/.local/bin}"
PYTHON_BIN="${MODEL_REGISTRY_INSTALL_PYTHON:-python3}"

case "${INSTALL_ROOT}" in
  ""|"/")
    echo "Refusing unsafe model-registry install root: ${INSTALL_ROOT}" >&2
    exit 2
    ;;
esac

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python 3.10 or newer is required to install model-registry" >&2
  exit 1
}

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required to install model-registry")
PY

wheel_count="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | wc -l | tr -d ' ')"
wheel_path="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name 'sima_neat_model_registry-*.whl' | sort | head -n 1)"
if [[ "${wheel_count}" != "1" || -z "${wheel_path}" ]]; then
  echo "Expected exactly one model-registry wheel, found ${wheel_count}" >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"
"${PYTHON_BIN}" -m venv "${INSTALL_ROOT}/venv"
"${INSTALL_ROOT}/venv/bin/python" -m pip install --upgrade --force-reinstall "${wheel_path}"

help_text="$("${INSTALL_ROOT}/venv/bin/model-registry" --help)"
module_help_text="$("${INSTALL_ROOT}/venv/bin/python" -m model_registry --help)"
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

launcher="${BIN_DIR}/model-registry"
if [[ -e "${launcher}" && ! -L "${launcher}" ]]; then
  echo "Refusing to replace non-symlink launcher: ${launcher}" >&2
  exit 1
fi
ln -sfn "${INSTALL_ROOT}/venv/bin/model-registry" "${launcher}"

"${launcher}" --version
echo "Installed author-safe model-registry: ${launcher}"
