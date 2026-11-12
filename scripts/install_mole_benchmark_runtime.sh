#!/usr/bin/env bash
# Install pinned MoLE for the GenAI performance stage; Core supplies the runtime.
# Usage: install_mole_benchmark_runtime.sh <runtime-json>
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <runtime-json>" >&2
  exit 2
fi
runtime_target="$("$REGISTRY_PYTHON" -c 'import json,sys; print(json.loads(sys.argv[1])["llima"]["install_target"])' "$1")"
mole_target="llima/mole@${runtime_target#llima@}"
export SIMA_MOLE_VENV_PATH="${SIMA_MOLE_VENV_PATH:-/media/nvme/llima/mole-venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/media/nvme/llima/pip-cache}"
export TMPDIR="${TMPDIR:-/media/nvme/llima/tmp}"
mole_python="${LLIMA_MOLE_PYTHON:-$SIMA_MOLE_VENV_PATH/bin/python}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

metadata="$(sima-cli neat install --env production "$mole_target" --json)"
expected_version="$("$REGISTRY_PYTHON" -c '
import json, sys
from urllib.request import urlopen
# sima-cli prints status messages before its JSON document.
metadata = json.loads(sys.argv[1][sys.argv[1].index("{"):])
with urlopen(metadata["metadata_url"], timeout=30) as response:
    print(json.load(response)["version"])
' "$metadata")"
installed_version="$("$mole_python" -c 'from importlib.metadata import version; from sima_lmm.mole.modalixlm import ModalixLM; print(version("sima-lmm"))' 2>/dev/null)" || installed_version=""
if [[ "$installed_version" == "$expected_version" ]]; then
  echo "MoLE already installed ($installed_version). Skipping install."
else
  package_dir="${WORK_DIR:-/media/nvme/neat-tests/mole-install-${runtime_target##*:}}/mole"
  sima-cli neat install --env production "$mole_target" --install-dir "$package_dir"
fi
"$mole_python" -c 'from sima_lmm.mole.modalixlm import ModalixLM; print("MoLE ready")'
dpkg-query -W 'sima-lmm-*'
