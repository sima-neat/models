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
sima-cli neat install --env production "$mole_target"
"${LLIMA_MOLE_PYTHON:-$HOME/sima-mole-venv/bin/python}" -c 'from sima_lmm.mole.modalixlm import ModalixLM; print("MoLE ready")'
dpkg-query -W 'sima-lmm-*'
