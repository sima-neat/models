#!/usr/bin/env bash
# Install a built model-registry package into a stand-in Model Compiler environment and check the
# command surface authors get: the author commands are present and the publication commands are not.
#
# Usage: verify_package.sh [dist_dir]   (default: dist/model-registry)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DIST_DIR="${1:-${ROOT_DIR}/dist/model-registry}"

model_compiler="$(mktemp -d)"
trap 'rm -rf "${model_compiler}"' EXIT

# The installer requires a Model Compiler environment: afe on the Python path and mla-masm beside it.
python -m venv "${model_compiler}"
"${model_compiler}/bin/python" -m pip install certifi jsonschema==4.18.4 PyYAML
"${model_compiler}/bin/python" -c 'from pathlib import Path; import sysconfig; Path(sysconfig.get_path("purelib"), "afe.py").touch()'
touch "${model_compiler}/bin/mla-masm"
chmod +x "${model_compiler}/bin/mla-masm"

MODEL_COMPILER_PYTHON="${model_compiler}/bin/python" bash "${DIST_DIR}/install_model_registry_package.sh"

model_registry="${model_compiler}/bin/model-registry"
"${model_registry}" --version
help_text="$("${model_registry}" --help)"
module_help_text="$("${model_compiler}/bin/python" -m model_registry --help)"
test "${help_text}" = "${module_help_text}"
for command in validate validate-models fetch fetch-calibration matrix resolve-compiler verify-mpk; do
  grep -q "${command}" <<< "${help_text}"
done
for command in package publish publish-report publish-catalog; do
  if "${model_registry}" "${command}" --help >/dev/null 2>&1; then
    echo "author CLI unexpectedly exposes ${command}" >&2
    exit 1
  fi
done
python -m json.tool "${DIST_DIR}/metadata.json" >/dev/null
