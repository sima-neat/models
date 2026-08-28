#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUT_DIR_INPUT="${1:-${ROOT_DIR}/dist/model-registry}"
PYTHON_BIN="${MODEL_REGISTRY_BUILD_PYTHON:-python3}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
BASE_VERSION="0.1.0"

if ! OUT_DIR="$("${PYTHON_BIN}" - "${ROOT_DIR}" "${OUT_DIR_INPUT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).expanduser().resolve(strict=False)
if output == root or output in root.parents:
    print(f"Refusing unsafe model-registry output directory: {output}", file=sys.stderr)
    raise SystemExit(2)
print(output)
PY
)"; then
  exit 2
fi

if [[ -n "${MODEL_REGISTRY_PACKAGE_VERSION:-}" ]]; then
  PACKAGE_VERSION="${MODEL_REGISTRY_PACKAGE_VERSION}"
elif [[ "${GITHUB_REF_TYPE:-}" == "tag" && "${GITHUB_REF_NAME:-}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  PACKAGE_VERSION="${GITHUB_REF_NAME#v}"
else
  commit="${GITHUB_SHA:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
  PACKAGE_VERSION="${BASE_VERSION}+${commit:0:12}"
fi

BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT
SOURCE_ROOT="${BUILD_ROOT}/source"
mkdir -p "${SOURCE_ROOT}"
cp -R "${ROOT_DIR}/tools/model_registry" "${SOURCE_ROOT}/model_registry"
find "${SOURCE_ROOT}/model_registry" -type d -name __pycache__ -prune -exec rm -rf {} +
for ci_module in benchmark_manifest benchmark_publication package registry; do
  rm "${SOURCE_ROOT}/model_registry/${ci_module}.py"
done
install -m 0644 "${ROOT_DIR}/packaging/model-registry/__main__.py" \
  "${SOURCE_ROOT}/model_registry/__main__.py"
sed -E "s/^version = \"[^\"]+\"/version = \"${PACKAGE_VERSION}\"/" \
  "${ROOT_DIR}/packaging/model-registry/pyproject.toml" > "${SOURCE_ROOT}/pyproject.toml"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
"${PYTHON_BIN}" -m build --wheel --outdir "${OUT_DIR}" "${SOURCE_ROOT}"

install -m 0755 "${ROOT_DIR}/packaging/model-registry/install_model_registry_package.sh" \
  "${OUT_DIR}/install_model_registry_package.sh"
install -m 0755 "${ROOT_DIR}/packaging/model-registry/resolve_model_compiler_python.sh" \
  "${OUT_DIR}/resolve_model_compiler_python.sh"

SIMA_CLI_CHECK_FOR_UPDATE=0 "${SIMA_CLI_BIN}" packages build "${OUT_DIR}" \
  --name "gh:sima-neat/models/model-registry" \
  --version "${PACKAGE_VERSION}" \
  --description "Author-safe local tooling for SiMa Neat model contributions" \
  --install-script "install_model_registry_package.sh" \
  --download-compatible-files-only

test -f "${OUT_DIR}/metadata.json"
"${PYTHON_BIN}" -m json.tool "${OUT_DIR}/metadata.json" >/dev/null
ls -lh "${OUT_DIR}"
