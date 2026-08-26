#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/dist/benchmarker}"
PYTHON_BIN="${BENCHMARKER_BUILD_PYTHON:-python3}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
BASE_VERSION="0.1.0"

case "${OUT_DIR}" in
  ""|"/"|"${ROOT_DIR}")
    echo "Refusing unsafe benchmarker output directory: ${OUT_DIR}" >&2
    exit 2
    ;;
esac

if [[ -n "${BENCHMARKER_PACKAGE_VERSION:-}" ]]; then
  PACKAGE_VERSION="${BENCHMARKER_PACKAGE_VERSION}"
elif [[ "${GITHUB_REF_TYPE:-}" == "tag" && "${GITHUB_REF_NAME:-}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  PACKAGE_VERSION="${GITHUB_REF_NAME#v}"
else
  commit="${GITHUB_SHA:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"
  PACKAGE_VERSION="${BASE_VERSION}+${commit:0:12}"
fi

BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT
SOURCE_ROOT="${BUILD_ROOT}/source"
mkdir -p "${SOURCE_ROOT}/benchmarker/schemas"

install -m 0644 "${ROOT_DIR}/tools/runtime/benchmarker.py" "${SOURCE_ROOT}/benchmarker/__init__.py"
install -m 0644 "${ROOT_DIR}/schemas/performance-test-v1.schema.json" \
  "${SOURCE_ROOT}/benchmarker/schemas/performance-test-v1.schema.json"
install -m 0644 "${ROOT_DIR}/schemas/performance-result-v1.schema.json" \
  "${SOURCE_ROOT}/benchmarker/schemas/performance-result-v1.schema.json"
sed -E "s/^version = \"[^\"]+\"/version = \"${PACKAGE_VERSION}\"/" \
  "${ROOT_DIR}/packaging/benchmarker/pyproject.toml" > "${SOURCE_ROOT}/pyproject.toml"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
"${PYTHON_BIN}" -m build --wheel --outdir "${OUT_DIR}" "${SOURCE_ROOT}"

install -m 0755 "${ROOT_DIR}/packaging/benchmarker/install_benchmarker_package.sh" \
  "${OUT_DIR}/install_benchmarker_package.sh"
install -m 0644 "${ROOT_DIR}/packaging/benchmarker/dependencies.json" "${OUT_DIR}/dependencies.json"
install -m 0644 "${ROOT_DIR}/schemas/performance-test-v1.schema.json" "${OUT_DIR}/"
install -m 0644 "${ROOT_DIR}/schemas/performance-result-v1.schema.json" "${OUT_DIR}/"

SIMA_CLI_CHECK_FOR_UPDATE=0 "${SIMA_CLI_BIN}" packages build "${OUT_DIR}" \
  --name "gh:sima-neat/models/benchmarker" \
  --version "${PACKAGE_VERSION}" \
  --description "Generic Python model benchmarker for Modalix DevKits" \
  --install-script "install_benchmarker_package.sh" \
  --download-compatible-files-only \
  --board-platform "modalix@==2.1.3"

test -f "${OUT_DIR}/metadata.json"
"${PYTHON_BIN}" -m json.tool "${OUT_DIR}/metadata.json" >/dev/null
ls -lh "${OUT_DIR}"
