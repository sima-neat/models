#!/usr/bin/env bash
# Install the published dataset-registry wheel into an existing Python environment.

set -euo pipefail

PYTHON_BIN="$1"
VULCAN_ENVIRONMENT="$2"
PACKAGE_DIR="${RUNNER_TEMP:-$(mktemp -d)}/dataset-registry-package"

download_wheel() {
  sima-cli neat install --help | grep -q -- '--download-only'
  sima-cli neat install \
    --env "$VULCAN_ENVIRONMENT" \
    --install-dir "$PACKAGE_DIR" \
    --download-only \
    dataset/dataset-registry@main:latest
}

find_wheel() {
  local wheels
  mapfile -t wheels < <(find "$PACKAGE_DIR" -maxdepth 1 -type f -name 'sima_neat_dataset_registry-*.whl' -print)
  if (( ${#wheels[@]} != 1 )); then
    echo "Expected one dataset-registry wheel, found ${#wheels[@]}" >&2
    exit 1
  fi
  echo "${wheels[0]}"
}

main() {
  local wheel
  download_wheel
  wheel="$(find_wheel)"
  "$PYTHON_BIN" -m pip install --disable-pip-version-check "$wheel"
  "$PYTHON_BIN" -m dataset_registry --version
}

main
