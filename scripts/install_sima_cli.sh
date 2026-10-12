#!/usr/bin/env bash
# Install sima-cli and export SIMA_CLI_BIN for the later steps of a GitHub Actions job.
#
# Usage: install_sima_cli.sh
#
# Needs GITHUB_ENV. packaging/model-registry/build.sh runs `sima-cli packages build` through SIMA_CLI_BIN.

set -euo pipefail

: "${GITHUB_ENV:?GITHUB_ENV is required}"

curl -fsSL https://docs.sima.ai/_static/tools/sima-cli-installer.sh | bash
sima_cli="${HOME}/.sima-cli/.venv/bin/sima-cli"
"${sima_cli}" packages build --help >/dev/null
echo "SIMA_CLI_BIN=${sima_cli}" >> "${GITHUB_ENV}"
