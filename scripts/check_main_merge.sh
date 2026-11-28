#!/usr/bin/env bash
# Check that merging develop into main changes nothing customers see.
#
# Usage: check_main_merge.sh <develop commit>
#
# Run it in a checkout of the pull request's merge result that also holds <develop commit>. It reads the
# registry without credentials and prints one line for each reason main would compile, show fewer results, or keep
# a promoted build of a build profile that is no longer declared.
# REGISTRY_API_URL selects another registry than production.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  sed -n '2,9s/^# \{0,1\}//p' "$0" >&2
  exit 2
fi

exec model-registry check-main-merge --repository "$(dirname "$0")/.." --head "$1" \
  --api-url "${REGISTRY_API_URL:-https://models-registry.neat.sima.ai}"
