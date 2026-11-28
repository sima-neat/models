#!/usr/bin/env bash
# Fail unless neat-core.revision in deps/manifest.json is the commit of the Core release tag install_target pins.
#
# Usage: check_neat_core_revision.sh [manifest]
#
# Catalogs and benchmark runs publish neat-core.revision as neat_core_revision. A Core branch package pins no
# release, so there is nothing to compare. Needs gh with a token that can read sima-neat/core.

set -euo pipefail

manifest="${1:-deps/manifest.json}"
target="$(jq -r '."neat-core".install_target' "$manifest")"
if [[ "$target" != core@v* ]]; then
  echo "Core is pinned to the branch package $target; there is no release commit to check."
  exit 0
fi
tag="${target#core@}"
expected="$(gh api "repos/sima-neat/core/commits/$tag" --jq .sha)"
pinned="$(jq -r '."neat-core".revision' "$manifest")"
if [[ "$pinned" != "$expected" ]]; then
  echo "neat-core.revision is $pinned, but Core $tag is commit $expected." >&2
  exit 1
fi
echo "neat-core.revision is the commit of Core $tag."
