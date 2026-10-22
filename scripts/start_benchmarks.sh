#!/usr/bin/env bash
# Start Vulcan Model Benchmarking for each build and print one line per build with its run link.
#
# Usage: start_benchmarks.sh <builds_json> [source_run_id]
#
# <builds_json> is one plan entry or a matrix ({"include": [...]}); each entry names model_id,
# build_profile_id, compile_invocation_id, and stages. A source run ID names
# the compile run whose output the benchmark downloads; without one, the benchmark restores the build
# from the registry. Needs GH_TOKEN with actions: write, GITHUB_REPOSITORY, GITHUB_REF_NAME,
# GITHUB_SHA, and GITHUB_STEP_SUMMARY. A failed start does not stop the others; the script then exits 1.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <builds_json> [source_run_id]" >&2
  exit 2
fi
SOURCE_RUN_ID="${2:-}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${GITHUB_STEP_SUMMARY:?GITHUB_STEP_SUMMARY is required}"

builds="$(jq -r 'if has("include") then .include[] else . end
  | [.model_id, .build_profile_id, .compile_invocation_id, (.stages | join(","))]
  | join("\u001f")' <<< "$1")"

failed=0
while IFS=$'\x1f' read -r model_id profile_id invocation_id stages; do
  [[ -n "$model_id" ]] || continue
  # The benchmark runs only when dispatched on its source branch.
  if output="$(gh workflow run vulcan-model-benchmarking.yml \
    --repo "$GITHUB_REPOSITORY" \
    --ref "$GITHUB_REF_NAME" \
    -f environment=staging \
    -f source_run_id="$SOURCE_RUN_ID" \
    -f source_sha="$GITHUB_SHA" \
    -f source_branch="$GITHUB_REF_NAME" \
    -f model_id="$model_id" \
    -f build_profile_id="$profile_id" \
    -f compile_invocation_id="$invocation_id" \
    -f stages="$stages" 2>&1)"; then
    run_url="$(grep -Eom1 'https://[^[:space:]]+/actions/runs/[0-9]+' <<< "$output" || true)"
    link="${run_url:-link unavailable}"
  else
    echo "$output" >&2
    link="not started"
    failed=1
  fi
  echo "DevKit run for $model_id/$profile_id/$invocation_id ($stages): $link" | tee -a "$GITHUB_STEP_SUMMARY"
done <<< "$builds"
exit "$failed"
