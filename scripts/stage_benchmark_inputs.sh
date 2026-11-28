#!/usr/bin/env bash
# Stage the current build's metadata, and its ordinary benchmark baseline's, for the DevKit to download by
# compile run.
# Usage: stage_benchmark_inputs.sh <source_checkout> <compiled_dir>
# Requires REGISTRY_API_URL, SOURCE_BRANCH, SOURCE_SHA, MODEL_ID, BUILD_PROFILE_ID,
# COMPILE_INVOCATION_ID and STAGES from the benchmark dispatch.
# Optional DEPLOYMENT_BRANCH selects another catalog for the compiled deployments.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <source_checkout> <compiled_dir>" >&2
  exit 2
fi
args=(--repository "$1" --output-dir "$2" --api-url "$REGISTRY_API_URL"
  --branch "$SOURCE_BRANCH" --catalog-revision "$SOURCE_SHA" --model-id "$MODEL_ID"
  --build-profile-id "$BUILD_PROFILE_ID" --compile-invocation-id "$COMPILE_INVOCATION_ID")
if [[ -n "${DEPLOYMENT_BRANCH:-}" ]]; then
  args+=(--deployment-branch "$DEPLOYMENT_BRANCH")
fi
model-registry stage-registry-build "${args[@]}"
if [[ ",$STAGES," == *,performance,* ]]; then
  model-registry stage-registry-build "${args[@]}" --benchmark-baseline
fi
