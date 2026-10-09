#!/usr/bin/env bash
# Cancel every workflow run of a branch that has not finished, so a deleted branch stops occupying runners.
#
# Usage: cancel_branch_runs.sh <branch>
#
# Needs GH_TOKEN with actions: write and GITHUB_REPOSITORY (owner/name).

set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "usage: $0 <branch>" >&2
  exit 2
fi
BRANCH="$1"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

error="$(mktemp)"
trap 'rm -f "$error"' EXIT
cancelled=0
for status in queued in_progress waiting pending requested; do
  while read -r run_id; do
    [[ -n "$run_id" ]] || continue
    if gh api -X POST "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/cancel" >/dev/null 2>"$error"; then
      cancelled=$((cancelled + 1))
    elif ! grep -q 'HTTP 409' "$error"; then
      # 409 means the run finished between listing and cancelling.
      cat "$error" >&2
      exit 1
    fi
  done < <(gh api -X GET --paginate "repos/$GITHUB_REPOSITORY/actions/runs" \
    -f branch="$BRANCH" -f status="$status" -F per_page=100 --jq '.workflow_runs[].id')
done
echo "Cancelled $cancelled unfinished workflow run(s) of $BRANCH."
