#!/usr/bin/env bash
# Verify the EC2 runner market and record its identity for compile provenance.
#
# Usage: scripts/record_compile_runner_identity.sh
#
# Requires curl and EC2 IMDSv2 access. Reads REQUESTED_RUNNER_MARKET, RUNNER_MARKET,
# RUNNER_NAME, RUNNER_APPROVAL_ID (optional), and RUNNER_APPROVAL_SPEC_REVISION.
# Appends verified identity to GITHUB_ENV and the runner summary to GITHUB_STEP_SUMMARY.
# Authentication, metadata reads, or a market mismatch fail before either file is changed.

set -euo pipefail
token="$(curl --fail --silent --show-error --connect-timeout 3 --max-time 10 -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)"
metadata() {
  curl --fail --silent --show-error --connect-timeout 3 --max-time 10 \
    -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$1"
}
instance_id="$(metadata instance-id)"
instance_type="$(metadata instance-type)"
actual_market="$(metadata instance-life-cycle)"
if [[ "$actual_market" != "$RUNNER_MARKET" ]]; then
  echo "::error::Runner market mismatch: selected $RUNNER_MARKET, EC2 reports $actual_market"
  exit 1
fi
{
  echo "RUNNER_INSTANCE_ID=$instance_id"
  echo "RUNNER_INSTANCE_TYPE=$instance_type"
  echo "RUNNER_ACTUAL_MARKET=$actual_market"
} >> "$GITHUB_ENV"
{
  echo "### Compile runner"
  echo "Requested market: $REQUESTED_RUNNER_MARKET; selected: $RUNNER_MARKET; actual: $actual_market"
  echo "Runner: $RUNNER_NAME; EC2: $instance_id ($instance_type)"
  echo "Approval: ${RUNNER_APPROVAL_ID:-none}; policy revision: $RUNNER_APPROVAL_SPEC_REVISION"
} >> "$GITHUB_STEP_SUMMARY"
