from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/record_compile_runner_identity.sh"


def run_identity(tmp_path: Path, market: str = "spot", actual: str = "spot", fail: str = ""):
    curl = tmp_path / "curl"
    curl.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ["CURL_CALLS"]).open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
key = args[-1].rsplit("/", 1)[-1]
if key == os.environ["FAIL_METADATA"]:
    sys.exit(22)
print({"token": "test-imds-token", "instance-id": "i-test", "instance-type": "c8gd.4xlarge",
       "instance-life-cycle": os.environ["ACTUAL_MARKET"]}[key])
''')
    curl.chmod(0o755)
    env_file = tmp_path / "env"
    summary = tmp_path / "summary"
    env_file.write_text("EXISTING=value\n")
    summary.write_text("Existing summary\n")
    env = {
        **os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "GITHUB_ENV": str(env_file), "GITHUB_STEP_SUMMARY": str(summary),
        "REQUESTED_RUNNER_MARKET": market, "RUNNER_MARKET": market,
        "RUNNER_NAME": "test-runner", "RUNNER_APPROVAL_ID": "sima-neat/models:test",
        "RUNNER_APPROVAL_SPEC_REVISION": "a" * 64,
        "ACTUAL_MARKET": actual, "FAIL_METADATA": fail, "CURL_CALLS": str(tmp_path / "calls"),
    }
    result = subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    return result, env_file.read_text(), summary.read_text(), calls


@pytest.mark.parametrize("market", ["spot", "on-demand"])
def test_records_verified_identity_and_authenticates_metadata(tmp_path, market):
    result, env, summary, calls = run_identity(tmp_path, market, market)
    assert result.returncode == 0, result.stderr
    assert env == (
        "EXISTING=value\nRUNNER_INSTANCE_ID=i-test\nRUNNER_INSTANCE_TYPE=c8gd.4xlarge\n"
        f"RUNNER_ACTUAL_MARKET={market}\n"
    )
    assert summary.startswith("Existing summary\n")
    assert f"Requested market: {market}; selected: {market}; actual: {market}" in summary
    assert "Runner: test-runner; EC2: i-test (c8gd.4xlarge)" in summary
    assert f"Approval: sima-neat/models:test; policy revision: {'a' * 64}" in summary
    assert len(calls) == 4
    assert calls[0][-1] == "http://169.254.169.254/latest/api/token"
    assert calls[0][calls[0].index("-X") + 1] == "PUT"
    assert "X-aws-ec2-metadata-token-ttl-seconds: 60" in calls[0]
    for call in calls[1:]:
        assert "X-aws-ec2-metadata-token: test-imds-token" in call
        assert "--fail" in call and "--max-time" in call
    assert "test-imds-token" not in result.stdout + result.stderr + env + summary


def test_market_mismatch_does_not_write_identity(tmp_path):
    result, env, summary, _ = run_identity(tmp_path, "on-demand", "spot")
    assert result.returncode != 0
    assert "Runner market mismatch: selected on-demand, EC2 reports spot" in result.stdout
    assert env == "EXISTING=value\n"
    assert summary == "Existing summary\n"


@pytest.mark.parametrize("endpoint", ["token", "instance-id", "instance-type", "instance-life-cycle"])
def test_metadata_failure_does_not_write_identity(tmp_path, endpoint):
    result, env, summary, _ = run_identity(tmp_path, fail=endpoint)
    assert result.returncode != 0
    assert env == "EXISTING=value\n"
    assert summary == "Existing summary\n"


def test_workflow_invokes_executable_script():
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulcan-model-compile.yml").read_text())
    steps = [step for job in workflow["jobs"].values() for step in job.get("steps", [])]
    identity = next(step for step in steps if step.get("name") == "Verify and record EC2 runner identity")
    assert identity["run"] == "scripts/record_compile_runner_identity.sh"
    assert os.access(SCRIPT, os.X_OK)
