#!/usr/bin/env python3
"""Usage: benchmark_promoted_models.py

Manually dispatch missing approved non-LLM/VLM measurements through Vulcan Model Benchmarking.
Requires REGISTRY_API_URL, STAGES, VULCAN_ENV and the standard GitHub Actions environment/token.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main():
    plan = json.loads(subprocess.check_output([
        "model-registry", "plan-promoted-benchmarks", "--api-url", os.environ["REGISTRY_API_URL"],
        "--branch", os.environ["GITHUB_REF_NAME"], "--stages", os.environ["STAGES"],
    ], text=True))
    queued = []
    try:
        for build in plan["include"]:
            inputs = {"environment": os.environ["VULCAN_ENV"], "source_sha": plan["source_sha"],
                      "source_branch": plan["source_branch"], "model_id": build["model_id"],
                      "build_profile_id": build["build_profile_id"],
                      "compile_invocation_id": build["compile_invocation_id"], "stages": ",".join(build["stages"])}
            subprocess.run([
                "gh", "api", "--method", "POST",
                f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/workflows/vulcan-model-benchmarking.yml/dispatches",
                "--input", "-",
            ], input=json.dumps({"ref": plan["source_branch"], "inputs": inputs}), text=True, check=True)
            queued.append(build)
    finally:
        lines = [f"## Promoted benchmark batch ({plan['source_branch']}, {os.environ['VULCAN_ENV']})", "",
                 f"Queued {len(queued)} of {len(plan['include'])} missing profiles; skipped {len(plan['skipped'])}.",
                 "LLMs and VLMs are excluded. Each queued profile runs separately in Vulcan Model Benchmarking.", "",
                 "| Model/profile | Stages or skip reason |", "| --- | --- |"]
        for build in [*queued, *plan["skipped"]]:
            detail = ", ".join(build["stages"]) if "stages" in build else build["reason"]
            lines.append(f"| {build['model_id']}/{build['build_profile_id']} | {detail} |")
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
