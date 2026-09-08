"""Publish a validated conversion through Vulcan and verify the consumer download."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from model_registry import _vulcan_onnx_artifacts as artifacts
from model_registry.fetch import Fetcher


def publish(result: dict, kms_key: str, output_dir: Path) -> dict:
    import boto3

    output_dir.mkdir(parents=True, exist_ok=False)
    export_dir = Path(result["output_dir"])
    provenance = json.loads((export_dir / "provenance.json").read_text())
    export_id = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
    provenance["workflow"] = {
        "repository": os.environ["GITHUB_REPOSITORY"],
        "sha": os.environ["GITHUB_SHA"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
    }
    prefix = f"models-onnx/{result['model_id']}/{result['fingerprint']}/{export_id}/"
    bundle, manifest = artifacts.package(export_dir, provenance)
    journal: list[dict] = []
    try:
        _, completion = artifacts.publish(
            boto3.client("s3"),
            "sima-neat-artifacts-staging",
            prefix,
            bundle,
            manifest,
            kms_key,
            journal,
        )
    finally:
        (output_dir / "publication.json").write_text(json.dumps(journal, indent=2) + "\n")
    source = {
        "type": "onnx_bundle",
        "uri": f"https://artifacts.stg.neat.sima.ai/{completion['key']}",
        "sha256": completion["sha256"],
        "size_bytes": completion["size_bytes"],
    }
    # Use the same verification path as compilation before emitting a usable source pin.
    path, _, _, _ = Fetcher(cache_dir=output_dir / "download").fetch_source(source)
    if artifacts.digest(path) != manifest["files"]["model.onnx"]["sha256"]:
        raise ValueError("Downloaded ONNX differs from the validated conversion")
    (output_dir / "source.yaml").write_text(yaml.safe_dump({"source": source}, sort_keys=False))
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--kms-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(json.loads(args.result.read_text()), args.kms_key, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
