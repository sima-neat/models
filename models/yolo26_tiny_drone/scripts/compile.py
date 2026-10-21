#!/usr/bin/env python3
"""Compile the qualified YOLO26-P2 tiny-drone source into its registry MPK."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from calibration import validate_calibration
from model_registry.fetch import Fetcher
from model_registry.model import load_model as load_metadata
from model_registry.source import SourceSpec
from output_labels import restore_public_output_labels
from quantized_snapshot import materialize_snapshot

MODEL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MODEL_DIR.parents[1]
TRAINING_DIR = REPOSITORY_ROOT / "training" / "yolo26_tiny_drone"
sys.path.insert(0, str(TRAINING_DIR))

from compile import (  # noqa: E402
    MODEL_NAME,
    audit_mpk,
    audit_quantized_model,
    preprocess_rgb_image,
)
from contract import DeploymentContract  # noqa: E402
from onnx_audit import audit_onnx  # noqa: E402


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = load_metadata(MODEL_DIR)
    matches = [item for item in metadata["build_profiles"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one YOLO26 tiny-drone variant named {variant_id!r}")
    profile = matches[0]
    binding = next(item for item in profile["components"] if item["id"] == "model")
    return {**profile, "parameters": binding["parameters"]}


def compile_variant(source: Path, output_dir: Path, variant_id: str, calibration_path: Path) -> Path:
    variant = variant_config(variant_id)
    parameters = variant["parameters"]
    if variant["target"] != "modalix" or parameters.get("precision") != "w8a8":
        raise ValueError("the qualified tiny-drone recipe supports only INT8 with MLA tensor layout")
    contract = DeploymentContract(model_size=int(parameters["model_size"]))
    source_audit = audit_onnx(source, contract)
    if source_audit["convolution_count"] != int(parameters["expected_convolutions"]):
        raise RuntimeError(
            "qualified source convolution count changed: "
            f"expected {parameters['expected_convolutions']}, found {source_audit['convolution_count']}"
        )

    calibration, calibration_report = validate_calibration(
        calibration_path,
        contract,
        expected_tensor_sha256=str(parameters["calibration_tensor_sha256"]),
        preprocess_rgb_image=preprocess_rgb_image,
        materialize=False,
    )
    if calibration is not None:
        raise RuntimeError("compile-only validation unexpectedly materialized calibration data")
    gc.collect()

    quantized_source = SourceSpec.from_mapping(parameters["quantized_source"])
    quantized_archive, _cache_hit = Fetcher().fetch(quantized_source)

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="yolo26-quantized-snapshot-") as temporary:
        snapshot_dir = Path(temporary) / "model"
        snapshot_report = materialize_snapshot(quantized_archive, snapshot_dir)
        quantized_report = audit_quantized_model(
            snapshot_dir / f"{MODEL_NAME}.sima.json",
            int(parameters["expected_convolutions"]),
        )
        from afe.apis.model import Model

        quantized = Model.load(MODEL_NAME, str(snapshot_dir))
        if quantized is None:
            raise RuntimeError("ModelSDK returned no model for the qualified quantized snapshot")
        output_label_report = restore_public_output_labels(quantized, contract.output_names())
        quantized.compile(
            output_path=str(output_dir),
            batch_size=1,
        )
    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one compiled MPK, found {archives}")
    # The qualified snapshot fixes the compiler model name; the published MPK carries the variant ID.
    archive = archives[0].rename(output_dir / f"yolo26_tiny_drone_{variant_id}_mpk.tar.gz")
    report = {
        "schema_version": 1,
        "model_id": "yolo26_tiny_drone",
        "variant_id": variant_id,
        "source": source_audit,
        "calibration": calibration_report,
        "quantized_snapshot": {
            **snapshot_report,
            "archive_sha256": quantized_source.sha256,
        },
        "quantized_model": quantized_report,
        "public_output_labels": output_label_report,
        "mpk": audit_mpk(archive, contract),
    }
    (output_dir / "yolo26-tiny-drone-audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-profile", required=True)
    parser.add_argument("--compile-invocation", required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="verified calibration snapshot")
    args = parser.parse_args()
    if args.compile_invocation != "model":
        parser.error(f"unsupported compile invocation: {args.compile_invocation}")
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    output_dir = args.output_dir.expanduser().resolve() / "model"
    archive = compile_variant(source, output_dir, args.build_profile, args.dataset.expanduser().resolve())
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
