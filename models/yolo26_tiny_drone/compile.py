#!/usr/bin/env python3
"""Compile the qualified YOLO26-P2 tiny-drone source into its registry MPK."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from calibration import validate_calibration
from model_registry.fetch import Fetcher
from model_registry.source import SourceSpec

MODEL_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODEL_DIR.parents[1]
TRAINING_DIR = REPOSITORY_ROOT / "training" / "yolo26_tiny_drone"
sys.path.insert(0, str(TRAINING_DIR))

from compile import (  # noqa: E402
    MODEL_NAME,
    audit_mpk,
    compile_model,
    preprocess_rgb_image,
)
from contract import DeploymentContract  # noqa: E402
from onnx_audit import audit_onnx  # noqa: E402


def variant_config(variant_id: str) -> dict[str, Any]:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    matches = [item for item in metadata["compile_variants"] if item["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one YOLO26 tiny-drone variant named {variant_id!r}")
    return matches[0]


def calibration_source() -> SourceSpec:
    metadata = yaml.safe_load((MODEL_DIR / "model.yaml").read_text(encoding="utf-8"))
    calibration = metadata["calibration"]
    if calibration.get("status") != "ready":
        raise ValueError("YOLO26 tiny-drone compilation requires pinned calibration data")
    return SourceSpec.from_mapping(calibration["source"])


def compile_variant(source: Path, output_dir: Path, variant_id: str) -> Path:
    variant = variant_config(variant_id)
    parameters = variant["parameters"]
    if (
        variant["target"] != "modalix"
        or parameters.get("precision") != "int8"
        or parameters.get("tessellation_owner") != "mla"
    ):
        raise ValueError("the qualified tiny-drone recipe supports only INT8 with MLA tensor layout")
    contract = DeploymentContract(model_size=int(parameters["model_size"]))
    source_audit = audit_onnx(source, contract)
    if source_audit["convolution_count"] != int(parameters["expected_convolutions"]):
        raise RuntimeError(
            "qualified source convolution count changed: "
            f"expected {parameters['expected_convolutions']}, found {source_audit['convolution_count']}"
        )

    calibration_override = os.environ.get("CALIBRATION_PATH", "").strip()
    if calibration_override:
        calibration_path = Path(calibration_override).expanduser().resolve()
        if not calibration_path.is_file():
            raise FileNotFoundError(f"calibration snapshot does not exist: {calibration_path}")
    else:
        calibration_path, _cache_hit = Fetcher().fetch(calibration_source())
    calibration, calibration_report = validate_calibration(
        calibration_path,
        contract,
        expected_tensor_sha256=str(parameters["calibration_tensor_sha256"]),
        preprocess_rgb_image=preprocess_rgb_image,
    )

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    archive, quantized_report, layout_report = compile_model(
        source,
        calibration,
        output_dir,
        MODEL_NAME,
        contract,
        int(parameters["expected_convolutions"]),
    )
    report = {
        "schema_version": 1,
        "model_id": "yolo26_tiny_drone",
        "variant_id": variant_id,
        "source": source_audit,
        "calibration": calibration_report,
        "quantized_model": quantized_report,
        "mla_layout": layout_report,
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
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    source = args.model.expanduser().resolve()
    if not source.is_file():
        parser.error(f"model does not exist: {source}")
    archive = compile_variant(source, args.output_dir.expanduser().resolve(), args.variant)
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
