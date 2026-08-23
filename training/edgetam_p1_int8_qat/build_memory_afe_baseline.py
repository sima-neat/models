#!/usr/bin/env python3
"""Materialize the EdgeTAM P1 memory-encoder AFE baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from afe.apis.defines import (
        CalibrationMethod,
        RequantizationMode,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model, onnx_source
    from afe.ir.tensor_type import ScalarType

    calibration = []
    bindings = []
    for sequence in ("dog", "cups"):
        encoder_path = args.cache / "encoder" / f"{sequence}_f00.npz"
        teacher_path = args.cache / "teacher" / f"{sequence}_prompt.npz"
        encoder = np.load(encoder_path, allow_pickle=False)
        teacher = np.load(teacher_path, allow_pickle=False)
        calibration.append(
            {
                # ModelSDK calibration receives the public TensorFlow/NHWC
                # boundary even though the source ONNX contract is NCHW.
                "pix_feat": np.ascontiguousarray(encoder["o3"]),
                "mask_for_mem": np.ascontiguousarray(teacher["mask"].astype(np.float32)),
            }
        )
        bindings.append(
            {
                "sequence": sequence,
                "encoder_sha256": sha256_file(encoder_path),
                "teacher_sha256": sha256_file(teacher_path),
            }
        )

    source = onnx_source(
        model_path=str(args.onnx),
        shape_dict={"pix_feat": (1, 256, 64, 64), "mask_for_mem": (1, 1, 1024, 1024)},
        dtype_dict={"pix_feat": ScalarType.float32, "mask_for_mem": ScalarType.float32},
    )
    config = (
        default_quantization.with_activation_quantization(quantization_scheme(True, False, 8))
        .with_weight_quantization(quantization_scheme(False, True, 8))
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("mse"))
        .with_quantization_mla(True)
        .with_tessellation_mla(True)
    )
    loaded = load_model(source, target=gen2_target)
    model = loaded.quantize(
        calibration_data=calibration,
        quantization_config=config,
        any_shape_on_mla=True,
        automatic_layout_conversion=False,
        model_name="edgetam_memory_encoder_p1_qat_baseline",
    )
    model.save(
        model_name="edgetam_memory_encoder_p1_qat_baseline",
        output_directory=str(args.output_dir),
        include_unquantized_net=True,
    )
    artifacts = []
    for artifact in sorted(args.output_dir.glob("*.sima")):
        artifacts.append(
            {
                "path": str(artifact),
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "BASELINE_MATERIALIZED_QAT_CATALOGUE_NOT_YET_APPLIED",
        "source_sha256": sha256_file(args.onnx),
        "calibration_bindings": bindings,
        "artifacts": artifacts,
        "compile": False,
        "board": False,
    }
    receipt_path = args.output_dir / "AFE_BASELINE_RECEIPT.json"
    temporary = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
