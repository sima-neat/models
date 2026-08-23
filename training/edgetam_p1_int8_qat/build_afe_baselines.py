#!/usr/bin/env python3
"""Materialize the two EdgeTAM P1 quantized AFE baselines.

The calibration tensors create the target graph and workspace controls only.
The selected QAT catalogue remains authoritative and is applied through the
typed Quantization Workspace before any compile or publication.
"""

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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-onnx", type=Path, required=True)
    parser.add_argument("--recurrent-onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--constants", type=Path, required=True)
    parser.add_argument("--attention-captures", type=Path, required=True)
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

    images = np.load(args.images, allow_pickle=False)
    constants = np.load(args.constants, allow_pickle=False)

    prompt_inputs = {
        "image": (1, 3, 1024, 1024),
        "dec_sparse_embedding": (1, 3, 256),
    }
    recurrent_inputs = {
        "image": (1, 3, 1024, 1024),
        "att_current_position": (1, 256, 64, 64),
        "att_memory_features": (1, 1, 512, 64),
        "att_memory_position": (1, 1, 512, 64),
        "att_object_pointers": (1, 1, 256),
        "dec_sparse_embedding": (1, 3, 256),
    }

    def source(path: Path, shapes: dict[str, tuple[int, ...]]):
        return onnx_source(
            model_path=str(path),
            shape_dict=shapes,
            dtype_dict={name: ScalarType.float32 for name in shapes},
        )

    prompt_calibration = []
    recurrent_calibration = []
    bindings = []
    for sequence in ("dog", "cups"):
        prompt_calibration.append(
            {
                "image": np.ascontiguousarray(images[f"{sequence}_f00"][None]),
                "dec_sparse_embedding": np.ascontiguousarray(constants[f"{sequence}_frame0_sparse_embedding"]),
            }
        )
        capture_path = args.attention_captures / f"{sequence}_f01_attention.npz"
        capture = np.load(capture_path, allow_pickle=False)
        recurrent_calibration.append(
            {
                "image": np.ascontiguousarray(images[f"{sequence}_f01"][None]),
                "att_current_position": np.ascontiguousarray(capture["current_position"]),
                "att_memory_features": np.ascontiguousarray(capture["memory_features"]),
                "att_memory_position": np.ascontiguousarray(capture["memory_position"]),
                "att_object_pointers": np.ascontiguousarray(capture["object_pointers"]),
                "dec_sparse_embedding": np.ascontiguousarray(constants["tracking_sparse_embedding"]),
            }
        )
        bindings.append(
            {
                "sequence": sequence,
                "attention_capture": str(capture_path),
                "attention_capture_sha256": sha256_file(capture_path),
            }
        )

    config = (
        default_quantization.with_activation_quantization(quantization_scheme(True, False, 8))
        .with_weight_quantization(quantization_scheme(False, True, 8))
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("mse"))
        .with_quantization_mla(True)
        .with_tessellation_mla(True)
    )
    entries = []
    for name, onnx_path, shapes, calibration in (
        ("image_prompt_p3", args.prompt_onnx, prompt_inputs, prompt_calibration),
        (
            "recurrent_s1_p1",
            args.recurrent_onnx,
            recurrent_inputs,
            recurrent_calibration,
        ),
    ):
        loaded = load_model(source(onnx_path, shapes), target=gen2_target)
        model = loaded.quantize(
            calibration_data=calibration,
            quantization_config=config,
            any_shape_on_mla=True,
            automatic_layout_conversion=False,
            model_name=f"edgetam_{name}_qat_baseline",
        )
        component_dir = args.output_dir / name
        component_dir.mkdir()
        model.save(
            model_name=f"edgetam_{name}_qat_baseline",
            output_directory=str(component_dir),
            include_unquantized_net=True,
        )
        artifacts = []
        for artifact in sorted(component_dir.glob("*.sima")):
            artifacts.append(
                {
                    "path": str(artifact),
                    "size": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            )
        entries.append(
            {
                "component": name,
                "source": str(onnx_path),
                "source_sha256": sha256_file(onnx_path),
                "calibration_samples": len(calibration),
                "artifacts": artifacts,
            }
        )
    receipt = {
        "schema_version": 1,
        "status": "BASELINES_MATERIALIZED_QAT_CATALOGUE_NOT_YET_APPLIED",
        "components": entries,
        "calibration_bindings": bindings,
        "images_sha256": sha256_file(args.images),
        "constants_sha256": sha256_file(args.constants),
        "compile": False,
        "board": False,
    }
    atomic_json(args.output_dir / "AFE_BASELINES_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
