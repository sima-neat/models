#!/usr/bin/env python3
"""Export the selected EdgeTAM QAT qparameter catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.resolve()
    sys.path[:0] = [
        str(bundle / "tools"),
        str(bundle / "runtime/site-packages"),
        str(bundle / "vendor"),
    ]
    import model_qat
    from target_real_affine_qat_v2 import (
        QAT_AFFINE,
        LearnedAffineA8,
        LearnedPerOutputW8,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    selected_state = checkpoint["model"]
    model, fold = model_qat.build_float_training_model("cpu")
    coverage = model_qat.convert_target_real(model)
    model_qat.materialize_qparams_from_state(model, selected_state)
    result = model.load_state_dict(selected_state, strict=True)
    if str(result) != "<All keys matched successfully>":
        raise RuntimeError(f"selected EdgeTAM state did not load strictly: {result}")
    audit = model_qat.qparam_audit(model)
    if not audit["pass"] or audit["a8_modules"] != 451 or audit["w8_modules"] != 221:
        raise RuntimeError(f"selected EdgeTAM qparameter audit failed: {audit}")

    activations = []
    weights = []
    affines = []
    for name, module in model.named_modules():
        if isinstance(module, LearnedAffineA8):
            scale, zero_point = module.qparams()
            activations.append(
                {
                    "module": name,
                    "dtype": "int8",
                    "qmin": -128,
                    "qmax": 127,
                    "scale": float(scale.detach()),
                    "zero_point": int(zero_point.detach().round()),
                    "rounding": "ties_to_even",
                }
            )
        elif isinstance(module, LearnedPerOutputW8):
            scales = module.log_scale.detach().exp().clamp_min(1e-8)
            weights.append(
                {
                    "module": name,
                    "dtype": "int8",
                    "qmin": -127,
                    "qmax": 127,
                    "axis": module.axis,
                    "zero_point": 0,
                    "scales": scales.tolist(),
                }
            )
        elif isinstance(module, QAT_AFFINE):
            affines.append(
                {
                    "module": name,
                    "kind": type(module).__name__,
                    "base_weight_sha256": semantic_tensor_sha256(module.base.weight),
                    "base_bias_sha256": (
                        semantic_tensor_sha256(module.base.bias) if module.base.bias is not None else None
                    ),
                    "input_quantizer": name + ".inq",
                    "weight_quantizer": name + ".wq",
                    "output_quantizer": name + ".outq",
                }
            )

    catalogue = {
        "schema_version": 1,
        "model": "edgetam_p1_int8_qat",
        "checkpoint": {
            "sha256": sha256_file(args.checkpoint),
            "stage": checkpoint["stage"],
            "initial_state_sha256": checkpoint["init_sha256"],
            "schedule_sha256": checkpoint["schedule_sha256"],
        },
        "source": {
            "checkpoint_sha256": coverage["source_checkpoint_sha256"],
            "config_sha256": coverage["source_config_sha256"],
        },
        "deployment_contract": {
            "prompt_component": "image_prompt_p3",
            "recurrent_component": "s1_p1",
            "memory_slots": 1,
            "object_pointer_contract": "p1",
            "upstream_s7_p16_equivalence_claimed": False,
        },
        "fold": fold,
        "coverage": coverage,
        "affines": affines,
        "a8": activations,
        "w8": weights,
        "qparam_audit": audit,
        "evidence_level": "PyTorch fake-quantized state catalogue; not AFE, MPK, simulator, or board evidence",
    }
    if len(affines) != 221 or len(activations) != 451 or len(weights) != 221:
        raise RuntimeError(
            f"unexpected EdgeTAM catalogue cardinality: {len(affines)}, {len(activations)}, {len(weights)}"
        )
    write_json(args.output_dir / "qat_catalogue.json", catalogue)
    print(
        json.dumps(
            {
                "status": "PASS",
                "affines": len(affines),
                "a8": len(activations),
                "w8": len(weights),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
