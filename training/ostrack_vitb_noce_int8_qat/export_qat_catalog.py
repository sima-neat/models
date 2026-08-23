#!/usr/bin/env python3
"""Export the exact selected OSTrack QAT catalogue and a bound preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import qat_base
import target_real_topology as topology
import torch


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
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--sample-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    qat_base.set_determinism(20260820, args.threads)

    model = qat_base.build_float_model(args.bundle)
    qat_base.replace_qat(model)
    layer_norms, gelus = topology.replace_ln_gelu(model)
    topology.attach_attention_residual(model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(checkpoint["model_state"], strict=True)
    if str(result) != "<All keys matched successfully>":
        raise RuntimeError(f"selected QAT state did not load strictly: {result}")
    topology.set_qat_mode(qat_base, model)
    model.eval()

    learned = []
    for name, module in qat_base.qat_modules(model):
        weight = module.weight.detach().float()
        reduction = tuple(range(1, weight.ndim))
        scales = weight.abs().amax(dim=reduction).clamp_min(1e-8) / 127.0
        learned.append(
            {
                "module": name,
                "kind": type(module).__name__,
                "activation": {
                    "dtype": "int8",
                    "qmin": -127,
                    "qmax": 127,
                    "zero_point": 0,
                    "scale": module.activation_scale(),
                },
                "weight": {
                    "dtype": "int8",
                    "axis": 0,
                    "qmin": -127,
                    "qmax": 127,
                    "zero_point": 0,
                    "scales": scales.tolist(),
                    "float_weight_sha256": semantic_tensor_sha256(weight),
                },
            }
        )

    boundaries = [
        {
            "module": name,
            "role": module.role,
            "dtype": "int8",
            "qmin": -127,
            "qmax": 127,
            "zero_point": 0,
            "scale": module.scale(),
        }
        for name, module in topology.extra_qmods(model)
    ]
    attention = []
    for index, block in enumerate(model.backbone.blocks):
        attention.append(
            {
                "block": index,
                "qk_program": "mla_qk_pow2/p64_hwc16_v2",
                "query_static_base": float(block.attn.q_base),
                "key_static_base": float(block.attn.k_base),
                "probability": {
                    "dtype": "int8",
                    "qmin": 0,
                    "qmax": 127,
                    "zero_point": 0,
                    "scale": 1 / 127,
                    "learnable": False,
                },
            }
        )

    sample = np.load(args.sample_npz, allow_pickle=False)
    template = torch.from_numpy(np.ascontiguousarray(sample["template"][:1])).float()
    search = torch.from_numpy(np.ascontiguousarray(sample["search"][:1])).float()
    with torch.inference_mode():
        output = model(template, search)
    preview_path = args.output_dir / "pytorch_qat_preview.npz"
    np.savez_compressed(
        preview_path,
        score_map=output["score_map"].numpy(),
        size_map=output["size_map"].numpy(),
        offset_map=output["offset_map"].numpy(),
        pred_boxes=output["pred_boxes"].numpy(),
    )

    catalogue = {
        "schema_version": 1,
        "model": "ostrack_vitb_noce_int8_qat",
        "checkpoint": {
            "sha256": qat_base.sha256_file(args.checkpoint),
            "label": checkpoint["label"],
            "update": checkpoint["update"],
        },
        "source_onnx": {
            "sha256": qat_base.sha256_file(args.source_onnx),
            "size_bytes": args.source_onnx.stat().st_size,
        },
        "contract": {
            "template_shape": [1, 3, 128, 128],
            "search_shape": [1, 3, 256, 256],
            "candidate_elimination": False,
        },
        "learned_affines": learned,
        "activation_boundaries": boundaries,
        "attention": attention,
        "coverage": {
            "learned_modules": len(learned),
            "learned_callsites": 65,
            "layer_norms": len(layer_norms),
            "gelus": len(gelus),
            "qk": len(attention),
            "av": len(attention),
            "residual_requantizations": 24,
        },
        "preview": {
            "sample_npz_sha256": qat_base.sha256_file(args.sample_npz),
            "payload_sha256": qat_base.sha256_file(preview_path),
        },
        "evidence_level": "PyTorch fake-quantized preview; not AFE, MPK, simulator, or board evidence",
    }
    if catalogue["coverage"] != {
        "learned_modules": 64,
        "learned_callsites": 65,
        "layer_norms": 25,
        "gelus": 12,
        "qk": 12,
        "av": 12,
        "residual_requantizations": 24,
    }:
        raise RuntimeError(f"unexpected OSTrack QAT coverage: {catalogue['coverage']}")
    write_json(args.output_dir / "qat_catalogue.json", catalogue)
    print(json.dumps({"status": "PASS", "coverage": catalogue["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
