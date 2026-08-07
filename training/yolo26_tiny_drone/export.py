#!/usr/bin/env python3
"""Export a qualified YOLO26-P2 QAT checkpoint to the eight-head INT8 source."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from contract import INPUT_NAME, DeploymentContract
from model import (
    EXPECTED_ATTENTION_MATMULS,
    EXPECTED_ATTENTION_SOFTMAXES,
    Yolo26TrainingDeployWrapper,
    fuse_batch_norm_for_qat,
    lower_attention_for_mla,
    select_deploy_graph,
    validate_attention_qat_scaffolding,
    validate_deploy_outputs,
)
from onnx_audit import audit_onnx, canonicalize_attention_for_modelsdk, sha256_file
from provenance import require_ultralytics_revision


def validate_checkpoint(checkpoint: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("unsupported QAT checkpoint schema")
    if checkpoint.get("architecture") != "yolo26n-p2-attention-one-class":
        raise RuntimeError("checkpoint architecture is not the deployment YOLO26-P2 variant")
    expected_attention = {
        "matmul_count": EXPECTED_ATTENTION_MATMULS,
        "softmax_count": EXPECTED_ATTENTION_SOFTMAXES,
    }
    if checkpoint.get("attention_scaffolding") != expected_attention:
        raise RuntimeError("QAT checkpoint lacks complete INT8 attention scaffolding evidence")
    try:
        DeploymentContract.from_dict(checkpoint.get("deployment"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint deployment contract is invalid") from exc
    if int(checkpoint.get("observer_images", 0)) < 128 and not args.allow_smoke:
        raise RuntimeError("QAT checkpoint did not observe 128 representative images")
    if not checkpoint.get("observers_frozen") and not args.allow_smoke:
        raise RuntimeError("QAT observers were not frozen before export")
    if not checkpoint.get("qualification_eligible") and not args.allow_smoke:
        raise RuntimeError("diagnostic QAT checkpoint cannot be published")
    if checkpoint.get("fp32_qualification_eligible") is not True and not args.allow_smoke:
        raise RuntimeError("QAT checkpoint descends from ineligible FP32 training")
    dataset_digest = checkpoint.get("dataset_sha256")
    if not isinstance(dataset_digest, str) or re.fullmatch(r"[0-9a-f]{64}", dataset_digest) is None:
        raise RuntimeError("QAT checkpoint has no valid dataset content digest")
    baseline = checkpoint.get("baseline") or {}
    final = checkpoint.get("final_metrics") or {}
    if not baseline or not final:
        raise RuntimeError("checkpoint lacks FP32 and QAT accuracy evidence")
    metric_names = ("ap50", "tiny_recall50", "ground_truth_count", "tiny_ground_truth_count")
    try:
        baseline_values = {name: float(baseline[name]) for name in metric_names}
        final_values = {name: float(final[name]) for name in metric_names}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint has incomplete FP32 or QAT accuracy evidence") from exc
    if not all(math.isfinite(value) for value in (*baseline_values.values(), *final_values.values())):
        raise RuntimeError("checkpoint accuracy evidence contains non-finite values")
    for stage, values in (("FP32", baseline_values), ("QAT", final_values)):
        if not 0.0 <= values["ap50"] <= 1.0 or not 0.0 <= values["tiny_recall50"] <= 1.0:
            raise RuntimeError(f"{stage} accuracy metrics must be in [0, 1]")
        if values["ground_truth_count"] < 0.0 or values["tiny_ground_truth_count"] < 0.0:
            raise RuntimeError(f"{stage} ground-truth counts must be non-negative")
        if not values["ground_truth_count"].is_integer() or not values["tiny_ground_truth_count"].is_integer():
            raise RuntimeError(f"{stage} ground-truth counts must be integers")
    if any(baseline_values[name] != final_values[name] for name in ("ground_truth_count", "tiny_ground_truth_count")):
        raise RuntimeError("FP32 and QAT evaluations used different ground-truth populations")
    tiny_ground_truth = int(final_values["tiny_ground_truth_count"])
    if tiny_ground_truth < args.minimum_tiny_ground_truth and not args.allow_smoke:
        raise RuntimeError(
            f"validation has {tiny_ground_truth} tiny objects; at least {args.minimum_tiny_ground_truth} are required"
        )
    ap50_drop = baseline_values["ap50"] - final_values["ap50"]
    tiny_recall_drop = baseline_values["tiny_recall50"] - final_values["tiny_recall50"]
    if ap50_drop > args.maximum_ap50_drop and not args.allow_smoke:
        raise RuntimeError(f"QAT AP50 drop {ap50_drop:.6f} exceeds {args.maximum_ap50_drop:.6f}")
    if tiny_recall_drop > args.maximum_tiny_recall_drop and not args.allow_smoke:
        raise RuntimeError(f"QAT tiny recall drop {tiny_recall_drop:.6f} exceeds {args.maximum_tiny_recall_drop:.6f}")
    if final_values["ap50"] < args.minimum_qat_ap50 and not args.allow_smoke:
        raise RuntimeError(f"QAT AP50 {final_values['ap50']:.6f} is below {args.minimum_qat_ap50:.6f}")
    if final_values["tiny_recall50"] < args.minimum_qat_tiny_recall50 and not args.allow_smoke:
        raise RuntimeError(
            f"QAT tiny recall50 {final_values['tiny_recall50']:.6f} is below {args.minimum_qat_tiny_recall50:.6f}"
        )
    return {
        "fp32_ap50": baseline_values["ap50"],
        "qat_ap50": final_values["ap50"],
        "ap50_drop": ap50_drop,
        "fp32_tiny_recall50": baseline_values["tiny_recall50"],
        "qat_tiny_recall50": final_values["tiny_recall50"],
        "tiny_recall50_drop": tiny_recall_drop,
        "tiny_ground_truth_count": float(tiny_ground_truth),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-ap50-drop", type=float, default=0.02)
    parser.add_argument("--maximum-tiny-recall-drop", type=float, default=0.05)
    parser.add_argument("--minimum-qat-ap50", type=float, default=0.30)
    parser.add_argument("--minimum-qat-tiny-recall50", type=float, default=0.50)
    parser.add_argument("--minimum-tiny-ground-truth", type=int, default=100)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.minimum_qat_ap50 <= 1.0:
        parser.error("minimum-qat-ap50 must be in [0, 1]")
    if not 0.0 <= args.minimum_qat_tiny_recall50 <= 1.0:
        parser.error("minimum-qat-tiny-recall50 must be in [0, 1]")
    if args.minimum_tiny_ground_truth < 1:
        parser.error("minimum-tiny-ground-truth must be positive")
    if not 0.0 <= args.maximum_ap50_drop <= 1.0 or not 0.0 <= args.maximum_tiny_recall_drop <= 1.0:
        parser.error("maximum accuracy drops must be in [0, 1]")
    if not args.checkpoint.is_file() or not args.fp32_checkpoint.is_file():
        parser.error("QAT and FP32 checkpoints must exist")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    accuracy = validate_checkpoint(checkpoint, args)
    contract = DeploymentContract.from_dict(checkpoint.get("deployment"))
    if int(checkpoint["arguments"]["batch_size"]) != 1:
        raise RuntimeError("QAT checkpoint was not captured for the batch-1 deployment contract")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import ultralytics
    from sima_qat.qat_api import sima_export_onnx, sima_finalize_qat_model, sima_prepare_qat_model
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    ultralytics_revision = require_ultralytics_revision(Path(ultralytics.__file__).resolve().parent)
    checkpoint_revision = checkpoint.get("versions", {}).get("ultralytics_revision")
    if checkpoint_revision != ultralytics_revision:
        raise RuntimeError(
            "QAT checkpoint Ultralytics revision does not match the export environment: "
            f"checkpoint={checkpoint_revision}, environment={ultralytics_revision}"
        )
    fp32_digest = sha256_file(args.fp32_checkpoint)
    if fp32_digest != checkpoint.get("fp32_checkpoint_sha256"):
        raise RuntimeError("FP32 checkpoint digest does not match the QAT lineage")

    detector = YOLO(str(args.fp32_checkpoint)).model
    detector.args = get_cfg(overrides=dict(getattr(detector, "args", {})))
    fuse_batch_norm_for_qat(detector)
    detector, lowered_attention_count = lower_attention_for_mla(detector)
    if lowered_attention_count != 2:
        raise RuntimeError(f"expected two YOLO26 attention modules, lowered {lowered_attention_count}")
    detector.to(device).requires_grad_(True).train()
    wrapped = Yolo26TrainingDeployWrapper(detector).to(device).train()
    example = (torch.rand(contract.input_shape, device=device),)
    qat_model = sima_prepare_qat_model(wrapped, example, device)
    attention_scaffolding = validate_attention_qat_scaffolding(qat_model)
    if attention_scaffolding != checkpoint.get("attention_scaffolding"):
        raise RuntimeError("export attention scaffolding does not match the QAT checkpoint")
    qat_model.load_state_dict(checkpoint["qat_state_dict"], strict=True)
    finalized = sima_finalize_qat_model(qat_model)
    deploy_graph = select_deploy_graph(finalized)
    validate_deploy_outputs(deploy_graph(*example), 1, contract)
    onnx_path = args.output_dir / f"yolo26n_p2_tiny_drone_qat_int8_{contract.model_size}.onnx"
    sima_export_onnx(
        deploy_graph,
        example,
        str(onnx_path),
        input_names=[INPUT_NAME],
        output_names=contract.output_names(),
        device=device,
    )
    canonicalization = canonicalize_attention_for_modelsdk(onnx_path)
    audit = audit_onnx(onnx_path, contract)
    report = {
        "schema_version": 1,
        "architecture": "yolo26n-p2-attention-one-class",
        "deployment": contract.to_dict(),
        "accuracy": accuracy,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fp32_checkpoint_sha256": fp32_digest,
        "fp32_lineage_sha256": checkpoint.get("fp32_lineage_sha256"),
        "fp32_qualification_eligible": checkpoint.get("fp32_qualification_eligible") is True,
        "manifest_sha256": checkpoint["manifest_sha256"],
        "dataset_sha256": checkpoint["dataset_sha256"],
        "observer_images": checkpoint["observer_images"],
        "source_qualification_eligible": checkpoint.get("qualification_eligible") is True,
        "qualification_eligible": checkpoint.get("qualification_eligible") is True and not args.allow_smoke,
        "diagnostic_export": args.allow_smoke,
        "onnx": {"file": onnx_path.name, "canonicalization": canonicalization, "audit": audit},
        "versions": checkpoint.get("versions", {}),
    }
    (args.output_dir / "export-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
