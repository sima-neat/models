#!/usr/bin/env python3
"""Finalize a qualified SSDLite checkpoint and emit the immutable source bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import onnx
import torch
from contract import INPUT_NAME, INPUT_SHAPE, OUTPUTS, output_names
from model import IMAGENET_MEAN, IMAGENET_STD, WEIGHTS_SHA256, WEIGHTS_URL, build_raw_model
from onnx import helper
from onnx_audit import audit_onnx, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-ap-drop", type=float, default=1.5)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def validate_checkpoint(checkpoint: dict[str, Any], *, allow_smoke: bool, maximum_ap_drop: float) -> dict[str, float]:
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("unsupported or missing QAT checkpoint schema")
    if checkpoint.get("architecture") != "torchvision.ssdlite320_mobilenet_v3_large":
        raise RuntimeError("checkpoint is not the pinned TorchVision SSDLite architecture")
    if checkpoint.get("weights_url") != WEIGHTS_URL:
        raise RuntimeError("checkpoint does not originate from the pinned TorchVision weights")
    checkpoint_weights_sha256 = checkpoint.get("weights_sha256")
    if checkpoint_weights_sha256 not in {None, WEIGHTS_SHA256}:
        raise RuntimeError("checkpoint TorchVision weight digest does not match the pinned source")
    if not checkpoint.get("observers_frozen") and not allow_smoke:
        raise RuntimeError("QAT observers were not frozen before export")
    if not checkpoint.get("qualification_eligible") and not allow_smoke:
        raise RuntimeError("diagnostic or partial-data QAT checkpoint cannot be published")
    baseline = checkpoint.get("baseline")
    history = checkpoint.get("history") or []
    final_coco = history[-1].get("coco") if history else None
    if baseline is None or final_coco is None:
        if not allow_smoke:
            raise RuntimeError("QAT checkpoint lacks FP32 and final COCO accuracy evidence")
        return {}
    ap_drop = float(baseline["ap"]) - float(final_coco["ap"])
    if ap_drop > maximum_ap_drop / 100.0 and not allow_smoke:
        raise RuntimeError(f"QAT accuracy drop is {ap_drop * 100.0:.3f} AP points; maximum is {maximum_ap_drop:.3f}")
    return {
        "fp32_ap": float(baseline["ap"]),
        "qat_ap": float(final_coco["ap"]),
        "ap_drop_points": ap_drop * 100.0,
    }


def set_provenance(path: Path, metadata: dict[str, str]) -> None:
    model = onnx.load(path, load_external_data=False)
    current = {item.key: item.value for item in model.metadata_props}
    current.update(metadata)
    helper.set_model_props(model, current)
    onnx.save(model, path)


def export_fp32(path: Path, device: torch.device) -> None:
    raw_model, _detector = build_raw_model(pretrained=True)
    raw_model.to(device).eval()
    example = torch.rand(INPUT_SHAPE, device=device)
    torch.onnx.export(
        raw_model,
        example,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=output_names(),
    )


def export_qat(path: Path, checkpoint: dict[str, Any], device: torch.device) -> None:
    from sima_qat.qat_api import sima_export_onnx, sima_finalize_qat_model, sima_prepare_qat_model

    raw_model, _detector = build_raw_model(pretrained=True)
    raw_model.train()
    qat_model = sima_prepare_qat_model(raw_model, (torch.rand(2, 3, 320, 320),), device)
    qat_model.load_state_dict(checkpoint["qat_state_dict"], strict=True)
    finalized = sima_finalize_qat_model(qat_model)
    example = (torch.rand(INPUT_SHAPE, device=device),)
    sima_export_onnx(
        finalized,
        example,
        str(path),
        input_names=[INPUT_NAME],
        output_names=output_names(),
        device=device,
    )


def deterministic_bundle(destination: Path, members: dict[str, bytes]) -> None:
    with destination.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(members):
                    payload = members[name]
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    member.mtime = 0
                    member.mode = 0o644
                    member.uid = 0
                    member.gid = 0
                    member.uname = ""
                    member.gname = ""
                    archive.addfile(member, fileobj=io.BytesIO(payload))


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    accuracy = validate_checkpoint(
        checkpoint,
        allow_smoke=args.allow_smoke,
        maximum_ap_drop=args.maximum_ap_drop,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = args.output_dir / "ssd_mobilenet_v3_torchvision_fp32.onnx"
    qat_path = args.output_dir / "ssd_mobilenet_v3_torchvision_qat_int8.onnx"
    export_fp32(fp32_path, device)
    export_qat(qat_path, checkpoint, device)
    provenance = {
        "sima.architecture": "torchvision.ssdlite320_mobilenet_v3_large",
        "sima.input.normalization": "imagenet_mean_std",
        "sima.output.order": "loc[0..5],confidence[0..5]",
        "sima.torchvision.weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        "sima.torchvision.weights_url": WEIGHTS_URL,
    }
    set_provenance(fp32_path, {**provenance, "sima.precision.source": "fp32"})
    set_provenance(
        qat_path,
        {
            **provenance,
            "sima.precision.source": "int8_qat_qdq",
            "sima.qat.checkpoint_sha256": checkpoint_sha256,
        },
    )
    fp32_audit = audit_onnx(fp32_path, expect_qat=False)
    qat_audit = audit_onnx(qat_path, expect_qat=True)
    manifest = {
        "schema_version": 1,
        "architecture": "torchvision.ssdlite320_mobilenet_v3_large",
        "accuracy": accuracy,
        "checkpoint_sha256": checkpoint_sha256,
        "input": {
            "dtype": "float32",
            "name": INPUT_NAME,
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
            "shape": list(INPUT_SHAPE),
        },
        "outputs": [{"name": name, "shape": list(shape)} for name, shape in OUTPUTS],
        "sources": {
            "bf16": {"file": fp32_path.name, "audit": fp32_audit},
            "int8": {"file": qat_path.name, "audit": qat_audit},
        },
        "training": {
            "accumulation_steps": int(checkpoint["arguments"]["accumulation_steps"]),
            "batch_size": int(checkpoint["arguments"]["batch_size"]),
            "completed_epochs": int(checkpoint["epoch"]),
            "freeze_observers_fraction": float(checkpoint["arguments"]["freeze_observers_fraction"]),
            "learning_rate": float(checkpoint["arguments"]["learning_rate"]),
            "lineage": checkpoint.get("lineage", []),
            "observer_warmup_images": int(checkpoint["arguments"]["observer_warmup_images"]),
            "seed": int(checkpoint["arguments"]["seed"]),
            "weight_decay": float(checkpoint["arguments"]["weight_decay"]),
        },
        "weights_sha256": WEIGHTS_SHA256,
        "weights_url": WEIGHTS_URL,
        "versions": checkpoint.get("versions", {}),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    bundle = args.output_dir / "ssd_mobilenet_v3_torchvision_qat_source.tar.gz"
    deterministic_bundle(
        bundle,
        {
            fp32_path.name: fp32_path.read_bytes(),
            qat_path.name: qat_path.read_bytes(),
            "manifest.json": manifest_bytes,
        },
    )
    report = {
        "bundle": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "manifest": manifest,
    }
    (args.output_dir / "export-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
