#!/usr/bin/env python3
"""Run target-exact SiMa INT8 QAT on a fused YOLO26n-P2 detector."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from contract import INPUT_SHAPE
from dataset import DroneDataset, FixedBatchCollator, load_manifest, sha256_file
from evaluate import evaluate_model
from model import Yolo26TrainingDeployWrapper, fuse_batch_norm_for_qat, validate_deploy_outputs
from provenance import require_ultralytics_revision
from torch import nn
from torch.ao.quantization import disable_observer
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--observer-warmup-images", type=int, default=128)
    parser.add_argument("--freeze-observers-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.observer_warmup_images) < 1:
        parser.error("epochs, batch size, accumulation steps, and observer images must be positive")
    if args.workers < 0:
        parser.error("workers must be >= 0")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay must be >= 0")
    if not 0.0 < args.freeze_observers_fraction < 1.0:
        parser.error("freeze-observers-fraction must be in (0, 1)")
    if args.observer_warmup_images < 128 and not args.smoke:
        parser.error("production QAT requires at least 128 representative observer images")
    if args.batch_size != 1:
        parser.error("target-exact YOLO26 QAT requires batch size 1; use accumulation steps")
    if args.epochs < 2 and not args.smoke:
        parser.error("production QAT needs at least two epochs so frozen observers receive refinement")
    if not args.manifest.is_file() or not args.fp32_checkpoint.is_file():
        parser.error("manifest and FP32 checkpoint must exist")
    return args


def qat_package_version() -> str:
    try:
        return importlib.metadata.version("swml-qat")
    except importlib.metadata.PackageNotFoundError:
        import sima_qat

        return str(getattr(sima_qat, "__version__", "unregistered-site-packages"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(dataset: DroneDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(args.seed + (1 if shuffle else 2))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=FixedBatchCollator(args.batch_size),
        generator=generator,
    )


def atomic_save(document: dict[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def warm_observers(model: nn.Module, data: DataLoader[Any], device: torch.device, required: int) -> int:
    model.eval()
    observed = 0
    for batch in data:
        model(batch["img"].to(device, non_blocking=True))
        observed += int(batch["valid_count"])
        if observed >= required:
            break
    if observed < required:
        raise RuntimeError(f"observer warm-up saw {observed} distinct images; required {required}")
    return observed


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = load_manifest(args.manifest)
    train_records = [record for record in records if record.split == "train"]
    if len(train_records) < args.observer_warmup_images:
        raise RuntimeError(
            f"manifest has {len(train_records)} distinct train frames; "
            f"observer warm-up requires {args.observer_warmup_images}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_data = loader(DroneDataset(records, "train", augment=True, seed=args.seed), args, True)
    observer_data = loader(DroneDataset(records, "train", augment=False, seed=args.seed), args, False)
    val_data = loader(DroneDataset(records, "val", augment=False, seed=args.seed), args, False)

    import ultralytics
    from sima_qat.qat_api import sima_prepare_qat_model
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    ultralytics_revision = require_ultralytics_revision(Path(ultralytics.__file__).resolve().parent)

    detector = YOLO(str(args.fp32_checkpoint)).model
    detector.args = get_cfg(overrides=dict(getattr(detector, "args", {})))
    fuse_batch_norm_for_qat(detector)
    detector.to(device).requires_grad_(True).train()
    criterion = detector.init_criterion()
    wrapped = Yolo26TrainingDeployWrapper(detector).to(device).train()
    baseline = evaluate_model(wrapped, detector.model[-1], val_data, device)
    example = (torch.rand((args.batch_size, *INPUT_SHAPE[1:]), device=device),)
    qat_model = sima_prepare_qat_model(wrapped, example, device)
    observed = warm_observers(qat_model, observer_data, device, args.observer_warmup_images)

    optimizer = torch.optim.SGD(
        qat_model.parameters(),
        lr=args.learning_rate,
        momentum=0.937,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    freeze_after = max(1, math.floor(args.epochs * args.freeze_observers_fraction))
    observers_frozen = False
    history = []
    final_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        if epoch > freeze_after and not observers_frozen:
            qat_model.apply(disable_observer)
            observers_frozen = True
        qat_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        started = time.monotonic()
        for step, batch in enumerate(train_data, start=1):
            images = batch["img"].to(device, non_blocking=True)
            for key in ("batch_idx", "cls", "bboxes"):
                batch[key] = batch[key].to(device, non_blocking=True)
            outputs = qat_model(images)
            validate_deploy_outputs(outputs["deploy"], images.shape[0])
            loss, components = criterion(outputs["predictions"], batch)
            scalar_loss = loss.sum()
            if not torch.isfinite(scalar_loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}")
            (scalar_loss / args.accumulation_steps).backward()
            if step % args.accumulation_steps == 0 or step == len(train_data):
                torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(scalar_loss.detach())
        scheduler.step()
        final_metrics = evaluate_model(qat_model, detector.model[-1], val_data, device)
        entry = {
            "epoch": epoch,
            "loss": loss_total / len(train_data),
            "metrics": final_metrics,
            "observers_frozen": observers_frozen,
            "seconds": time.monotonic() - started,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        atomic_save(
            {
                "schema_version": 1,
                "architecture": "yolo26n-p2-int8-conv-one-class",
                "qat_state_dict": qat_model.state_dict(),
                "epoch": epoch,
                "arguments": vars(args),
                "baseline": baseline,
                "final_metrics": final_metrics,
                "history": history,
                "observers_frozen": observers_frozen,
                "observer_images": observed,
                "manifest_sha256": sha256_file(args.manifest),
                "fp32_checkpoint_sha256": sha256_file(args.fp32_checkpoint),
                "qualification_eligible": not args.smoke and observers_frozen,
                "versions": {
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                    "ultralytics_revision": ultralytics_revision,
                    "swml_qat": qat_package_version(),
                },
            },
            args.output_dir / "checkpoint-latest.pt",
        )
    if not observers_frozen and not args.smoke:
        raise RuntimeError("production QAT completed without a frozen-observer refinement epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
