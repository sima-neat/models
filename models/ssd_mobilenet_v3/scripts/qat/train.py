#!/usr/bin/env python3
"""Fine-tune TorchVision SSDLite320-MobileNetV3 with SiMa PT2E QAT."""

from __future__ import annotations

import argparse
import hashlib
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
from dataset import (
    build_coco_dataset,
    collate_detection,
    deterministic_subset,
    seed_worker,
)
from evaluate import evaluate_coco
from model import (
    WEIGHTS_SHA256,
    WEIGHTS_URL,
    build_raw_model,
    compute_losses,
    resize_batch,
    validate_raw_outputs,
)
from torch import Tensor, nn
from torch.ao.quantization import disable_observer
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--restart-optimizer",
        action="store_true",
        help="continue a completed QAT stage with a new learning rate while preserving learned observers",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--observer-warmup-images", type=int, default=128)
    parser.add_argument("--freeze-observers-fraction", type=float, default=0.9)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.accumulation_steps <= 0:
        parser.error("epochs, batch size, and accumulation steps must be positive")
    if not 0.0 < args.freeze_observers_fraction <= 1.0:
        parser.error("--freeze-observers-fraction must be in (0, 1]")
    if args.observer_warmup_images < 128 and not args.smoke:
        parser.error("production QAT requires at least 128 observer warm-up images")
    if (args.max_train_samples is not None or args.max_val_samples is not None) and not args.smoke:
        parser.error("sample limits are diagnostic only and require --smoke")
    if args.restart_optimizer and args.resume is None:
        parser.error("--restart-optimizer requires --resume")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    return [{name: value.to(device, non_blocking=True) for name, value in target.items()} for target in targets]


def atomic_torch_save(document: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def make_loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=drop_last,
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.inference_mode()
def warm_observers(
    qat_model: nn.Module,
    detector: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    required_images: int,
) -> None:
    observed = 0
    qat_model.eval()
    for images, _targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        image_list, _targets, _sizes = resize_batch(detector, images, None)
        outputs = qat_model(image_list.tensors)
        validate_raw_outputs(outputs, len(images))
        observed += len(images)
        if observed >= required_images:
            break
    if observed < required_images:
        raise RuntimeError(f"observer warm-up saw {observed} images; required {required_images}")
    print(f"Initialized QAT observers with {observed} representative training images", flush=True)


def train_epoch(
    qat_model: nn.Module,
    detector: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    epoch: int,
) -> dict[str, float]:
    qat_model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"bbox_regression": 0.0, "classification": 0.0, "total": 0.0}
    started = time.monotonic()
    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        image_list, resized_targets, _original_sizes = resize_batch(detector, images, targets)
        if resized_targets is None:
            raise RuntimeError("training targets disappeared during resize")
        outputs = qat_model(image_list.tensors)
        validate_raw_outputs(outputs, len(images))
        losses = compute_losses(detector, image_list, resized_targets, outputs)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}: {losses}")
        (loss / accumulation_steps).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        totals["bbox_regression"] += float(losses["bbox_regression"].detach())
        totals["classification"] += float(losses["classification"].detach())
        totals["total"] += float(loss.detach())
        if step == 1 or step % 100 == 0 or step == len(loader):
            elapsed = max(time.monotonic() - started, 1e-6)
            images_seen = step * loader.batch_size
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={float(loss):.5f} "
                f"images_per_second={images_seen / elapsed:.2f}",
                flush=True,
            )
    return {name: value / len(loader) for name, value in totals.items()}


def checkpoint_document(
    *,
    qat_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict[str, Any]],
    baseline: dict[str, float] | None,
    observers_frozen: bool,
    qualification_eligible: bool,
    lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "torchvision.ssdlite320_mobilenet_v3_large",
        "weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "epoch": epoch,
        "qat_state_dict": qat_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "observers_frozen": observers_frozen,
        "qualification_eligible": qualification_eligible,
        "history": history,
        "baseline": baseline,
        "arguments": vars(args),
        "lineage": lineage,
        "versions": {
            "python": ".".join(map(str, os.sys.version_info[:3])),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "swml_qat": importlib.metadata.version("swml-qat"),
        },
    }


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_base = build_coco_dataset(args.coco_root, "train2017", training=True)
    val_base = build_coco_dataset(args.coco_root, "val2017", training=False)
    train_dataset = deterministic_subset(train_base, args.max_train_samples, args.seed)
    val_dataset = deterministic_subset(val_base, args.max_val_samples, args.seed + 1)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        drop_last=False,
        seed=args.seed + 1,
    )

    raw_model, detector = build_raw_model(pretrained=True)
    raw_model.eval()
    detector.to(device).eval()
    baseline = None
    if not args.skip_baseline_eval:
        raw_model.to(device).eval()
        baseline = evaluate_coco(raw_model, detector, val_loader, val_base.coco, device)
        print(f"FP32 baseline: {json.dumps(baseline, sort_keys=True)}", flush=True)
        raw_model.to("cpu")
        torch.cuda.empty_cache()

    from sima_qat.qat_api import sima_prepare_qat_model

    raw_model.train()
    example = (torch.rand(2, 3, 320, 320),)
    qat_model = sima_prepare_qat_model(raw_model, example, device)
    detector.to(device).eval()
    optimizer = torch.optim.SGD(
        qat_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1
    history: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    observers_frozen = False
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != "torchvision.ssdlite320_mobilenet_v3_large":
            raise RuntimeError("resume checkpoint architecture does not match SSDLite-MobileNetV3")
        planned_epochs = int(checkpoint.get("arguments", {}).get("epochs", args.epochs))
        checkpoint_epoch = int(checkpoint["epoch"])
        if planned_epochs != args.epochs and not args.restart_optimizer:
            raise RuntimeError(
                f"resume must preserve the original {planned_epochs}-epoch cosine schedule; received {args.epochs}"
            )
        if args.restart_optimizer and args.epochs <= checkpoint_epoch:
            raise RuntimeError(
                f"restarted QAT stage must extend beyond checkpoint epoch {checkpoint_epoch}; received {args.epochs}"
            )
        qat_model.load_state_dict(checkpoint["qat_state_dict"], strict=True)
        if args.restart_optimizer:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - checkpoint_epoch,
            )
            digest = hashlib.sha256(args.resume.read_bytes()).hexdigest()
            lineage = [*checkpoint.get("lineage", [])]
            lineage.append(
                {
                    "checkpoint_sha256": digest,
                    "completed_epoch": checkpoint_epoch,
                    "learning_rate": args.learning_rate,
                    "optimizer_restarted": True,
                    "target_epoch": args.epochs,
                }
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            lineage = list(checkpoint.get("lineage", []))
        history = list(checkpoint.get("history", []))
        baseline = checkpoint.get("baseline") or baseline
        observers_frozen = bool(checkpoint.get("observers_frozen", False))
        if observers_frozen:
            qat_model.apply(disable_observer)
        start_epoch = checkpoint_epoch + 1
    else:
        warm_observers(
            qat_model,
            detector,
            train_loader,
            device,
            args.observer_warmup_images,
        )

    freeze_after_epoch = max(1, math.floor(args.epochs * args.freeze_observers_fraction))
    qualification_eligible = (
        not args.smoke
        and baseline is not None
        and len(train_base) == len(train_dataset)
        and len(val_base) == len(val_dataset)
        and args.observer_warmup_images >= 128
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch > freeze_after_epoch and not observers_frozen:
            qat_model.apply(disable_observer)
            observers_frozen = True
            print(f"Froze QAT observers at epoch {epoch}", flush=True)
        losses = train_epoch(
            qat_model,
            detector,
            train_loader,
            optimizer,
            device,
            args.accumulation_steps,
            epoch,
        )
        if epoch == args.epochs and not observers_frozen:
            qat_model.apply(disable_observer)
            observers_frozen = True
            print(f"Froze QAT observers after final epoch {epoch}", flush=True)
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "losses": losses,
        }
        if epoch == args.epochs or epoch % args.eval_every == 0:
            record["coco"] = evaluate_coco(qat_model, detector, val_loader, val_base.coco, device)
            print(f"QAT epoch {epoch}: {json.dumps(record['coco'], sort_keys=True)}", flush=True)
        history.append(record)
        checkpoint = checkpoint_document(
            qat_model=qat_model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            history=history,
            baseline=baseline,
            observers_frozen=observers_frozen,
            qualification_eligible=qualification_eligible and epoch == args.epochs,
            lineage=lineage,
        )
        atomic_torch_save(checkpoint, args.output_dir / "checkpoint-latest.pt")
        if "coco" in record:
            atomic_torch_save(checkpoint, args.output_dir / f"checkpoint-epoch-{epoch:03d}.pt")
        report = {
            "baseline": baseline,
            "history": history,
            "qualification_eligible": qualification_eligible and epoch == args.epochs,
        }
        (args.output_dir / "training-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
