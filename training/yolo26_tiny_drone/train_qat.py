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
from dataset import DroneDataset, FixedBatchCollator, FrameRecord, load_manifest, sha256_file
from evaluate import evaluate_model
from model import Yolo26TrainingDeployWrapper, fuse_batch_norm_for_qat, validate_deploy_outputs
from provenance import require_ultralytics_revision
from torch import nn
from torch.ao.quantization import disable_observer, enable_observer
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--observer-warmup-images", type=int, default=128)
    parser.add_argument("--freeze-observers-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.observer_warmup_images) < 1:
        parser.error("epochs, batch size, accumulation steps, and observer images must be positive")
    if args.workers < 0:
        parser.error("workers must be >= 0")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay must be >= 0")
    if not 0.0 <= args.freeze_observers_fraction < 1.0:
        parser.error("freeze-observers-fraction must be in [0, 1)")
    if args.observer_warmup_images < 128 and not args.smoke:
        parser.error("production QAT requires at least 128 representative observer images")
    if args.batch_size != 1:
        parser.error("target-exact YOLO26 QAT requires batch size 1; use accumulation steps")
    if args.epochs < 2 and not args.smoke:
        parser.error("production QAT needs at least two epochs so frozen observers receive refinement")
    if not args.manifest.is_file() or not args.fp32_checkpoint.is_file():
        parser.error("manifest and FP32 checkpoint must exist")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        parser.error("resume checkpoint must exist")
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


def representative_observer_records(records: list[FrameRecord], required: int, seed: int) -> list[FrameRecord]:
    """Select a reproducible, non-contiguous observer sample without replacement."""
    train_records = sorted(
        (record for record in records if record.split == "train"),
        key=lambda record: str(record.image),
    )
    if len(train_records) < required:
        raise RuntimeError(
            f"manifest has {len(train_records)} distinct train frames; observer warm-up requires {required}"
        )
    return random.Random(seed).sample(train_records, required)


def atomic_save(document: dict[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_resume_checkpoint(checkpoint: dict[str, Any], manifest: Path, fp32_checkpoint: Path) -> None:
    """Reject a resume source that is not the same frozen production contract."""
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("resume checkpoint has an unsupported schema")
    if checkpoint.get("architecture") != "yolo26n-p2-int8-conv-one-class":
        raise RuntimeError("resume checkpoint has the wrong architecture")
    if not checkpoint.get("observers_frozen"):
        raise RuntimeError("resume checkpoint must have frozen observers")
    if int(checkpoint.get("observer_images", 0)) < 128:
        raise RuntimeError("resume checkpoint did not observe 128 representative images")
    if not checkpoint.get("qualification_eligible"):
        raise RuntimeError("resume checkpoint is not eligible for production qualification")
    if checkpoint.get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("resume checkpoint manifest digest does not match")
    if checkpoint.get("fp32_checkpoint_sha256") != sha256_file(fp32_checkpoint):
        raise RuntimeError("resume checkpoint FP32 digest does not match")
    if not isinstance(checkpoint.get("qat_state_dict"), dict):
        raise RuntimeError("resume checkpoint has no QAT state dictionary")
    if not isinstance(checkpoint.get("baseline"), dict) or not isinstance(checkpoint.get("final_metrics"), dict):
        raise RuntimeError("resume checkpoint has incomplete accuracy evidence")
    observer_selection = checkpoint.get("observer_selection")
    if not isinstance(observer_selection, dict) or not isinstance(observer_selection.get("images"), list):
        raise RuntimeError("resume checkpoint has incomplete observer evidence")
    if len(observer_selection["images"]) != int(checkpoint["observer_images"]):
        raise RuntimeError("resume checkpoint observer count does not match its evidence")


@torch.inference_mode()
def warm_observers(model: nn.Module, data: DataLoader[Any], device: torch.device, required: int) -> list[str]:
    model.eval()
    observed: list[str] = []
    seen: set[str] = set()
    for batch in data:
        model(batch["img"].to(device, non_blocking=True))
        valid_count = int(batch["valid_count"])
        for image in batch["images"][:valid_count]:
            if image in seen:
                raise RuntimeError(f"observer warm-up repeated representative image: {image}")
            seen.add(image)
            observed.append(image)
        if len(observed) >= required:
            break
    if len(observed) != required:
        raise RuntimeError(f"observer warm-up saw {len(observed)} distinct images; required exactly {required}")
    return observed


def evaluate_without_observer_updates(
    model: nn.Module,
    head: nn.Module,
    data: DataLoader[Any],
    device: torch.device,
    *,
    observers_frozen: bool,
) -> dict[str, float]:
    """Keep held-out validation data out of the QAT observer state."""
    if not observers_frozen:
        model.apply(disable_observer)
    try:
        return evaluate_model(model, head, data, device)
    finally:
        if not observers_frozen:
            model.apply(enable_observer)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = load_manifest(args.manifest)
    resume = None
    if args.resume_checkpoint is not None:
        resume = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(resume, args.manifest, args.fp32_checkpoint)
    observer_records = (
        representative_observer_records(records, args.observer_warmup_images, args.seed) if resume is None else []
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    train_data = loader(DroneDataset(records, "train", augment=True, seed=args.seed), args, True)
    observer_data = (
        loader(DroneDataset(observer_records, "train", augment=False, seed=args.seed), args, False)
        if resume is None
        else None
    )
    val_data = loader(DroneDataset(records, "val", augment=False, seed=args.seed), args, False)

    import ultralytics
    from sima_qat.qat_api import sima_prepare_qat_model
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    ultralytics_revision = require_ultralytics_revision(Path(ultralytics.__file__).resolve().parent)
    if resume is not None and resume.get("versions", {}).get("ultralytics_revision") != ultralytics_revision:
        raise RuntimeError("resume checkpoint Ultralytics revision does not match the environment")

    detector = YOLO(str(args.fp32_checkpoint)).model
    detector.args = get_cfg(overrides=dict(getattr(detector, "args", {})))
    fuse_batch_norm_for_qat(detector)
    detector.to(device).requires_grad_(True).train()
    criterion = detector.init_criterion()
    wrapped = Yolo26TrainingDeployWrapper(detector).to(device).train()
    baseline = evaluate_model(wrapped, detector.model[-1], val_data, device) if resume is None else resume["baseline"]
    example = (torch.rand((args.batch_size, *INPUT_SHAPE[1:]), device=device),)
    qat_model = sima_prepare_qat_model(wrapped, example, device)
    if resume is None:
        if observer_data is None:
            raise RuntimeError("fresh QAT has no observer data")
        observed_paths = warm_observers(qat_model, observer_data, device, args.observer_warmup_images)
        observer_evidence = {
            "count": len(observed_paths),
            "images": [{"path": path, "sha256": sha256_file(Path(path))} for path in observed_paths],
            "method": "seeded_random_without_replacement",
            "seed": args.seed,
        }
        initial_qat_metrics = evaluate_without_observer_updates(
            qat_model,
            detector.model[-1],
            val_data,
            device,
            observers_frozen=False,
        )
        start_epoch = 0
        history: list[dict[str, Any]] = []
        observers_frozen = False
        resume_digest = None
        resume_start_metrics = None
    else:
        qat_model.load_state_dict(resume["qat_state_dict"], strict=True)
        qat_model.apply(disable_observer)
        observed_paths = [str(image["path"]) for image in resume["observer_selection"]["images"]]
        observer_evidence = resume["observer_selection"]
        initial_qat_metrics = resume.get("initial_qat_metrics", resume["final_metrics"])
        resume_start_metrics = resume["final_metrics"]
        start_epoch = int(resume["epoch"])
        history = list(resume.get("history", []))
        observers_frozen = True
        resume_digest = sha256_file(args.resume_checkpoint)

    optimizer = torch.optim.SGD(
        qat_model.parameters(),
        lr=args.learning_rate,
        momentum=0.937,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    freeze_after = math.floor(args.epochs * args.freeze_observers_fraction)
    best_frozen_priority: tuple[float, float] | None = None
    final_metrics: dict[str, float] = {}
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        if local_epoch > freeze_after and not observers_frozen:
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
            window_start = ((step - 1) // args.accumulation_steps) * args.accumulation_steps
            window_size = min(args.accumulation_steps, len(train_data) - window_start)
            (scalar_loss / window_size).backward()
            if step % args.accumulation_steps == 0 or step == len(train_data):
                torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(scalar_loss.detach())
        scheduler.step()
        final_metrics = evaluate_without_observer_updates(
            qat_model,
            detector.model[-1],
            val_data,
            device,
            observers_frozen=observers_frozen,
        )
        entry = {
            "epoch": epoch,
            "loss": loss_total / len(train_data),
            "metrics": final_metrics,
            "observers_frozen": observers_frozen,
            "seconds": time.monotonic() - started,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        checkpoint = {
            "schema_version": 1,
            "architecture": "yolo26n-p2-int8-conv-one-class",
            "qat_state_dict": qat_model.state_dict(),
            "epoch": epoch,
            "arguments": vars(args),
            "baseline": baseline,
            "initial_qat_metrics": initial_qat_metrics,
            "resume_start_metrics": resume_start_metrics,
            "final_metrics": final_metrics,
            "history": history,
            "observers_frozen": observers_frozen,
            "observer_images": len(observed_paths),
            "observer_selection": observer_evidence,
            "manifest_sha256": sha256_file(args.manifest),
            "fp32_checkpoint_sha256": sha256_file(args.fp32_checkpoint),
            "resume_checkpoint_sha256": resume_digest,
            "qualification_eligible": not args.smoke and observers_frozen,
            "versions": {
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "ultralytics_revision": ultralytics_revision,
                "swml_qat": qat_package_version(),
            },
        }
        atomic_save(checkpoint, args.output_dir / "checkpoint-latest.pt")
        if observers_frozen:
            priority = (float(final_metrics["ap50"]), float(final_metrics["tiny_recall50"]))
            if best_frozen_priority is None or priority > best_frozen_priority:
                best_frozen_priority = priority
                checkpoint["selection"] = {
                    "policy": "maximum_frozen_ap50_then_tiny_recall50",
                    "selected_epoch": epoch,
                }
                atomic_save(checkpoint, args.output_dir / "checkpoint-best-frozen.pt")
    if not observers_frozen and not args.smoke:
        raise RuntimeError("production QAT completed without a frozen-observer refinement epoch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
