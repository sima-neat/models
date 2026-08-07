#!/usr/bin/env python3
"""Run target-exact SiMa INT8 QAT on a fused YOLO26n-P2 detector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from contract import DEFAULT_MODEL_SIZE, DeploymentContract
from dataset import (
    DroneDataset,
    FixedBatchCollator,
    FrameRecord,
    dataset_content_sha256,
    load_manifest,
    require_unchanged_dataset,
    sha256_file,
)
from evaluate import evaluate_model
from model import (
    EXPECTED_ATTENTION_MATMULS,
    EXPECTED_ATTENTION_SOFTMAXES,
    Yolo26TrainingDeployWrapper,
    fuse_batch_norm_for_qat,
    lower_attention_for_mla,
    validate_attention_qat_scaffolding,
    validate_deploy_outputs,
)
from paths import natural_path_key
from provenance import (
    SWML_QAT_PACKAGE_SHA256,
    SWML_QAT_REVISION,
    ULTRALYTICS_PACKAGE_SHA256,
    ULTRALYTICS_REVISION,
    YOLO26N_SHA256,
    activate_verified_swml_qat_runtime,
    require_ultralytics_revision,
    snapshot_file,
)
from torch import nn
from torch.ao.quantization import disable_observer, enable_observer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fp32-checkpoint", type=Path, required=True)
    parser.add_argument("--fp32-lineage", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--trainable-scope", choices=("all", "head", "classification"), default="all")
    parser.add_argument("--class-distillation-weight", type=float, default=0.0)
    parser.add_argument("--observer-warmup-images", type=int, default=128)
    parser.add_argument("--max-train-images", type=int)
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
    if args.class_distillation_weight < 0.0:
        parser.error("class-distillation-weight must be >= 0")
    if not 0.0 <= args.freeze_observers_fraction < 1.0:
        parser.error("freeze-observers-fraction must be in [0, 1)")
    if args.observer_warmup_images < 128 and not args.smoke:
        parser.error("production QAT requires at least 128 representative observer images")
    if args.max_train_images is not None and args.max_train_images < (1 if args.smoke else 128):
        parser.error("max-train-images must be at least 128 for production QAT")
    if args.batch_size != 1:
        parser.error("target-exact YOLO26 QAT requires batch size 1; use accumulation steps")
    try:
        DeploymentContract(args.image_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.epochs < 2 and not args.smoke:
        parser.error("production QAT needs at least two epochs so frozen observers receive refinement")
    if not args.manifest.is_file() or not args.fp32_checkpoint.is_file() or not args.fp32_lineage.is_file():
        parser.error("manifest, FP32 checkpoint, and FP32 lineage must exist")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        parser.error("resume checkpoint must exist")
    if args.trainable_scope != "all" and args.resume_checkpoint is None:
        parser.error("restricted-scope recovery requires --resume-checkpoint")
    if args.class_distillation_weight > 0.0 and args.trainable_scope != "classification":
        parser.error("class distillation requires --trainable-scope classification")
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


def select_qat_training_records(
    records: list[FrameRecord], maximum: int | None
) -> tuple[list[FrameRecord], dict[str, Any]]:
    """Select temporal coverage from every train sequence without frame leakage."""
    train_records = sorted(
        (record for record in records if record.split == "train"),
        key=lambda record: (record.sequence, natural_path_key(record.image)),
    )
    if not train_records:
        raise RuntimeError("manifest has no QAT training frames")
    grouped: dict[str, list[FrameRecord]] = {}
    for record in train_records:
        grouped.setdefault(record.sequence, []).append(record)
    if maximum is None or maximum >= len(train_records):
        selected = train_records
        method = "full_train_split"
    else:
        if maximum < len(grouped):
            raise RuntimeError(f"max-train-images={maximum} cannot cover all {len(grouped)} training sequences")
        quotas = {sequence: 1 for sequence in grouped}
        remaining = maximum - len(grouped)
        capacities = {sequence: len(items) - 1 for sequence, items in grouped.items()}
        total_capacity = sum(capacities.values())
        ideals = {sequence: remaining * capacity / total_capacity for sequence, capacity in capacities.items()}
        for sequence, ideal in ideals.items():
            quotas[sequence] += math.floor(ideal)
        leftover = maximum - sum(quotas.values())
        remainder_order = sorted(
            grouped,
            key=lambda sequence: (
                -(ideals[sequence] - math.floor(ideals[sequence])),
                sequence,
            ),
        )
        for sequence in remainder_order[:leftover]:
            quotas[sequence] += 1
        selected = []
        for sequence, items in sorted(grouped.items()):
            quota = quotas[sequence]
            indices = [(2 * index + 1) * len(items) // (2 * quota) for index in range(quota)]
            selected.extend(items[index] for index in indices)
        method = "proportional_per_sequence_temporal_bins"
    digest = hashlib.sha256()
    per_sequence: dict[str, int] = {}
    for record in selected:
        digest.update(f"{record.sequence}\0{record.image.name}\n".encode())
        per_sequence[record.sequence] = per_sequence.get(record.sequence, 0) + 1
    evidence = {
        "method": method,
        "requested_maximum": maximum,
        "source_train_images": len(train_records),
        "selected_images": len(selected),
        "selected_sequences": len(per_sequence),
        "selection_sha256": digest.hexdigest(),
        "per_sequence_images": per_sequence,
    }
    return selected, evidence


def atomic_save(document: dict[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_fp32_lineage(
    path: Path,
    fp32_checkpoint: Path,
    manifest: Path,
    contract: DeploymentContract,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Require the immutable training lineage and bind it to this QAT run."""
    try:
        lineage = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read FP32 lineage: {path}") from exc
    if lineage.get("schema_version") != 2 or lineage.get("architecture") != "yolo26n-p2-attention-one-class":
        raise RuntimeError("FP32 lineage has an unsupported schema or architecture")
    architecture = Path(__file__).with_name("yolo26n-p2-deploy.yaml")
    if lineage.get("architecture_sha256") != sha256_file(architecture):
        raise RuntimeError("FP32 lineage architecture digest does not match the canonical QAT graph")
    try:
        recorded_contract = DeploymentContract.from_dict(lineage.get("deployment"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("FP32 lineage has an invalid deployment contract") from exc
    if recorded_contract != contract:
        raise RuntimeError("FP32 lineage deployment contract does not match QAT")
    if lineage.get("best_weights_sha256") != sha256_file(fp32_checkpoint):
        raise RuntimeError("FP32 checkpoint digest does not match its lineage")
    if lineage.get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("QAT manifest digest does not match FP32 lineage")
    if lineage.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("dataset contents do not match FP32 lineage")
    if type(lineage.get("qualification_eligible")) is not bool:
        raise RuntimeError("FP32 lineage has no explicit qualification eligibility")
    if lineage.get("ultralytics_revision") != ULTRALYTICS_REVISION:
        raise RuntimeError("FP32 lineage Ultralytics revision does not match the QAT environment")
    if lineage.get("ultralytics_package_sha256") != ULTRALYTICS_PACKAGE_SHA256:
        raise RuntimeError("FP32 lineage Ultralytics package contents do not match the QAT environment")
    if lineage["qualification_eligible"] and lineage.get("base_weights_sha256") != YOLO26N_SHA256:
        raise RuntimeError("eligible FP32 lineage does not descend from the pinned YOLO26 base weights")
    return lineage


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    manifest: Path,
    fp32_checkpoint: Path,
    fp32_lineage: Path,
    dataset_sha256: str,
    contract: DeploymentContract,
    training_selection: dict[str, Any],
) -> None:
    """Reject a resume source that is not the same frozen production contract."""
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("resume checkpoint has an unsupported schema")
    if checkpoint.get("architecture") != "yolo26n-p2-attention-one-class":
        raise RuntimeError("resume checkpoint has the wrong architecture")
    expected_attention = {
        "matmul_count": EXPECTED_ATTENTION_MATMULS,
        "softmax_count": EXPECTED_ATTENTION_SOFTMAXES,
    }
    if checkpoint.get("attention_scaffolding") != expected_attention:
        raise RuntimeError("resume checkpoint lacks complete INT8 attention scaffolding evidence")
    try:
        resume_contract = DeploymentContract.from_dict(checkpoint.get("deployment"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("resume checkpoint has an invalid deployment contract") from exc
    if resume_contract != contract:
        raise RuntimeError(
            "resume checkpoint deployment shape does not match: "
            f"checkpoint={resume_contract.input_shape}, requested={contract.input_shape}"
        )
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
    if checkpoint.get("fp32_lineage_sha256") != sha256_file(fp32_lineage):
        raise RuntimeError("resume checkpoint FP32 lineage digest does not match")
    if checkpoint.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("resume checkpoint dataset digest does not match")
    if not isinstance(checkpoint.get("qat_state_dict"), dict):
        raise RuntimeError("resume checkpoint has no QAT state dictionary")
    if not isinstance(checkpoint.get("baseline"), dict) or not isinstance(checkpoint.get("final_metrics"), dict):
        raise RuntimeError("resume checkpoint has incomplete accuracy evidence")
    observer_selection = checkpoint.get("observer_selection")
    if not isinstance(observer_selection, dict) or not isinstance(observer_selection.get("images"), list):
        raise RuntimeError("resume checkpoint has incomplete observer evidence")
    if len(observer_selection["images"]) != int(checkpoint["observer_images"]):
        raise RuntimeError("resume checkpoint observer count does not match its evidence")
    recorded_selection = checkpoint.get("training_selection")
    if recorded_selection is None and training_selection["method"] == "full_train_split":
        return
    if recorded_selection != training_selection:
        raise RuntimeError("resume checkpoint QAT training selection does not match")


@dataclass(frozen=True)
class VerifiedQATSources:
    fp32_lineage: dict[str, Any]
    resume: dict[str, Any] | None
    detector: nn.Module
    teacher_detector: nn.Module | None
    fp32_checkpoint_sha256: str
    fp32_lineage_sha256: str
    resume_checkpoint_sha256: str | None
    ultralytics_version: str
    ultralytics_revision: str


def load_verified_qat_sources(
    args: argparse.Namespace,
    contract: DeploymentContract,
    dataset_sha256: str,
    training_selection: dict[str, Any],
) -> VerifiedQATSources:
    """Validate, snapshot, and fully consume every mutable QAT source."""
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    ultralytics_revision = require_ultralytics_revision(Path(ultralytics.__file__).resolve().parent)
    fp32_digest = sha256_file(args.fp32_checkpoint)
    lineage_digest = sha256_file(args.fp32_lineage)
    resume_digest = sha256_file(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    with tempfile.TemporaryDirectory(prefix="yolo26-qat-inputs-") as temporary:
        snapshot_dir = Path(temporary)
        fp32_snapshot = snapshot_file(args.fp32_checkpoint, snapshot_dir / "fp32.pt", fp32_digest)
        lineage_snapshot = snapshot_file(args.fp32_lineage, snapshot_dir / "lineage.json", lineage_digest)
        fp32_lineage = validate_fp32_lineage(
            lineage_snapshot,
            fp32_snapshot,
            args.manifest,
            contract,
            dataset_sha256,
        )
        resume = None
        if args.resume_checkpoint is not None and resume_digest is not None:
            resume_snapshot = snapshot_file(
                args.resume_checkpoint,
                snapshot_dir / "resume.pt",
                resume_digest,
            )
            resume = torch.load(resume_snapshot, map_location="cpu", weights_only=False)
            validate_resume_checkpoint(
                resume,
                args.manifest,
                fp32_snapshot,
                lineage_snapshot,
                dataset_sha256,
                contract,
                training_selection,
            )
            shutil.copyfile(resume_snapshot, args.output_dir / "checkpoint-best-frozen.pt")
        if resume is not None and resume.get("versions", {}).get("ultralytics_revision") != ultralytics_revision:
            raise RuntimeError("resume checkpoint Ultralytics revision does not match the environment")
        if (
            resume is not None
            and resume.get("versions", {}).get("ultralytics_package_sha256") != ULTRALYTICS_PACKAGE_SHA256
        ):
            raise RuntimeError("resume checkpoint Ultralytics package contents do not match the environment")
        if resume is not None and resume.get("versions", {}).get("swml_qat_revision") != SWML_QAT_REVISION:
            raise RuntimeError("resume checkpoint swml-qat revision does not match the environment")
        if resume is not None and resume.get("versions", {}).get("swml_qat_package_sha256") != SWML_QAT_PACKAGE_SHA256:
            raise RuntimeError("resume checkpoint swml-qat package contents do not match the environment")
        detector = YOLO(str(fp32_snapshot)).model
        detector.args = get_cfg(overrides=dict(getattr(detector, "args", {})))
        teacher_detector = None
        if args.class_distillation_weight > 0.0:
            teacher_detector = YOLO(str(fp32_snapshot)).model
            teacher_detector.args = get_cfg(overrides=dict(getattr(teacher_detector, "args", {})))
    return VerifiedQATSources(
        fp32_lineage=fp32_lineage,
        resume=resume,
        detector=detector,
        teacher_detector=teacher_detector,
        fp32_checkpoint_sha256=fp32_digest,
        fp32_lineage_sha256=lineage_digest,
        resume_checkpoint_sha256=resume_digest,
        ultralytics_version=ultralytics.__version__,
        ultralytics_revision=ultralytics_revision,
    )


def select_trainable_parameters(model: nn.Module, scope: str) -> tuple[list[nn.Parameter], int]:
    """Select full, detection-head, or class-head-only recovery parameters."""
    if scope not in {"all", "head", "classification"}:
        raise ValueError(f"unsupported trainable scope: {scope}")
    selected = []
    selected_elements = 0
    for name, parameter in model.named_parameters():
        in_head = name.startswith("detector_model_29_")
        in_class_head = in_head and ("_cv3_" in name or "_one2one_cv3_" in name)
        trainable = scope == "all" or (scope == "head" and in_head) or (scope == "classification" and in_class_head)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
            selected_elements += parameter.numel()
    if not selected:
        raise RuntimeError(f"trainable scope {scope} selected no parameters")
    return selected, selected_elements


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
    model_size: int,
) -> dict[str, float]:
    """Keep held-out validation data out of the QAT observer state."""
    if not observers_frozen:
        model.apply(disable_observer)
    try:
        return evaluate_model(model, head, data, device, model_size=model_size)
    finally:
        if not observers_frozen:
            model.apply(enable_observer)


def main() -> int:
    args = parse_args()
    contract = DeploymentContract(args.image_size)
    swml_qat_runtime = activate_verified_swml_qat_runtime()
    records = load_manifest(args.manifest)
    dataset_digest = dataset_content_sha256(args.manifest, records)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    training_records, training_selection = select_qat_training_records(records, args.max_train_images)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = load_verified_qat_sources(args, contract, dataset_digest, training_selection)
    fp32_lineage = sources.fp32_lineage
    resume = sources.resume
    observer_records = (
        representative_observer_records(records, args.observer_warmup_images, args.seed) if resume is None else []
    )
    train_data = loader(
        DroneDataset(
            training_records,
            "train",
            augment=True,
            seed=args.seed,
            model_size=contract.model_size,
        ),
        args,
        True,
    )
    observer_data = (
        loader(
            DroneDataset(
                observer_records,
                "train",
                augment=False,
                seed=args.seed,
                model_size=contract.model_size,
            ),
            args,
            False,
        )
        if resume is None
        else None
    )
    val_data = loader(
        DroneDataset(
            records,
            "val",
            augment=False,
            seed=args.seed,
            model_size=contract.model_size,
        ),
        args,
        False,
    )

    from sima_qat.qat_api import sima_prepare_qat_model

    ultralytics_revision = sources.ultralytics_revision
    detector = sources.detector
    fuse_batch_norm_for_qat(detector)
    detector, lowered_attention_count = lower_attention_for_mla(detector)
    if lowered_attention_count != 2:
        raise RuntimeError(f"expected two YOLO26 attention modules, lowered {lowered_attention_count}")
    detector.to(device).requires_grad_(True).train()
    criterion = detector.init_criterion()
    wrapped = Yolo26TrainingDeployWrapper(detector).to(device).train()
    baseline = (
        evaluate_model(
            wrapped,
            detector.model[-1],
            val_data,
            device,
            model_size=contract.model_size,
        )
        if resume is None
        else resume["baseline"]
    )
    example = (
        torch.rand(
            (args.batch_size, *contract.input_shape[1:]),
            device=device,
        ),
    )
    qat_model = sima_prepare_qat_model(wrapped, example, device)
    attention_scaffolding = validate_attention_qat_scaffolding(qat_model)
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
            model_size=contract.model_size,
        )
        start_epoch = 0
        history: list[dict[str, Any]] = []
        observers_frozen = False
        resume_digest = None
        resume_start_metrics = None
        resume_priority = None
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
        resume_digest = sources.resume_checkpoint_sha256
        resume_priority = (
            float(resume["final_metrics"]["ap50"]),
            float(resume["final_metrics"]["tiny_recall50"]),
        )

    teacher = None
    if args.class_distillation_weight > 0.0:
        teacher_detector = sources.teacher_detector
        if teacher_detector is None:
            raise RuntimeError("class distillation requested without a verified FP32 teacher")
        fuse_batch_norm_for_qat(teacher_detector)
        teacher_detector.to(device).requires_grad_(False).eval()
        teacher = Yolo26TrainingDeployWrapper(teacher_detector).to(device).eval()

    trainable_parameters, trainable_parameter_count = select_trainable_parameters(qat_model, args.trainable_scope)
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.learning_rate,
        momentum=0.937,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    freeze_after = math.floor(args.epochs * args.freeze_observers_fraction)
    best_frozen_priority: tuple[float, float] | None = resume_priority
    final_metrics: dict[str, float] = {}
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        if local_epoch > freeze_after and not observers_frozen:
            qat_model.apply(disable_observer)
            observers_frozen = True
        qat_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        distillation_total = 0.0
        started = time.monotonic()
        for step, batch in enumerate(train_data, start=1):
            images = batch["img"].to(device, non_blocking=True)
            for key in ("batch_idx", "cls", "bboxes"):
                batch[key] = batch[key].to(device, non_blocking=True)
            outputs = qat_model(images)
            validate_deploy_outputs(outputs["deploy"], images.shape[0], contract)
            loss, components = criterion(outputs["predictions"], batch)
            detection_loss = loss.sum()
            distillation_loss = torch.zeros((), device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_predictions = teacher(images)["predictions"]
                student_predictions = outputs["predictions"]
                distillation_loss = torch.stack(
                    [
                        F.smooth_l1_loss(
                            student_predictions[branch]["scores"],
                            teacher_predictions[branch]["scores"],
                        )
                        for branch in ("one2many", "one2one")
                    ]
                ).mean()
            scalar_loss = detection_loss + args.class_distillation_weight * distillation_loss
            if not torch.isfinite(scalar_loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}")
            window_start = ((step - 1) // args.accumulation_steps) * args.accumulation_steps
            window_size = min(args.accumulation_steps, len(train_data) - window_start)
            (scalar_loss / window_size).backward()
            if step % args.accumulation_steps == 0 or step == len(train_data):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 10.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(detection_loss.detach())
            distillation_total += float(distillation_loss.detach())
        scheduler.step()
        final_metrics = evaluate_without_observer_updates(
            qat_model,
            detector.model[-1],
            val_data,
            device,
            observers_frozen=observers_frozen,
            model_size=contract.model_size,
        )
        entry = {
            "epoch": epoch,
            "loss": loss_total / len(train_data),
            "class_distillation_loss": distillation_total / len(train_data),
            "class_distillation_weight": args.class_distillation_weight,
            "metrics": final_metrics,
            "observers_frozen": observers_frozen,
            "seconds": time.monotonic() - started,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        qualification_eligible = fp32_lineage["qualification_eligible"] and not args.smoke and observers_frozen
        if qualification_eligible:
            require_unchanged_dataset(
                args.manifest,
                records,
                manifest_sha256=fp32_lineage["manifest_sha256"],
                dataset_sha256=dataset_digest,
                context="QAT",
            )
        checkpoint = {
            "schema_version": 1,
            "architecture": "yolo26n-p2-attention-one-class",
            "deployment": contract.to_dict(),
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
            "training_selection": training_selection,
            "attention_scaffolding": attention_scaffolding,
            "manifest_sha256": fp32_lineage["manifest_sha256"],
            "dataset_sha256": dataset_digest,
            "fp32_checkpoint_sha256": sources.fp32_checkpoint_sha256,
            "fp32_lineage_sha256": sources.fp32_lineage_sha256,
            "fp32_qualification_eligible": fp32_lineage["qualification_eligible"],
            "resume_checkpoint_sha256": resume_digest,
            "trainable_scope": args.trainable_scope,
            "trainable_parameter_count": trainable_parameter_count,
            "qualification_eligible": qualification_eligible,
            "versions": {
                "torch": torch.__version__,
                "ultralytics": sources.ultralytics_version,
                "ultralytics_revision": ultralytics_revision,
                "ultralytics_package_sha256": ULTRALYTICS_PACKAGE_SHA256,
                "swml_qat": qat_package_version(),
                "swml_qat_revision": SWML_QAT_REVISION,
                "swml_qat_package_sha256": swml_qat_runtime.package_sha256,
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
