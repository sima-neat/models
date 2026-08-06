#!/usr/bin/env python3
"""Train a YOLO26n-P2 teacher, then recover the INT8-safe student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import DEFAULT_MODEL_SIZE, DeploymentContract
from dataset import load_manifest, sha256_file, write_ultralytics_dataset
from provenance import ULTRALYTICS_REVISION, YOLO26N_SHA256, require_ultralytics_revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--teacher-init-weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--recovery-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.teacher_epochs, args.recovery_epochs, args.batch_size) < 1:
        parser.error("teacher epochs, recovery epochs, and batch size must be positive")
    if args.workers < 0:
        parser.error("workers must be >= 0")
    try:
        contract = DeploymentContract(args.image_size)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.base_weights.is_file():
        parser.error(f"base weights do not exist: {args.base_weights}")
    if args.teacher_init_weights is not None and not args.teacher_init_weights.is_file():
        parser.error(f"teacher initialization does not exist: {args.teacher_init_weights}")
    digest = sha256_file(args.base_weights)
    if digest != YOLO26N_SHA256 and not args.smoke:
        parser.error(f"base weight SHA-256 mismatch: expected {YOLO26N_SHA256}, got {digest}")

    records = load_manifest(args.manifest)
    train_count = sum(record.split == "train" for record in records)
    val_count = sum(record.split == "val" for record in records)
    if (train_count < 128 or val_count < 1) and not args.smoke:
        parser.error("production training requires at least 128 train frames and a non-empty validation split")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    generated = args.output_dir / "dataset"
    data_yaml = write_ultralytics_dataset(args.manifest, generated)

    import ultralytics
    from ultralytics import YOLO

    package_root = Path(ultralytics.__file__).resolve().parent
    require_ultralytics_revision(package_root)
    if not (package_root / "cfg" / "models" / "26" / "yolo26-p2.yaml").is_file():
        raise RuntimeError("pinned Ultralytics package does not contain YOLO26-P2")
    teacher_architecture = package_root / "cfg" / "models" / "26" / "yolo26-p2.yaml"
    student_architecture = Path(__file__).with_name("yolo26n-p2-int8.yaml")
    if not student_architecture.is_file():
        raise RuntimeError(f"INT8-safe student architecture is missing: {student_architecture}")

    common = {
        "data": str(data_yaml),
        "imgsz": contract.model_size,
        "batch": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "deterministic": True,
        "single_cls": True,
        "optimizer": "MuSGD",
        "close_mosaic": 10,
        "project": str(args.output_dir),
        "exist_ok": False,
    }
    teacher_initialization = args.teacher_init_weights or args.base_weights
    if args.teacher_init_weights is None:
        teacher = YOLO(str(teacher_architecture)).load(str(teacher_initialization))
    else:
        teacher = YOLO(str(teacher_initialization))
        head = teacher.model.model[-1]
        checkpoint_classes = int(teacher.model.yaml.get("nc", -1))
        feature_heads = len(getattr(head, "cv2", ()))
        if checkpoint_classes != 1 or feature_heads != 4:
            raise RuntimeError(
                "teacher initialization must be a one-class YOLO26-P2 checkpoint: "
                f"classes={checkpoint_classes}, feature_heads={feature_heads}"
            )
    teacher_results = teacher.train(
        **common,
        epochs=args.teacher_epochs,
        name="teacher",
    )
    teacher_best = args.output_dir / "teacher" / "weights" / "best.pt"
    if not teacher_best.is_file():
        raise RuntimeError(f"teacher training did not produce {teacher_best}")

    # This is the graph-surgery boundary. Matching P2 backbone, neck, and head
    # tensors transfer from the converged teacher; only the two attention sites
    # are reinitialized as compiler-friendly convolutional blocks.
    student = YOLO(str(student_architecture)).load(str(teacher_best))
    student_results = student.train(
        **common,
        epochs=args.recovery_epochs,
        name="student",
    )
    best = args.output_dir / "student" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"student recovery did not produce {best}")
    lineage = {
        "schema_version": 1,
        "architecture": "yolo26n-p2-int8-conv-one-class",
        "deployment": contract.to_dict(),
        "base_weights_sha256": digest,
        "teacher_initialization": {
            "file": teacher_initialization.name,
            "sha256": sha256_file(teacher_initialization),
            "warm_started": args.teacher_init_weights is not None,
        },
        "teacher_weights_sha256": sha256_file(teacher_best),
        "best_weights_sha256": sha256_file(best),
        "manifest_sha256": sha256_file(args.manifest),
        "qualification_eligible": not args.smoke,
        "split_counts": {split: sum(record.split == split for record in records) for split in ("train", "val", "test")},
        "ultralytics_revision": ULTRALYTICS_REVISION,
        "teacher_results": getattr(teacher_results, "results_dict", {}),
        "student_results": getattr(student_results, "results_dict", {}),
    }
    (args.output_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
