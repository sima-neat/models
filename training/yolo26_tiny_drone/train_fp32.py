#!/usr/bin/env python3
"""Train the canonical deployable YOLO26n-P2 graph in FP32."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from contract import DEFAULT_MODEL_SIZE, DeploymentContract
from dataset import (
    dataset_content_sha256,
    load_manifest,
    require_unchanged_dataset,
    sha256_file,
    write_ultralytics_dataset,
)
from provenance import ULTRALYTICS_REVISION, YOLO26N_SHA256, require_ultralytics_revision


def snapshot_file(source: Path, destination: Path, expected_sha256: str) -> Path:
    """Copy exactly the bytes that passed a provenance check and make them read-only."""
    shutil.copyfile(source, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise RuntimeError(
            f"source changed while creating its private snapshot: expected {expected_sha256}, got {actual}"
        )
    destination.chmod(0o400)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch size must be positive")
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
    if args.initial_weights is not None and not args.initial_weights.is_file():
        parser.error(f"initial weights do not exist: {args.initial_weights}")
    digest = sha256_file(args.base_weights)
    if digest != YOLO26N_SHA256 and not args.smoke:
        parser.error(f"base weight SHA-256 mismatch: expected {YOLO26N_SHA256}, got {digest}")

    manifest_digest = sha256_file(args.manifest)
    records = load_manifest(args.manifest)
    if sha256_file(args.manifest) != manifest_digest:
        raise RuntimeError("manifest changed while loading FP32 dataset records")
    dataset_digest = dataset_content_sha256(args.manifest, records)
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
    architecture = Path(__file__).with_name("yolo26n-p2-deploy.yaml")
    if not architecture.is_file():
        raise RuntimeError(f"canonical deployment architecture is missing: {architecture}")
    architecture_digest = sha256_file(architecture)

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
    initialization = args.initial_weights or args.base_weights
    initialization_digest = digest if initialization == args.base_weights else sha256_file(initialization)
    with tempfile.TemporaryDirectory(prefix="yolo26-fp32-inputs-") as temporary:
        snapshot_dir = Path(temporary)
        architecture_snapshot = snapshot_file(
            architecture,
            snapshot_dir / architecture.name,
            architecture_digest,
        )
        initialization_snapshot = snapshot_file(
            initialization,
            snapshot_dir / initialization.name,
            initialization_digest,
        )
        source = YOLO(str(initialization_snapshot))
        if args.initial_weights is not None:
            head = source.model.model[-1]
            checkpoint_classes = int(source.model.yaml.get("nc", -1))
            feature_heads = len(getattr(head, "cv2", ()))
            if checkpoint_classes != 1 or feature_heads != 4:
                raise RuntimeError(
                    "initial weights must be a one-class YOLO26-P2 checkpoint: "
                    f"classes={checkpoint_classes}, feature_heads={feature_heads}"
                )
        model = YOLO(str(architecture_snapshot))
        source_state = source.model.float().state_dict()
        target_state = model.model.state_dict()
        compatible = {
            name
            for name, tensor in source_state.items()
            if name in target_state and tensor.shape == target_state[name].shape
        }
        if not compatible:
            raise RuntimeError("initial weights contain no tensors compatible with the deployment graph")
        model.load(str(initialization_snapshot))
    results = model.train(
        **common,
        epochs=args.epochs,
        name="fp32",
    )
    best = args.output_dir / "fp32" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"FP32 training did not produce {best}")
    require_unchanged_dataset(
        args.manifest,
        records,
        manifest_sha256=manifest_digest,
        dataset_sha256=dataset_digest,
        context="FP32 training",
    )
    lineage = {
        "schema_version": 2,
        "architecture": "yolo26n-p2-attention-one-class",
        "architecture_sha256": architecture_digest,
        "deployment": contract.to_dict(),
        "base_weights_sha256": digest,
        "initialization": {
            "file": initialization.name,
            "sha256": initialization_digest,
            "task_specific": args.initial_weights is not None,
            "compatible_tensors": len(compatible),
            "target_tensors": len(target_state),
        },
        "best_weights_sha256": sha256_file(best),
        "manifest_sha256": manifest_digest,
        "dataset_sha256": dataset_digest,
        "qualification_eligible": not args.smoke,
        "split_counts": {split: sum(record.split == split for record in records) for split in ("train", "val", "test")},
        "ultralytics_revision": ULTRALYTICS_REVISION,
        "fp32_results": getattr(results, "results_dict", {}),
    }
    (args.output_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
