#!/usr/bin/env python3
"""Materialize the Anti-UAV Track 3 corpus with sequence-disjoint splits."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dataset import sha256_file

DATASET_URL = "https://huggingface.co/datasets/wish44165/StrongBaseline_YOLOv12-BoT-SORT-ReID"
ARCHIVE_SHA256 = "c98069bf37b60b2fdc35f12e3514d619e7e06e550c90575500c2087bb2077074"
SEQUENCE_PATTERN = re.compile(r"^(MultiUAV-\d{3})_(\d+)\.jpg$")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass(frozen=True)
class SourceFrame:
    image: Path
    label: Path | None
    sequence: str
    frame_number: int
    boxes: int
    clipped_boxes: int
    dropped_boxes: int
    duplicate_boxes: int
    maximum_boundary_overflow: float


@dataclass(frozen=True)
class SequenceStats:
    sequence: str
    frames: int
    boxes: int


def _label_values(path: Path) -> list[tuple[float, float, float, float]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected class and normalized xywh")
        try:
            class_id, x, y, width, height = (float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: label contains a non-numeric value") from exc
        if class_id != 0.0:
            raise ValueError(f"{path}:{line_number}: expected the one-class id 0")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"{path}:{line_number}: normalized values must be finite")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: box dimensions must be in (0, 1]")
        values.append((x, y, width, height))
    return values


def _validate_label(path: Path) -> tuple[int, int, int, int, float]:
    boxes = 0
    clipped_boxes = 0
    dropped_boxes = 0
    duplicate_boxes = 0
    maximum_overflow = 0.0
    seen: set[tuple[float, float, float, float]] = set()
    for x, y, width, height in _label_values(path):
        overflow = max(
            0.0,
            width / 2 - x,
            x + width / 2 - 1,
            height / 2 - y,
            y + height / 2 - 1,
        )
        maximum_overflow = max(maximum_overflow, overflow)
        clipped_width = min(1.0, x + width / 2) - max(0.0, x - width / 2)
        clipped_height = min(1.0, y + height / 2) - max(0.0, y - height / 2)
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            dropped_boxes += 1
            continue
        canonical = (
            (max(0.0, x - width / 2) + min(1.0, x + width / 2)) / 2,
            (max(0.0, y - height / 2) + min(1.0, y + height / 2)) / 2,
            clipped_width,
            clipped_height,
        )
        if canonical in seen:
            duplicate_boxes += 1
            continue
        seen.add(canonical)
        clipped_boxes += int(overflow > 0.0)
        boxes += 1
    return boxes, clipped_boxes, dropped_boxes, duplicate_boxes, maximum_overflow


def _canonical_label_text(path: Path) -> str:
    lines = []
    seen: set[tuple[float, float, float, float]] = set()
    for x, y, width, height in _label_values(path):
        left = max(0.0, x - width / 2)
        right = min(1.0, x + width / 2)
        top = max(0.0, y - height / 2)
        bottom = min(1.0, y + height / 2)
        clipped_width = right - left
        clipped_height = bottom - top
        if clipped_width <= 0.0 or clipped_height <= 0.0:
            continue
        canonical = ((left + right) / 2, (top + bottom) / 2, clipped_width, clipped_height)
        if canonical in seen:
            continue
        seen.add(canonical)
        lines.append(f"0 {(left + right) / 2:.9f} {(top + bottom) / 2:.9f} {clipped_width:.9f} {clipped_height:.9f}")
    return "\n".join(lines) + ("\n" if lines else "")


def discover_frames(source_root: Path) -> list[SourceFrame]:
    """Merge the publisher's leaky frame splits and validate every pair."""
    by_name: dict[str, SourceFrame] = {}
    for publisher_split in ("train", "val"):
        directory = source_root / publisher_split
        if not directory.is_dir():
            raise ValueError(f"missing publisher split directory: {directory}")
        for image in sorted(directory.glob("*.jpg")):
            match = SEQUENCE_PATTERN.fullmatch(image.name)
            if match is None:
                raise ValueError(f"unexpected image name: {image}")
            if image.stem in by_name:
                raise ValueError(f"duplicate frame across publisher splits: {image.stem}")
            label = image.with_suffix(".txt")
            boxes, clipped_boxes, dropped_boxes, duplicate_boxes, maximum_overflow = (
                _validate_label(label) if label.is_file() else (0, 0, 0, 0, 0.0)
            )
            by_name[image.stem] = SourceFrame(
                image=image,
                label=label if label.is_file() else None,
                sequence=match.group(1),
                frame_number=int(match.group(2)),
                boxes=boxes,
                clipped_boxes=clipped_boxes,
                dropped_boxes=dropped_boxes,
                duplicate_boxes=duplicate_boxes,
                maximum_boundary_overflow=maximum_overflow,
            )
        image_stems = {path.stem for path in directory.glob("*.jpg")}
        orphan_labels = sorted(path for path in directory.glob("*.txt") if path.stem not in image_stems)
        if orphan_labels:
            raise ValueError(f"label without an image: {orphan_labels[0]}")
    if not by_name:
        raise ValueError("the source dataset contains no JPEG frames")
    return sorted(by_name.values(), key=lambda frame: (frame.sequence, frame.frame_number))


def summarize_sequences(frames: list[SourceFrame]) -> list[SequenceStats]:
    totals: dict[str, list[int]] = {}
    for frame in frames:
        aggregate = totals.setdefault(frame.sequence, [0, 0])
        aggregate[0] += 1
        aggregate[1] += frame.boxes
    return [
        SequenceStats(sequence, frames_and_boxes[0], frames_and_boxes[1])
        for sequence, frames_and_boxes in sorted(totals.items())
    ]


def split_sequences(
    sequences: list[SequenceStats],
    seed: int,
    fixed: dict[str, str] | None = None,
) -> dict[str, str]:
    """Greedily balance whole sequences by sequence, frame, and box counts."""
    if len(sequences) < len(SPLIT_RATIOS):
        raise ValueError("at least three source sequences are required")
    totals = {
        "sequences": len(sequences),
        "frames": sum(item.frames for item in sequences),
        "boxes": sum(item.boxes for item in sequences),
    }
    by_name = {item.sequence: item for item in sequences}
    fixed = dict(fixed or {})
    unknown = sorted(set(fixed) - set(by_name))
    if unknown:
        raise ValueError(f"fixed split references an unknown sequence: {unknown[0]}")
    invalid_splits = sorted(set(fixed.values()) - set(SPLIT_RATIOS))
    if invalid_splits:
        raise ValueError(f"fixed sequence has an invalid split: {invalid_splits[0]}")
    rng = random.Random(seed)
    tie_breakers = {item.sequence: rng.random() for item in sequences}
    ordered = sorted(
        (item for item in sequences if item.sequence not in fixed),
        key=lambda item: (-item.boxes, -item.frames, tie_breakers[item.sequence], item.sequence),
    )
    current = {split: {"sequences": 0, "frames": 0, "boxes": 0} for split in SPLIT_RATIOS}
    assignments: dict[str, str] = dict(fixed)
    for sequence, split in fixed.items():
        item = by_name[sequence]
        current[split]["sequences"] += 1
        current[split]["frames"] += item.frames
        current[split]["boxes"] += item.boxes

    def score(candidate: str, item: SequenceStats) -> float:
        result = 0.0
        for split, ratio in SPLIT_RATIOS.items():
            for metric, increment in (
                ("sequences", 1),
                ("frames", item.frames),
                ("boxes", item.boxes),
            ):
                value = current[split][metric]
                if split == candidate:
                    value += increment
                target = totals[metric] * ratio
                result += ((value - target) / target) ** 2
        return result

    for item in ordered:
        selected = min(SPLIT_RATIOS, key=lambda split: (score(split, item), split))
        assignments[item.sequence] = selected
        current[selected]["sequences"] += 1
        current[selected]["frames"] += item.frames
        current[selected]["boxes"] += item.boxes
    if set(assignments.values()) != set(SPLIT_RATIOS):
        raise RuntimeError("sequence balancing produced an empty split")
    return assignments


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def materialize(
    frames: list[SourceFrame],
    assignments: dict[str, str],
    destination: Path,
    *,
    archive: Path,
    seed: int,
    fixed_assignments: dict[str, str],
) -> Path:
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}")
    if staging.exists():
        raise ValueError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    manifest_frames = []
    split_statistics = {
        split: {
            "sequences": set(),
            "frames": 0,
            "boxes": 0,
            "negative_frames": 0,
            "missing_label_frames": 0,
            "boundary_clipped_boxes": 0,
            "non_intersecting_boxes_dropped": 0,
            "duplicate_boxes_removed": 0,
            "maximum_boundary_overflow": 0.0,
        }
        for split in SPLIT_RATIOS
    }
    try:
        for frame in frames:
            split = assignments[frame.sequence]
            relative_image = Path("images") / frame.sequence / frame.image.name
            relative_label = Path("labels") / frame.sequence / f"{frame.image.stem}.txt"
            _hardlink(frame.image, staging / relative_image)
            label_destination = staging / relative_label
            if frame.label is None:
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.touch()
            elif frame.clipped_boxes or frame.dropped_boxes or frame.duplicate_boxes:
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.write_text(_canonical_label_text(frame.label), encoding="utf-8")
            else:
                _hardlink(frame.label, label_destination)
            manifest_frames.append(
                {
                    "image": relative_image.as_posix(),
                    "label": relative_label.as_posix(),
                    "sequence": frame.sequence,
                    "split": split,
                }
            )
            stats = split_statistics[split]
            stats["sequences"].add(frame.sequence)
            stats["frames"] += 1
            stats["boxes"] += frame.boxes
            stats["negative_frames"] += int(frame.boxes == 0)
            stats["missing_label_frames"] += int(frame.label is None)
            stats["boundary_clipped_boxes"] += frame.clipped_boxes
            stats["non_intersecting_boxes_dropped"] += frame.dropped_boxes
            stats["duplicate_boxes_removed"] += frame.duplicate_boxes
            stats["maximum_boundary_overflow"] = max(
                stats["maximum_boundary_overflow"], frame.maximum_boundary_overflow
            )
        serializable_stats = {
            split: {
                **stats,
                "sequences": len(stats["sequences"]),
            }
            for split, stats in split_statistics.items()
        }
        manifest = {
            "schema_version": 1,
            "source": {
                "name": "Anti-UAV Track 3 train/validation",
                "url": DATASET_URL,
                "license": "CC-BY-4.0",
                "archive": archive.name,
                "archive_sha256": ARCHIVE_SHA256,
            },
            "split_policy": {
                "method": "whole_sequence_greedy_balance",
                "seed": seed,
                "ratios": SPLIT_RATIOS,
                "balance_metrics": ["sequences", "frames", "boxes"],
                "fixed_assignments": fixed_assignments,
            },
            "annotation_policy": {
                "missing_label": "explicit_empty_label",
                "out_of_frame_box": "clip_xyxy_then_recompute_normalized_xywh",
                "non_intersecting_box": "drop",
                "duplicate_box": "keep_first_exact_canonical_box",
                "canonical_decimal_places": 9,
            },
            "split_statistics": serializable_stats,
            "frames": manifest_frames,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--test-sequence",
        action="append",
        default=[],
        help="source sequence to force into the held-out test split; repeatable",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    archive = args.source_archive.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if not archive.is_file():
        parser.error(f"source archive does not exist: {archive}")
    digest = sha256_file(archive)
    if digest != ARCHIVE_SHA256:
        parser.error(f"source archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {digest}")
    frames = discover_frames(source_root)
    sequences = summarize_sequences(frames)
    fixed_assignments = {sequence: "test" for sequence in args.test_sequence}
    assignments = split_sequences(sequences, args.seed, fixed_assignments)
    manifest = materialize(
        frames,
        assignments,
        destination,
        archive=archive,
        seed=args.seed,
        fixed_assignments=fixed_assignments,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "frames": len(frames),
                "boxes": sum(frame.boxes for frame in frames),
                "sequences": len(sequences),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
